"""Explicitly install and verify the local PyTorch backend using uv (no driver installs)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]

# Executed only in a fresh interpreter, after uv has finished replacing packages.
VERIFY_SCRIPT = r'''
import json
import sys

def verify():
    import torch
    import torchvision

    expected, requested = json.loads(sys.argv[1]), sys.argv[2]
    print(f"Runtime builds: torch {torch.__version__}; torchvision {torchvision.__version__}", flush=True)
    tags = []
    for name, module in (("torch", torch), ("torchvision", torchvision)):
        public, _, tag = module.__version__.partition("+")
        if public != expected[name]:
            raise RuntimeError(f"{name} version mismatch: expected {expected[name]}, got {module.__version__}")
        tags.append(tag)
    cuda, hip = torch.version.cuda, torch.version.hip
    print(f"Runtime CUDA: {cuda}; HIP: {hip}", flush=True)
    if getattr(torch.version, "xpu", None) or any("xpu" in tag for tag in tags):
        raise RuntimeError("XPU builds are unsupported by the project's hardware policy; rerun with --backend cpu")
    if hip:
        kind = "rocm"
        if tags[0] != tags[1] or not tags[0].startswith("rocm"):
            raise RuntimeError("torch/torchvision ROCm backend build mismatch")
        # HIP includes compiler/build details; compare the runtime major/minor only.
        if tags[0][4:].split(".")[:2] != hip.split(".")[:2]:
            raise RuntimeError("ROCm wheel tag does not match the HIP runtime")
    elif cuda:
        kind = "cuda"
        if tags != ["cu" + cuda.replace(".", "")] * 2:
            raise RuntimeError("torch/torchvision CUDA backend build mismatch")
    else:
        kind = "cpu"
        if any(tag not in ("", "cpu") for tag in tags):
            raise RuntimeError("torch/torchvision CPU backend build mismatch")
    if requested == "cpu" and kind != "cpu":
        raise RuntimeError("CPU requested but an accelerator build was installed")
    if requested not in ("auto", "cpu"):
        # uv validates backend names. Compare its requested identifier to the installed tag.
        def normalized(value):
            if value.startswith("rocm"):
                parts = value[4:].split(".")
                while parts and parts[-1] == "0":
                    parts.pop()
                return "rocm" + ".".join(parts)
            return value
        if normalized(requested) != normalized(tags[0]):
            raise RuntimeError(f"Requested backend {requested} does not match installed build {tags[0] or kind}")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)] if available else []
    print(f"Visible CUDA/ROCm GPUs: {count}; names: {names}; runtime available: {available}", flush=True)
    if kind in ("cuda", "rocm") and (not available or not count):
        reason = "no usable visible GPU"
        try:
            torch.cuda.init()
        except Exception as error:
            reason = str(error)
        raise RuntimeError(f"Accelerator build installed but unavailable: {reason}. Check drivers and "
                           "CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES, or rerun with --backend cpu")
    mps = getattr(torch.backends, "mps", None)
    device = ("cpu" if requested == "cpu" else "cuda:0" if count
              else "mps" if mps and mps.is_available() else "cpu")
    torch.set_num_threads(2)
    def tensor_check(target):
        value = (torch.tensor([1., 2.], device=target) * 2).sum().item()
        if value != 6.:
            raise RuntimeError(f"Tensor computation failed on {target}: {value}")
    def nms_check(target):
        boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device=target)
        scores = torch.tensor([.9, .8], device=target)
        kept = torchvision.ops.nms(boxes, scores, .5).tolist()
        if kept != [0]:
            raise RuntimeError(f"Native torchvision NMS failed on {target}: {kept}")
    tensor_check("cpu")
    nms_check("cpu")
    if device != "cpu":
        tensor_check(device)
        if device != "mps":
            nms_check(device)
    print(f"Verified device: {device}; tensor compute passed; native torchvision NMS passed on "
          + ("cpu (MPS tensor verified separately; MPS NMS not claimed)" if device == "mps"
             else "cpu and " + device if device != "cpu" else "cpu"), flush=True)

try:
    verify()
except Exception as error:
    print(f"Verification failed: {error}. Close Python processes using torch and rerun setup; "
          "use --backend cpu for a CPU-only recovery.", file=sys.stderr, flush=True)
    sys.exit(1)
'''


def project_interpreter(root: Path) -> str:
    """Refuse system/conda environments and interpreters outside the real checkout venv."""
    expected = root / ".venv"
    executable = Path(os.path.abspath(sys.executable))
    bindir = expected / ("Scripts" if os.name == "nt" else "bin")
    if (sys.prefix == sys.base_prefix or expected.resolve() != expected
            or Path(sys.prefix).resolve() != expected
            or executable.parent != bindir or not executable.is_file()
            or not (expected / "pyvenv.cfg").is_file() or (expected / "conda-meta").exists()):
        raise ValueError(f"Run setup with this project's real .venv interpreter: {bindir / 'python'}; "
                         "system, conda, redirected .venv, and other environments are refused")
    return str(executable)


def requirements(root: Path) -> list[str]:
    """Read the two unambiguous exact public pins from project metadata, never installed state."""
    with (root / "pyproject.toml").open("rb") as source:
        dependencies = tomllib.load(source)["project"]["dependencies"]
    result = []
    for name in ("torch", "torchvision"):
        matches = [dep for dep in dependencies
                   if re.match(rf"^{name}(?:\s|[<>=!~;\[@]|$)", dep, re.IGNORECASE)]
        if len(matches) != 1 or not re.fullmatch(rf"{name}==\d+\.\d+\.\d+", matches[0]):
            raise ValueError(f"pyproject.toml must contain exactly one public {name}==X.Y.Z pin")
        result.append(matches[0])
    return result


def backend_name(value: str) -> str:
    """Reject option injection; uv remains the authority on supported backend identifiers."""
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*", value):
        raise argparse.ArgumentTypeError("backend must be a plain uv backend name, e.g. auto, cpu, cu130")
    return value


def subprocess_environment() -> dict[str, str]:
    """Copy transport/trust settings without inheriting uv installation policy."""
    safe_uv = {
        "UV_SYSTEM_CERTS", "UV_NATIVE_TLS", "UV_OFFLINE", "UV_CACHE_DIR", "UV_NO_CACHE",
        "UV_HTTP_TIMEOUT", "UV_HTTP_CONNECT_TIMEOUT", "UV_HTTP_RETRIES", "UV_REQUEST_TIMEOUT",
        "UV_NO_PROGRESS", "UV_NO_WRAP",
    }
    # uv ignores pip-specific settings today; don't leak destination controls to children.
    pip_destination = {"PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_USER", "PIP_CONFIG_FILE", "PYTHONUSERBASE"}
    return {name: value for name, value in os.environ.copy().items()
            if (not name.upper().startswith("UV_") or name.upper() in safe_uv)
            and name.upper() not in pip_destination}


def main(argv: list[str] | None = None) -> int:
    """Install the exact pair only on explicit invocation; dry-run delegates resolution to uv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=backend_name, default="auto",
                        help="uv torch backend (default: auto); cpu forces CPU wheels")
    parser.add_argument("--dry-run", action="store_true", help="Resolve with uv without installing or verifying")
    args = parser.parse_args(argv)
    try:
        python = project_interpreter(ROOT)
        pins = requirements(ROOT)
        uv = shutil.which("uv")
        if uv is None:
            raise ValueError("uv is missing from PATH. Install uv 0.11.25 yourself, then rerun setup")
        # --python does not override target/prefix. Disable discovery AND UV_CONFIG_FILE.
        env = subprocess_environment()
        command = [uv, "--no-config", "pip", "install", "--python", python, "--torch-backend", args.backend,
                   "--reinstall-package", "torch", "--reinstall-package", "torchvision", *pins]
        if args.dry_run:
            command.append("--dry-run")
        print(f"Target environment: {ROOT / '.venv'}", flush=True)
        print("Command: " + (subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)), flush=True)
        result = subprocess.run(command, check=False, env=env)
        if result.returncode:
            print("Setup install failed. Check uv's backend/wheel/driver diagnostics above; no fallback was "
                  "attempted. On Windows close processes holding torch DLLs before retrying. "
                  "Replacement is not transactional; rerun setup (or --backend cpu) to recover.", file=sys.stderr)
            return result.returncode if result.returncode > 0 else 1
        if args.dry_run:
            print("Dry-run only: no packages installed; runtime verification skipped.", flush=True)
            return 0
        versions = dict(pin.split("==") for pin in pins)
        result = subprocess.run([python, "-I", "-c", VERIFY_SCRIPT, json.dumps(versions), args.backend],
                                check=False, timeout=120, env=env)
        if result.returncode:
            print("Setup runtime verification failed; installation alone does not establish GPU usability. "
                  "Check diagnostics above; rerun setup or use --backend cpu.", file=sys.stderr)
        return result.returncode if result.returncode >= 0 else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"Setup failed: {error}. Use the project's .venv, check pyproject.toml and uv, "
              "then rerun setup. No automatic driver/tool installation is performed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())