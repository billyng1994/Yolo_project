"""Installer contracts plus real offline uv dry-runs; no package installation."""

from __future__ import annotations

import builtins
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from safety_yolo import setup


@pytest.fixture
def installer(monkeypatch, tmp_path):
    """Use a fake tool path and process boundary, not the live package installer."""
    for name in tuple(os.environ):
        if name.upper().startswith(("UV_", "PIP_")) or name.upper() == "PYTHONUSERBASE":
            monkeypatch.delenv(name)
    monkeypatch.setenv("UV_SYSTEM_CERTS", "true")
    python = str(tmp_path / ".venv" / "python")
    uv = str(tmp_path / "tool with spaces" / "uv.exe")
    calls = []
    monkeypatch.setattr(setup, "project_interpreter", lambda root: python)
    monkeypatch.setattr(setup.shutil, "which", lambda name: uv if name == "uv" else None)
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(setup.subprocess, "run", run)
    return python, uv, calls


@pytest.mark.parametrize("backend", ["auto", "cpu", "cu130", "rocm7.2", "xpu", "unknown"])
def test_command_and_fresh_verification(installer, capsys, backend):
    python, uv, calls = installer
    assert setup.main(["--backend", backend]) == 0
    pins = setup.requirements(setup.ROOT)
    assert calls[0] == ([uv, "--no-config", "pip", "install", "--python", python, "--torch-backend", backend,
                         "--reinstall-package", "torch", "--reinstall-package", "torchvision", *pins],
                        {"check": False, "env": dict(os.environ)})
    command, kwargs = calls[1]
    assert command[:4] == [python, "-I", "-c", setup.VERIFY_SCRIPT]
    assert json.loads(command[4]) == dict(pin.split("==") for pin in pins)
    assert command[5] == backend and kwargs == {"check": False, "timeout": 120, "env": dict(os.environ)}
    output = capsys.readouterr().out
    assert "Target environment:" in output and "Command:" in output
    assert "--no-config pip install" in output
    assert "--system" not in calls[0][0] and "shell" not in calls[0][1]


def test_command_printed_before_install_and_always_reinstalls(installer, monkeypatch, capsys):
    """Same public CPU versions must not bypass backend replacement."""
    def run(command, **kwargs):
        assert "Command:" in capsys.readouterr().out
        assert command.count("--reinstall-package") == 2
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.main(["--backend", "cpu", "--dry-run"]) == 0


def test_dry_run_never_verifies(installer, capsys):
    assert setup.main(["--dry-run"]) == 0
    assert len(installer[2]) == 1 and installer[2][0][0][-1] == "--dry-run"
    assert "verification skipped" in capsys.readouterr().out


def test_subprocess_environment_is_copied_and_destination_controls_removed(installer, monkeypatch, tmp_path):
    """Both children retain transport/trust, not inherited installation policy."""
    unsafe = {
        name: str(tmp_path / name) for name in (
            "UV_CONFIG_FILE", "UV_TARGET", "UV_PREFIX", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT",
            "UV_PROJECT", "UV_WORKING_DIR", "UV_WORKING_DIRECTORY", "UV_FUTURE_DESTINATION",
            "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_CONFIG_FILE", "PYTHONUSERBASE",
        )
    }
    unsafe.update(UV_SYSTEM_PYTHON="true", UV_BREAK_SYSTEM_PACKAGES="true", PIP_USER="true",
                  UV_NO_CONFIG="false", UV_INSECURE_HOST="example.invalid")
    safe = {
        "UV_SYSTEM_CERTS": "true", "UV_NATIVE_TLS": "true", "UV_OFFLINE": "true",
        "UV_CACHE_DIR": str(tmp_path / "cache"), "UV_NO_CACHE": "false", "UV_NO_PROGRESS": "1",
        "UV_HTTP_TIMEOUT": "30", "UV_HTTP_CONNECT_TIMEOUT": "10", "UV_HTTP_RETRIES": "2",
        "UV_REQUEST_TIMEOUT": "30", "UV_NO_WRAP": "1",
        "HTTPS_PROXY": "http://proxy.example.invalid:8080", "HTTP_PROXY": "http://proxy.example.invalid:8080",
        "ALL_PROXY": "http://proxy.example.invalid:8080", "NO_PROXY": "localhost",
        "SSL_CERT_FILE": str(tmp_path / "ca.pem"), "SSL_CERT_DIR": str(tmp_path / "certs"),
        "SSL_CLIENT_CERT": str(tmp_path / "client.pem"), "REQUESTS_CA_BUNDLE": str(tmp_path / "ca.pem"),
        "CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0",
    }
    for name, value in (unsafe | safe).items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    assert setup.main([]) == 0
    assert dict(os.environ) == before
    child = installer[2][0][1].get("env")
    assert child is not None, "uv must receive an explicit isolated child environment"
    assert child is not os.environ
    assert installer[2][1][1]["env"] is child
    assert not unsafe.keys() & child.keys()
    assert all(child[name] == value for name, value in safe.items())
    assert child == {name: value for name, value in before.items() if name not in unsafe}


@pytest.mark.parametrize("setting", ["target", "prefix"])
@pytest.mark.parametrize("source", ["explicit", "project", "user"])
def test_real_uv_dry_run_ignores_destination_config(tmp_path, source, setting):
    """Exercise native config parsing/discovery against repo pins, never install."""
    if shutil.which("uv") is None:
        pytest.skip("real uv executable is not installed")
    try:
        python = setup.project_interpreter(setup.ROOT)
    except ValueError:
        pytest.skip("real uv regression requires this checkout's .venv interpreter")
    # Keep the existing wheel cache and real transport settings; fixtures affect only a child.
    env = {name: value for name, value in os.environ.items() if not name.upper().startswith("UV_")}
    env.update(UV_SYSTEM_CERTS="true", UV_OFFLINE="true", UV_NO_PROGRESS="1",
               RUST_LOG="uv=debug", PYTHONDONTWRITEBYTECODE="1")
    outside = tmp_path / ("outside-" + setting)
    cwd = tmp_path / "working"
    cwd.mkdir()
    if source == "explicit":
        config = tmp_path / "explicit.toml"
        env["UV_CONFIG_FILE"] = str(config)
    elif source == "project":
        config = cwd / "uv.toml"
    else:
        # Documented user config roots: APPDATA on Windows, XDG_CONFIG_HOME on Unix.
        user_config = tmp_path / "user-config"
        env["APPDATA" if os.name == "nt" else "XDG_CONFIG_HOME"] = str(user_config)
        config = user_config / "uv" / "uv.toml"
        config.parent.mkdir(parents=True)
    config.write_text(f"[pip]\n{setting} = '{outside.as_posix()}'\n", encoding="utf-8")
    # These are defensive removals, not a claim that uv supports every pip/UV alias.
    env_outside = tmp_path / "outside-env"
    for name in ("UV_TARGET", "UV_PREFIX", "UV_PROJECT_ENVIRONMENT", "PIP_TARGET", "PIP_PREFIX",
                 "PIP_ROOT", "PYTHONUSERBASE"):
        env[name] = str(env_outside)
    env["UV_SYSTEM_PYTHON"] = "true"
    before = {name: version(name) for name in ("torch", "torchvision")}
    # main retains its real ROOT while uv discovers configuration from a disposable cwd.
    script = (f"import sys; sys.path.insert(0, {str(setup.ROOT)!r}); "
              "from safety_yolo.setup import main; raise SystemExit(main(['--dry-run']))")
    result = subprocess.run([python, "-B", "-c", script], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    assert not outside.exists(), output  # Even dry-run scaffolding is an isolation failure.
    assert not env_outside.exists(), output
    assert {name: version(name) for name in before} == before
    assert "--no-config pip install" in result.stdout, output
    assert str(outside) not in output and str(env_outside) not in output, output
    if result.returncode and ("network connectivity is disabled" in output
                              or "not found in the cache" in output):
        pytest.skip("offline wheel metadata for the exact repo pins is not cached")
    assert result.returncode == 0, output
    assert "verification skipped" in result.stdout
    assert "Would" in result.stderr and "install" in result.stderr
    assert "Using --target directory" not in output and "Using --prefix directory" not in output
    assert str(setup.ROOT / ".venv") in output
    for pin in setup.requirements(setup.ROOT):
        assert pin in result.stdout


@pytest.mark.parametrize("bad", ["--system", "cpu;echo", "cpu extra", "../cpu", "", "CPU", "cu130\n"])
def test_backend_rejects_unsafe_names(installer, bad):
    with pytest.raises(SystemExit) as error:
        setup.main(["--backend=" + bad])
    assert error.value.code == 2 and not installer[2]


@pytest.mark.parametrize("code", [1, 7, -9])
def test_install_failure_propagates_without_verification(installer, monkeypatch, capsys, code):
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.main([]) == (code if code > 0 else 1)
    assert len(calls) == 1
    assert "not transactional" in capsys.readouterr().err


def test_verification_failure_propagates(installer, monkeypatch, capsys):
    results = iter([0, 9])
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=next(results)))
    assert setup.main([]) == 9
    assert "does not establish GPU usability" in capsys.readouterr().err


def test_verification_timeout_after_successful_install(installer, monkeypatch, capsys):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            assert kwargs["timeout"] == 120
            raise subprocess.TimeoutExpired("fresh verification", 120)
        assert "timeout" not in kwargs  # Large downloads have no imposed deadline.
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.main([]) == 2 and len(calls) == 2
    error = capsys.readouterr().err
    assert "timed out" in error and "Traceback" not in error


def test_installer_parent_never_imports_runtime_packages(installer, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision", "ultralytics", "PIL", "yaml"}:
            raise AssertionError("Installer parent imported " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert setup.main([]) == 0
    assert len(installer[2]) == 2


@pytest.mark.parametrize("failure", [FileNotFoundError("uv disappeared"), PermissionError("DLL in use"),
                                   subprocess.TimeoutExpired("verification", 120)])
def test_process_errors_are_actionable(installer, monkeypatch, capsys, failure):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.main([]) == 2
    error = capsys.readouterr().err
    assert "Setup failed:" in error and "Traceback" not in error


def test_missing_uv_does_not_bootstrap(installer, monkeypatch, capsys):
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    assert setup.main([]) == 2 and not installer[2]
    assert "uv is missing" in capsys.readouterr().err


@pytest.fixture
def virtualenv(monkeypatch, tmp_path):
    venv = tmp_path / ".venv"
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir(parents=True)
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    python.touch()
    (venv / "pyvenv.cfg").touch()
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base-conda-is-okay"))
    monkeypatch.setattr(sys, "executable", str(python))
    return tmp_path, venv, python


def test_real_project_venv_is_required(virtualenv):
    root, _, python = virtualenv
    assert setup.project_interpreter(root) == str(python)


@pytest.mark.parametrize("problem", ["system", "conda", "wrong-prefix", "wrong-executable", "no-config"])
def test_refuse_wrong_environments(virtualenv, monkeypatch, problem):
    root, venv, python = virtualenv
    if problem == "system":
        monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    elif problem == "conda":
        (venv / "conda-meta").mkdir()
    elif problem == "wrong-prefix":
        monkeypatch.setattr(sys, "prefix", str(root / "other"))
    elif problem == "wrong-executable":
        monkeypatch.setattr(sys, "executable", str(root / python.name))
    else:
        (venv / "pyvenv.cfg").unlink()
    with pytest.raises(ValueError, match="refused"):
        setup.project_interpreter(root)


def test_wrong_environment_blocks_all_subprocesses(installer, monkeypatch):
    def reject(root):
        raise ValueError("wrong environment")
    monkeypatch.setattr(setup, "project_interpreter", reject)
    assert setup.main([]) == 2 and not installer[2]


def test_redirected_venv_refused(virtualenv, monkeypatch):
    root, venv, _ = virtualenv
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        return root / "elsewhere" if path == venv else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="redirected"):
        setup.project_interpreter(root)


def test_requirements_come_from_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["torch==8.1.2", "torchvision==9.2.3", "torchmetrics==4.5.6"]\n')
    assert setup.requirements(tmp_path) == ["torch==8.1.2", "torchvision==9.2.3"]


@pytest.mark.parametrize("deps", [["torch>=2.0", "torchvision==0.1.2"],
                                  ["torch==2.0.0+cpu", "torchvision==0.1.2"],
                                  ["torch==2.0.0", "torch==2.1.0", "torchvision==0.1.2"],
                                  ["torch==2.0.0; python_version > '3'", "torchvision==0.1.2"], []])
def test_refuse_missing_ambiguous_or_nonexact_pins(tmp_path, deps):
    (tmp_path / "pyproject.toml").write_text("[project]\ndependencies = " + json.dumps(deps))
    with pytest.raises(ValueError, match="exactly one public"):
        setup.requirements(tmp_path)


def test_help_is_stdlib_only_and_does_not_install():
    script = '''
import builtins, runpy, subprocess, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'ultralytics', 'PIL', 'yaml'}:
        raise AssertionError('Heavy import: ' + name)
    return original(name, *args, **kwargs)
def no_process(*args, **kwargs):
    raise AssertionError('Help tried to install')
builtins.__import__ = guarded
subprocess.run = no_process
sys.argv = ['setup', '--help']
runpy.run_module('safety_yolo.setup', run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--backend" in result.stdout and "--dry-run" in result.stdout


# Only these minimal modules are injected in an actual fresh verification subprocess.
# No imports of the installed pair, network calls, or package writes are needed.
FAKE_RUNTIME = r'''
import json, sys
from types import SimpleNamespace as NS
options = json.loads(sys.argv[3])
expected = json.loads(sys.argv[1])
events = []
class Tensor:
    def __init__(self, value, device):
        self.value, self.device = value, device
    def __mul__(self, value):
        return Tensor([x * value for x in self.value], self.device)
    def sum(self):
        return Tensor(sum(self.value), self.device)
    def item(self):
        return self.value
    def tolist(self):
        return self.value
def tensor(value, device):
    events.append('tensor:' + device)
    return Tensor(value, device)
def nms(boxes, scores, threshold):
    if options.get('bad_ops') or options.get('bad_ops_device') == boxes.device:
        raise RuntimeError('operator torchvision::nms does not exist')
    events.append('nms:' + boxes.device)
    return Tensor([0], boxes.device)
def init():
    raise RuntimeError('driver unavailable or device hidden')
def version(name):
    return options.get(name + '_version', expected[name]) + options.get(name + '_tag', options.get('tag', '+cpu'))
torch = NS(__version__=version('torch'), version=NS(cuda=options.get('cuda'), hip=options.get('hip')),
           cuda=NS(is_available=lambda: options.get('available', False),
                   device_count=lambda: options.get('count', 0), get_device_name=lambda i: 'Fake GPU', init=init),
           backends=NS(mps=NS(is_available=lambda: options.get('mps', False))),
           set_num_threads=lambda count: None, tensor=tensor)
vision = NS(__version__=version('torchvision'), ops=NS(nms=nms))
sys.modules.update(torch=torch, torchvision=None if options.get('missing_vision') else vision)
'''


def verify_fake(options, backend="auto"):
    versions = dict(pin.split("==") for pin in setup.requirements(setup.ROOT))
    script = FAKE_RUNTIME + "\nexec(" + repr(setup.VERIFY_SCRIPT) + ")\nprint('EVENTS:', events)"
    return subprocess.run([sys.executable, "-I", "-c", script, json.dumps(versions), backend,
                           json.dumps(options)], capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("options,backend,device", [
    ({}, "auto", "cpu"), ({}, "cpu", "cpu"),
    ({"cuda":"13.0", "tag":"+cu130", "available":True, "count":1}, "auto", "cuda:0"),
    ({"cuda":"13.0", "tag":"+cu130", "available":True, "count":1}, "cu130", "cuda:0"),
    ({"hip":"7.2.0-build", "tag":"+rocm7.2", "available":True, "count":1}, "rocm7.2", "cuda:0"),
    ({"hip":"7.2.0-build", "tag":"+rocm7.2.0", "available":True, "count":1}, "rocm7.2", "cuda:0"),
    ({"tag":"", "mps":True}, "auto", "mps"),
    ({"mps":True}, "cpu", "cpu"),
    ({"available":True, "count":1}, "cpu", "cpu"),
])
def test_fresh_runtime_verifies_native_compute(options, backend, device):
    result = verify_fake(options, backend)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Verified device: {device}" in result.stdout
    assert "nms:cpu" in result.stdout
    if device == "cuda:0":
        assert "nms:cuda:0" in result.stdout and "Fake GPU" in result.stdout
    elif device == "mps":
        assert "tensor:mps" in result.stdout and "nms:mps" not in result.stdout
        assert "MPS NMS not claimed" in result.stdout
    else:
        assert "tensor:cuda" not in result.stdout and "nms:cuda" not in result.stdout


@pytest.mark.parametrize("options,backend,message", [
    ({"torch_version":"0.0.0"}, "auto", "version mismatch"),
    ({"torchvision_version":"0.0.0"}, "auto", "version mismatch"),
    ({"cuda":"13.0", "tag":"+cu130", "torchvision_tag":"+cpu"}, "auto", "build mismatch"),
    ({"hip":"7.2.0", "tag":"+rocm7.1"}, "auto", "HIP runtime"),
    ({"hip":"7.2.0", "tag":"+rocm7.20"}, "auto", "HIP runtime"),
    ({"tag":"+xpu"}, "auto", "--backend cpu"),
    ({"tag":"+cu130"}, "auto", "CPU backend build mismatch"),
    ({"cuda":"13.0", "tag":"+cu130"}, "cpu", "CPU requested"),
    ({}, "cu130", "does not match"),
    ({"cuda":"13.0", "tag":"+cu130", "count":1}, "auto", "driver unavailable or device hidden"),
    ({"cuda":"13.0", "tag":"+cu130", "available":True, "count":0}, "auto", "unavailable"),
    ({"bad_ops":True}, "cpu", "torchvision::nms"),
        ({"cuda":"13.0", "tag":"+cu130", "available":True, "count":1,
            "bad_ops_device":"cuda:0"}, "auto", "torchvision::nms"),
        ({"missing_vision":True}, "auto", "torchvision"),
])
def test_fresh_runtime_rejects_mismatches_and_unusable_gpu(options, backend, message):
    result = verify_fake(options, backend)
    assert result.returncode == 1
    assert message in result.stderr and "Traceback" not in result.stderr
    assert "Verified device:" not in result.stdout
    if message == "driver unavailable or device hidden":
        assert "Visible CUDA/ROCm GPUs: 1" in result.stdout and "runtime available: False" in result.stdout