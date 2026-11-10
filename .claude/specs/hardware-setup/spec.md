# Automatic local PyTorch setup

## Request

Detect the local accelerator and explicitly install the matching PyTorch build into this project's environment. This feature is independent of the blocked safety-yolo dataset-preparation acceptance group; it does not authorize fixes or extra reviews for that group.

## Requirements

- Add `python -m safety_yolo.setup` after normal `uv sync --frozen`. No torch, Ultralytics or third-party imports in the installer process before package replacement. Help must work with stdlib only.
- Use installed uv's supported `uv pip install --torch-backend auto` rather than maintaining a CUDA/driver lookup table. This selects the wheel backend, not the number of training GPUs. Keep existing hardware selection unchanged.
- Replace the incompatible current pins with `torch==2.14.1`, `torchvision==0.29.1`. Both have verified official Windows/Python 3.12 CUDA wheels. Regenerate the universal lock; do not encode this computer's CUDA backend as a universal default.
- Read exact torch/torchvision requirements from `pyproject.toml`; do not duplicate them as setup constants. Restrict changes to project `.venv`, refuse a global/wrong interpreter, invoke subprocesses without a shell and noninteractively, and do not install drivers or global tooling.
- Default automatic backend, explicit `--backend cpu` or uv-supported backend override, and `--dry-run` that resolves/reports but neither installs nor verifies an unmodified environment. Display the target environment and exact command before running.
- Force the pair's reinstall so an existing CPU build with the same public versions cannot cause a false no-op. Use uv's own supported options, fail clearly on unsupported backend/missing wheels/missing uv, and do not silently substitute versions or downgrade drivers.
- Verify in a fresh subprocess after installation: pinned public versions and matching backend builds, torch tensor compute and torchvision ops on the selected supported device. Report runtime builds, visible GPU names/count, selected device, and practical failures. No model training, pretrained downloads, dataset operations, camera access, or tracker use.
- CPU-only hosts and macOS MPS work where official wheels support the pinned platform. AMD/ROCm support is delegated to uv/PyTorch; do not claim untested hardware validation. Existing training has no XPU path; explicitly report this limitation rather than claim Intel GPU training works.
- Setup is opt-in. Ordinary train/predict/validate must never install packages. `uv sync`/`uv run` may restore locked default wheels; document rerunning setup or using `uv run --no-sync` after hardware setup. Preserve virtual-environment isolation and universal lock semantics.
- Test command construction, env safety, backend overrides, dry-run, missing tools, install/verification failure, fresh-process verification, lazy help, and real local CUDA compute if wheels install successfully. Preserve the existing suite and CPU smoke with the new pair.

## Non-goals

No dataset safety fixes, full training, broad dependency modernization, bespoke accelerator compatibility catalog, automatic driver installs, global package installs, or new application framework.