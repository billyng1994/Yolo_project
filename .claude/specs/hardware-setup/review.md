# Independent hardware-setup review

## Cycle 1 — 2026-11-10

Reviewing: Group 1 — explicit automatic local PyTorch setup. This is the first independent cycle for this feature only (maximum three nonresetting cycles). The separate safety-yolo group's terminal C2 review was neither read nor reopened.

### Spec Alignment

- Reviewed this feature's complete `spec.md`, `design.md`, `tasks.md`, `safety_yolo/setup.py`, `tests/test_setup.py`, `pyproject.toml`, `README.md`, and universal lock, with read-only hardware/import/test-fixture integration checks.
- Exact public pins, targeted reinstalls, native uv auto/explicit backend selection, lazy stdlib help, separate verification process, opt-in installation, actionable failures, and honest accelerator limitations match the design.
- The hard requirement that installation remain inside this project's real `.venv` is not satisfied when uv inherits destination configuration; see C1.
- Completion notes accurately distinguish implementation evidence from independent review. Historical editable-metadata repair is documented; it is not evidence that the documented fresh `uv sync --frozen` path is broken. Current installed metadata passes dependency checks.

### Critical

- **C1 — inherited uv configuration bypasses the installation-destination guard** — [`safety_yolo/setup.py:161-167`] **Runtime-verified, non-mutating reproduction.** The command supplies an absolute `--python` but inherits configuration and the entire environment. In uv 0.11.25, `[pip].target` or `[pip].prefix` changes the installation destination independently of that interpreter. With `UV_CONFIG_FILE` pointing at a disposable TOML file containing either setting, the real `python -m safety_yolo.setup --dry-run` exits **0**, prints this project's `.venv` as its target, but uv logs `Using --target directory at .../outside-target` or `Using --prefix directory at .../outside-prefix` and plans **13 installs into that external destination**, rather than two replacements in `.venv`. uv even creates the destination lock/directory scaffolding during dry-run. Thus a real invocation can write outside the explicitly authorized environment; the later verifier still examines `.venv`, not the redirected destination, and cannot prevent those writes. No actual package installation was executed to demonstrate this.
  - **Recommended fix:** isolate uv's destination configuration before launching it. One empirically validated approach is a copied child environment with `UV_CONFIG_FILE` removed, together with `--no-config` to disable discovery of project/user/system configuration. Preserve legitimate certificate trust settings such as `UV_SYSTEM_CERTS`. Reject or neutralize any other supported destination controls rather than relying solely on `--python`. **`--no-config` alone is insufficient:** the explicit `UV_CONFIG_FILE` still won in the observed uv version.
  - **Regression coverage:** extend the command/environment boundary tests starting at [`tests/test_setup.py:34`] to cover inherited explicit configuration and discovered target/prefix settings. The existing suite checks the running interpreter and argv, but mocks away uv's configuration behavior; it passes despite this escape. Regression checks must prove rejection before launching uv or prove the effective child invocation cannot install outside `.venv`, without performing a real install.

### Warning

None additional. The missing regression belongs to C1 rather than a duplicate finding.

### Suggestion

None.

### Cross-Task Consistency

- Native `hardware.plan(device='0')` succeeds with `HardwarePlan(device='0', batch=16, workers=0, amp=True, ranks=1)`; no model training was invoked.
- Lock entries and project metadata agree on torch 2.14.1 / torchvision 0.29.1. All registry sources are PyPI; Windows CPython 3.12 wheels are present and no machine-specific `tool.uv.sources` mapping exists.
- README correctly documents TLS system trust, optional setup, overlay restoration by sync, MPS CPU-only NMS verification, unsupported XPU training, and unvalidated hardware. No additional cross-platform hardware proof is demanded.
- Explicit interpreter selection resisted the tested `UV_PYTHON`, `UV_PROJECT_ENVIRONMENT`, and documented `UV_SYSTEM_PYTHON=true` overrides. `UV_SYSTEM`, `UV_TARGET`, and `UV_PREFIX` probes did not redirect installation and are **not** findings. The demonstrated escape uses real supported TOML target/prefix settings via `UV_CONFIG_FILE`.

### Tests

- [x] Independent default suite passing: `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` — **185 passed, 1 skipped in 60.44s**. `PYTHONDONTWRITEBYTECODE=1`; the skip is the opt-in CPU training smoke. Existing tests use mocked installers and temporary synthetic artifacts, not real prepared data.
- [ ] Isolation regression coverage adequate — C1 is not covered.
- Fresh runtime verifier: project Python launched `[sys.executable, '-I', '-B', '-c', VERIFY_SCRIPT, json.dumps(dict(pin.split('==') for pin in requirements(ROOT))), 'auto']`, with `check=False, timeout=120` — **exit 0**. Observed torch **2.14.1+cu130**, torchvision **0.29.1+cu130**, CUDA **13.0**, HIP **None**, one **NVIDIA RTX 3500 Ada Generation Laptop GPU**; CPU and `cuda:0` tensor computation and native torchvision NMS passed.
- `uv --version` — **0.11.25**. `uv pip check --python <absolute project .venv\Scripts\python.exe>` — **47 packages compatible**. `uv lock --check --offline` — **66 packages resolved, exit 0**; no sync or lock rewrite.
- Baseline `.venv\Scripts\python.exe -B -m safety_yolo.setup --dry-run` with child-only `UV_SYSTEM_CERTS=true`, `UV_OFFLINE=false`, and `UV_NO_PROGRESS=1` — **exit 0**, only the two cu130 replacements planned, runtime verification skipped.
- C1 reproduction (both target and prefix): create disposable `%TEMP%` TOML fixtures with `[pip]` and `target = '<temporary external directory>'` or `prefix = '<temporary external directory>'`; pass their paths through child-only `UV_CONFIG_FILE`; run `.venv\Scripts\python.exe -B -m safety_yolo.setup --dry-run` with `UV_SYSTEM_CERTS=true`, `UV_OFFLINE=true`, `UV_NO_PROGRESS=1`, `RUST_LOG=uv=debug`. Both **exit 0**, log the external destination, and report `Would install 13 packages`. Target scaffolding contained `.lock`; prefix scaffolding contained `.lock` and `Lib/site-packages`. All disposable fixtures were removed automatically.
- Mitigation experiment: with `UV_CONFIG_FILE` removed from the child environment, run `uv --no-config pip install --python <absolute project Python> --torch-backend auto --reinstall-package torch --reinstall-package torchvision torch==2.14.1 torchvision==0.29.1 --dry-run` offline — **exit 0**, logs `.venv`, plans two uninstalls/two installs for both fixtures. A preceding experiment retaining `UV_CONFIG_FILE` with `--no-config` failed the isolation assertion, confirming that the flag alone does not suffice.
- Editor diagnostics: no errors in setup, its tests, or project metadata. Source inspection found no embedded credentials or shell invocation in the new installer.

### Verification Limits and Scope

- No installs/reinstalls, `uv sync`, driver changes, full/GPU model training, pretrained downloads, real dataset preparation, or changes to real data/cache/run outputs were performed. Network-enabled dry-runs used verified OS certificate trust, never an insecure TLS bypass; uv metadata caches outside real data could be updated.
- Prior implementation CPU smoke evidence (one pass) was read, not rerun: this review independently exercised the default suite and tiny native CPU/CUDA operations only. MPS/ROCm, POSIX installation, multi-GPU and fresh installation remain outside the locally exercised runtime; their accepted limitations are not new blockers.
- No subagents or plugin reviewers were spawned, as requested. Only this new `review.md` was authored; implementation, tasks, old reviews, and other specs were not edited.

### Verdict: FAIL

Critical: **1**. Warning: **0**. Suggestion: **0**.

Reason: empirically demonstrated uv configuration escape violates the required project-only installation boundary despite passing tests and native CUDA verification.

## Cycle 2 — 2026-11-10

Reviewing: Group 1 — independent hardware-setup feature, verification of C1 mitigation and regressions. This is **Cycle 2 of the same nonresetting maximum three cycles**, not a new group or reset. The unrelated safety-yolo group's terminal C2 remains out of scope and unchanged. Cycle 1 above remains historical FAIL evidence.

### Spec Alignment

- Read this feature's complete spec/design/tasks/review and current `setup.py`, `test_setup.py`, and README; also checked project pins and test fixtures. No implementation or task-file changes were made.
- **C1 resolved, runtime-verified:** `subprocess_environment()` copies the caller's environment, removes `UV_CONFIG_FILE` and non-allowlisted uv policy/destination/interpreter controls, and removes pip target/prefix/root/user/config and `PYTHONUSERBASE` controls. The uv command additionally uses global `--no-config`. Both installation and fresh verification receive the same explicit filtered environment; the caller is not mutated. Neither measure is treated as sufficient by itself.
- Certificate trust, standard proxies, offline/cache/HTTP settings and GPU visibility remain available. No insecure-host bypass or forced system-trust default was added. The deliberate removal of uv index/config policy is documented; uv's native backend selection remains the resolver authority.
- Exact metadata pins, forced pair reinstall, explicit backend handling, dry-run verification skip, stdlib-only parent/help, and actionable install/runtime failure behavior remain covered. No new Critical/Warning regression was found.
- The implementation's historical **seven pre-fix failures** were read from its completion record, not independently recreated by reverting code. Current green results and native effective-destination assertions were independently executed below.

### Critical

None outstanding. Cycle 1 C1 is closed by the combined configuration/environment isolation and independent native probes.

### Warning

None.

### Suggestion

None.

### Cross-Task Consistency

- README accurately describes ignored configuration, retained trust/proxy controls, opt-in installation and overlay/sync behavior. The mitigation introduces no new dependency, public pin, hardware policy, or universal-lock change.
- Current installed builds still match the public pins and each other: torch **2.14.1+cu130**, torchvision **0.29.1+cu130**. Fresh verification reports CUDA **13.0**, HIP **None**, one **NVIDIA RTX 3500 Ada Generation Laptop GPU**, runtime available **True**, selected **cuda:0**.
- Environment-boundary tests assert preserved transport/certificate values, removed destination/config controls, unchanged parent environment, and shared filtered environment for both children. Native tests exercise explicit configuration plus project/user discovery rather than trusting argv mocks alone.

### Tests

All commands ran from `C:\Users\Billy.Ng\Desktop\Projects\Yolo_Project`, using its absolute `.venv\Scripts\python.exe` (Python **3.12.8**) and uv **0.11.25**. `PYTHONDONTWRITEBYTECODE=1`, Python `-B`, and pytest `-p no:cacheprovider` prevented bytecode/pytest-cache writes.

- [x] Setup regressions: `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_setup.py` — **70 passed in 24.24s**, no skips; all six real uv cases executed.
- [x] Full default suite: `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` — **192 passed, 1 skipped in 64.00s**. Only the opt-in training smoke was skipped.
- [x] C1 regression coverage adequate for the demonstrated escape: independently executed an additional six-case native matrix, beyond the existing tests. Each case used `[pip] target = '<temporary outside path>'` or `prefix = '<temporary outside path>'`, provided through **explicit `UV_CONFIG_FILE`**, disposable working-directory **`uv.toml`**, or disposable **`APPDATA\uv\uv.toml`**. The copied child environment additionally supplied `UV_NO_CONFIG=false`, `UV_SYSTEM_PYTHON=true`, and hostile `UV_TARGET`, `UV_PREFIX`, `UV_PROJECT_ENVIRONMENT`, `PIP_TARGET`, `PIP_PREFIX`, `PIP_ROOT`, and `PYTHONUSERBASE` paths. Every case used `UV_SYSTEM_CERTS=true`, `UV_OFFLINE=true`, `UV_NO_PROGRESS=1`, and `RUST_LOG=uv=debug`, preserving other standard transport settings. Fixtures used `tempfile.TemporaryDirectory(prefix='hardware-c2-')` and were automatically removed.
  - Exact outer child invocation: `[python, '-B', '-c', code]`, with `cwd=<temporary working directory>`, `env=<copied hostile environment>`, `capture_output=True`, `text=True`, `timeout=120`; `code` was `import sys; sys.path.insert(0, 'C:\\Users\\Billy.Ng\\Desktop\\Projects\\Yolo_Project'); from safety_yolo.setup import main; raise SystemExit(main(['--dry-run']))`.
  - Exact effective installer argv in every case: `C:\Users\Billy.Ng\AppData\Local\miniforge3\Scripts\uv.EXE --no-config pip install --python C:\Users\Billy.Ng\Desktop\Projects\Yolo_Project\.venv\Scripts\python.exe --torch-backend auto --reinstall-package torch --reinstall-package torchvision torch==2.14.1 torchvision==0.29.1 --dry-run`.
  - **All six exited 0.** Native uv debug output, not merely the setup banner, reported `Using Python 3.12.8 environment at: C:\Users\Billy.Ng\Desktop\Projects\Yolo_Project\.venv`. Assertions required that effective path, no target/prefix destination log, no foreign path in output, both hostile destinations absent, two planned uninstalls/two installs, exact cu130 builds, unchanged installed pair, and unchanged parent environment.
  - uv read cached responses from `https://download.pytorch.org/whl/cu130/torch/` and `.../torchvision/`, selecting official Windows CPython 3.12 cu130 wheel candidates. It reported **Would download 2 packages / Would uninstall 2 packages / Would install 2 packages**. This proves offline cached-metadata resolution and effective destination isolation, **not cached wheel-payload availability or an actual installation**. No downloads or package writes were performed.
- [x] Fresh runtime: project Python ran `[sys.executable, '-I', '-B', '-c', VERIFY_SCRIPT, json.dumps(dict(pin.split('==') for pin in requirements(ROOT))), 'auto']`, with `check=False`, `timeout=120`, and `env=subprocess_environment()` supplemented only with `UV_SYSTEM_CERTS=true`, `UV_OFFLINE=true`, `UV_NO_PROGRESS=1`, `PYTHONDONTWRITEBYTECODE=1` — **exit 0**. Tiny tensor computation and native torchvision NMS passed on **CPU and cuda:0**; this was not model training.
- `uv --version` — **0.11.25**. `uv --no-config pip check --python C:\Users\Billy.Ng\Desktop\Projects\Yolo_Project\.venv\Scripts\python.exe` — **47 packages compatible**, exit 0. `uv --no-config lock --check --offline` — **66 packages resolved**, exit 0. Both checks received the same filtered/offline environment; no sync or lock regeneration occurred.
- Editor diagnostics: no errors in setup or its tests. Before appending this section, SHA-256 comparisons confirmed all **26** baseline source/spec/metadata/README files unchanged, including `pyproject.toml`, `uv.lock`, tasks, unrelated modules/specs and the original **8,263-byte** review. Sorted installed distribution name/version snapshots were identical before and after verification. Final append validation checks the original review byte prefix and permits only this review file to differ.

### Verification Limits and Scope

- No actual install/reinstall, sync, package/version changes, full/GPU training, smoke training, pretrained downloads, dataset preparation, driver/global configuration changes, subagents or plugin reviewers. Only this review appendix was authored. Test/probe fixtures were disposable temporary files; real `data/`, `.cache/`, and `runs/` were not targeted.
- Fresh installation and wheel download/extraction were deliberately not exercised. This cycle does not independently re-establish historical CPU training smoke, MPS/ROCm hardware, POSIX installation, multi-GPU, or GPU model-training evidence. These accepted limits are not asserted as new failures.
- The unrelated terminal C2 blocker was neither reopened nor reset. No additional review cycle is requested or implied by this PASS.

### Verdict: PASS

Critical: **0**. Warning: **0**. Suggestion: **0**.

Reason: C1's native configuration escape is closed in all six independently exercised target/prefix cases; safety contracts, full default suite, fresh CPU/CUDA operations and dependency/lock checks pass without package changes.