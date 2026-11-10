# Tasks

## Current acceptance status

**Blocked after terminal Cycle 3.** Implementation and verification history below is retained, not an acceptance claim. Independent review reproduced C2: native-cache invalidation can delete an accepted local manifest source reached through an alias. No further fixes or review cycles are authorized without the user's decision. Do not run real dataset preparation until resolved. Latest suite retry: 122 passed, 1 skipped; CPU smoke: 1 passed. These green tests do not cover C2.

## Group 1

- [x] [coding] Build and verify the modular local YOLO workflow | `safety_yolo/`, `configs/`, `tests/`, `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `README.md` | Meet spec.md, preserve the real export, and pass offline tests. Run: `.venv\Scripts\python.exe -m pytest -q`
	> Done. Flat typed package, verified cached preparation, hardware/config policy, train/resume/predict/validate CLIs, pinned environment/lock, offline tests and README implemented directly without subagents. Final `.venv\Scripts\python.exe -m pytest -q`: 86 passed, 1 opt-in smoke skipped. With `SAFETY_YOLO_SMOKE=1`, `.venv\Scripts\python.exe -m pytest -q -s tests/test_smoke.py`: 1 passed (126.22s), exercising offline preparation, one CPU epoch, weights, streamed prediction, val and test metrics. All six help paths pass heavy-import guards. Real read-only inspect: 2,801 images, splits 2,605/114/82, 38,352 boxes, 24 backgrounds. Exactly one real export image verified quietly; no full download/training. Python 3.12.8 in `.venv`, Ultralytics 8.4.208, torch 2.15.0+cpu, torchvision 0.29.2; `uv sync --locked` and `uv pip check` passed. Editor diagnostics clear. GPU/MPS/video/live-camera execution and real interrupted-checkpoint continuation remain unverified. Group 2 and independent review remain open.

### Cycle 1 scoped fix wave

> Done. One direct fix wave for C1 and W1-W4 only. No subagents or git operations. `review.md` and the real source export, `data/`, `.cache/`, and `runs/` were not modified. Cycle 1's FAIL is not replaced by this implementation note. Independent review remains open.

- C1. Replaced recursive `.tmp-*` deletion with exact manifest-derived destinations in `.prepare-staging/` and an output path/type allowlist. Legitimate temporary-prefix images/labels survive offline reruns without the download cache. Unknown files remain intact and preparation refuses them before invalidating an existing YAML.
- W1. Fresh train, predict, and validate resolve project paths from the working directory before forwarding. Default, relative custom, and absolute paths are checked against real upstream `get_save_dir`. Resume forwarding remains unchanged.
- W2. Pinned `cloudpickle==3.1.1`, regenerated `uv.lock`, and synced only `.venv`. Real upstream `generate_ddp_file`, companion `torch_load`, model execution, callback execution, and cleanup pass with `YOLO_AUTOINSTALL=false`. This is not a GPU/DDP training run.
- W3. Supported serializable lifecycle callbacks reapply requested CPU threads after upstream initialization. Scoped configuration restores the caller's prior setting on success or exception. Real prediction/validation compute observes one requested thread; real training forwards observe two requested threads. Standalone validation warmup remains before its start callback. Resume still rejects fresh-run CPU-thread settings.
- W4. A complete staged initial fingerprint can recover before its first rename after the stopped child's stale lock is removed. Hard-exit tests also pass at image, label, summary, and YAML publication. Mismatched initial staging and foreign staging content are refused without deletion.

Verification on Windows, Python 3.12.8, torch 2.15.0+cpu. `$fixTmp` is `C:\Users\Billy.Ng\AppData\Local\Temp\safety-yolo-fix-4920953f78864cb488392061215b8134`.

- `.\.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider --basetemp (Join-Path $fixTmp 'wave-final')` passed with **110 passed, 1 opt-in skipped in 32.92s**. Includes all six lazy help paths.
- With `SAFETY_YOLO_SMOKE=1`, `.\.venv\Scripts\python.exe -m pytest -q -s --tb=short -p no:cacheprovider --basetemp (Join-Path $fixTmp 'smoke-final') tests/test_smoke.py` passed with **1 passed in 25.06s**. Actual one-epoch train, streamed prediction, val/test metrics, and training thread probes. Real-framework fixtures block socket connections and seed bundled local fonts in temporary settings directories.
- `uv pip check --python .\.venv\Scripts\python.exe` passed for 47 installed packages. `uv lock --check --python .\.venv\Scripts\python.exe --no-progress` passed for 66 resolved packages. The invoked uv executable was `C:\Users\Billy.Ng\AppData\Local\miniforge3\Scripts\uv.exe`.
- Source/test editor diagnostics are clear. Real dataset download verification is reserved for the separate runtime verifier. CUDA, Linux/WSL2 multi-GPU, MPS, actual interrupted-checkpoint continuation, live camera, and real video execution remain unverified.

### Cycle 2 scoped W5 fix wave

- [x] [coding] Permit repeat preparation after native disk image caching | `safety_yolo/dataset.py`, `tests/test_dataset.py`, `tests/test_upstream.py`, `.claude/specs/safety-yolo/tasks.md` | Accept only exact manifest-derived regular-file cache paths, preserve expected files and parent directories, reject foreign artifacts, and preserve C1/W4 safeguards. Run: `.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider`
	> Done. W5 only. The preparer derives native image caches with `image.with_suffix('.npy')`, excludes every expected manifest file and allowed directory, and uses the existing link refusal and file/type allowlist before any invalidation. After verifying the source, it removes only that image's eligible regenerable binary cache before publishing the image. Every successful preparation invalidates these caches because local pixels can change without a manifest change, even without `refresh`. It never loads NumPy caches or deletes manifest source/prepared images or URL-keyed download images. Native label-cache behavior and staging ownership checks are unchanged.

The real `YOLODataset(cache='disk')` regression creates caches for all three synthetic splits, verifies native image reads, then reruns preparation offline without a download cache. It asserts byte-for-byte source/image/label preservation and exact cache invalidation. It also changes local pixels with and without `refresh`, rebuilds real native caches, and verifies that the reader sees the new pixels. Preparation runs with `numpy.load` forbidden. Additional tests preserve manifest directories such as `frame.npy/other.png` and `frame.npy/nested/other.png`, reject file/directory type mismatches, and retain unknown `.npy` files unchanged while refusing preparation. C1 temporary-prefix and all five W4 hard-exit regressions pass.

Verification used only project-local Python 3.12.8 in `.venv`. `$w5Tmp` is `C:\Users\Billy.Ng\AppData\Local\Temp\safety-yolo-w5-511d40fb2aae4ea588c08162302f79a1`. Environment included `PYTHONDONTWRITEBYTECODE=1`, `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false`, `WANDB_MODE=disabled`, `OMP_NUM_THREADS=2`, and `MKL_NUM_THREADS=2`, with settings and matplotlib output under that temporary root. Native subprocesses block socket connections. No packages were installed.

- Before the source fix, `.\.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider --basetemp (Join-Path $w5Tmp 'red') tests/test_upstream.py::test_actual_disk_image_cache_allows_offline_reprepare` failed as expected with **1 failed in 4.64s**, at `DatasetError: Unexpected dataset artifacts; use a clean output directory`. Log is `red.log`.
- After the fix, `.\.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider --basetemp (Join-Path $w5Tmp 'focused') tests/test_dataset.py tests/test_upstream.py::test_actual_disk_image_cache_allows_offline_reprepare` passed with **82 passed in 9.64s**. Log is `focused.log`.
- `.\.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider --basetemp (Join-Path $w5Tmp 'all')` passed with **122 passed, 1 opt-in skipped in 34.18s**. This includes the previous 110 passing tests and all native integration/help tests. Log is `all.log`.
- With `SAFETY_YOLO_SMOKE=1`, `.\.venv\Scripts\python.exe -m pytest -q -s --tb=short -p no:cacheprovider --basetemp (Join-Path $w5Tmp 'smoke') tests/test_smoke.py` passed with **1 passed in 23.40s**. The flag was removed afterward. Actual tiny CPU training, prediction, val/test metrics, and training thread assertions passed. Log is `smoke.log`.
- Editor diagnostics reported no errors in the three changed Python files.
- After verification, all 8,405 `data/`, `.cache/`, and `runs/` snapshot entries matched the pre-edit baseline. Checks covered directory entries and each file's SHA256, size, and last-write timestamp. `review.md` retained SHA256 `E14511EF3E804D0AD286091B88CCBC72F90128E46B7D57917503BB6331AEFE81`.

No subagents, new review cycle, full training, or real-data preparation were performed. `review.md` is not edited and its Cycle 2 FAIL remains the independent verdict pending any separately authorized Cycle 3. Group 2 and physical hardware limitations remain open.

## Group 2

- [!] [coding] Exercise the real dataset and bounded CPU pipeline | generated `data/`, `.cache/`, `runs/` | Inspect all manifest records, prepare real images where URLs work, prove a tiny train-predict-validate CPU path, and record any hardware gaps. Run: `.venv\Scripts\python.exe -m pytest -q`

	> Blocked. Partial real image/cache artifacts exist, but complete preparation and offline repeat verification are not established. Preparation is paused for C2; the real CPU pipeline passed with synthetic fixtures only. No full model training was started.

## Review

- [!] Independent code and documentation review. All three cycles consumed; terminal Cycle 3 FAIL, critical C2 unresolved. See `review.md` for reproduction and evidence. Await user authorization before any additional fix/review work. No git operations or PR without request.