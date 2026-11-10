# Construction safety detection

Prepare an Ultralytics detection NDJSON export, train YOLO26n, stream predictions, and evaluate the original validation or test split. Preparation is explicit and completes before training or distributed workers start. No cloud account, API key, or `.env` is needed.

## Install

Use **Python 3.12.8** and **uv 0.11.25**. Install uv using its official installer or package manager; project dependencies stay in `.venv`. Run all commands from this project directory.

PowerShell:

```powershell
uv python install 3.12.8
uv sync --frozen
.\.venv\Scripts\Activate.ps1
python -m safety_yolo.setup
python -m safety_yolo.train --help
```

Bash, including Linux, WSL2, and macOS:

```bash
uv python install 3.12.8
uv sync --frozen
source .venv/bin/activate
python -m safety_yolo.setup
python -m safety_yolo.train --help
```

If PowerShell activation is restricted, use `.\.venv\Scripts\python.exe` instead of `python`; no execution-policy change is required. Select `.venv` as the VS Code interpreter too.

Direct pins are Ultralytics **8.4.208**, torch **2.14.1**, torchvision **0.29.1**, Pillow **13.0.0**, PyYAML **6.0.3**, cloudpickle **3.1.1**, and pytest **9.2.0**. Build backend: hatchling **1.33.0**. `uv.lock` pins the transitive resolution. Cloudpickle supports native DDP script/companion serialization without runtime dependency installation. The installed release includes YOLO26 architecture definitions. `yolo26n.pt` downloads official pretrained weights on first use; `yolo26n.yaml` with `pretrained: false` starts randomly initialized weights.

### Automatic local PyTorch setup (explicit opt-in)

`python -m safety_yolo.setup` delegates local driver/backend detection to **uv 0.11.25** using its native `--torch-backend auto`. It reads the exact pair from `pyproject.toml`, displays the target and command, and reinstalls only torch/torchvision (plus resolver-required dependencies) into this checkout's real `.venv`. Same-public-version CPU builds are deliberately reinstalled, not mistaken for a usable accelerator install. No custom driver-to-wheel table, GPU-count mapping, driver installation, elevation, or global/tool installation is performed. Ordinary train/predict/validate never install dependencies.

Setup ignores uv configuration files (`UV_CONFIG_FILE` and discovered project/user/system configuration) using a filtered child environment and `uv --no-config`. Only uv certificate, HTTP timeout/retry, cache, offline, and progress/wrapping environment settings are retained; other `UV_*` settings and pip destination/config controls are removed from both installer and verification children without changing the caller's environment. Standard proxy and certificate variables, including `UV_SYSTEM_CERTS`, remain available; configure managed-network trust as described below, not in an ignored uv config file.

```bash
# Resolve/report only; may contact package indexes, but does not install or verify
python -m safety_yolo.setup --dry-run
# Force CPU wheels, even on a GPU host
python -m safety_yolo.setup --backend cpu
# Optional explicit backend understood by the installed uv
python -m safety_yolo.setup --backend cu130 --dry-run
```

uv chooses an available supported backend rather than blindly matching a driver's maximum CUDA version. For example, uv 0.11.25 supports `cu130`, not `cu132`; a CUDA-13.2-capable driver can use the CUDA 13.0 wheels. Older drivers/platforms may have no matching wheels for this pinned release: setup fails with uv diagnostics rather than guessing another release or silently falling back. Missing uv must be installed separately. On Windows, stop Python processes using torch before replacement to avoid loaded-DLL errors; replacement is not transactional, so rerun setup (or `--backend cpu`) after addressing failures.

If uv reports `UnknownIssuer` on a managed network whose CA is already trusted by the OS, enable uv's system trust store for the setup invocation (`$env:UV_SYSTEM_CERTS = 'true'` in PowerShell, or prefix the command with `UV_SYSTEM_CERTS=true` in Bash). Restore the prior setting afterward. Keep certificate verification enabled; do not use insecure-host/TLS bypasses. This local installation required system trust; it is not forced into project configuration.

After actual installation a **fresh subprocess** checks public versions and backend build alignment, prints runtime builds and visible GPU names/count, and runs tiny torch tensor and native torchvision NMS checks on CPU and CUDA/ROCm when usable. A CUDA/ROCm build with an unusable GPU fails with driver/visibility guidance; installation is not proof of GPU access. Auto CPU fallback is valid without a GPU. MPS verifies its tensor separately and CPU NMS only (not native MPS NMS). XPU is explicitly unsupported by the existing training hardware policy; use `--backend cpu`. Only the local NVIDIA accelerator has been hardware-validated; CPU/MPS/ROCm/XPU branches also have mocks, and non-NVIDIA accelerators are not hardware-validated.

`uv.lock` remains a universal baseline without a machine-specific CUDA source. A subsequent `uv sync` **or syncing `uv run`** may restore locked default wheels. Rerun setup afterward, use the activated venv directly, or use `uv run --no-sync` to retain the hardware overlay. Setup selects the wheel backend, not training GPU count: the existing automatic selection, `--device`, and `--gpus` policies below are unchanged. This feature does not resolve the separate terminal C2 dataset-preparation blocker or establish real dataset preparation completion.

## Inspect and prepare

The Downloads path below is an example, not a hardcoded application path:

```powershell
python -m safety_yolo.dataset inspect "$HOME\Downloads\construction-site-safety.ndjson"
python -m safety_yolo.dataset prepare "$HOME\Downloads\construction-site-safety.ndjson"
```

```bash
python -m safety_yolo.dataset inspect "$HOME/Downloads/construction-site-safety.ndjson"
python -m safety_yolo.dataset prepare "$HOME/Downloads/construction-site-safety.ndjson"
```

`inspect` reads every record without torch, image decoding, network requests, or writes. The verified export has **2,801 images: train 2,605, val 114, test 82; 38,352 boxes; 24 backgrounds**. IDs 0 through 9 are Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, and vehicle.

Preparation preserves those IDs, normalized coordinates, splits, image bytes, and background images with empty labels. Omitted annotations indicate a background; malformed annotation objects fail. Filenames must be safe relative POSIX paths with supported image extensions. Local sources use `file` relative to the manifest directory, or a relative `url` field pointing within that directory. Remote sources require HTTPS. All record validation finishes before downloads begin.

Default output:

```text
data/construction-site-safety/
  images/{train,val,test}/
  labels/{train,val,test}/
  data.yaml
  summary.json
  .manifest.sha256
  .prepare-staging/
```

Defaults are `--threads 12`, `--timeout 30` seconds per blocking network operation, `--retries 2`, and `--cache .cache/images`. Threads are bounded to 1..32. Each request uses a unique temporary file, a 100 MiB download cap, and Pillow decoding/dimension checks before atomic publication. The full URL, including its query, hashes into the cache key; signatures are not printed or stored as cache filenames. There is no total wall-clock deadline for a server that continuously supplies data.

- `--output PATH` selects another prepared dataset directory.
- `--offline` uses verified prepared images or cache only, plus available local sources. A missing/corrupt artifact fails.
- `--refresh` re-fetches remote images instead of reusing prepared images/cache. It cannot be combined with `--offline`.
- Repeating preparation verifies existing artifacts and rewrites labels/YAML. A changed manifest is refused in an existing output directory. Use a new `--output` after exporting again; the URL cache can still be shared.
- Preparation removes `data.yaml` while updating and publishes it last, only after every image and label succeeds. Partial artifacts remain reusable, but no success YAML is published. Do not run training concurrently with preparation.
- One preparer owns each output via `.prepare.lock`. After an interrupted process, confirm it is stopped before manually removing that lock. Retry against the same manifest. Atomic publication uses exact manifest-derived paths under `.prepare-staging/`, not filename-prefix cleanup. A complete staged initial fingerprint permits recovery before the first fingerprint rename. Unknown files, links, or a mismatched fingerprint are refused without deletion. Keep unknown artifacts intact and use a clean output directory. Legitimate `.tmp-*` image/label names are preserved.
- HTTP 401/403/404 can mean expired signed URLs. Obtain a fresh export and use a new output directory. `--refresh` cannot renew a signature. Other failures may be connectivity, corrupt images, or dimension mismatches.
- The source export in Downloads is never moved, modified, or deleted. Keep a private backup if needed; exports and generated data are ignored by version control.

`data.yaml` stores an **absolute** dataset root. After moving a prepared dataset, rerun prepare against the same manifest with the new `--output` path, using `--offline` for remote-backed data already present. Local-backed exports still need their local sources. Preparation also removes stale Ultralytics label caches. Other relative command/config paths resolve from your working directory, not the config file's directory. Training/evaluation accept local YAML directories only and reject YAML download instructions.

## Train

Commands below start real training when **you** run them. The default configuration is `configs/train.yaml`, but it is used only when passed explicitly. Code defaults match its core settings.

```bash
# Conservative CPU start
python -m safety_yolo.train --config configs/train.yaml --device cpu --epochs 10 --batch 4 --workers 0 --cpu-threads 4

# One CUDA GPU
python -m safety_yolo.train --config configs/train.yaml --device 0 --batch 16

# Two visible CUDA GPUs, Linux or WSL2 only
python -m safety_yolo.train --config configs/train.yaml --device 0,1 --batch 16
# Equivalent count selector: first N visible GPUs
python -m safety_yolo.train --config configs/train.yaml --gpus 2 --batch 16
```

The same one-line commands work in activated PowerShell. Without `--device` or `--gpus`, selection is CUDA 0, then MPS, then CPU. `cuda:0` is accepted. `--gpus N` means the first N *visible* CUDA devices, respecting torch's device visibility. Explicit unavailable/duplicate devices fail. Native Windows multi-GPU fails early with Linux/WSL2 guidance. No custom launcher is needed; Ultralytics owns DDP.

**Batch is global**, so batch 16 over two GPUs means 8 per GPU. Multi-GPU requires a positive integer divisible by rank count. AutoBatch `--batch -1` or a memory fraction strictly between 0 and 1 is allowed only on one CUDA GPU. `--batch 1.0` means fixed batch 1. CPU/MPS use fixed batches and AMP is disabled for fresh runs. Windows workers default to 0; elsewhere the default is CPU count divided by ranks, capped at 8. `--workers 0` is useful when troubleshooting multiprocessing. Reduce fixed batch/imgsz on memory errors.

Precedence is code defaults, then YAML, then **only explicitly supplied** CLI values. CLI `device` replaces YAML `gpus` and vice versa; supplying both together is an error. CLI booleans take `true` or `false`, for example `--amp false --cache false`. Cache accepts `false`, `true`, `ram`, or `disk`. Inspect `python -m safety_yolo.train --help` for CLI options.

YAML additionally accepts `lrf`, `weight_decay`, `momentum`, `warmup_epochs`, `cos_lr`, `close_mosaic`, `mosaic`, `mixup`, `copy_paste`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`, `hsv_h`, `hsv_s`, `hsv_v`, `multi_scale`, `rect`, `deterministic`, `fraction`, `freeze`, `save`, `save_period`, `exist_ok`, `plots`, `val`, `pretrained`, and `verbose`. Unknown keys and null values are rejected. For key semantics see the [Ultralytics training reference](https://docs.ultralytics.com/modes/train/).

Default training output is `runs/train/safety/`, including `weights/best.pt`, `weights/last.pt`, `args.yaml`, and `results.csv`. Existing run names normally get a numeric suffix. Keep `save` enabled to retain checkpoints; use `--project` and `--name` to control locations.

Fresh training, prediction, and validation resolve default and relative `--project` paths to absolute working-directory paths before passing them to Ultralytics. They are not nested beneath its task-specific runs directory.

`--cpu-threads N` (YAML `cpu_threads`) requires a positive integer. Supported lifecycle callbacks reapply the setting after device selection and before the training or inference loop. Training also reapplies it for final validation. The caller's prior torch thread count is restored on completion or failure. Standalone validation warmup precedes its start callback and can use upstream's thread count. Resume does not accept `cpu_threads`; it retains the existing resume override restrictions.

### Resume an interrupted run

```bash
python -m safety_yolo.train --resume runs/train/safety/weights/last.pt
python -m safety_yolo.train --resume runs/train/safety/weights/last.pt --device cpu --workers 0 --batch 4
```

Resume loads a trusted local checkpoint and calls `train(resume=True)`. It preserves checkpoint epochs, optimizer/scheduler state, dataset, and run configuration instead of merging new-run defaults. The wrapper accepts only explicit `device`/`gpus`, `workers`, and `batch` changes. This subset was checked against `BaseTrainer.check_resume` in the pinned release. No `--config`, `--epochs`, `--imgsz`, `--data`, or other fresh-run settings are accepted with resume. Hardware and batch validity are checked against the effective checkpoint values before training.

A completed/stripped checkpoint without optimizer and epoch state is rejected; use an interrupted `last.pt`. To fine-tune completed weights, start a new run with `--model path/to/best.pt`. Preserve the original dataset paths for resume. Changing to CPU or Windows may need the explicit hardware overrides shown above.

## Predict and evaluate

Prediction and evaluation require an existing trusted local `.pt` file and one device. Predictions consume the streaming generator rather than collecting all frames in memory. Sources are local images, directories, videos, or a numeric camera index:

```bash
python -m safety_yolo.predict --model runs/train/safety/weights/best.pt --source data/construction-site-safety/images/test --device cpu
python -m safety_yolo.predict --model runs/train/safety/weights/best.pt --source input.mp4 --device 0 --save-txt
python -m safety_yolo.predict --model runs/train/safety/weights/best.pt --source 0 --device cpu --no-save
python -m safety_yolo.validate --model runs/train/safety/weights/best.pt --data data/construction-site-safety/data.yaml --split val --device cpu
python -m safety_yolo.validate --model runs/train/safety/weights/best.pt --data data/construction-site-safety/data.yaml --split test --device cpu
```

Prediction defaults to `runs/predict/safety/` and saves annotated media unless `--no-save` is used. It also exposes `--conf`, `--iou`, `--imgsz`, and `--batch`. Evaluation defaults to `runs/val/safety/` and writes `metrics.json` with precision, recall, mAP50, mAP50-95, and fitness, alongside native Ultralytics outputs. Both support `--project`, `--name`, and `--cpu-threads`; evaluation has an opt-in `--plots`. Use validation for tuning and reserve test evaluation for final assessment.

## Verification

```powershell
python -m pytest -q
$env:SAFETY_YOLO_SMOKE = '1'
python -m pytest -q -s tests/test_smoke.py
Remove-Item Env:SAFETY_YOLO_SMOKE
```

```bash
python -m pytest -q
SAFETY_YOLO_SMOKE=1 python -m pytest -q -s tests/test_smoke.py
```

The default suite uses synthetic local images and mocked HTTPS transport, not real exports, GPU computation, or external images. Setup regressions additionally run native offline uv dry-runs with temporary destination configurations (driver detection only, no installation); they skip if uv, this checkout's `.venv`, or exact-pin cached metadata is unavailable. Help and inspect tests forbid heavy imports. Subprocess contracts exercise real save-directory resolution, native DDP script generation/companion loading (including the thread callback), and real CPU prediction/validation thread counts. Hard-exit fixtures check recovery before fingerprint, image, label, summary, and YAML publication. The opt-in CPU test generates eight 64x64 images, prepares them offline, trains random-initialized YOLO26n for one epoch with batch 2/workers 0/AMP false, predicts, and evaluates both splits. It also checks thread counts during real training forwards. Real-framework fixtures block socket connections, disable automatic dependency installation, and provide a bundled local font in temporary config directories. No pretrained weights are needed. Contract subprocesses have a 120-second timeout; smoke subcommands have a 300-second timeout. Artifacts remain in pytest's temporary directory, printed on success. A one-epoch synthetic result is a plumbing check, not model-quality evidence. The DDP contract is a CPU serialization check, not a distributed training run.

Hardware setup was exercised on Windows/Python 3.12.8 with uv 0.11.25: auto selected **torch 2.14.1+cu130 / torchvision 0.29.1+cu130**, CUDA runtime **13.0**, and **one NVIDIA RTX 3500 Ada Generation Laptop GPU** (driver **596.71**). Fresh-child tensor computation and native torchvision NMS passed on CPU and `cuda:0`. The offline suite and temporary two-thread CPU pipeline also passed with this pair; dependency compatibility and lock freshness were checked without syncing away the hardware overlay. The pre-install CPU baseline was 2.15.0+cpu / 0.29.2+cpu. Full command evidence and resolved setup failures are recorded in `.claude/specs/hardware-setup/tasks.md`.

Earlier dataset checks are not new preparation acceptance evidence: the separate terminal C2 blocker remains blocked. No real dataset preparation, full training, or GPU model training was performed for this feature. Multi-GPU, MPS/ROCm hardware, live cameras, and real video files have not been exercised; hardware policy and camera streaming are covered with fakes. Real resume continuation is source-checked and mock-tested, not demonstrated by the completed one-epoch smoke checkpoint.

## Trust, licensing, and limitations

Load **only trusted checkpoints**. PyTorch model files may execute code while loading. Keep raw exports, signed URLs, images, caches, weights, and runs private. Ultralytics may create its own user settings/cache files; this project does not configure tracker credentials.

Ultralytics is distributed under **AGPL-3.0**, with commercial licensing options. Review its terms, model licenses, and the dataset's own license before distributing or deploying derived work. This repository does not grant rights to the source dataset.

This is a detection workflow, not a certified safety or compliance system. Detections can be wrong, biased, or missed; evaluate on representative deployment data. The labels do not establish which equipment belongs to which person, so do not infer person-level PPE compliance from co-occurring boxes. Human review and independent safety controls remain necessary.

### Module map

- `safety_yolo/dataset.py`: frozen manifest/image/box records, validation, caching, preparation.
- `safety_yolo/hardware.py`: device, global batch, workers, and AMP policy.
- `safety_yolo/setup.py`: opt-in local uv backend installation and fresh-process runtime verification.
- `safety_yolo/config.py`: YAML/CLI precedence and local data/model boundaries.
- `safety_yolo/train.py`, `predict.py`, `validate.py`: guarded command entry points using supported YOLO APIs.