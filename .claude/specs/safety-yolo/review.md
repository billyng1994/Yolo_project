# Independent safety YOLO review

## Cycle 1 — 2026-11-10

Reviewing: Group 1, complete local workflow. Cycle 1 of the maximum 3. Group 2's concurrent real-data download was excluded from execution.

### Spec Alignment

Read `spec.md`, `design.md`, `tasks.md`, every `safety_yolo` module, every test, `configs/train.yaml`, `pyproject.toml`, and `README.md`. Also inspected `.python-version`, `.gitignore`, lock metadata, and the installed upstream implementations. No implementation, tests, configuration, lock, dataset, project cache, or project runs were changed. No agents, global installations, preparation of real data, or git operations were used.

The immutable manifest model, complete pre-download validation, split/background preservation, sanitized errors, bounded downloads, persistent full-URL cache identity, guarded CLIs, and ordinary CPU workflow are implemented. The five findings below prevent acceptance. Complete real-dataset preparation is still Group 2 work, not a result of this review.

### Critical

1. **C1. Rerun cleanup deletes valid manifest images.** [`safety_yolo/dataset.py:306-308`] Verified runtime data deletion in a temporary fixture. `safe_relative` accepts `.tmp-frame.png`, and initial preparation successfully publishes it. The next preparation recursively unlinks every `.tmp-*` file before computing the manifest's expected destinations. This deletes `images/train/.tmp-frame.png` and its label even though both belong to the unchanged manifest. After removing only the temporary fixture's URL cache, an offline rerun changed the verified image from present to absent, failed with `Preparation failed for 1 images`, and removed `data.yaml`. An expired remote URL can make the removed prepared copy unrecoverable. Exclude all expected manifest destinations from cleanup and distinguish preparer-owned staging files from user filenames. Add a regression covering a valid temporary-prefix filename, missing cache, and offline reuse. Do not solve this by silently dropping the record.

### Warning

1. **W1. Relative project paths do not match the documented working-directory contract.** [`safety_yolo/config.py:19`, `safety_yolo/config.py:127`, `safety_yolo/train.py:57`] Verified with the installed `ultralytics.cfg.get_save_dir`, without creating run directories. With fresh settings, `project='runs/train'` resolves to `<cwd>/runs/detect/runs/train/safety`, not `<cwd>/runs/train/safety`. Prediction and validation similarly resolve to `runs/detect/runs/predict/safety` and `runs/detect/runs/val/safety`. Consequently the default-output descriptions and downstream checkpoint examples at `README.md:104-125` reference the wrong location. Relative custom projects also depend on upstream user settings rather than solely on the working directory. Normalize project paths to absolute working-directory paths before passing them upstream, consistently across all commands, and test the documented defaults. The real smoke test uses an absolute project at `tests/test_smoke.py:35`, so it cannot detect this mismatch.

2. **W2. The locked environment omits a required DDP serialization dependency.** [`pyproject.toml:10-16`] Verified by executing the installed `ultralytics.utils.dist.generate_ddp_file` with a temporary CPU `Linear` model and trainer argument container. `importlib.metadata` reports `cloudpickle` missing, and generation raises `ModuleNotFoundError: No module named 'cloudpickle'` with automatic installation disabled. Upstream `utils/dist.py:71-72` checks and then imports `cloudpickle>=3.1.1`; it serializes the trainer/model/callback state into the companion `.pt` file. No GPU is necessary to reproduce this prerequisite failure. An online default launch may auto-install it, but that is an unpinned mutation outside the supposedly reproducible dependency set; a local/offline launch fails. Pin and lock the DDP requirement or provide a pinned documented DDP extra. Test generation and companion-state loading without automatic installation. The installed loader explicitly defaults `weights_only=False`; there is no evidence for a separate Torch weights-only loading defect. Actual multi-GPU execution was not performed.

3. **W3. CPU thread overrides are overwritten before inference/training.** [`safety_yolo/config.py:110-117`, `safety_yolo/train.py:50`, `safety_yolo/predict.py:21`, `safety_yolo/validate.py:21`] Verified through the real prediction and validation entry functions using temporary smoke weights/images, not fake models. Both commands with `--cpu-threads 1` called `torch.set_num_threads` with `[1, 8]` and finished with 8 effective threads. The installed `select_device('cpu')` resets the limit to upstream `NUM_THREADS`; it also does so for MPS. A direct probe likewise changed 2 to 8. Training calls the helper before its trainer selects a device, so it has the same ordering problem. Apply the limit after upstream device/model initialization through a supported lifecycle hook, or remove the unsupported promise. Assert the thread count at the actual compute boundary, rather than only testing that the helper ran. This is a proved configuration defect, not a measured performance regression.

4. **W4. An interrupted initial state publication cannot recover through the documented retry.** [`safety_yolo/dataset.py:300-309`, `README.md:73`] Verified by running a temporary preparer subprocess and replacing `os.replace` in that subprocess with `os._exit(73)`. The process exits during the first atomic `.manifest.sha256` write, before any image fetch, leaving `.prepare.lock` and `.tmp-<random>` but no state file. After confirming subprocess termination and removing its lock, the retry fails with `Output belongs to a different or unknown manifest; use a new output directory`. The unknown-output check runs before orphan cleanup, contrary to the documented orphan cleanup on retry. Provide a safe recovery path for an interrupted initialization containing only provably preparer-owned staging artifacts, while retaining refusal of genuinely foreign datasets. Add a hard-exit regression, not only a test that places an orphan inside an already initialized output.

### Suggestion

- The tests are meaningful but incomplete. `tests/test_commands.py:29-59` checks forwarding with a fake checkpoint, and `tests/test_hardware.py` checks policy with fake availability. Neither proves upstream DDP serialization or actual hardware execution. Extend integration coverage at the boundaries identified by W1-W3. Keep the existing clear hardware limitations.
- Consider adding the real interrupted-checkpoint continuation probe described below as an opt-in test. This review exercised that behavior successfully, whereas the committed smoke test's completed checkpoints are correctly rejected for resume.

#### No-comments read-only pass

Performed before substantive behavioral review. Files touched: none. Actual deletions: 0. Optional deletions: 6 comment/docstring instances across 6 files, listed below. No `MUST KILL` code-refactor flags, restorations, encoding changes, or reruns were required.

- `safety_yolo/__init__.py:1`. The package summary repeats its name and implementation layout.
- `safety_yolo/hardware.py:1`. The module summary repeats the module's role.
- `safety_yolo/config.py:59`. The selector-precedence comment narrates the immediately following branches.
- `safety_yolo/dataset.py:303`. The symlink comment narrates the immediately following check.
- `tests/test_smoke.py:1`. The test summary repeats the test name and visible fixture setup.
- `configs/train.yaml:8`. The workers comment repeats the documented default policy.

Retain module docstrings used as argparse help, the `DatasetError` public error contract, the path-resolution rule, the signed-error privacy rationale, the real-export omitted-annotations explanation, and the upstream label-cache explanation. The initial preliminary chat inventory was narrowed after tracing which docstrings actually supply CLI help. These are optional deletions only and do not affect the verdict.

### Cross-Task Consistency

- Signed URL queries are omitted from record representations, cache filenames, summary output, and handled transport-error output. Existing tests exercised HTTP 403, transport errors, corrupt images, dimension failures, cache identity, refresh, retry, and concurrent fetching. No signed URL privacy violation was proved. These mocked transports are not proof that the concurrent real export download completed.
- Config precedence and explicit false values work in the offline tests. CLI flags in the README are accepted by the corresponding parser definitions and six guarded help tests. The output-path mismatch remains W1.
- Native Windows multi-GPU refusal, fixed global-batch divisibility, duplicate/unavailable IDs, and single-CUDA AutoBatch policy have useful fake-backed coverage. A separate probe of the installed upstream selector with CUDA calls explicitly mocked and `CUDA_VISIBLE_DEVICES='3,7'` preserved that variable, used visible indices `0,1`, and selected visible index 1 for an explicit single-device request. No device-visibility bug was proved. No claim is made that physical CUDA/MPS devices or Linux DDP ran.
- A genuine interrupted CPU checkpoint saved `device='cpu'` as `str`, batch `2` as `int`, epoch `0`, and 366 optimizer state entries. Serialized EMA tensors were FP16; `YOLO(last.pt)` loaded FP32 tensors as intended. `train.run({}, resume=last)` continued from epoch 2 of 2 and preserved the original run, yielding CSV epochs `[1, 2]`. No fresh-run-default overwrite or saved device/dtype defect was observed. This used an in-memory upstream `NUM_THREADS=2` patch solely to bound the temporary reproduction despite W3; application files were unchanged.
- `tasks.md` accurately disclosed that Group 1 had not demonstrated real interrupted resume or GPU execution. This review supplies CPU resume evidence only. It does not complete Group 2.

### Tests

- [x] Offline suite passed independently. `86 passed, 1 skipped`.
- [x] Opt-in real CPU smoke passed independently. `1 passed`. It launches `python -m safety_yolo.dataset`, `.train`, `.predict`, and `.validate`, verifies weights and annotated media, and reads metrics from both splits.
- [x] Installed dependency integrity passed. `46 packages`, all compatible.
- [x] Editor diagnostics reported no errors in application or tests.
- [x] Temporary runtime reproductions proved C1 and W1-W4.
- [x] Genuine CPU checkpoint interruption and continuation passed.
- [ ] Test coverage adequate for acceptance. Missing regressions are specified in C1 and W1-W4.
- [ ] Physical CUDA, Linux/WSL2 DDP, MPS, live camera, and video execution. Not available/executed in this review.

Verified interpreter was project-local Python 3.12.8. Installed metadata matched Ultralytics 8.4.208, torch 2.15.0 with runtime build `2.15.0+cpu`, torchvision 0.29.2, Pillow 13.0.0, PyYAML 6.0.3, and pytest 9.2.0. These versions were verified, not dismissed based on release-date assumptions.

#### Verification commands and reproduction recipe

Commands ran in noninteractive PowerShell from the project root using `.venv\Scripts\python.exe`. Set `PYTHONDONTWRITEBYTECODE=1`, `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false`, `WANDB_MODE=disabled`, and `YOLO_CONFIG_DIR` beneath a fresh OS temporary directory. Do not point any reproduction at real data/cache/runs.

```powershell
$reviewTmp = Join-Path ([IO.Path]::GetTempPath()) ('safety-yolo-review-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $reviewTmp | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:WANDB_MODE = 'disabled'
$env:YOLO_CONFIG_DIR = Join-Path $reviewTmp 'yolo'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp (Join-Path $reviewTmp 'offline')
$env:SAFETY_YOLO_SMOKE = '1'
try {
    .\.venv\Scripts\python.exe -m pytest -q -s -p no:cacheprovider --basetemp (Join-Path $reviewTmp 'smoke') tests/test_smoke.py
} finally {
    Remove-Item Env:SAFETY_YOLO_SMOKE -ErrorAction SilentlyContinue
}
& 'C:\Users\Billy.Ng\AppData\Local\miniforge3\Scripts\uv.exe' pip check --python '.\.venv\Scripts\python.exe'
```

Two preliminary terminal attempts were interrupted, including a `KeyboardInterrupt` during pytest import. The completed retry produced the passing result above; interrupted attempts were not counted as passes.

Minimal upstream repros, runnable with the same environment, make no project output writes:

```powershell
@'
from types import SimpleNamespace
import torch
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils.dist import generate_ddp_file
for mode in ('train', 'predict', 'val'):
    args = get_cfg(overrides={'task':'detect', 'mode':mode, 'project':f'runs/{mode}', 'name':'safety'})
    print(mode, get_save_dir(args))
torch.set_num_threads(2)
from ultralytics.utils.torch_utils import select_device
select_device('cpu', verbose=False)
print('Effective threads', torch.get_num_threads())
trainer = SimpleNamespace(args=SimpleNamespace(device='0,1', batch=16), model=torch.nn.Linear(2,1), callbacks={})
generate_ddp_file(trainer)
'@ | .\.venv\Scripts\python.exe -
```

For C1, use a temporary NDJSON with one 32x24 remote-backed training image named `.tmp-frame.png`, valid normalized boxes, and a synthetic HTTPS URL. Mock only `dataset.build_opener` to return `io.BytesIO` containing a valid 32x24 PNG. Prepare once, remove that fixture's hashed `.img` cache entry, then prepare the unchanged manifest with `offline=True`. The image is deleted and the rerun fails. For W4, replace `dataset.os.replace` with `lambda *args: os._exit(73)` in an isolated subprocess preparing a fresh temporary output. After it exits, remove that fixture's lock and retry. Both steps are before any need for real network access.

Real command thread probes called `predict.main` and `validate.main` with the smoke fixture's local model, `--device cpu --imgsz 64 --cpu-threads 1`, and an absolute temporary `--project`. A wrapping spy on the real `torch.set_num_threads` recorded `[1, 8]` for each, without replacing model inference.

For the genuine resume probe, train `yolo26n.yaml` with the temporary smoke dataset, total `epochs=2`, batch 2, workers 0, AMP/pretrained/plots false, and an absolute temporary project. A factory returns a real `YOLO` instance with an `on_model_save` callback that raises a distinctive exception after the first real checkpoint write. Inspect that checkpoint with the installed trusted loader, discard the callback-bearing instance, then invoke `train.run({}, resume=str(last))`. Assert CSV epochs `[1, 2]`.

This review's retained temporary evidence is under `C:/Users/Billy.Ng/AppData/Local/Temp/safety-yolo-review-24b5c11755e0487c89234cbe4bef285a/`. The detailed resume evidence is `resume-repro/evidence.log`; smoke artifacts are under `smoke/test_real_cpu_pipeline0/`. Temporary evidence is not a permanent test fixture and may be removed by OS cleanup.

### Verdict: FAIL

Reason: 1 Critical and 4 Warning findings remain. Passing baseline tests and CPU smoke do not cover the demonstrated failures. No Severe/High security vulnerability was proved.

Principle applied: **Prove It Works** changed the review from source/fake-only inference to pinned-upstream probes, real module smoke, hard-exit fixtures, and genuine checkpoint continuation. Its leaf skill was read in full. No implementation fixes were made.

## Cycle 2, 2026-11-10

Reviewing: Group 1, the scoped C1 and W1-W4 fix wave and its integration regressions. Cycle 2 of the nonresetting maximum 3. Cycle 1 remains intact.

### Spec Alignment

Read `poteto-mode/SKILL.md` in full, including Principles, before investigation. Read the spec, design, tasks, prior review, all seven package files, all six test files, README, training configuration, dependency manifest, runtime pin, ignore rules, and relevant lock entries from current disk. Read the installed native predictor, validator, trainer, dataset base, and DDP generator. Inspected the actual save-directory resolver, thread selector, and checkpoint loader. No implementation, tests, configuration, lock, real `data/`, project `.cache/`, or project `runs/` were modified. The sole workspace edit is this append. Verification scripts, fonts, settings, images, caches, checkpoints, and logs are OS temporary artifacts. No installations, git operations, real-data preparation, or full training were performed.

The original C1 and W1-W4 findings are closed for their demonstrated failure cases. One new Warning is reproduced below. The successful baseline and smoke runs do not cover that regression. Group 2 remains outside this review's completion claim.

### Critical

None. C1's manifest-image deletion is no longer reproduced. No Severe/High security vulnerability was proved.

### Warning

1. **W5. Native disk image caches make repeat preparation fail.** [`safety_yolo/dataset.py:314-322`] **Runtime reproduced locally, not awaiting live validation.** The new output allowlist includes the three native label caches but excludes native image `.npy` caches. The installed `ultralytics/data/base.py:206,215-216,407-414` creates these files beside manifest images for the supported `cache='disk'` setting. Prepare an unchanged local fixture successfully, construct the real `YOLODataset` with `cache='disk'`, then prepare it again offline. The new call raises `Unexpected dataset artifacts; use a clean output directory` at line 322. The native cache was `images/train/train.npy`, 2,432 bytes. The existing YAML, original image, and cache remained byte-identical, so this is not data loss. A second reproduction through the real `safety_yolo.train.run` wrapper completed a tiny one-epoch `cache='disk'` run, created six train/val `.npy` files, and produced the same failure on repeat preparation. This breaks the combination of repeat preparation at `README.md:72` and supported disk caching at `README.md:101`. Recognize and safely invalidate or reuse only exact manifest-derived native image-cache files, with file/type and expected-directory collision checks. Preserve refusal of genuinely foreign files. Do not reintroduce wildcard deletion. Add a real native disk-cache-to-prepare regression; checking only label caches misses this contract. Evidence is in `cycle2-probe.log` and `cycle2-success-probe.log` under the temporary root below.

### Suggestion

No new suggestions in this fix-verification cycle. Cycle 1's optional suggestions are not new blocking findings.

### Cross-Task Consistency

- **C1 verified fixed.** [`safety_yolo/dataset.py:212-230,303-338,375-376`] Publication stages exact metadata/image/label destinations under a separate root namespace. No recursive filename-prefix deletion remains. The committed remote-backed `.tmp-*` regression passed without its URL cache. An additional fixture used `.tmp-frame.png`, `.tmp-folder/.tmp-nested.png`, `.prepare-staging/frame.png`, `summary.json/frame.png`, `.manifest.sha256/frame.png`, and `train.cache/frame.png`. All six valid manifest images and six labels survived an offline rerun byte-for-byte after removing only that fixture's download cache. These names do not collide with root metadata, root staging, or the split-level label-cache paths. W5 is a separate allowlist regression, not a recurrence of C1.
- **W1 verified fixed.** [`safety_yolo/config.py:112-115`, `safety_yolo/train.py:51`, `safety_yolo/predict.py:22`, `safety_yolo/validate.py:22`] All three fresh-run entry points normalize project paths before forwarding. The committed tests compare default, custom relative, and absolute projects against the real `get_save_dir`, covering nine command/path combinations. Native `ultralytics/cfg/__init__.py:563-573` only prepends its settings directory to relative projects, so forwarded absolute projects bypass the original unwanted nesting. Resume forwarding remains unchanged and its regression passed.
- **W2 verified fixed.** [`pyproject.toml:16`, `uv.lock:74,754,769`, `tests/test_upstream.py:36-66`] Installed metadata, direct dependency, and lock all specify cloudpickle 3.1.1. Real `generate_ddp_file`, native companion `torch_load`, restored model inference, restored callback execution, and cleanup passed with `YOLO_AUTOINSTALL=false`. An additional subprocess executed the actual generated `.py` script and loaded its companion state with a CPU trainer test double. It restored deliberately set Linear weights, executed the callback with one thread, and reported `{"threads": 1, "shape": [1, 1], "autoinstall": false}`. Both generated files were removed by native `ddp_cleanup`. The real loader at `ultralytics/utils/patches.py:197-221` defaults to `weights_only=False`. This proves serialization prerequisites and generated-script loading, not distributed GPU training.
- **W3 verified fixed at compute boundaries.** [`safety_yolo/config.py:118-139`, `safety_yolo/train.py:53-63`, `safety_yolo/predict.py:33-41`, `safety_yolo/validate.py:27-34`] Real prediction and validation tests observed exactly one requested thread during compute and restored the prior count of two. Smoke training forwards observed exactly two requested threads. A separate complete tiny training run observed `[1, 1]` at its actual loss-forward calls and restored the prior count of three. Additional injected exceptions at real prediction inference, validation model forward after preprocessing, and training loss forward each observed one requested thread and restored three. The partial callback also survived both native companion loading and execution of the generated DDP script. Native prediction invokes its start callback before warmup/inference; trainer selection precedes its start hooks. Standalone validation warmup, including the final standalone validation invoked by training, still precedes `on_val_start`. The documented warmup exception is retained; no claim is made that this pre-callback initialization always uses the requested count.
- **W4 verified fixed for the specified hard exits.** [`safety_yolo/dataset.py:303-338`, `tests/test_dataset.py:238-306`] All five subprocess cases really exited with code 73 before replacement of the initial fingerprint, image, label, summary, or YAML. After subprocess completion and manual removal of only their temporary stale lock, retries succeeded. Initial adoption requires the complete matching staged fingerprint. Mismatched initial content, unknown output files, foreign staged files, and foreign staged images were refused without deletion or invalidating an existing success YAML. An incomplete initial fingerprint is deliberately not treated as proof of ownership. These tests establish process-exit recovery at the selected boundaries, not power-loss durability or recovery of every partial initial write.
- **Completion notes checked.** `tasks.md` accurately reports 110 passing offline tests, one skipped opt-in test, the real smoke, cloudpickle pin, and hardware limitations. The independent reruns agree. Its claim of complete acceptance needs the W5 native-cache interaction covered. The new allowlist's treatment of label caches alone is not adequate verification of supported native cache modes.
- **No-comments pass.** Reviewed the fix wave's callback and staging comments/docstrings read-only. No files or comments were deleted, no suppression was added, and no new blocking comment finding was raised. Cycle 1's optional deletions were not reopened.
- **Security scope.** The secret-assignment scan found only the synthetic privacy sentinel at `tests/test_dataset.py:127`. Signed-error/cache tests passed. Trusted checkpoint loading remains an explicit boundary, not a safe-unpickle guarantee. No AWS resources are in scope. Semgrep was not installed; no scanner installation or network-backed scan was attempted.

### Tests

- [x] Complete offline suite. **110 passed, 1 skipped in 33.98s**.
- [x] Opt-in real CLI CPU smoke. **1 passed in 26.34s**. Preparation, one tiny epoch, checkpoints, streamed prediction, and val/test metric files passed. Training compute thread assertions passed.
- [x] Installed dependency integrity. **47 packages**, all compatible.
- [x] Lock consistency checked offline. **66 packages resolved**, exit code 0.
- [x] Real native DDP generation/loading/callback tests, real inference thread tests, and all six lazy help tests passed within the suite.
- [x] Additional temporary namespace, generated-script, successful training restoration, and real compute-exception probes passed their assertions. Native disk-cache rejection was independently reproduced twice.
- [x] Editor diagnostics reported no source/test errors.
- [ ] Test coverage adequate for acceptance. The suite does not cover W5; its green result misses a real supported-workflow regression.
- [ ] Physical CUDA, Linux/WSL2 multi-GPU, MPS, live camera, and real video execution. Not executed here.

#### Exact commands and retained evidence

PowerShell working directory was `C:\Users\Billy.Ng\Desktop\Projects\Yolo_Project`. The explicitly configured project interpreter was Python 3.12.8 at `.venv\Scripts\python.exe`. Runtime metadata confirmed Ultralytics 8.4.208, torch 2.15.0 with runtime build 2.15.0+cpu, torchvision 0.29.2, cloudpickle 3.1.1, and pytest 9.2.0.

`$reviewTmp` was `C:\Users\Billy.Ng\AppData\Local\Temp\safety-yolo-review-c2-e57b96d36ac749b3aaea7f79d9aab3e1`.

The environment was `PYTHONDONTWRITEBYTECODE=1`, `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false`, `WANDB_MODE=disabled`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and `UV_OFFLINE=true`. `YOLO_CONFIG_DIR` was `$reviewTmp\yolo`; `MPLCONFIGDIR` was `$reviewTmp\mpl`. Real-framework test/probe subprocesses also blocked socket connections and used temporary bundled local fonts. No automatic dependency installation was allowed.

Executed commands, with results above:

1. `.\.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider --basetemp (Join-Path $reviewTmp 'offline') 2>&1 | Tee-Object -FilePath (Join-Path $reviewTmp 'offline.log')`.
2. Set `$env:SAFETY_YOLO_SMOKE = '1'`, then `.\.venv\Scripts\python.exe -m pytest -q -s --tb=short -p no:cacheprovider --basetemp (Join-Path $reviewTmp 'smoke') tests/test_smoke.py 2>&1 | Tee-Object -FilePath (Join-Path $reviewTmp 'smoke.log')`. Removed `SAFETY_YOLO_SMOKE` in `finally`.
3. `& 'C:\Users\Billy.Ng\AppData\Local\miniforge3\Scripts\uv.exe' pip check --python .\.venv\Scripts\python.exe`.
4. `& 'C:\Users\Billy.Ng\AppData\Local\miniforge3\Scripts\uv.exe' lock --check --python .\.venv\Scripts\python.exe --no-progress --offline`.
5. `.\.venv\Scripts\python.exe (Join-Path $reviewTmp 'cycle2_probe.py') $reviewTmp 2>&1 | Tee-Object -FilePath (Join-Path $reviewTmp 'cycle2-probe.log')`.
6. `.\.venv\Scripts\python.exe (Join-Path $reviewTmp 'cycle2_success_probe.py') $reviewTmp 2>&1 | Tee-Object -FilePath (Join-Path $reviewTmp 'cycle2-success-probe.log')`.

Each command checked `$LASTEXITCODE` and completed with exit code 0. The expected W5 exceptions were explicitly asserted inside the probe scripts; they were not mistaken for passing preparation. Native API source and exact finding citations were retained in `native-source-evidence.log`. The complete probe scripts remain beside their logs. Their temporary directories are intentionally single-use; use a fresh temporary root and smoke fixture to rerun them. OS cleanup may remove these artifacts.

#### Limitations and review integrity

The original genuine interrupted-checkpoint continuation evidence remains in Cycle 1. It was not repeated because that path was not changed; existing resume policy/forwarding tests were rerun. No claim is made that this review completed the real dataset download, full training, or physical GPU/MPS validation. DDP execution used a CPU trainer test double, not `torch.distributed` workers. Real exception tests injected deliberate compute failures; they prove restoration on those exceptions, not every possible hardware failure. Hardware availability tests remain mocked. Independent plugin-review and teammate-message tools were unavailable, so no additional agent verdict or implementer clarification is claimed. No subagents were spawned.

The recorded SHA256 of the entire pre-append `review.md` was `06FC920B7C0361963D43959EF0562869937F96F4A260D40B29582B5CFBA2CE09`. This cycle appends evidence rather than rewriting the previous verdict.

### Verdict: FAIL

Reason: **0 Critical, 1 Warning, 0 new Suggestions.** C1 and W1-W4 are verified fixed, but W5 breaks repeat preparation after supported native disk caching. Cycle 2 is complete and does not reset the budget. Any subsequent independent review is Cycle 3, the terminal cycle.

Principle applied: **Prove It Works** changed the review from trusting the reported green suite to executing the installed DDP script, probing actual compute/restoration, testing staging-like legitimate names, and reproducing the native disk-cache interaction through real training. Its leaf skill was read in full. No implementation fixes were made.

## Cycle 3, 2026-11-10

Reviewing: Group 1, W5 native disk-cache fix and C1/W4 safety regressions. This is the terminal cycle of the nonresetting maximum 3. Per the explicit handoff, the preceding model-validation failure occurred before spawn and produced no reviewer or artifacts. It did not consume this invocation. No further review cycle or automatic fix wave is authorized.

### Spec Alignment

Read `poteto-mode/SKILL.md` in full, including Principles, before investigation. Read the spec, design, tasks, both previous cycles, current `dataset.py`, dataset/upstream tests, shared fixtures, CPU smoke, dependency manifest, and installed native `BaseDataset`. Loaded the required spec-workflow skill and verification principle leaves. The review stays within W5 and existing source-preservation/recovery requirements. Other original fixes retain Cycle 2's evidence unless the rerun detects a regression. Group 2 is not completed by this review.

The data under review consists of immutable manifest image records, their resolved local source paths, expected prepared files/directories, and derived native cache paths. Native image caches use `Path(image).with_suffix('.npy')`. The fix subtracts prepared destinations and directories from the deletion set, but does not subtract resolved manifest sources. That missing distinction causes the verified data-loss finding below.

Throughput checkpoint is not applicable to this read-only investigation. No subagents, installations, implementation edits, real-data preparation, git operations, or full training were performed. Only this Cycle 3 section is appended in the workspace. All generated settings, fixtures, native caches, logs, weights, smoke runs, and junctions are beneath the OS temporary evidence root.

### Critical

1. **C2. W5 invalidation can delete a verified manifest source that resolves to a native-cache path.** [`safety_yolo/dataset.py:321-322,357-362`] **Runtime reproduced on Windows in a temporary fixture.** `image_caches` excludes `expected` files and manifest directories, not `ImageRecord.local` paths. Local sources are resolved at lines 187-188. An accepted manifest-local alias can resolve inside the prepared output while remaining inside the manifest directory. The output link scan does not catch an alias located outside the output. After `fetch_image` verifies that local source, line 360 unlinks it; line 362 then tries to read the deleted source. Preparation fails, the source is gone, and the destination was never published. This is actual source-file loss, not an unexecuted security hypothesis.

    Reproduction used the ordinary public `read_manifest` and `prepare` functions without mocking either function or altering implementation. A one-image manifest specifies `file='frame.png'`, `split='train'`, dimensions 32x24, and local `url='source-alias/frame.npy'`. First prepare offline with that source missing. The normal failure leaves the matching fingerprint and no success YAML. Create a Windows directory junction `source-alias` next to the manifest, targeting `prepared/images/train`, and write a valid 107-byte PNG to that directory as `frame.npy`. Read the unchanged manifest again. Assert that its resolved local source equals `prepared/images/train/frame.npy` and run the real `verify_image` successfully. Retry preparation offline with one worker. The retry reports one failed image, the verified source changes from present to absent, and `images/train/frame.png` remains absent. No foreign file or link was injected inside the output. Source extensions are not restricted by the local-source contract, and this file passed the actual decoder and dimension checks.

    Preserve all resolved manifest source identities across every worker before treating a path as deletable cache, or reject such overlaps non-destructively before publication/invalidation. The preservation check must consider every record's source, not only the current worker's source. Add a regression for the accepted alias/retry case. This recommendation identifies the blocker; it does not authorize implementation work after the terminal verdict. Evidence is `alias-collision.log` under the temporary root below.

### Warning

None separately rated. The initial full-suite failure is disclosed in Tests. Its root cause and relationship to W5 were not established, so this review does not invent a second regression finding.

### Suggestion

No new suggestions. Cycle 1's optional comment/style suggestions are not reopened.

### Cross-Task Consistency

- **Original W5 failure case passes.** `tests/test_upstream.py:36-87` really constructs `YOLODataset(cache='disk')` for train, val, and test. It checks populated native cache files and compares actual reader pixels with OpenCV decoding. Preparation runs with `numpy.load` forbidden, without a download cache, and preserves the original sources, prepared images, and labels byte-for-byte. The changed-pixel loop tests both refresh settings. The next iteration reads the previously changed pixels through a new real reader; the final reader checks the final changed pixels. This is behavioral evidence, not a fake native cache or an assertion only about helper calls.
- **Native naming and stale-read risk confirmed.** Installed `ultralytics/data/base.py` derives `npy_files` with `with_suffix('.npy')`, creates disk caches, and reads those files through `np.load`. Invalidating exact caches is justified when pixels change without a manifest change. C2 concerns ownership of the file being deleted, not the need to invalidate genuine caches.
- **C1 and W4 selected safeguards pass.** The focused run exercises remote-backed temporary-prefix image/label preservation without a download cache, all five actual hard-exit boundaries and stale-lock retries, matching initial staging, foreign staging refusal, and foreign-file preservation. The test subprocesses actually exit 73; these are not simulated orphan-file-only assertions. The new invalidator does not restore wildcard cleanup. This does not negate C2's separate resolved-source deletion.
- **Prepared-directory and file-type collisions pass.** Cases include `frame.npy/other.png`, nested children, and the Windows case variant. The exact candidate-directory subtraction preserves these directories and their manifest files. Both file/directory mismatch cases refuse preparation. Unknown `.npy` variants, wrong-split paths, and foreign output/staging files remain unchanged on refusal. Source aliases are absent from this coverage, which is why the green focused suite misses C2.
- **Other original fixes retain verification.** The complete suite rerun includes native DDP serialization/loading and callback execution, real predict/val thread-boundary tests, command project-path comparisons, resume forwarding/policy, and lazy help paths. The actual CPU smoke rerun includes training thread assertions. Physical distributed training and genuine interrupted-checkpoint continuation were not repeated; prior-cycle evidence and stated limitations remain unchanged.
- **Read-only comment pass.** No comments or suppressions were edited. The scoped comments do not resolve or conceal the missing local-source exclusion. No new blocking comment finding was raised.

### Tests

- [x] Final complete suite rerun passed. **122 passed, 1 skipped in 33.16s**.
- [x] Focused dataset/native-disk-cache run passed. **82 passed in 9.49s**.
- [x] Opt-in real CLI CPU smoke passed. **1 passed in 23.46s**. It prepared synthetic data, trained one tiny epoch, verified checkpoints and saved prediction output, validated both splits, and checked training compute threads.
- [x] Dependency integrity passed offline. **47 packages**, all compatible. Lock consistency passed offline with **66 packages resolved**.
- [x] Editor diagnostics reported no errors in the three changed Python files.
- [x] Independent C2 probe reproduced deletion with an actual accepted manifest, actual verified PNG source, and actual Windows junction outside the prepared output.
- [ ] Every suite execution passed. The first complete run returned **1 failed, 121 passed, 1 skipped in 33.97s**. `tests/test_dataset.py:163`, `test_retry_and_corrupt_cache_recovery`, observed four transport calls instead of three. It passed in the focused run and the complete retry without any code/test change. The cause remains unverified; the first failure is not erased by the retry.
- [ ] Source-preservation coverage adequate for acceptance. Existing tests miss C2.
- [ ] Real dataset completion, physical CUDA, Linux/WSL2 multi-GPU, MPS, live camera, and real video execution. Not performed here.

#### Commands, evidence, and limitations

Working directory was `C:\Users\Billy.Ng\Desktop\Projects\Yolo_Project`. The explicitly configured project `.venv\Scripts\python.exe` reports Python 3.12.8. Installed metadata reports Ultralytics 8.4.208, torch 2.15.0, torchvision 0.29.2, cloudpickle 3.1.1, and pytest 9.2.0. No dependency was installed or updated.

`$c3Tmp` is `C:\Users\Billy.Ng\AppData\Local\Temp\safety-yolo-review-c3-ff9410637f4946e1b4fbc16a6822a896`. Environment variables were `PYTHONDONTWRITEBYTECODE=1`, `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false`, `WANDB_MODE=disabled`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and `UV_OFFLINE=true`. `YOLO_CONFIG_DIR` and `MPLCONFIGDIR` point beneath that temporary root. Native test and smoke subprocesses block socket connections. The synthetic local-source probe has no remote records and uses offline preparation.

Executed commands and outcomes:

1. `.\.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider --basetemp (Join-Path $c3Tmp 'all')`. Exit 1, initial failure detailed above. Log `all.log`.
2. `.\.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider --basetemp (Join-Path $c3Tmp 'focused') tests/test_dataset.py tests/test_upstream.py::test_actual_disk_image_cache_allows_offline_reprepare`. Exit 0, 82 passed. Log `focused.log`.
3. Set `SAFETY_YOLO_SMOKE=1`, then `.\.venv\Scripts\python.exe -m pytest -q -s --tb=short -p no:cacheprovider --basetemp (Join-Path $c3Tmp 'smoke') tests/test_smoke.py`. Exit 0, 1 passed. Removed the opt-in flag in `finally`. Log `smoke.log`.
4. `& 'C:\Users\Billy.Ng\AppData\Local\miniforge3\Scripts\uv.exe' pip check --python .\.venv\Scripts\python.exe`. Exit 0. Log `deps.log`.
5. `& 'C:\Users\Billy.Ng\AppData\Local\miniforge3\Scripts\uv.exe' lock --check --python .\.venv\Scripts\python.exe --no-progress --offline`. Exit 0. Log `lock.log`.
6. `.\.venv\Scripts\python.exe -m pytest -q --tb=short -p no:cacheprovider --basetemp (Join-Path $c3Tmp ('all-retry-' + [guid]::NewGuid().ToString('N')))`. Exit 0, 122 passed and 1 skipped. Log `all-retry.log`.
7. `.\.venv\Scripts\python.exe -c $aliasProbe` ran before and after `New-Item -ItemType Junction -Path (Join-Path $collisionRoot 'source-alias') -Target (Join-Path $collisionRoot 'prepared/images/train')`. Both calls exited 0 after asserting the expected failed preparation and reproduced source loss. `$collisionRoot` is `$c3Tmp\alias-collision-ad4682acc450486793a9a2f6f68fbaa8`. Log `alias-collision.log` ends with `REPRODUCED_LOCAL_SOURCE_DELETION`. The fixture recipe is given in C2; the PowerShell probe variable is session-local.

An earlier combined terminal attempt was interrupted. Subsequent probe attempts encountered an existing temporary directory and then the correctly enforced direct `images`-component path ban. None counts as a successful reproduction. The final alias probe uses fresh state and passes its explicit deletion assertions. The direct-path rejection was considered before rating C2, not hidden. No implementation or test was changed to obtain any result.

Immediately before this append, SHA256/size/last-write and directory-entry comparison found **zero changes across 8,443 snapshot entries**, including **8,404 `data/`, `.cache/`, and `runs/` entries covered by the snapshot**. Source, tests, configuration, specs, lock, and root files were unchanged. The pre-append `review.md` SHA256 was `E14511EF3E804D0AD286091B88CCBC72F90128E46B7D57917503BB6331AEFE81`. The final integrity check verifies that these bytes remain the exact prefix and that this file is the sole changed workspace file.

No additional reviewer/plugin verdict or implementer clarification is claimed. No subagents were used, as requested. No AWS resources are in scope. This review does not establish power-loss durability, concurrent trainer/preparer safety, or hardware behavior beyond the stated CPU tests. Temporary evidence may be removed by OS cleanup.

### Verdict: FAIL

Reason: **1 Critical, 0 separately rated Warnings, 0 new Suggestions.** Ordinary native disk-cache reruns and refreshed-pixel reads now pass, but exact-name cache invalidation can delete an accepted resolved local source. Source preservation is an existing requirement. **Group 1 is blocked for the user's decision. Cycle 3 is terminal. No fixes, fix agents, further changes, or new review cycles are authorized by this verdict.** The task-status blocker is recorded here rather than editing `tasks.md`, which is outside the allowed write scope.

Principles applied: **Prove It Works** changed the decision from trusting green suites to testing actual file survival at the deletion boundary. **Test Behavior, Not Implementation** changed the assessment of the native regression to check actual reader pixels across both refresh modes, and exposed the missing source-survival assertion. Both leaf skills were read in full this session. No implementation fixes were made.