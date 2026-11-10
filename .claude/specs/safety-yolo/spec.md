# Construction safety YOLO workflow

## Requirements

Provide a Python project that prepares the user's Ultralytics detection NDJSON export, trains a configurable YOLO detection model, predicts on images and videos, and evaluates validation or test data. Keep these commands in separate importable modules. Support CPU, one CUDA GPU, multiple CUDA GPUs on Linux or WSL2, and Apple MPS when available. Do not start a full training run automatically.

The verified input is `C:/Users/Billy.Ng/Downloads/construction-site-safety.ndjson`. Its first record has `type=dataset`, `task=detect`, and a contiguous `class_names` mapping of ten classes. It contains 2,801 image records with `type`, `file`, signed HTTPS `url`, `width`, `height`, `split`, and `annotations.boxes`. Boxes are already `[class_id, center_x, center_y, width, height]` with normalized coordinates. Preserve all supplied class IDs, coordinates, backgrounds, and splits. Expected counts are train 2,605, val 114, test 82. Never print signed query parameters.

## Acceptance

- Dataset inspect is read-only and does not import torch or download images.
- Dataset preparation validates the entire manifest before network requests. Validate filenames and collisions, split values, class IDs, finite normalized boxes, positive dimensions, and detection task. Do not silently drop bad records or reshuffle splits.
- Preparation supports local image sources as well as export URLs. Remote downloads use bounded concurrency, URL-keyed persistent disk cache, timeouts, retries, atomic writes, and image decoding/dimension checks. Failed responses do not enter the cache. Provide refresh and offline behavior. Repeated preparation reuses verified data. Do not log full URLs or transport exceptions containing signatures. Expired URLs get actionable export-refresh advice.
- Produce standard `images/{split}`, `labels/{split}`, a dataset YAML, and a sanitized summary. Do not publish a success YAML after partial failure. Avoid stale output mixing when the input manifest changes. Preserve source export in data if copying, never move/delete Downloads.
- Training accepts YAML configuration plus explicit CLI overrides for model, data, epochs, imgsz, batch, device, workers, cache, AMP, seed, project/name, and useful Ultralytics settings. Paths and defaults are documented. Use supported YOLO APIs, not a custom trainer.
- Device selection defaults to one available CUDA GPU, otherwise MPS, otherwise CPU. Accept explicit CPU, MPS, GPU IDs, or GPU count. Reject unavailable or duplicate GPUs. Native Windows multi-GPU gets a clear Linux/WSL2 message. Require a positive fixed global batch divisible by GPU count for multi-GPU; AutoBatch is single-CUDA only. Windows workers default to zero. Prediction uses a single device and streams results.
- Resume restores a local checkpoint with `resume=True`, does not silently overwrite checkpoint settings with fresh-run defaults, and explains permitted hardware overrides. Use main guards for multiprocessing.
- Validation supports val/test explicitly and writes useful metrics. Prediction requires explicit model/source and supports images, directories, videos, and camera indices.
- Pin project dependencies and Python runtime in an isolated `.venv`; create a reproducible lock. Do not globally install packages or force one CUDA wheel onto every machine. Document platform-specific torch installation.
- Include offline tests for malformed inputs, preservation, cache reuse/failure, path safety, hardware policy and configuration precedence/resume. Include a small opt-in real CPU smoke test of train/predict/validate.
- README has runnable Windows and Linux examples, setup, preparation, hardware variants, resume, outputs, tests, and limitations. No API key is required for local workflow, so no `.env` is needed.

## Non-goals

No Azure or AWS services, dashboards, experiment-tracker credentials, custom model architecture, deployment, full training, git initialization, commit, or PR creation.

## References

- https://docs.ultralytics.com/modes/train
- https://docs.ultralytics.com/datasets/detect/
- https://docs.ultralytics.com/modes/predict/
- https://docs.ultralytics.com/modes/val/
- https://docs.ultralytics.com/quickstart/