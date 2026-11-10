# Design

## Selected shape

Use a flat `safety_yolo` Python package runnable with `python -m safety_yolo.dataset`, `.train`, `.predict`, and `.validate`. Domain structures are frozen manifest, image, and box records. Parse external NDJSON once into those structures. The dataset module owns all annotation rules and preparation state. A shared hardware module owns device selection and batch/worker policy. Keep configuration handling small and CLI wrappers thin.

## Alternatives

| Candidate | Strength | Cost |
| --- | --- | --- |
| Importable package with explicit preparation | Offline recovery, validated data, deterministic tests | Maintains a small downloader |
| Root scripts with native NDJSON conversion | Less converter code | Version-dependent completion and recovery behavior |

Two independent design proposals and a cross-judge favored the package. Graft native YAML configuration and supported Ultralytics train, predict, and val APIs from the root-script alternative. Do not add a custom distributed launcher, generic plugin registry, or parallel root entry points.

## Throughput checkpoint

- Blocking first steps. Inspect real NDJSON schema, current editor interpreter, official train documentation, and design alternatives before implementation.
- Independent workstreams. Read-only design proposals can run independently. One implementation owner handles the coupled package and tests.
- Shared mutable state. Only the implementation owner edits application files. Download workers get unique per-request cache and temporary files. Preparation completes before distributed training.
- Smallest safe decomposition. Keep dataset rules together and use one implementation owner to avoid config and hardware policy drift. Independent review follows implementation.

## Verification

Run the offline test suite, all command help paths, inspect the real export, and a bounded real CPU smoke run with a tiny local fixture. If dependencies or hardware block the real smoke run, record the exact gap. Mocked hardware tests are not proof that multi-GPU ran. Inspect diagnostics using the project interpreter.

## Security considerations

Do not print or commit signed URLs, raw exports, images, caches, model weights, or runs. Validate paths before writes. Verify TLS and image decoding. Atomic writes protect interrupted downloads. Unknown or invalid annotations fail explicitly. Local model checkpoints are trusted pickle inputs, so document loading only trusted weights. No cloud credentials are required.