"""Train a new detector or resume an interrupted trusted local checkpoint."""

from __future__ import annotations

import argparse

from .config import (boolean, cache_value, cpu_threads, load_config, local_data, project_path,
                     training_options, trusted_model)
from .hardware import plan

RESUME_KEYS = {"device", "gpus", "workers", "batch"}


def run(options: dict, *, resume: str | None = None, factory=None, backend=None):
    if resume:
        if set(options) - RESUME_KEYS:
            raise ValueError("Resume permits only device/gpus, workers, and batch overrides; no fresh-run config")
        model_path = trusted_model(resume)
        if factory is None:
            from ultralytics import YOLO

            factory = YOLO
        model = factory(model_path)
        checkpoint: object = model.ckpt
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint is not resumable: need an interrupted last.pt with optimizer and epoch")
        epoch = checkpoint.get("epoch")
        if type(epoch) is not int or epoch < 0 or checkpoint.get("optimizer") is None:
            raise ValueError("Checkpoint is not resumable: need an interrupted last.pt with optimizer and epoch")
        saved = checkpoint.get("train_args")
        if not isinstance(saved, dict) or "data" not in saved or "batch" not in saved:
            raise ValueError("Checkpoint has no usable saved training configuration")
        local_data(saved["data"], ("train", "val"))
        hardware = plan(device=options.get("device", saved.get("device")) if "gpus" not in options else None,
                        gpus=options.get("gpus"), batch=options.get("batch", saved["batch"]),
                        workers=options.get("workers", saved.get("workers")), backend=backend)
        overrides: dict[str, object] = {"resume": True}
        if "device" in options or "gpus" in options:
            overrides["device"] = hardware.device
        if "batch" in options:
            overrides["batch"] = hardware.batch
        if "workers" in options:
            overrides["workers"] = hardware.workers
        return model.train(**overrides)
    options = dict(options)
    options["data"] = local_data(options["data"], ("train", "val"))
    hardware = plan(device=options.pop("device", None), gpus=options.pop("gpus", None),
                    batch=options.pop("batch"), workers=options.pop("workers", None),
                    amp=options.pop("amp"), backend=backend)
    threads = options.pop("cpu_threads", None)
    options["project"] = project_path(options["project"])
    model_path = options.pop("model")
    with cpu_threads(threads) as callback:
        if factory is None:
            from ultralytics import YOLO

            factory = YOLO
        model = factory(model_path)
        if callback is not None:
            for event in ("on_pretrain_routine_start", "on_train_start", "on_val_start"):
                model.add_callback(event, callback)
        return model.train(**options, device=hardware.device, batch=hardware.batch,
                           workers=hardware.workers, amp=hardware.amp)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", help="Training YAML; explicit CLI arguments win")
    parser.add_argument("--resume", help="Interrupted trusted local last.pt, not a fresh run")
    for key in ("model", "data", "device", "project", "name", "optimizer"):
        parser.add_argument(f"--{key}")
    for key in ("epochs", "imgsz", "gpus", "workers", "seed", "patience", "cpu-threads"):
        parser.add_argument(f"--{key}", type=int)
    for key in ("batch", "lr0"):
        parser.add_argument(f"--{key}", type=float)
    parser.add_argument("--cache", type=cache_value)
    parser.add_argument("--amp", type=boolean)
    for key in ("plots", "val", "exist-ok"):
        parser.add_argument(f"--{key}", type=boolean)
    args = vars(parser.parse_args(argv))
    resume = args.pop("resume", None)
    config = args.pop("config", None)
    try:
        if resume and config:
            raise ValueError("Resume must not use a fresh-run --config")
        options = args if resume else training_options(load_config(config), args)
        run(options, resume=resume)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()