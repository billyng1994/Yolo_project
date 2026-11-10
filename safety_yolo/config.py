"""Small CLI/config boundaries; paths are relative to the working directory."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

TRAIN_KEYS = {
    "model", "data", "epochs", "imgsz", "batch", "device", "gpus", "workers", "cache", "amp",
    "seed", "project", "name", "patience", "optimizer", "lr0", "lrf", "weight_decay", "momentum",
    "warmup_epochs", "cos_lr", "close_mosaic", "mosaic", "mixup", "copy_paste", "degrees", "translate",
    "scale", "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "multi_scale",
    "rect", "deterministic", "fraction", "freeze", "save", "save_period", "exist_ok", "plots", "val",
    "pretrained", "verbose", "cpu_threads",
}
DEFAULTS = {"model": "yolo26n.pt", "data": "data/construction-site-safety/data.yaml",
            "epochs": 100, "imgsz": 640, "batch": 16, "amp": True, "seed": 0,
            "project": "runs/train", "name": "safety", "patience": 25}


def boolean(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() == "true"


def cache_value(value: str) -> bool | str:
    return boolean(value) if value.lower() in {"true", "false"} else value.lower()


def load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    try:
        result = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raise ValueError("Cannot read training YAML configuration") from None
    if not isinstance(result, dict) or set(result) - TRAIN_KEYS:
        raise ValueError("Configuration must be a mapping containing supported training keys only")
    if any(v is None for v in result.values()):
        raise ValueError("Omit unset configuration keys instead of setting null")
    return result


def training_options(config: dict, explicit: dict) -> dict:
    result = DEFAULTS | config | explicit
    if set(result) - TRAIN_KEYS:
        raise ValueError("Unsupported training configuration key")
    for key in ("model", "data", "project", "name", "optimizer"):
        if key in result and (not isinstance(result[key], str) or not result[key].strip()):
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("amp", "cos_lr", "rect", "deterministic", "save", "exist_ok", "plots", "val", "verbose"):
        if key in result and type(result[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    if "device" in result and "gpus" in result:
        # A CLI hardware selector replaces a config selector, but two CLI selectors conflict.
        if "device" in explicit and "gpus" not in explicit:
            result.pop("gpus")
        elif "gpus" in explicit and "device" not in explicit:
            result.pop("device")
        else:
            raise ValueError("Choose device or gpus, not both")
    if type(result["epochs"]) is not int or result["epochs"] < 1:
        raise ValueError("Epochs must be a positive integer")
    if type(result["imgsz"]) is not int or result["imgsz"] < 32:
        raise ValueError("Image size must be an integer >=32")
    if result.get("cache", False) not in (True, False, "ram", "disk"):
        raise ValueError("Cache must be true, false, ram, or disk")
    return result


def trusted_model(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model must be an explicit, trusted local .pt checkpoint")
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise ValueError("Model must be an explicit, trusted local .pt checkpoint")
    return str(path)


def local_data(value: str, splits: tuple[str, ...]) -> str:
    import yaml

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Data must be an existing local dataset YAML")
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
        raise ValueError("Data must be an existing local dataset YAML; prepare NDJSON first")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "download" in data:
            raise ValueError("Dataset YAML must be local and must not contain download instructions")
        root = Path(data.get("path") or path.parent)
        if not root.is_absolute():
            raise ValueError("Dataset YAML path must be absolute; rerun prepare after moving the dataset")
        for split in splits:
            relative = data.get(split)
            if not isinstance(relative, str) or not (root / relative).is_dir():
                raise ValueError(f"Dataset {split} must reference an existing local image directory")
            if not any((root / relative).rglob("*.*")):
                raise ValueError(f"Dataset {split} is empty")
    except (OSError, TypeError, yaml.YAMLError):
        raise ValueError("Invalid local dataset YAML") from None
    return str(path)


def project_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Project must be a nonempty path")
    return str(Path(value).expanduser().resolve())


def _set_cpu_threads(runner: Any, *, count: int) -> None:
    import torch

    torch.set_num_threads(count)


@contextmanager
def cpu_threads(value: int | None):
    """Bound initialization and compute, then restore the caller's PyTorch setting."""
    if value is None:
        yield None
        return
    if type(value) is not int or value < 1:
        raise ValueError("CPU threads must be positive")
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(value)
        yield partial(_set_cpu_threads, count=value)
    finally:
        torch.set_num_threads(previous)


def inference_parser(description: str, *, validation: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, help="Trusted local .pt checkpoint")
    parser.add_argument("--device", default=None, help="auto, cpu, mps, 0, or cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1 if not validation else 16)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--project", default="runs/val" if validation else "runs/predict")
    parser.add_argument("--name", default="safety")
    return parser