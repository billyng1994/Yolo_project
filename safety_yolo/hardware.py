"""One hardware policy shared by all model commands."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
from typing import Any


@dataclass(frozen=True)
class HardwarePlan:
    device: str
    batch: int | float
    workers: int
    amp: bool
    ranks: int


def plan(*, device: str | int | None = None, gpus: int | None = None,
         batch: int | float = 16, workers: int | None = None, amp: bool = True,
         training: bool = True, backend: Any = None, platform: str | None = None,
         cpu_count: int | None = None) -> HardwarePlan:
    if backend is None:
        import torch

        backend = torch
    platform = platform or sys.platform
    if device is not None and gpus is not None:
        raise ValueError("Choose device or gpus, not both")
    available = backend.cuda.device_count() if backend.cuda.is_available() else 0
    if gpus is not None:
        if type(gpus) is not int or not 1 <= gpus <= available:
            raise ValueError("GPU count must be positive and no greater than visible CUDA devices")
        device = ",".join(map(str, range(gpus)))
    if device is None or device in ("", "auto"):
        device = "0" if available else "mps" if backend.backends.mps.is_available() else "cpu"
    device = str(device).lower().strip()
    if device.startswith("cuda:"):
        device = device[5:]
    ranks = 1
    cuda = device not in {"cpu", "mps"}
    if cuda:
        if not all(p.isascii() and p.isdecimal() for p in device.split(",")):
            raise ValueError("Device must be cpu, mps, a CUDA index, or comma-separated CUDA indices")
        ids = [int(p) for p in device.split(",")]
        if len(set(ids)) != len(ids) or any(i >= available for i in ids):
            raise ValueError("Duplicate or unavailable CUDA device")
        ranks = len(ids)
        device = ",".join(map(str, ids))
        if ranks > 1:
            if not training:
                raise ValueError("Prediction and evaluation require one device")
            if platform == "win32":
                raise ValueError("Native Windows multi-GPU is unsupported; use Linux or WSL2")
    elif device == "mps" and not backend.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if type(batch) not in (int, float) or not math.isfinite(batch):
        raise ValueError("Batch must be a finite number")
    if float(batch).is_integer():
        batch = int(batch)
    automatic = batch == -1 or 0 < batch < 1
    if automatic:
        if not training or not cuda or ranks != 1:
            raise ValueError("AutoBatch is supported only for single-CUDA training")
    elif not isinstance(batch, int) or batch < 1 or batch % ranks:
        raise ValueError("Batch must be a positive fixed global integer divisible by GPU count")
    if workers is None:
        workers = 0 if platform == "win32" else min(8, max(0, (cpu_count or os.cpu_count() or 1) // ranks))
    if type(workers) is not int or workers < 0:
        raise ValueError("Workers must be a nonnegative integer")
    if type(amp) is not bool:
        raise ValueError("AMP must be true or false")
    return HardwarePlan(device, batch, workers, amp and cuda, ranks)