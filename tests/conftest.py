from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest


@pytest.fixture
def export(tmp_path):
    def make(rows=None, header=None):
        header = header or {"type": "dataset", "task": "detect", "class_names": {"0": "hat", "1": "person"}}
        if rows is None:
            rows = []
            for index, split in enumerate(("train", "val", "test")):
                file = f"frame{index}.png"
                Image.new("RGB", (32, 24), color=(index * 50, 80, 120)).save(tmp_path / file)
                rows.append({"type": "image", "file": file, "width": 32, "height": 24,
                             "split": split, "annotations": {"boxes": [[1, .5, .5, .25, .75]]}})
            rows[-1].pop("annotations")  # The real export omits annotations for backgrounds.
        path = tmp_path / "export.ndjson"
        path.write_text("\n".join(json.dumps(r) for r in [header, *rows]), encoding="utf-8")
        return path
    return make


@pytest.fixture
def row():
    return {"type": "image", "file": "frame.png", "width": 32, "height": 24,
            "split": "train", "annotations": {"boxes": [[0, .5, .5, .25, .75]]}}


@pytest.fixture
def fake_torch():
    from types import SimpleNamespace

    def make(cuda=0, mps=False):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: bool(cuda), device_count=lambda: cuda),
                               backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))
    return make