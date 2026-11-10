"""Opt-in real CLI integration. No pretrained weights and no external images."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw
import pytest
import yaml


@pytest.mark.smoke
@pytest.mark.skipif(os.environ.get("SAFETY_YOLO_SMOKE") != "1", reason="Set SAFETY_YOLO_SMOKE=1 for real CPU smoke")
def test_real_cpu_pipeline(tmp_path):
    import matplotlib

    fonts = tmp_path / "yolo/Ultralytics"
    fonts.mkdir(parents=True)
    for name in ("Arial.ttf", "Arial.Unicode.ttf"):
        shutil.copyfile(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf", fonts / name)
    rows = [{"type":"dataset", "task":"detect", "class_names":{"0":"hat"}}]
    for split, count in (("train",4), ("val",2), ("test",2)):
        for i in range(count):
            file = f"{split}{i}.png"
            image = Image.new("RGB", (64,64), "navy")
            ImageDraw.Draw(image).rectangle((16,16,48,48), fill="orange")
            image.save(tmp_path / file)
            rows.append({"type":"image", "file":file, "split":split, "width":64, "height":64,
                         "annotations":{"boxes":[[0,.5,.5,.5,.5]]}})
    manifest = tmp_path / "tiny.ndjson"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    output = tmp_path / "prepared"
    project = tmp_path / "runs"
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({
        "model":"yolo26n.yaml", "data":str(output / "data.yaml"), "epochs":1, "imgsz":64,
        "batch":2, "device":"cpu", "workers":0, "amp":False, "plots":False, "pretrained":False,
        "project":str(project), "name":"train", "cpu_threads":2, "close_mosaic":0,
    }))
    env = os.environ | {"YOLO_OFFLINE":"true", "YOLO_AUTOINSTALL":"false", "WANDB_MODE":"disabled",
                        "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2", "YOLO_CONFIG_DIR":str(tmp_path / "yolo")}
    def command(module, *args):
        script = """
import runpy, socket, sys
def offline(*args, **kwargs):
    raise AssertionError('CPU smoke attempted network access')
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = offline
module = sys.argv.pop(1)
if module == 'train':
    import torch
    from ultralytics.nn.tasks import DetectionModel
    original = DetectionModel.forward
    observed = []
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict) and self.training:
            observed.append(torch.get_num_threads())
        return original(self, x, *args, **kwargs)
    DetectionModel.forward = forward
runpy.run_module('safety_yolo.' + module, run_name='__main__')
if module == 'train':
    assert observed and set(observed) == {2}, observed
"""
        result = subprocess.run([sys.executable, "-c", script, module, *map(str,args)],
                                capture_output=True, text=True, errors="replace", env=env, timeout=300)
        assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    command("dataset", "prepare", manifest, "--output", output, "--cache", tmp_path / "cache", "--offline")
    command("train", "--config", config)
    weights = project / "train/weights/best.pt"
    assert weights.is_file() and (weights.parent / "last.pt").is_file()
    command("predict", "--model", weights, "--source", tmp_path / "val0.png", "--device", "cpu",
            "--imgsz", 64, "--cpu-threads", 2, "--project", project, "--name", "predict")
    assert (project / "predict/val0.jpg").is_file() or (project / "predict/val0.png").is_file()
    for split in ("val", "test"):
        command("validate", "--model", weights, "--data", output / "data.yaml", "--split", split,
                "--device", "cpu", "--imgsz",64, "--batch",2, "--workers",0, "--cpu-threads",2,
                "--project", project, "--name", split)
        assert json.loads((project / split / "metrics.json").read_text())
    print(f"CPU smoke artifacts: {project}")