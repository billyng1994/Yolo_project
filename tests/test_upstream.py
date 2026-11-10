from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from safety_yolo import dataset


def run_contract(tmp_path, script, *args):
    import matplotlib

    fonts = tmp_path / "yolo/Ultralytics"
    fonts.mkdir(parents=True, exist_ok=True)
    for name in ("Arial.ttf", "Arial.Unicode.ttf"):
        shutil.copyfile(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf", fonts / name)
    env = os.environ | {"YOLO_OFFLINE":"true", "YOLO_AUTOINSTALL":"false", "WANDB_MODE":"disabled",
                        "YOLO_CONFIG_DIR":str(tmp_path / "yolo"), "PYTHONDONTWRITEBYTECODE":"1",
                        "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2"}
    guard = """
import socket
def offline(*args, **kwargs):
    raise AssertionError('Offline contract attempted network access')
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = offline
"""
    result = subprocess.run([sys.executable, "-c", guard + script, *map(str, args)], env=env,
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, errors="replace", timeout=120)
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]


def test_actual_disk_image_cache_allows_offline_reprepare(tmp_path, export):
    manifest_path = export()
    run_contract(tmp_path, """
import sys
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from ultralytics.data.dataset import YOLODataset
from safety_yolo import dataset as ds

root, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
manifest = ds.read_manifest(manifest_path)
output, cache = root / 'prepared', root / 'download-cache'
data = {'names': dict(enumerate(manifest.classes)), 'channels': 3}
yaml_path = ds.prepare(manifest, output, cache=cache, offline=True)
sources = {image.local: image.local.read_bytes() for image in manifest.images}
prepared = {p: p.read_bytes() for kind in ('images', 'labels')
            for p in (output / kind).rglob('*') if p.is_file()}

def native_readers():
    readers = [YOLODataset(img_path=str(output / 'images' / split), data=data,
                          imgsz=32, batch_size=1, augment=False, cache='disk')
               for split in ds.SPLITS]
    caches = {Path(file).with_suffix('.npy') for reader in readers for file in reader.im_files}
    assert len(caches) == len(manifest.images)
    assert all(p.is_file() and p.stat().st_size > 0 for p in caches)
    for reader in readers:
        for i, file in enumerate(reader.im_files):
            assert np.array_equal(reader.load_image(i)[0], cv2.imread(file))
    return caches

caches = native_readers()
assert not cache.exists(), 'Fixture must not rely on a download cache'
with patch('numpy.load', side_effect=AssertionError('Preparation must not load native caches')):
    assert ds.prepare(manifest, output, cache=cache, offline=True) == yaml_path
assert all(not p.exists() for p in caches), 'Regenerable native caches must be invalidated'
assert all(p.read_bytes() == content for p, content in sources.items())
assert all(p.read_bytes() == content for p, content in prepared.items())

for refresh in (False, True):
    caches = native_readers()
    Image.new('RGB', (32, 24), 'lime' if refresh else 'purple').save(root / 'frame0.png')
    with patch('numpy.load', side_effect=AssertionError('Preparation must not load native caches')):
        assert ds.prepare(manifest, output, cache=cache, offline=not refresh, refresh=refresh) == yaml_path
    assert all(not p.exists() for p in caches)
    sources[root / 'frame0.png'] = (root / 'frame0.png').read_bytes()
    prepared[output / 'images/train/frame0.png'] = sources[root / 'frame0.png']
    assert all(p.read_bytes() == content for p, content in sources.items())
    assert all(p.read_bytes() == content for p, content in prepared.items())
native_readers()
""", tmp_path, manifest_path)


def test_actual_ddp_generation_and_companion_loading(tmp_path):
    run_contract(tmp_path, """
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from ultralytics.utils import AUTOINSTALL, USER_CONFIG_DIR
from ultralytics.utils.dist import generate_ddp_file, ddp_cleanup
from ultralytics.utils.patches import torch_load
from safety_yolo.config import cpu_threads

assert os.environ['YOLO_AUTOINSTALL'] == 'false' and not AUTOINSTALL
assert USER_CONFIG_DIR.resolve().is_relative_to(Path(os.environ['YOLO_CONFIG_DIR']).resolve())
trainer = SimpleNamespace(args=SimpleNamespace(device='0,1', batch=16),
                          model=torch.nn.Linear(2, 1), callbacks={})
with cpu_threads(1) as callback:
    trainer.callbacks = {'on_train_start':[callback]}
script = Path(generate_ddp_file(trainer))
try:
    state = torch_load(script.with_suffix('.pt'), map_location='cpu')
    assert state['trainer'] is SimpleNamespace
    assert state['args'] == {'device':'0,1', 'batch':16}
    torch.set_num_threads(2)
    state['callbacks']['on_train_start'][0](SimpleNamespace(device=torch.device('cpu')))
    assert torch.get_num_threads() == 1
    assert torch.equal(state['model'].weight, trainer.model.weight)
    assert state['model'](torch.ones(1, 2)).shape == (1, 1)
finally:
    ddp_cleanup(trainer, str(script))
assert not script.exists() and not script.with_suffix('.pt').exists()
""")


@pytest.mark.parametrize("mode", ["predict", "val"])
def test_actual_cpu_threads_at_inference_boundary(tmp_path, export, mode):
    data = dataset.prepare(dataset.read_manifest(export()), tmp_path / "prepared", cache=tmp_path / "cache")
    run_contract(tmp_path, """
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from safety_yolo import predict, validate

root, data, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
torch.set_num_threads(2)
weights = root / 'random.pt'
YOLO('yolo26n.yaml').save(weights)
observed = []
active = False
original_predict = BasePredictor.inference
original_preprocess = DetectionValidator.preprocess
original_forward = AutoBackend.forward
def inference(self, *args, **kwargs):
    observed.append(torch.get_num_threads())
    return original_predict(self, *args, **kwargs)
def preprocess(self, *args, **kwargs):
    global active
    active = True
    return original_preprocess(self, *args, **kwargs)
def forward(self, *args, **kwargs):
    if active:
        observed.append(torch.get_num_threads())
    return original_forward(self, *args, **kwargs)
args = ['--model', str(weights), '--device', 'cpu', '--imgsz', '64', '--cpu-threads', '1',
        '--project', str(root / 'runs')]
with patch.object(BasePredictor, 'inference', inference), \
     patch.object(DetectionValidator, 'preprocess', preprocess), \
     patch.object(AutoBackend, 'forward', forward):
    if mode == 'predict':
        predict.main([*args, '--source', str(root / 'frame1.png'), '--no-save'])
    else:
        validate.main([*args, '--data', data, '--batch', '1', '--workers', '0'])
assert observed and set(observed) == {1}, (mode, observed)
assert torch.get_num_threads() == 2, 'Caller thread setting was not restored'
""", tmp_path, data, mode)