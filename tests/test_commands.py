from __future__ import annotations

import subprocess
import sys
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from safety_yolo import config, dataset, predict, train, validate


def test_config_precedence_and_false_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("epochs: 3\ngpus: 2\namp: true\ncache: ram\n")
    options = config.training_options(config.load_config(str(path)), {"epochs":1, "amp":False, "device":"cpu"})
    assert options["epochs"] == 1 and options["amp"] is False and "gpus" not in options
    assert options["model"] == "yolo26n.pt"
    assert config.training_options({"device":"cpu"}, {"gpus":1})["gpus"] == 1
    with pytest.raises(ValueError):
        config.training_options({}, {"device":"cpu", "gpus":1})
    for content in ("[]", "model: null", "unknown: 1", "a: ["):
        path.write_text(content)
        with pytest.raises(ValueError):
            config.load_config(str(path))
    for invalid in ({"data":[]}, {"model":False}, {"amp":"false"}, {"epochs":True}, {"imgsz":0}):
        with pytest.raises(ValueError):
            config.training_options(invalid, {})


def test_train_and_resume_forwarding(export, tmp_path, fake_torch):
    data = dataset.prepare(dataset.read_manifest(export()), tmp_path / "prepared", cache=tmp_path / "cache")
    calls = []
    saved = {"data": str(data), "device":"cpu", "batch":4, "workers":0, "epochs":88}
    class Model:
        ckpt = {"epoch":2, "optimizer":{}, "train_args":saved}
        def __init__(self, path):
            calls.append(path)
        def train(self, **kwargs):
            calls.append(kwargs)
    backend = fake_torch()
    train.run(config.training_options({}, {"data":str(data), "device":"cpu", "workers":0, "amp":True}),
              factory=Model, backend=backend)
    assert calls[-1]["amp"] is False
    assert calls[-1]["epochs"] == 100
    checkpoint = tmp_path / "last.pt"
    checkpoint.touch()
    train.run({}, resume=str(checkpoint), factory=Model, backend=backend)
    assert calls[-1] == {"resume":True}
    train.run({"workers":0, "device":"cpu", "batch":2.0}, resume=str(checkpoint), factory=Model, backend=backend)
    assert calls[-1] == {"resume":True, "workers":0, "device":"cpu", "batch":2}
    for illegal in ({"epochs":1}, {"model":"yolo26n.pt"}, {"amp":False}, {"imgsz":64}):
        with pytest.raises(ValueError, match="Resume permits"):
            train.run(illegal, resume=str(checkpoint), factory=Model, backend=backend)
    Model.ckpt = {"epoch":-1, "optimizer":None}
    with pytest.raises(ValueError, match="not resumable"):
        train.run({}, resume=str(checkpoint), factory=Model, backend=backend)
    Model.ckpt = {"epoch":0, "optimizer":{}}
    with pytest.raises(ValueError, match="saved training configuration"):
        train.run({}, resume=str(checkpoint), factory=Model, backend=backend)


def test_resume_rejects_config_and_cli_defaults(tmp_path):
    with pytest.raises(SystemExit) as error:
        train.main(["--resume", "last.pt", "--config", "configs/train.yaml"])
    assert error.value.code == 2


@pytest.mark.parametrize("module,extra", [
    ("dataset", []), ("dataset", ["inspect"]), ("dataset", ["prepare"]),
    ("train", []), ("predict", []), ("validate", []),
])
def test_help_is_lazy(module, extra):
    script = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics', 'PIL', 'yaml'}:
        raise AssertionError('Heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
module = sys.argv.pop(1)
runpy.run_module(module, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script, f"safety_yolo.{module}", *extra, "--help"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_streamed_camera_and_test_metrics(monkeypatch, tmp_path, export):
    model = tmp_path / "model.pt"
    model.touch()
    data = dataset.prepare(dataset.read_manifest(export()), tmp_path / "data", cache=tmp_path / "cache")
    calls, yielded = [], []
    class Model:
        def __init__(self, path):
            assert path == str(model)
        def predict(self, **kwargs):
            calls.append(kwargs)
            for i in range(3):
                yielded.append(i)
                yield i
        def val(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(results_dict={"metrics/mAP50(B)":.5}, save_dir=tmp_path)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    hardware = SimpleNamespace(device="cpu", workers=0)
    monkeypatch.setattr(predict, "plan", lambda **kw: hardware)
    monkeypatch.setattr(validate, "plan", lambda **kw: hardware)
    predict.main(["--model", str(model), "--source", "0", "--no-save"])
    assert calls[-1]["source"] == 0 and calls[-1]["stream"] is True
    assert yielded == [0, 1, 2]
    validate.main(["--model", str(model), "--data", str(data), "--split", "test"])
    assert calls[-1]["split"] == "test" and calls[-1]["half"] is False
    assert (tmp_path / "metrics.json").is_file()


def test_data_boundary(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("download: echo dangerous\n")
    with pytest.raises(ValueError, match="download"):
        config.local_data(str(path), ("train",))
    path.write_text("path: relative\ntrain: images/train\n")
    with pytest.raises(ValueError, match="absolute"):
        config.local_data(str(path), ("train",))
    with pytest.raises(ValueError, match="trusted"):
        config.trusted_model("yolo26n.pt")


@pytest.mark.parametrize("project", [None, "custom/results", "absolute"])
def test_command_projects_match_real_save_dir(monkeypatch, tmp_path, export, fake_torch, project):
    data = dataset.prepare(dataset.read_manifest(export()), tmp_path / "prepared", cache=tmp_path / "cache")
    model = tmp_path / "model.pt"
    model.touch()
    monkeypatch.chdir(tmp_path)
    if project == "absolute":
        project = str(tmp_path / "absolute")
    extra = [] if project is None else ["--project", project]
    calls = {}
    class Model:
        def __init__(self, path):
            pass
        def train(self, **kwargs):
            calls["train"] = kwargs
        def predict(self, **kwargs):
            calls["predict"] = kwargs
            return iter(())
        def val(self, **kwargs):
            calls["val"] = kwargs
            return SimpleNamespace(results_dict={}, save_dir=tmp_path)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    monkeypatch.setattr(train, "plan", lambda **kwargs: SimpleNamespace(device="cpu", batch=2, workers=0, amp=False))
    for module in (predict, validate):
        monkeypatch.setattr(module, "plan", lambda **kwargs: SimpleNamespace(device="cpu", workers=0))
    train.main(["--data", str(data), "--device", "cpu", *extra])
    predict.main(["--model", str(model), "--source", "0", "--no-save", *extra])
    validate.main(["--model", str(model), "--data", str(data), *extra])
    contracts = []
    for mode, kwargs in calls.items():
        expected = Path(project).resolve() if project else tmp_path / "runs" / mode
        assert Path(kwargs["project"]) == expected
        contracts.append(({"task":"detect", "mode":mode, "project":kwargs["project"],
                           "name":"safety", "exist_ok":True}, str(expected / "safety")))
    assert set(calls) == {"train", "predict", "val"}
    script = """
import json, socket, sys
def offline(*args, **kwargs):
    raise AssertionError('Save-directory contract attempted network access')
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = offline
from pathlib import Path
from ultralytics.cfg import get_cfg, get_save_dir
for overrides, expected in json.loads(sys.argv[1]):
    assert get_save_dir(get_cfg(overrides=overrides)) == Path(expected)
"""
    env = os.environ | {"YOLO_CONFIG_DIR":str(tmp_path / "yolo"), "YOLO_OFFLINE":"true",
                        "YOLO_AUTOINSTALL":"false", "WANDB_MODE":"disabled"}
    result = subprocess.run([sys.executable, "-c", script, json.dumps(contracts)],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_thread_scope_restores_on_error(monkeypatch):
    current = [3]
    backend = SimpleNamespace(get_num_threads=lambda: current[0],
                              set_num_threads=lambda value: current.__setitem__(0, value))
    monkeypatch.setitem(sys.modules, "torch", backend)
    with pytest.raises(RuntimeError, match="compute failed"):
        with config.cpu_threads(1) as callback:
            assert current[0] == 1
            current[0] = 8
            callback(None)
            assert current[0] == 1
            raise RuntimeError("compute failed")
    assert current[0] == 3
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive"):
            with config.cpu_threads(value):
                pytest.fail("Invalid thread count accepted")
    assert current[0] == 3