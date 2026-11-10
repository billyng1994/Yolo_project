import pytest

from safety_yolo.hardware import plan


def test_automatic(fake_torch):
    assert plan(backend=fake_torch(), platform="win32").device == "cpu"
    assert plan(backend=fake_torch(mps=True)).device == "mps"
    assert plan(backend=fake_torch(cuda=2, mps=True)).device == "0"
    assert plan(backend=fake_torch(), platform="win32").workers == 0
    assert not plan(backend=fake_torch()).amp
    assert not plan(backend=fake_torch(mps=True)).amp
    assert plan(backend=fake_torch(cuda=1)).amp


def test_fixed_ddp_and_count(fake_torch):
    result = plan(gpus=2, batch=8, backend=fake_torch(cuda=4), platform="linux", cpu_count=8)
    assert (result.device, result.ranks, result.batch, result.workers) == ("0,1", 2, 8, 4)
    assert plan(device="cuda:0", batch=1.0, backend=fake_torch(cuda=1)).batch == 1
    assert type(plan(batch=1.0, backend=fake_torch()).batch) is int
    assert plan(device="1", backend=fake_torch(cuda=2)).device == "1"


@pytest.mark.parametrize("kwargs", [
    {"device":"0,0"}, {"device":"2"}, {"device":"-1"}, {"device":"cuda"},
    {"device":"0,1", "batch":3}, {"device":"0,1", "batch":-1},
    {"device":"0,1", "batch":.5}, {"device":"0,1", "training":False},
    {"device":"cpu", "gpus":1}, {"gpus":3}, {"gpus":0}, {"workers":-1},
    {"batch":0}, {"batch":2.5}, {"batch":float("nan")}, {"batch":True},
    {"device":"cpu", "batch":-1}, {"device":"mps"}, {"amp":"false"},
])
def test_rejections(fake_torch, kwargs):
    with pytest.raises(ValueError):
        plan(backend=fake_torch(cuda=2), platform="linux", **kwargs)


def test_windows_and_single_auto(fake_torch):
    with pytest.raises(ValueError, match="WSL2"):
        plan(device="0,1", backend=fake_torch(cuda=2), platform="win32")
    for batch in (-1, .5):
        assert plan(batch=batch, backend=fake_torch(cuda=1)).batch == batch
    with pytest.raises(ValueError, match="AutoBatch"):
        plan(batch=.5, training=False, backend=fake_torch(cuda=1))