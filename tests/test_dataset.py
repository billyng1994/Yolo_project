from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
from urllib.error import HTTPError, URLError

from PIL import Image
import pytest
import yaml

from safety_yolo import dataset as ds


def image_bytes(size=(32, 24)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "orange").save(buffer, format="PNG")
    return buffer.getvalue()


def transport(monkeypatch, response):
    class Opener:
        def open(self, request, timeout):
            return response(request, timeout)
    monkeypatch.setattr(ds, "build_opener", lambda *a: Opener())
    monkeypatch.setattr(ds.time, "sleep", lambda seconds: None)


def test_preservation_and_idempotency(export, tmp_path):
    manifest = ds.read_manifest(export())
    assert manifest.summary() == {"images": 3, "splits": {"train": 1, "val": 1, "test": 1},
                                  "boxes": 2, "backgrounds": 1,
                                  "classes": {0: {"name": "hat", "boxes": 0},
                                              1: {"name": "person", "boxes": 2}}}
    output = tmp_path / "prepared"
    config = ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True)
    assert yaml.safe_load(config.read_text())["names"] == {0: "hat", 1: "person"}
    assert (output / "labels/train/frame0.txt").read_text() == "1 0.5 0.5 0.25 0.75\n"
    assert (output / "labels/test/frame2.txt").read_bytes() == b""
    assert (output / "images/train/frame0.png").read_bytes() == (tmp_path / "frame0.png").read_bytes()
    (output / "labels/train.cache").write_bytes(b"old parsed label cache")
    staged = output / ".prepare-staging/images/train/frame0.png"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"incomplete")
    assert ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True) == config
    assert not (output / "labels/train.cache").exists()
    assert not list(output.rglob(".tmp-*"))
    assert not (output / ".prepare.lock").exists()


@pytest.mark.parametrize("patch", [
    {"split": "validation"}, {"type": "dataset"}, {"task": "segment"},
    {"width": 0}, {"width": True}, {"height": 2.5}, {"height": None},
    {"file": "../escape.png"}, {"file": "/escape.png"}, {"file": "C:/escape.png"},
    {"file": "folder\\escape.png"}, {"file": "a//b.png"}, {"file": "a/./b.png"},
    {"file": "a/../b.png"}, {"file": "CON.png"}, {"file": "a.png:secret"},
    {"file": "file.png "}, {"file": "images/x.png"}, {"file": "x.exe"},
    {"url": "http://example.test/a.png?secret=hidden"},
    {"url": "https://user:password@example.test/a.png"}, {"url": 42},
    {"url": "../elsewhere.png"}, {"annotations": None},
    {"annotations": {"segments": []}}, {"annotations": {"boxes": None}},
])
def test_bad_records(export, row, patch, monkeypatch):
    monkeypatch.setattr(ds, "build_opener", lambda *a: pytest.fail("Network before validation"))
    row.update(patch)
    with pytest.raises(ds.DatasetError):
        ds.read_manifest(export([row]))


@pytest.mark.parametrize("box", [[2,.5,.5,.2,.2], [-1,.5,.5,.2,.2], [.5,.5,.5,.2,.2],
                                 [0,float("nan"),.5,.2,.2], [0,.5,.5,float("inf"),.2],
                                 [0,.5,.5,0,.2], [0,1.1,.5,.2,.2], [0,True,.5,.2,.2],
                                 [0,.5,.5,.2], "box"])
def test_bad_boxes(export, row, box):
    row["annotations"]["boxes"] = [box]
    with pytest.raises(ds.DatasetError):
        ds.read_manifest(export([row]))


@pytest.mark.parametrize("names", [{}, {"1":"hat"}, {"0":None}, {"0":""}, {"0":"hat\n"}])
def test_bad_classes(export, row, names):
    with pytest.raises(ds.DatasetError):
        ds.read_manifest(export([row], {"type":"dataset", "task":"detect", "class_names": names}))


def test_schema_and_collision(export, row):
    for suffix in (".png", ".jpg", ".PNG"):
        with pytest.raises(ds.DatasetError, match="Duplicate"):
            ds.read_manifest(export([row, row | {"file": "frame" + suffix}]))
    for files in (("frame.png", "frame.png/child.png"), ("frame.txt/a.png", "frame.jpg")):
        for order in (files, files[::-1]):
            with pytest.raises(ds.DatasetError, match="collision"):
                ds.read_manifest(export([row | {"file":f} for f in order]))
    path = export([row])
    path.write_text('{"type":"dataset","type":"image"}\n{}')
    with pytest.raises(ds.DatasetError, match="line 1"):
        ds.read_manifest(path)
    path.write_text('{"type":"dataset","task":"segment"}\n{}')
    with pytest.raises(ds.DatasetError, match="task=detect"):
        ds.read_manifest(path)


def test_full_url_identity_cache_and_refresh(export, row, tmp_path, monkeypatch):
    calls = []
    def response(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(image_bytes())
    transport(monkeypatch, response)
    urls = [f"https://example.test/frame.png?signature={n}" for n in range(2)]
    manifest = ds.read_manifest(export([row | {"file": f"{i}.png", "url": url} for i, url in enumerate(urls)]))
    cache = tmp_path / "cache"
    ds.prepare(manifest, tmp_path / "one", cache=cache, timeout=7)
    assert sorted(calls) == [(u, 7) for u in urls]
    assert len(list(cache.glob("*.img"))) == 2
    ds.prepare(manifest, tmp_path / "two", cache=cache, offline=True)
    assert len(calls) == 2
    ds.prepare(manifest, tmp_path / "two", cache=cache, refresh=True)
    assert len(calls) == 4
    assert "signature" not in repr(manifest)
    assert "signature" not in (tmp_path / "one/summary.json").read_text()


@pytest.mark.parametrize("failure", ["corrupt", "dimension", "403", "transport"])
def test_failed_downloads_publish_nothing(export, row, tmp_path, monkeypatch, capsys, failure):
    secret = "signature=private-never-print"
    def response(request, timeout):
        if failure == "403":
            raise HTTPError(request.full_url, 403, secret, {}, None)
        if failure == "transport":
            raise URLError(secret)
        return io.BytesIO(b"invalid" if failure == "corrupt" else image_bytes((20, 20)))
    transport(monkeypatch, response)
    path = export([row | {"url": "https://example.test/image?" + secret}])
    output, cache = tmp_path / "prepared", tmp_path / "cache"
    with pytest.raises(SystemExit) as result:
        ds.main(["prepare", str(path), "--output", str(output), "--cache", str(cache), "--retries", "0"])
    assert result.value.code == 2
    assert secret not in capsys.readouterr().err
    assert not (output / "data.yaml").exists()
    assert not list(cache.glob("*.img"))
    assert not list(cache.glob(".download-*"))


def test_retry_and_corrupt_cache_recovery(export, row, tmp_path, monkeypatch):
    calls = []
    def response(request, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise URLError("transient")
        return io.BytesIO(image_bytes())
    transport(monkeypatch, response)
    manifest = ds.read_manifest(export([row | {"url": "https://example.test/a"}]))
    cache = tmp_path / "cache"
    ds.prepare(manifest, tmp_path / "out", cache=cache)
    assert len(calls) == 2
    cached = next(cache.glob("*.img"))
    cached.write_bytes(b"corrupt")
    with pytest.raises(ds.DatasetError):
        ds.prepare(manifest, tmp_path / "new", cache=cache, offline=True)
    ds.prepare(manifest, tmp_path / "new", cache=cache)
    assert len(calls) == 3


def test_changed_manifest_failure_and_lock(export, tmp_path):
    original = ds.read_manifest(export())
    output = tmp_path / "out"
    ds.prepare(original, output, cache=tmp_path / "cache")
    new = ds.Manifest(("changed", "person"), original.images, "different")
    with pytest.raises(ds.DatasetError, match="different"):
        ds.prepare(new, output)
    (output / ".prepare.lock").touch()
    with pytest.raises(ds.DatasetError, match="locked"):
        ds.prepare(original, output)
    (output / ".prepare.lock").unlink()
    (tmp_path / "frame0.png").write_bytes(b"corrupt")
    with pytest.raises(ds.DatasetError):
        ds.prepare(original, output)
    assert not (output / "data.yaml").exists()


def test_bounded_parallel_downloads(export, row, tmp_path, monkeypatch):
    barrier = threading.Barrier(3)
    threads = set()
    def response(request, timeout):
        threads.add(threading.get_ident())
        barrier.wait(timeout=10)
        return io.BytesIO(image_bytes())
    transport(monkeypatch, response)
    rows = [row | {"file":f"{i}.png", "url":f"https://example.test/{i}"} for i in range(3)]
    ds.prepare(ds.read_manifest(export(rows)), tmp_path / "out", cache=tmp_path / "cache", threads=3)
    assert len(threads) == 3


def test_inspect_no_heavy_import_or_network(export):
    path = export()
    script = """
import builtins, socket, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics', 'PIL', 'yaml'}:
        raise AssertionError('Heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network'))
socket.socket.connect_ex = socket.socket.connect
from safety_yolo.dataset import main
main(['inspect', sys.argv[1]])
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["backgrounds"] == 1


def test_redirect_refuses_http():
    with pytest.raises(ds.DatasetError):
        ds._HTTPSRedirect().redirect_request(None, None, 302, "", {}, "http://insecure.test/a")


def test_temporary_prefix_manifest_destinations_survive_offline_rerun(export, row, tmp_path, monkeypatch):
    transport(monkeypatch, lambda *args: io.BytesIO(image_bytes()))
    names = (".tmp-frame.png", ".tmp-folder/.tmp-nested.png")
    manifest = ds.read_manifest(export([
        row | {"file": name, "url": f"https://example.test/{i}"} for i, name in enumerate(names)
    ]))
    output, cache = tmp_path / "out", tmp_path / "cache"
    ds.prepare(manifest, output, cache=cache)
    before = {p.relative_to(output): p.read_bytes() for folder in ("images", "labels")
              for p in (output / folder).rglob("*") if p.is_file()}
    for cached in cache.glob("*.img"):
        cached.unlink()
    monkeypatch.setattr(ds, "build_opener", lambda *a: pytest.fail("Offline prepared image requested network"))
    assert ds.prepare(manifest, output, cache=cache, offline=True).is_file()
    assert all((output / p).read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("stop", [".manifest.sha256", "images/train/frame0.png", "labels/train/frame0.txt",
                                  "summary.json", "data.yaml"])
def test_hard_exit_recovers_after_stale_lock_removed(export, tmp_path, stop):
    path = export()
    output = tmp_path / "out"
    script = """
import os, sys
from pathlib import Path
from safety_yolo import dataset as ds
original = ds.os.replace
def replace(source, destination):
    if Path(destination) == Path(sys.argv[2]) / sys.argv[4]:
        os._exit(73)
    return original(source, destination)
ds.os.replace = replace
ds.prepare(ds.read_manifest(sys.argv[1]), Path(sys.argv[2]), cache=Path(sys.argv[3]), offline=True, threads=1)
"""
    result = subprocess.run([sys.executable, "-c", script, str(path), str(output), str(tmp_path / "cache"), stop],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 73, result.stderr
    assert (output / ".manifest.sha256").is_file() is (stop != ".manifest.sha256")
    assert (output / ".prepare.lock").is_file()
    assert (output / ".prepare-staging" / stop).is_file()
    (output / ".prepare.lock").unlink()
    assert ds.prepare(ds.read_manifest(path), output, cache=tmp_path / "cache", offline=True).is_file()
    assert (output / "images/train/frame0.png").read_bytes() == (tmp_path / "frame0.png").read_bytes()


@pytest.mark.parametrize("name", [".tmp-customer", "images/train/foreign.png", "images/train/foreign.npy",
                                   "images/train/frame0.png.npy", "images/val/frame0.npy", ".prepare-staging/customer.txt",
                                   ".prepare-staging/../customer.txt"])
@pytest.mark.parametrize("initialized", [False, True])
def test_unknown_output_is_never_deleted_or_adopted(export, tmp_path, name, initialized):
    manifest = ds.read_manifest(export())
    output = tmp_path / "out"
    if initialized:
        ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True)
    foreign = output / name
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"customer data")
    with pytest.raises(ds.DatasetError):
        ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True)
    assert foreign.read_bytes() == b"customer data"
    assert (output / ".manifest.sha256").is_file() is initialized
    assert (output / "data.yaml").is_file() is initialized


@pytest.mark.parametrize("nested", ["frame.npy/other.png", "frame.npy/nested/other.png", "FRAME.npy/other.png"])
def test_native_cache_collision_preserves_manifest_directory(export, row, tmp_path, nested):
    for name in ("frame.png", nested):
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(image_bytes())
    manifest = ds.read_manifest(export([row | {"file": name} for name in ("frame.png", nested)]))
    output = tmp_path / "out"
    config = ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True)
    before = {p: p.read_bytes() for kind in ("images", "labels")
              for p in (output / kind).rglob("*") if p.is_file()}
    assert ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True) == config
    assert (output / "images/train" / nested).parent.is_dir()
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("expected_directory", [False, True])
def test_native_cache_file_directory_type_mismatch_is_refused(export, row, tmp_path, expected_directory):
    (tmp_path / "frame.png").write_bytes(image_bytes())
    rows = [row]
    if expected_directory:
        rows.append(row | {"file": "frame.npy/other.png"})
    manifest = ds.read_manifest(export(rows))
    output = tmp_path / "out"
    collision = output / "images/train/frame.npy"
    collision.parent.mkdir(parents=True)
    if expected_directory:
        collision.write_bytes(b"not a manifest directory")
    else:
        collision.mkdir()
    with pytest.raises(ds.DatasetError, match="Unexpected"):
        ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True)
    if expected_directory:
        assert collision.read_bytes() == b"not a manifest directory"
    else:
        assert collision.is_dir()
    assert not (output / "data.yaml").exists()


@pytest.mark.parametrize("initialized", [False, True])
def test_staging_namespace_rejects_foreign_content(export, tmp_path, initialized):
    manifest = ds.read_manifest(export())
    output = tmp_path / "out"
    if initialized:
        ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True)
    foreign = output / ".prepare-staging/images/train/customer.png"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"customer image")
    with pytest.raises(ds.DatasetError, match="Unexpected"):
        ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True)
    assert foreign.read_bytes() == b"customer image"
    assert (output / "data.yaml").exists() is initialized


def test_initial_staging_requires_matching_manifest(export, tmp_path):
    manifest = ds.read_manifest(export())
    output = tmp_path / "out"
    pending = output / ".prepare-staging/.manifest.sha256"
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"unknown ownership")
    with pytest.raises(ds.DatasetError):
        ds.prepare(manifest, output, cache=tmp_path / "cache", offline=True)
    assert pending.read_bytes() == b"unknown ownership"
    assert not (output / ".manifest.sha256").exists()
    assert not (output / "data.yaml").exists()