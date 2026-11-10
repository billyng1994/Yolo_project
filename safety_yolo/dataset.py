"""Validate detection exports and publish verified local YOLO datasets."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DatasetError(ValueError):
    """A sanitized dataset boundary failure."""


@dataclass(frozen=True)
class Box:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    def label(self) -> str:
        return " ".join(map(str, (self.class_id, self.cx, self.cy, self.width, self.height)))


@dataclass(frozen=True)
class ImageRecord:
    file: str
    split: str
    width: int
    height: int
    boxes: tuple[Box, ...]
    url: str | None = field(repr=False)
    local: Path | None = field(repr=False)


@dataclass(frozen=True)
class Manifest:
    classes: tuple[str, ...]
    images: tuple[ImageRecord, ...]
    fingerprint: str

    def summary(self) -> dict:
        counts = Counter(box.class_id for image in self.images for box in image.boxes)
        return {
            "images": len(self.images),
            "splits": {s: sum(i.split == s for i in self.images) for s in SPLITS},
            "boxes": sum(counts.values()),
            "backgrounds": sum(not i.boxes for i in self.images),
            "classes": {i: {"name": n, "boxes": counts[i]} for i, n in enumerate(self.classes)},
        }


def safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DatasetError("Image paths must be nonempty relative POSIX paths")
    parts = value.split("/")
    for part in parts:
        if (part in {"", ".", ".."} or part.endswith((" ", "."))
                or re.search(r'[<>:"|?*\x00-\x1f\x7f]', part)
                or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)):
            raise DatasetError("Unsafe image path")
    if "images" in [p.lower() for p in parts[:-1]]:
        raise DatasetError("Nested images directories conflict with YOLO label lookup")
    return value


def validate_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.hostname and not parsed.username
                 and not parsed.password and not parsed.fragment and parsed.port != 0
                 and not re.search(r"[\x00-\x20\x7f]", url))
    except ValueError:
        valid = False
    if not valid:
        raise DatasetError("Remote image sources must be HTTPS URLs without credentials or fragments")


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError("Duplicate JSON key")
        result[key] = value
    return result


def _positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise DatasetError("Image dimensions must be positive integers")
    return value


def _box(row: object, class_count: int) -> Box:
    if not isinstance(row, list) or len(row) != 5:
        raise DatasetError("Each box must contain class_id,cx,cy,width,height")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
        raise DatasetError("Box values must be finite numbers")
    cls, cx, cy, width, height = row
    if not float(cls).is_integer() or not 0 <= cls < class_count:
        raise DatasetError("Unknown or noninteger box class ID")
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise DatasetError("Boxes must have normalized centers and positive normalized sizes")
    return Box(int(cls), cx, cy, width, height)


def read_manifest(path: str | Path) -> Manifest:
    path = Path(path).resolve()
    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        raise DatasetError("Cannot read UTF-8 manifest") from None
    records = []
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line, object_pairs_hook=_object)
            if not isinstance(record, dict):
                raise DatasetError("Record must be an object")
            records.append(record)
        except (ValueError, RecursionError):
            raise DatasetError(f"Invalid JSON object at line {number}") from None
    if len(records) < 2:
        raise DatasetError("Manifest requires a dataset header and at least one image")
    header = records[0]
    if header.get("type") != "dataset" or header.get("task") != "detect":
        raise DatasetError("First record must be a dataset with task=detect")
    names = header.get("class_names")
    if (not isinstance(names, dict) or not names
            or set(names) != {str(i) for i in range(len(names))}
            or any(not isinstance(n, str) or not n.strip() or any(ord(c) < 32 for c in n)
                   for n in names.values())):
        raise DatasetError("class_names must map contiguous string IDs starting at 0 to names")
    classes = tuple(names[str(i)] for i in range(len(names)))
    images = []
    destinations: set[str] = set()
    directories: set[str] = set()
    for number, row in enumerate(records[1:], 2):
        try:
            if row.get("type") != "image" or row.get("split") not in SPLITS:
                raise DatasetError("Expected an image with split train, val, or test")
            if "task" in row and row["task"] != "detect":
                raise DatasetError("Image task must be detect")
            file = safe_relative(row.get("file"))
            if PurePosixPath(file).suffix.lower() not in IMAGE_SUFFIXES:
                raise DatasetError("Unsupported image filename extension")
            for kind, name in (("images", file), ("labels", str(PurePosixPath(file).with_suffix(".txt")))):
                key = f"{kind}/{row['split']}/{name}".casefold()
                parents = {str(p) for p in PurePosixPath(key).parents}
                if key in destinations or key in directories or parents & destinations:
                    raise DatasetError("Duplicate image/label destination or file/directory collision")
                destinations.add(key)
                directories.update(parents)
            width, height = _positive_int(row.get("width")), _positive_int(row.get("height"))
            annotations = row.get("annotations", {"boxes": []})
            if (not isinstance(annotations, dict) or set(annotations) != {"boxes"}
                    or not isinstance(annotations["boxes"], list)):
                raise DatasetError("Detection annotations must contain only a boxes list")
            boxes = tuple(_box(b, len(classes)) for b in annotations["boxes"])
            source = row.get("url")
            local = None
            url = None
            if source and isinstance(source, str) and urlsplit(source).scheme:
                validate_url(source)
                url = source
            else:
                if source is not None and not isinstance(source, str):
                    raise DatasetError("Image source must be a string")
                relative = safe_relative(source or file)
                local = (path.parent / relative).resolve()
                if not local.is_relative_to(path.parent):
                    raise DatasetError("Local source escapes the manifest directory")
            images.append(ImageRecord(file, row["split"], width, height, boxes, url, local))
        except (ValueError, TypeError, OverflowError) as error:
            message = str(error) if isinstance(error, DatasetError) else "Malformed image record"
            raise DatasetError(f"Line {number}: {message}") from None
    return Manifest(classes, tuple(images), hashlib.sha256(raw).hexdigest())


def verify_image(path: Path, image: ImageRecord) -> None:
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            opened.load()
            size = ImageOps.exif_transpose(opened).size
            if size != (image.width, image.height):
                raise DatasetError("Image dimensions do not match manifest")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise DatasetError("Image failed decoding or dimension verification") from None


def atomic_write(path: Path, content: bytes, *, staging_path: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if staging_path is None:
        fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        temp = Path(name)
    else:
        temp = staging_path
        temp.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def cache_key(url: str) -> str:
    return hashlib.sha256(("GET " + url).encode()).hexdigest()


def fetch_image(image: ImageRecord, cache: Path, *, offline: bool, refresh: bool,
                timeout: float, retries: int) -> Path:
    if image.local is not None:
        verify_image(image.local, image)
        return image.local
    assert image.url is not None
    cached = cache / (cache_key(image.url) + ".img")
    if cached.is_file() and not refresh:
        try:
            verify_image(cached, image)
            return cached
        except DatasetError:
            if offline:
                raise DatasetError("Offline cache entry failed verification") from None
    if offline:
        raise DatasetError("Offline cache miss; prepare online with a current export first")
    cache.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        fd, name = tempfile.mkstemp(prefix=".download-", dir=cache)
        temp = Path(name)
        status = None
        try:
            opener = build_opener(_HTTPSRedirect())
            request = Request(image.url, headers={"User-Agent": "safety-yolo/0.1"})
            with os.fdopen(fd, "wb") as handle:
                with opener.open(request, timeout=timeout) as response:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 100 * 1024 * 1024:
                            raise DatasetError("Image exceeds 100 MiB download limit")
                        handle.write(chunk)
            verify_image(temp, image)
            os.replace(temp, cached)
            return cached
        except HTTPError as error:
            status = error.code
            error.close()
        except (URLError, OSError, ValueError, HTTPException):
            pass
        finally:
            temp.unlink(missing_ok=True)
        if status in {401, 403, 404}:
            raise DatasetError(f"HTTP {status}; signed export may have expired. Export again and use a new output directory")
        if attempt < retries:
            time.sleep(min(2 ** attempt, 4))
    raise DatasetError("Image download failed after retries; check connectivity or renew the signed export")


def prepare(manifest: Manifest, output: Path, *, cache: Path = Path(".cache/images"),
            offline: bool = False, refresh: bool = False, threads: int = 12,
            timeout: float = 30, retries: int = 2) -> Path:
    """Publish verified images, invalidating only collision-free native .npy image caches."""
    if (offline and refresh) or not 1 <= threads <= 32 or not 0 < timeout <= 300 or not 0 <= retries <= 5:
        raise DatasetError("Use 1..32 threads, timeout (0,300], retries 0..5; offline conflicts with refresh")
    output = output.resolve()
    cache = cache.resolve()
    if cache == output or cache.is_relative_to(output):
        raise DatasetError("Cache must be outside the prepared dataset")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".prepare.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise DatasetError("Preparation is locked; if no process is active, remove .prepare.lock and retry") from None
    try:
        state = output / ".manifest.sha256"
        staging = output / ".prepare-staging"
        if any(p.is_symlink() or p.is_junction() for p in output.rglob("*")):
            raise DatasetError("Prepared output must not contain links or junctions")
        yaml_path = output / "data.yaml"
        expected = set()
        for image in manifest.images:
            expected.add(output / "images" / image.split / image.file)
            expected.add(output / "labels" / image.split / Path(image.file).with_suffix(".txt"))
        metadata = {state, yaml_path, output / "summary.json"}
        label_caches = {output / "labels" / f"{s}.cache" for s in SPLITS}
        staged = {staging / p.relative_to(output) for p in expected | metadata}
        files = expected | metadata | label_caches | staged | {lock}
        directories = {output, staging} | {output / kind / s for kind in ("images", "labels") for s in SPLITS}
        for path in files | directories.copy():
            directories.update(p for p in path.parents if p.is_relative_to(output))
        image_caches = {(output / "images" / image.split / image.file).with_suffix(".npy")
                        for image in manifest.images} - expected - directories
        files.update(image_caches)
        for path in output.rglob("*"):
            if not ((path.is_file() and path in files) or (path.is_dir() and path in directories)):
                raise DatasetError("Unexpected dataset artifacts; use a clean output directory")
        if state.is_file():
            if state.read_text().strip() != manifest.fingerprint:
                raise DatasetError("Output belongs to a different or unknown manifest; use a new output directory")
        else:
            initial_files = {lock, staging / state.name}
            if any(p not in initial_files and p != staging for p in output.rglob("*")):
                raise DatasetError("Output belongs to a different or unknown manifest; use a new output directory")
            pending = staging / state.name
            if pending.exists() and pending.read_bytes() != manifest.fingerprint.encode():
                raise DatasetError("Output belongs to a different or unknown manifest; use a new output directory")

        def publish(path: Path, content: bytes) -> None:
            atomic_write(path, content, staging_path=staging / path.relative_to(output))

        publish(state, manifest.fingerprint.encode())
        yaml_path.unlink(missing_ok=True)

        def one(image: ImageRecord) -> bool:
            destination = output / "images" / image.split / image.file
            try:
                if destination.is_file() and image.url and not refresh:
                    try:
                        verify_image(destination, image)
                        source = destination
                    except DatasetError:
                        source = fetch_image(image, cache, offline=offline, refresh=refresh,
                                             timeout=timeout, retries=retries)
                else:
                    source = fetch_image(image, cache, offline=offline, refresh=refresh,
                                         timeout=timeout, retries=retries)
                cached_image = destination.with_suffix(".npy")
                if cached_image in image_caches:
                    # Local pixels can change without changing the manifest fingerprint.
                    cached_image.unlink(missing_ok=True)
                if source != destination:
                    publish(destination, source.read_bytes())
                verify_image(destination, image)
                label = output / "labels" / image.split / Path(image.file).with_suffix(".txt")
                content = "".join(b.label() + "\n" for b in image.boxes).encode()
                publish(label, content)
                return label.read_bytes() == content
            except (OSError, ValueError):
                return False

        with ThreadPoolExecutor(max_workers=threads) as pool:
            successful = sum(pool.map(one, manifest.images))
        if successful != len(manifest.images):
            raise DatasetError(f"Preparation failed for {len(manifest.images) - successful} images. "
                               "No data.yaml published. Check local images/cache or renew expired export URLs")
        import yaml

        config = {"path": output.as_posix(), **{s: f"images/{s}" for s in SPLITS},
                  "names": dict(enumerate(manifest.classes))}
        for s in SPLITS:
            (output / "images" / s).mkdir(parents=True, exist_ok=True)
            (output / "labels" / s).mkdir(parents=True, exist_ok=True)
        for cached_labels in label_caches - expected:
            cached_labels.unlink(missing_ok=True)  # Ultralytics caches include absolute paths and parsed label contents.
        publish(output / "summary.json", json.dumps(manifest.summary(), indent=2).encode())
        publish(yaml_path, yaml.safe_dump(config, sort_keys=False).encode())
        return yaml_path
    finally:
        lock.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate every record and print sanitized counts")
    inspect.add_argument("manifest", type=Path)
    prep = commands.add_parser("prepare", help="Verify images and publish a local dataset")
    prep.add_argument("manifest", type=Path)
    prep.add_argument("--output", type=Path, default=Path("data/construction-site-safety"))
    prep.add_argument("--cache", type=Path, default=Path(".cache/images"))
    prep.add_argument("--offline", action="store_true")
    prep.add_argument("--refresh", action="store_true")
    prep.add_argument("--threads", type=int, default=12)
    prep.add_argument("--timeout", type=float, default=30)
    prep.add_argument("--retries", type=int, default=2)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        manifest = read_manifest(args.pop("manifest"))
        if command == "prepare":
            print(f"Verified dataset YAML: {prepare(manifest, **args)}")
        print(json.dumps(manifest.summary(), indent=2))
    except (DatasetError, OSError):
        # OS and transport exceptions may contain private paths or signed URLs.
        import sys

        error = sys.exception()
        parser.exit(2, f"error: {error if isinstance(error, DatasetError) else 'Dataset filesystem operation failed'}\n")


if __name__ == "__main__":
    main()