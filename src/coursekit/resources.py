import json
import os
import stat
import tempfile
import urllib.request
import zipfile
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock

from coursekit.common import (
    CourseError,
    digest,
    fingerprint,
    read_toml,
    safe_path,
    write_json,
)


def manifest(path: Path) -> dict:
    value = read_toml(path)
    if value.get("schema_version") != 1 or not value.get("files"):
        raise CourseError(
            f"Resources are not published: {path}. Ask the instructor for assets."
        )
    names = []
    for entry in value["files"]:
        name = entry["path"]
        safe_path(name)
        if name.startswith(".") or len(entry["sha256"]) != 64 or int(entry["size"]) < 0:
            raise CourseError(f"Invalid resource entry: {name}")
        names.append(name.casefold())
    if len(set(names)) != len(names):
        raise CourseError("Duplicate resource paths.")
    downloads = [value["archive"]] if "archive" in value else value["files"]
    for entry in downloads:
        if (
            not isinstance(entry.get("url"), str)
            or not entry["url"].startswith(("https://", "http://", "file://"))
            or len(entry["sha256"]) != 64
            or type(entry["size"]) is not int
            or entry["size"] < 0
        ):
            raise CourseError("Invalid resource download metadata.")
    for name in names:
        if any(
            parent.as_posix().casefold() in names
            for parent in Path(name).parents
            if str(parent) != "."
        ):
            raise CourseError("Conflicting resource paths.")
    return value


def _target(root: Path, name: str) -> Path:
    path = root / safe_path(name)
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise CourseError(f"Resource path contains a symbolic link: {path}")
    return path


def ready(root: Path, spec: dict) -> str:
    marker = root / ".ready.json"
    expected = fingerprint(spec)
    if not marker.exists() or json.loads(marker.read_text()).get("manifest") != expected:
        raise CourseError("Resources are incomplete; run step_0_update_resources.")
    for entry in spec["files"]:
        path = _target(root, entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["size"]
            or digest(path) != entry["sha256"]
        ):
            raise CourseError(
                f"Resource changed: {entry['path']}; run step_0_update_resources."
            )
    return expected


def _store(path: Path, entry: dict, source) -> None:
    """Check bounded bytes before replacing one managed file."""
    with source:
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".download-")
        try:
            size = 0
            with os.fdopen(fd, "wb") as output:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > entry["size"]:
                        raise CourseError(f"Download exceeds declared size: {path.name}")
                    output.write(block)
            if size != entry["size"] or digest(Path(temporary)) != entry["sha256"]:
                raise CourseError(f"Corrupted download: {path.name}; rerun preparation.")
            os.replace(temporary, path)
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
            raise CourseError(f"Invalid resource archive: {path.name}") from exc
        finally:
            Path(temporary).unlink(missing_ok=True)


def sync(root: Path, spec: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".update.lock"), timeout=0):
        state_path = root / ".managed.json"
        old = set(json.loads(state_path.read_text())) if state_path.exists() else set()
        current = {entry["path"] for entry in spec["files"]}
        for name in old | current:
            _target(root, name)
        (root / ".ready.json").unlink(missing_ok=True)
        # The union survives interruption, including a subsequent manifest change.
        write_json(state_path, sorted(old | current))
        missing = []
        for entry in spec["files"]:
            path = _target(root, entry["path"])
            if (
                path.is_file()
                and path.stat().st_size == entry["size"]
                and digest(path) == entry["sha256"]
            ):
                continue
            missing.append(entry)
        with tempfile.TemporaryDirectory(dir=root) as temporary, ExitStack() as stack:
            archive = None
            if missing and "archive" in spec:
                entry = spec["archive"]
                path = Path(temporary) / "resources.zip"
                print(f"Downloading resource archive ({entry['size']} bytes)")
                _store(path, entry, urllib.request.urlopen(entry["url"], timeout=60))
                try:
                    archive = stack.enter_context(zipfile.ZipFile(path))
                except zipfile.BadZipFile as exc:
                    raise CourseError("Invalid resource archive.") from exc
                members = archive.infolist()
                sizes = {e["path"]: e["size"] for e in spec["files"]}
                if (
                    len(members) != len(sizes)
                    or {m.filename for m in members} != set(sizes)
                    or any(
                        m.file_size != sizes[m.filename]
                        or stat.S_IFMT(m.external_attr >> 16) not in (0, stat.S_IFREG)
                        or m.flag_bits & 1
                        for m in members
                    )
                ):
                    raise CourseError("Archive members do not match the resource manifest.")
            for entry in missing:
                path = _target(root, entry["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Preparing {entry['path']} ({entry['size']} bytes)")
                source = (
                    archive.open(entry["path"])
                    if archive is not None
                    else urllib.request.urlopen(entry["url"], timeout=60)
                )
                _store(path, entry, source)
        for name in old - current:
            _target(root, name).unlink(missing_ok=True)
        write_json(state_path, sorted(current))
        write_json(root / ".ready.json", {"manifest": fingerprint(spec)})
        print(f"Resources ready: {root}")
