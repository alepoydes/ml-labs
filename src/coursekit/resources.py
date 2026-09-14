import json
import os
import tempfile
import urllib.request
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
        for entry in spec["files"]:
            path = _target(root, entry["path"])
            if (
                path.is_file()
                and path.stat().st_size == entry["size"]
                and digest(path) == entry["sha256"]
            ):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {entry['path']} ({entry['size']} bytes)")
            fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".download-")
            try:
                size = 0
                with (
                    os.fdopen(fd, "wb") as output,
                    urllib.request.urlopen(entry["url"], timeout=60) as response,
                ):
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > entry["size"]:
                            raise CourseError(
                                f"Download exceeds declared size: {entry['path']}"
                            )
                        output.write(block)
                if size != entry["size"] or digest(Path(temporary)) != entry["sha256"]:
                    raise CourseError(
                        f"Corrupted download: {entry['path']}; rerun resource preparation."
                    )
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        for name in old - current:
            _target(root, name).unlink(missing_ok=True)
        write_json(state_path, sorted(current))
        write_json(root / ".ready.json", {"manifest": fingerprint(spec)})
        print(f"Resources ready: {root}")
