import hashlib
import json
import os
import re
import tempfile
import tomllib
from pathlib import Path, PurePosixPath


class CourseError(Exception):
    pass


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def safe_path(name: str) -> Path:
    """Portable relative path shared by ZIP entries and resource manifests."""
    parts = PurePosixPath(name).parts
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
    }
    if (
        not parts
        or name != "/".join(parts)
        or name.startswith("/")
        or any(p in {".", ".."} or p.endswith(".") for p in parts)
        or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in parts)
        or any(p.split(".")[0].upper() in reserved for p in parts)
    ):
        raise CourseError(f"Unsafe relative path: {name!r}")
    return Path(*parts)


def file_hashes(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CourseError(f"Symbolic link is not allowed: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
    return result
