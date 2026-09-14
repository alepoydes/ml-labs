import hashlib
import json
import re
import shutil
import stat
import struct
import tempfile
import tomllib
import zipfile
import zlib
from pathlib import Path

from coursekit.common import CourseError, digest, file_hashes, safe_path
from coursekit.labs import require_lab

MAX_ARCHIVE = 256 * 1024**2
MAX_EXPANDED = 512 * 1024**2
MAX_FILES = 4096
REQUIRED = {
    "submission.toml",
    "train.py",
    "solution.py",
    "config.toml",
    "artifacts/provenance.json",
}


def metadata(raw: bytes) -> dict:
    if len(raw) > 65536:
        raise CourseError("Submission metadata exceeds 64 KiB.")
    value = tomllib.loads(raw.decode("utf-8"))
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise CourseError("Unsupported submission schema.")
    require_lab(value.get("lab"), value.get("api_version"))
    if not isinstance(value.get("variant"), str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value["variant"]
    ):
        raise CourseError(
            "Variant must be a short alphanumeric name (hyphens and underscores allowed)."
        )
    ai = value.get("ai_usage", {})
    if (
        not isinstance(ai, dict)
        or not isinstance(ai.get("access_tier"), str)
        or ai["access_tier"]
        not in {
            "none_or_free",
            "basic_subscription",
            "professional_subscription",
        }
    ):
        raise CourseError("Declare a supported ai_usage.access_tier.")
    if (
        not isinstance(ai.get("tools"), list)
        or not all(isinstance(x, str) for x in ai["tools"])
        or not isinstance(ai.get("notes"), str)
    ):
        raise CourseError("AI tools must be a list of names and notes must be text.")
    return value


def _allowed(name: str) -> bool:
    path = safe_path(name)
    return name in REQUIRED | {"package.json"} or (
        path.parts[0] in {"lib", "artifacts"}
        and len(path.parts) > 1
        and not any(p.startswith(".") or p == "__pycache__" for p in path.parts)
        and path.suffix not in {".pyc", ".pyo", ".npy"}
    )


def inspect(archive: Path) -> dict:
    """Inspect bounded bytes and hashes only. Never import a submitted module."""
    if archive.stat().st_size > MAX_ARCHIVE:
        raise CourseError("Archive exceeds 256 MiB.")
    try:
        # Bound the central directory before ZipFile allocates its entry objects.
        # Our size limits never require ZIP64 or multi-disk archives.
        with archive.open("rb") as stream:
            stream.seek(max(0, archive.stat().st_size - 65557))
            tail = stream.read(65557)
        end = tail.rfind(b"PK\x05\x06")
        if end < 0 or len(tail) - end < 22:
            raise CourseError("Missing ZIP end record.")
        _, disk, start_disk, count, total, size, offset, comment = struct.unpack_from(
            "<4s4H2LH", tail, end
        )
        if (
            disk
            or start_disk
            or count != total
            or total > MAX_FILES
            or size > 4 * 1024**2
            or end + 22 + comment != len(tail)
        ):
            raise CourseError("Unsupported or oversized ZIP directory.")
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if len(entries) > MAX_FILES or sum(e.file_size for e in entries) > MAX_EXPANDED:
                raise CourseError("Archive exceeds expanded size or file count limit.")
            names = set()
            hashes = {}
            small = {}
            for entry in entries:
                name = entry.filename
                if not _allowed(name) or name.casefold() in names:
                    raise CourseError(f"Invalid or duplicate archive member: {name}")
                mode = entry.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG) or entry.flag_bits & 1:
                    raise CourseError(f"Special or encrypted archive member: {name}")
                names.add(name.casefold())
                if entry.file_size < 0 or entry.file_size > MAX_EXPANDED:
                    raise CourseError("Invalid expanded file size.")
                hasher, size = hashlib.sha256(), 0
                data = bytearray()
                collect = name in {
                    "submission.toml",
                    "package.json",
                    "artifacts/provenance.json",
                    "config.toml",
                }
                with source.open(entry) as stream:
                    while block := stream.read(1024 * 1024):
                        size += len(block)
                        if size > entry.file_size or (collect and size > 1024**2):
                            raise CourseError(f"Invalid or oversized member: {name}")
                        hasher.update(block)
                        if collect:
                            data.extend(block)
                if size != entry.file_size:
                    raise CourseError(f"Truncated member: {name}")
                hashes[name] = hasher.hexdigest()
                if collect:
                    small[name] = bytes(data)
            for name in names:
                if any(
                    p.as_posix().casefold() in names
                    for p in Path(name).parents
                    if str(p) != "."
                ):
                    raise CourseError("Conflicting archive paths.")
            if not REQUIRED | {"package.json"} <= hashes.keys():
                raise CourseError(
                    "Archive is missing required solution files or provenance."
                )
            declared = json.loads(small["package.json"])
            if declared != {k: v for k, v in hashes.items() if k != "package.json"}:
                raise CourseError("Package file hashes do not match.")
            info = metadata(small["submission.toml"])
            tomllib.loads(small["config.toml"].decode())
            if json.loads(small["artifacts/provenance.json"]).get("status") != "complete":
                raise CourseError("Training provenance is incomplete.")
            return info
    except (
        zipfile.BadZipFile,
        ValueError,
        KeyError,
        UnicodeError,
        RuntimeError,
        NotImplementedError,
        TypeError,
        AttributeError,
        EOFError,
        zlib.error,
    ) as exc:
        raise CourseError(f"Invalid archive: {exc}") from exc


def extract(archive: Path, destination: Path) -> dict:
    info = inspect(archive)
    if destination.exists():
        with zipfile.ZipFile(archive) as source:
            expected = json.loads(source.read("package.json"))
        actual = file_hashes(destination)
        actual.pop("package.json", None)
        if (
            actual != expected
            or json.loads((destination / "package.json").read_text()) != expected
        ):
            raise CourseError(
                "Cached extraction was modified; remove this cache entry and retry."
            )
        return info
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".unpacking-"
    ) as temporary:
        root = Path(temporary) / "solution"
        root.mkdir()
        with zipfile.ZipFile(archive) as source:
            if (
                len(source.infolist()) > MAX_FILES
                or sum(e.file_size for e in source.infolist()) > MAX_EXPANDED
            ):
                raise CourseError("Archive changed during extraction.")
            for entry in source.infolist():
                if not _allowed(entry.filename):
                    raise CourseError("Archive changed during extraction.")
                path = root / safe_path(entry.filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
        # Recheck content before exposing the completed directory.
        expected = json.loads((root / "package.json").read_text())
        actual = file_hashes(root)
        actual.pop("package.json")
        if actual != expected:
            raise CourseError("Extraction hashes do not match.")
        root.rename(destination)
    return info


def pack(solution: Path, output: Path) -> Path:
    from coursekit.training import check_provenance

    info = metadata((solution / "submission.toml").read_bytes())
    check_provenance(solution)
    paths = {}
    candidates = [solution / name for name in REQUIRED if "/" not in name]
    for directory in (solution / "lib", solution / "artifacts"):
        if directory.is_symlink():
            raise CourseError(f"Symbolic link in solution: {directory.name}")
        candidates.extend(directory.rglob("*"))
    for path in sorted(candidates):
        relative = path.relative_to(solution).as_posix()
        if path.is_symlink():
            raise CourseError(f"Symbolic link in solution: {relative}")
        if path.is_file() and relative != "package.json" and _allowed(relative):
            paths[relative] = path
    if not REQUIRED <= paths.keys():
        raise CourseError("Solution is missing required files; train it before packaging.")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        archive = Path(temporary) / "submission.zip"
        hashes = {name: digest(path) for name, path in paths.items()}
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for name in sorted(paths.keys() | {"package.json"}):
                entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                entry.external_attr = (stat.S_IFREG | 0o644) << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                if name == "package.json":
                    target.writestr(entry, json.dumps(hashes, sort_keys=True))
                else:
                    with paths[name].open("rb") as src, target.open(entry, "w") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
        inspect(archive)
        destination = output / f"{info['lab']}-{info['variant']}-{digest(archive)[:12]}.zip"
        archive.replace(destination)
    print(f"{destination} ({destination.stat().st_size} bytes)")
    return destination
