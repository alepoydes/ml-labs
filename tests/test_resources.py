import stat
import zipfile

import pytest

from coursekit.common import CourseError, digest
from coursekit.resources import ready, sync


def entry(path, name):
    return {
        "path": name,
        "url": path.as_uri(),
        "sha256": digest(path),
        "size": path.stat().st_size,
    }


def test_updates_reuse_hashes_and_preserve_unmanaged_files(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"first")
    root = tmp_path / "data"
    spec = {"schema_version": 1, "files": [entry(source, "a.bin")]}
    sync(root, spec)
    ready(root, spec)
    (root / "student.txt").write_text("keep")
    source.unlink()  # An unchanged active resource must not attempt a download.
    sync(root, spec)
    source.write_bytes(b"second")
    changed = {"schema_version": 1, "files": [entry(source, "b.bin")]}
    sync(root, changed)
    assert not (root / "a.bin").exists()
    assert (root / "student.txt").read_text() == "keep"
    ready(root, changed)


def test_failed_update_recovers_and_blocks_readers(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"valid")
    root = tmp_path / "data"
    spec = {"schema_version": 1, "files": [entry(source, "a.bin")]}
    source.write_bytes(b"wrong")
    with pytest.raises(CourseError, match="Corrupted"):
        sync(root, spec)
    with pytest.raises(CourseError, match="incomplete"):
        ready(root, spec)
    source.write_bytes(b"valid")
    sync(root, spec)
    ready(root, spec)
    (root / "a.bin").write_bytes(b"wrong")
    with pytest.raises(CourseError, match="changed"):
        ready(root, spec)


def test_resource_symlink_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"secret")
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.bin").symlink_to(source)
    with pytest.raises(CourseError, match="symbolic"):
        sync(root, {"files": [entry(source, "a.bin")]})


def bundle(tmp_path, members=None):
    source = tmp_path / "source.txt"
    source.write_bytes(b"sample")
    archive = tmp_path / "resources.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, content in members or [("train/example.txt", b"sample")]:
            target.writestr(name, content)
    file = entry(source, "train/example.txt")
    del file["url"]
    return {
        "schema_version": 1,
        "archive": entry(archive, "resources.zip"),
        "files": [file],
    }


def test_archive_roundtrip_repair_and_no_persistent_duplicate(tmp_path):
    spec = bundle(tmp_path)
    root = tmp_path / "data"
    sync(root, spec)
    ready(root, spec)
    assert (root / "train/example.txt").read_bytes() == b"sample"
    assert not list(root.rglob("*.zip"))
    (root / "train/example.txt").write_bytes(b"broken")
    sync(root, spec)
    ready(root, spec)
    (tmp_path / "resources.zip").unlink()
    sync(root, spec)  # An intact dataset does not download the archive again.
    ready(root, spec)
    raw = tmp_path / "replacement.txt"
    raw.write_text("new")
    (root / "student.txt").write_text("keep")
    sync(root, {"files": [entry(raw, "replacement.txt")]})
    assert not (root / "train/example.txt").exists()
    assert (root / "student.txt").read_text() == "keep"


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "traversal",
        "duplicate",
        "symlink",
        "member_hash",
        "archive_hash",
        "not_zip",
    ],
)
def test_invalid_archive_never_becomes_ready(tmp_path, damage):
    members = [("train/example.txt", b"sample")]
    if damage in {"extra", "traversal"}:
        members += [("extra.txt" if damage == "extra" else "../escape.txt", b"extra")]
    elif damage == "duplicate":
        members *= 2
    elif damage == "member_hash":
        members = [("train/example.txt", b"wrong!")]
    elif damage == "symlink":
        link = zipfile.ZipInfo("train/example.txt")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        members = [(link, b"sample")]
    if damage == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            spec = bundle(tmp_path, members)
    else:
        spec = bundle(tmp_path, members)
    if damage in {"archive_hash", "not_zip"}:
        archive = tmp_path / "resources.zip"
        archive.write_bytes(b"not a zip")
        if damage == "not_zip":
            spec["archive"] = entry(archive, "resources.zip")
    root = tmp_path / "data"
    with pytest.raises(CourseError):
        sync(root, spec)
    with pytest.raises(CourseError, match="incomplete"):
        ready(root, spec)
    assert not (tmp_path / "escape.txt").exists()
    assert not (root / "train/example.txt").exists()
