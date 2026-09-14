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
