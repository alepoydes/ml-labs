import json
import stat
import zipfile

import pytest

from coursekit.common import CourseError, digest
from coursekit.submissions import extract, inspect, pack


def test_deterministic_pack_and_roundtrip(solution, tmp_path):
    first = pack(solution, tmp_path / "dist")
    previous = first.read_bytes()
    assert pack(solution, tmp_path / "dist").read_bytes() == previous
    destination = tmp_path / "extracted"
    assert extract(first, destination)["lab"] == "blindjack"
    assert (destination / "solution.py").read_bytes() == (
        solution / "solution.py"
    ).read_bytes()
    assert (
        json.loads((destination / "artifacts/provenance.json").read_text())["resource_hash"]
        == "training-resource-fingerprint"
    )
    extract(first, destination)


def test_stale_training_is_not_relabelled(solution, tmp_path):
    (solution / "config.toml").write_text("selected_cards = 4\n")
    with pytest.raises(CourseError, match="changed"):
        pack(solution, tmp_path / "dist")


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/absolute.py",
        "C:/file",
        "lib/../file",
        "lib\\file",
        "lib/CON.py",
        "lib/a//b.py",
    ],
)
def test_unsafe_zip_paths_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(name, "bad")
    with pytest.raises(CourseError):
        inspect(archive)


def test_symlink_and_duplicate_paths(tmp_path):
    archive = tmp_path / "bad.zip"
    entry = zipfile.ZipInfo("lib/link")
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(entry, "/etc/passwd")
    with pytest.raises(CourseError, match="Special"):
        inspect(archive)
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("lib/a.py", "a")
        target.writestr("lib/A.py", "a")
    with pytest.raises(CourseError, match="duplicate"):
        inspect(archive)


def test_corrupt_hash_and_cached_extraction(solution, tmp_path):
    archive = pack(solution, tmp_path / "dist")
    destination = tmp_path / "cached"
    extract(archive, destination)
    (destination / "solution.py").write_text("raise RuntimeError('tampered')")
    declared = json.loads((destination / "package.json").read_text())
    declared["solution.py"] = digest(destination / "solution.py")
    (destination / "package.json").write_text(json.dumps(declared))
    with pytest.raises(CourseError, match="modified"):
        extract(archive, destination)


def test_inspection_does_not_execute_code(solution, tmp_path):
    marker = tmp_path / "executed"
    (solution / "solution.py").write_text(f"raise RuntimeError({str(marker)!r})\n")
    from coursekit.training import train

    train(solution, tmp_path, "resources", 0)
    archive = pack(solution, tmp_path / "dist")
    inspect(archive)
    assert not marker.exists()


@pytest.mark.parametrize(
    "replacement", ["variant = 12", "ai_usage = 12", "ai_usage = {access_tier = []}"]
)
def test_malformed_metadata_types_are_user_errors(replacement):
    from coursekit.submissions import metadata

    raw = 'schema_version = 1\nlab = "blindjack"\napi_version = 1\n'
    if not replacement.startswith("variant"):
        raw += 'variant = "test"\n'
    with pytest.raises(CourseError):
        metadata((raw + replacement).encode())


def test_zip_directory_limit_before_extraction(tmp_path):
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as target:
        for i in range(4097):
            target.writestr(f"lib/{i}.py", "")
    with pytest.raises(CourseError, match="oversized ZIP directory"):
        inspect(archive)
