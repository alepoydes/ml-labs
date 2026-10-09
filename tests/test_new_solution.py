import shutil
import sys
from pathlib import Path

import pytest

from coursekit.cli import main
from coursekit.common import file_hashes, read_toml


@pytest.fixture
def project(tmp_path, monkeypatch):
    lab = Path(__file__).resolve().parents[1] / "labs/blindjack"
    for template in ("starter", "baseline"):
        shutil.copytree(
            lab / template,
            tmp_path / "labs/blindjack" / template,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "arguments,template,name",
    [
        ([], "starter", "starter"),
        (["--template", "baseline"], "baseline", "baseline"),
        (["my-model"], "starter", "my-model"),
        (["my-model", "--template", "baseline"], "baseline", "my-model"),
        (["baseline", "--template", "baseline"], "baseline", "baseline"),
    ],
)
def test_copy_template(project, monkeypatch, arguments, template, name):
    source = project / "labs/blindjack" / template
    original = file_hashes(source)
    monkeypatch.setattr(sys, "argv", ["coursekit", "new", "blindjack", *arguments])
    main()
    destination = project / "work/blindjack" / name
    expected = read_toml(source / "submission.toml") | {"variant": name}
    assert read_toml(destination / "submission.toml") == expected
    actual = file_hashes(destination)
    assert actual.keys() == original.keys()
    assert all(
        actual[path] == value
        for path, value in original.items()
        if path != "submission.toml"
    )
    assert file_hashes(source) == original

    (destination / "solution.py").write_text("# Student changes\n")
    before_retry = file_hashes(destination)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert file_hashes(destination) == before_retry
    assert file_hashes(source) == original


@pytest.mark.parametrize("name", ["", "../outside", "nested/name", 'name"\nextra = "value'])
def test_invalid_copy_name_is_rejected(project, monkeypatch, name):
    monkeypatch.setattr(sys, "argv", ["coursekit", "new", "blindjack", name])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert not (project / "work").exists()


def test_builtin_baseline_reuses_verified_parameters_and_preserves_old_copies(tmp_path):
    from coursekit import resources, training

    source = Path(__file__).resolve().parents[1] / "labs/line_following"
    shutil.copytree(source, tmp_path / "labs/line_following")
    resources.sync(
        tmp_path / "data/line_following",
        resources.manifest(tmp_path / "labs/line_following/resources.toml"),
    )
    installed = tmp_path / "labs/line_following/baseline"
    original = file_hashes(installed)
    managed = training.prepare_baseline(tmp_path, "line_following")
    provenance = managed / "artifacts/provenance.json"
    before = provenance.stat().st_mtime_ns
    assert training.prepare_baseline(tmp_path, "line_following") == managed
    assert provenance.stat().st_mtime_ns == before
    assert training.check_provenance(managed, artifacts=True)["status"] == "complete"
    params = managed / "artifacts/controller.json"
    params.write_text(params.read_text().replace("0.12", "0.10"))
    changed = params.read_bytes()
    repaired = training.prepare_baseline(tmp_path, "line_following")
    assert repaired == managed
    assert (
        next(managed.parent.glob("previous-*/artifacts/controller.json")).read_bytes()
        == changed
    )
    assert file_hashes(installed) == original
    assert not (tmp_path / "work").exists()
    old_identity = training.controller_identity(managed)
    # A course update selects a different managed copy and preserves old evidence.
    config = installed / "config.toml"
    config.write_text(config.read_text().replace("0.12", "0.11"))
    updated = training.prepare_baseline(tmp_path, "line_following")
    assert updated != managed
    assert training.controller_identity(managed) == old_identity
    assert training.controller_identity(updated)["version"] != old_identity["version"]
