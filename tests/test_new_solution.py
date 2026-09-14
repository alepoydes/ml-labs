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
