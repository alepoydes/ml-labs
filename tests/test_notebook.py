import json
import os
import subprocess
import sys

import numpy as np
import pytest

from coursekit.common import CourseError, file_hashes
from coursekit.submissions import extract, pack
from coursekit.training import load_module, load_training, train_local


def command(project, *args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=project,
        env=os.environ | {"MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def baseline(student_project):
    result = command(
        student_project,
        "-m",
        "coursekit.cli",
        "new",
        "blindjack",
        "experiment",
        "--template",
        "baseline",
    )
    assert result.returncode == 0, result.stderr
    solution = student_project / "work/blindjack/experiment"
    (solution / "config.toml").write_text(
        "resolution = 4\nhidden_units = 4\nmax_iter = 3\nlearning_rate = 0.001\n"
        'patience = 5\nactivation = "tanh"\n'
    )
    return solution


def test_baseline_train_notebook_and_archive(student_project, baseline):
    result = command(
        student_project, "-m", "coursekit.cli", "train", str(baseline), "--seed", "7"
    )
    assert result.returncode == 0, result.stderr
    run_dir = student_project / "runs/blindjack/experiment"
    history, saved = load_training(baseline, run_dir)
    np.testing.assert_array_equal(history["epoch"], [1, 2, 3])
    assert history["loss"].shape == history["validation_accuracy"].shape == (3,)
    assert saved["provenance"]["seed"] == 7
    assert saved["config"]["resolution"] == 4

    before = file_hashes(baseline / "artifacts"), file_hashes(run_dir)
    # Execute the real notebook, with a hard failure if opening it starts training.
    analysis = """
import runpy
import sys
import numpy as np
from coursekit import training
from coursekit.labs.blindjack import benchmark

def forbidden(*args, **kwargs):
    raise AssertionError('Opening a notebook must not train')
training.train_local = forbidden
namespace = runpy.run_path(sys.argv[1])
_, values = namespace['app'].run()
solution = values['solution']
images, labels, hands = benchmark(values['data_dir'])
predict = values['predict_classes']
expected = predict(solution.model, images[np.arange(len(images))], solution.resolution)
np.testing.assert_array_equal(values['predictions'], expected)
reloaded = training.load_solution(values['solution_dir'])
np.testing.assert_array_equal(values['mask'], reloaded.select(images[hands[0]]))
print('Notebook analysis passed')
"""
    result = command(student_project, "-c", analysis, str(baseline / "notebook.py"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Notebook analysis passed" in result.stdout
    assert "Public validation accuracy" in result.stdout
    assert before == (file_hashes(baseline / "artifacts"), file_hashes(run_dir))

    # The notebook's explicit training call produces the same inference artifacts.
    train_local(baseline, student_project, seed=7)
    assert before[0] == file_hashes(baseline / "artifacts")
    archive = pack(baseline, student_project / "dist")
    extracted = student_project / "extracted"
    extract(archive, extracted)
    assert not (extracted / "notebook.py").exists()
    assert not (extracted / "training.npz").exists()
    result = command(student_project, "-m", "coursekit.cli", "evaluate", str(baseline))
    assert result.returncode == 0, result.stderr
    expected = json.loads((run_dir / "evaluate.json").read_text())
    result = command(student_project, "-m", "coursekit.cli", "check", str(archive))
    assert result.returncode == 0, result.stderr
    assert json.loads((run_dir / "check.json").read_text()) == expected
    assert expected["hands"] == 330


def test_untrained_notebook_gives_instructions(student_project, baseline):
    result = command(student_project, str(baseline / "notebook.py"))
    assert result.returncode != 0
    assert "No saved training result" in result.stderr
    assert "uncomment train_local" in result.stderr
    assert not (baseline / "artifacts").exists()
    assert not (student_project / "runs").exists()


@pytest.mark.parametrize("changed", ["history", "model"])
def test_saved_results_are_not_silently_mixed(student_project, baseline, changed):
    train_local(baseline, student_project)
    run_dir = student_project / "runs/blindjack/experiment"
    path = {
        "history": run_dir / "training.npz",
        "model": baseline / "artifacts/model.joblib",
    }[changed]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(CourseError, match="changed"):
        load_training(baseline, run_dir)


def test_failed_retraining_invalidates_completion(student_project, baseline):
    train_local(baseline, student_project)
    (baseline / "train.py").write_text(
        "def train(*args):\n    raise ValueError('training interrupted')\n"
    )
    with pytest.raises(ValueError, match="interrupted"):
        train_local(baseline, student_project)
    assert not (baseline / "artifacts/provenance.json").exists()
    with pytest.raises(CourseError, match="No saved training result"):
        load_training(baseline, student_project / "runs/blindjack/experiment")
    with pytest.raises(CourseError, match="provenance"):
        pack(baseline, student_project / "dist")


def test_source_edits_allow_analysis_but_require_retraining_for_pack(
    student_project, baseline, capsys
):
    train_local(baseline, student_project)
    run_dir = student_project / "runs/blindjack/experiment"
    before, saved = load_training(baseline, run_dir)
    config = baseline / "config.toml"
    config.write_text(config.read_text().replace("resolution = 4", "resolution = 8"))
    source = baseline / "solution.py"
    source.write_text(source.read_text() + "\n# An experiment with the selection code.\n")
    history, restored = load_training(baseline, run_dir)
    assert "changed since training" in capsys.readouterr().out
    assert restored == saved and restored["config"]["resolution"] == 4
    np.testing.assert_array_equal(history["loss"], before["loss"])
    result = command(student_project, str(baseline / "notebook.py"))
    assert result.returncode == 0, result.stderr
    assert "Public validation accuracy" in result.stdout
    with pytest.raises(CourseError, match="changed"):
        pack(baseline, student_project / "dist")


def test_another_variant_loads_its_own_helpers(tmp_path):
    for name in ("first", "second"):
        solution = tmp_path / name
        (solution / "lib").mkdir(parents=True)
        (solution / "lib/__init__.py").write_text("")
        (solution / "lib/model.py").write_text(f"value = {name!r}\n")
        (solution / "solution.py").write_text("from lib.model import value\n")
        assert load_module(solution, "solution").value == name


@pytest.mark.parametrize("entry", ["solution", "train"])
def test_same_size_edits_reload_source_and_helpers(tmp_path, entry):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/__init__.py").write_text("")
    (tmp_path / "train.py").write_text("from solution import local, value\n")
    source, helper = tmp_path / "solution.py", tmp_path / "lib/model.py"
    for value in (1, 2):
        source.write_text(f"from lib.model import value\nlocal = {value}\n")
        helper.write_text(f"value = {value}\n")
        for path in (source, helper):
            os.utime(path, (1700000000, 1700000000))
        module = load_module(tmp_path, entry)
        assert (module.local, module.value) == (value, value)
