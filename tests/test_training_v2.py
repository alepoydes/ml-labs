import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from coursekit.common import CourseError, file_hashes
from coursekit.submissions import extract, pack
from coursekit.training import load_solution, load_training, train
from coursekit.training_v2 import adapt


def test_composite_stages_archive_reload_and_failed_replacement(tmp_path, data):
    source = Path(__file__).resolve().parents[1] / "examples/composite_blindjack"
    solution = tmp_path / "solution"
    shutil.copytree(source, solution, ignore=shutil.ignore_patterns("__pycache__"))
    runs = tmp_path / "runs"
    events = []
    shared = train(
        solution, data, "fixture", 0, run_dir=runs, stage="encoder", report=events.append
    )
    assert not (solution / "artifacts").exists()
    assert events[-1]["step"] == 5
    members = [
        train(
            solution,
            data,
            "fixture",
            seed,
            run_dir=runs,
            stage="member",
            inputs={"encoder": shared},
        )
        for seed in (1, 2)
    ]
    complete = train(
        solution,
        data,
        "fixture",
        3,
        run_dir=runs,
        inputs={"encoder": shared, "member_1": members[0], "member_2": members[1]},
    )
    original = file_hashes(solution / "artifacts")
    history, _ = load_training(solution, shared.parent)
    assert len(history["encoder"]["step"]) == 5
    history, record = load_training(solution, members[0].parent)
    assert len(history["head"]["step"]) == 10
    assert record["inputs"]["encoder"] == file_hashes(shared)
    assert file_hashes(complete) == original
    archive = pack(solution, tmp_path / "dist")
    extracted = tmp_path / "extracted"
    extract(archive, extracted)
    images = np.full((7, 128, 128, 3), 255, np.uint8)
    np.testing.assert_array_equal(
        load_solution(solution).select(images), load_solution(extracted).select(images)
    )
    with pytest.raises(ValueError, match="Assembly requires"):
        train(solution, data, "fixture", 4, run_dir=runs, inputs={"member_1": members[0]})
    assert file_hashes(solution / "artifacts") == original
    failed = [json.loads(p.read_text()) for p in runs.glob("*/training.json")]
    assert sum(r["status"] == "failed" for r in failed) == 1


def test_adaptation_starts_from_base_and_inputs_are_immutable(solution, data, tmp_path):
    metadata = solution / "submission.toml"
    metadata.write_text(metadata.read_text().replace("api_version = 1", "api_version = 2"))
    (solution / "train.py").write_text("""
import numpy as np

def train(data, output, config, seed, **kwargs):
    np.savez_compressed(output / 'state.npz', value=np.array([1]))

def adapt(data, source, output, config, seed, **kwargs):
    with np.load(source / 'state.npz') as arrays:
        value = arrays['value'] + 1
    np.savez_compressed(output / 'state.npz', value=value)
    if seed == 99:
        (source / 'state.npz').unlink()
""")
    (solution / "solution.py").write_text("""
import numpy as np
class Solution:
    def __init__(self, artifacts, config, seed, *, device):
        with np.load(artifacts / 'state.npz') as arrays:
            self.value = int(arrays['value'][0])
""")
    base = train(solution, data, "fixture", 0, run_dir=tmp_path / "runs")
    before = file_hashes(base)
    limits = dict(wall_seconds=1, memory_mb=512, artifact_bytes=10000, gpu_memory_mb=0)
    for _ in range(2):
        adapted = adapt(solution, data, base, run_dir=tmp_path / "adapted", limits=limits)
        assert load_solution(solution, artifact_dir=adapted).value == 2
        assert load_solution(solution).value == 1
        assert file_hashes(base) == before
    with pytest.raises(CourseError, match="modified its input"):
        adapt(solution, data, base, run_dir=tmp_path / "bad", limits=limits, seed=99)
    assert file_hashes(base) == before
    with pytest.raises(CourseError, match="artifact limit"):
        adapt(
            solution,
            data,
            base,
            run_dir=tmp_path / "large",
            limits=limits | {"artifact_bytes": 1},
        )
    assert load_solution(solution).value == 1


def test_switching_solutions_does_not_import_previous_helpers(tmp_path):
    from coursekit.training import load_module

    for name, regular in [("old", True), ("new", False)]:
        root = tmp_path / name
        (root / "lib").mkdir(parents=True)
        if regular:
            (root / "lib/__init__.py").write_text("")
            (root / "lib/previous.py").write_text("value = 'must not leak'\n")
        (root / "lib/model.py").write_text(f"value = {name!r}\n")
        (root / "solution.py").write_text("from lib.model import value\n")
        assert load_module(root, "solution").value == name
    (root / "solution.py").write_text("from lib.previous import value\n")
    with pytest.raises(ModuleNotFoundError):
        load_module(root, "solution")


def test_stage_metadata_is_opt_in_without_reinterpreting_v1(solution):
    from coursekit.submissions import metadata

    raw = "training = []\n" + (solution / "submission.toml").read_text()
    assert metadata(raw.encode())["api_version"] == 1
    with pytest.raises(CourseError, match="training must be a table"):
        metadata(raw.replace("api_version = 1", "api_version = 2").encode())
