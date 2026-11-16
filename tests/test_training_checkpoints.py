# Imported pytest fixtures are intentionally rebound by injection below.
# ruff: noqa: F811
import base64
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from test_visual_queries import queries  # noqa: F401

from coursekit.common import CourseError, digest
from coursekit.training_checkpoints import (
    AdaptationTimeout,
    Checkpoints,
    checkpoint_writer,
)
from coursekit.training_v2 import adapt


def test_checkpoint_commit_keeps_previous_bundle_until_hashes_match(tmp_path):
    output, history = tmp_path / "run/artifacts", tmp_path / "run/history"
    output.mkdir(parents=True)
    limits = dict(wall_seconds=10, artifact_bytes=100000)
    received = Checkpoints(tmp_path / "accepted", limits)
    publish = checkpoint_writer(output, history, 3, limits, received)
    for name in ("first.json", "second.json"):
        (output / name).write_text('{"version":1}')
    receipt = publish({"adapt": dict(step=np.array([1]), loss=np.array([0.5]))})
    before = received.last
    assert receipt["index"] == 1 and receipt["seconds"] >= 0
    files = {"artifacts/first.json": dict(size=3, sha256=digest(output / "first.json"))}
    received(dict(kind="begin", files=files))
    received(
        dict(
            kind="chunk",
            path="artifacts/first.json",
            offset=0,
            data=base64.b64encode(b"bad").decode(),
        )
    )
    with pytest.raises(CourseError, match="hash mismatch"):
        received(dict(kind="commit"))
    assert received.last == before
    assert json.loads((before / "artifacts/second.json").read_text()) == {"version": 1}
    # Abandon the incomplete transfer: time expiry cannot promote it.
    received.deadline = time.monotonic() - 1
    with pytest.raises(AdaptationTimeout):
        received(dict(kind="commit"))
    assert received.last == before and received.receipt == receipt
    with np.load(before / "history/adapt.npz", allow_pickle=False) as saved:
        assert saved["loss"].tolist() == [0.5]


@pytest.mark.parametrize(
    "name", ["../escape", "artifacts/provenance.json", "artifacts/x.npy"]
)
def test_checkpoint_rejects_unsafe_outputs_before_creating_files(tmp_path, name):
    received = Checkpoints(tmp_path / "accepted", dict(wall_seconds=10, artifact_bytes=10))
    with pytest.raises(CourseError):
        received(dict(kind="begin", files={name: dict(size=1, sha256="0" * 64)}))
    assert received.last is None and not list(received.directory.iterdir())


def test_local_adaptation_uses_checkpoint_only_for_time_limit(tmp_path):
    solution = tmp_path / "solution"
    shutil.copytree(Path(__file__).parents[1] / "labs/visual_queries/starter", solution)
    (solution / "submission.toml").write_text(
        'schema_version=1\napi_version=2\nlab="blindjack"\nvariant="fixture"\n'
        "[training]\nsupports_adapt=true\n"
        '[ai_usage]\naccess_tier="none_or_free"\ntools=[]\nnotes=""\n'
    )
    (solution / "train.py").write_text("""
import time
import numpy as np
from pathlib import Path

def adapt(data, initial, output, config, seed, *, device, limits, checkpoint, report):
    if seed != 2:
        (output / "a.json").write_text('{"version":1}')
        (output / "b.json").write_text('{"version":1}')
        checkpoint({"adapt": {"step": np.array([1]), "loss": np.array([0.5])}})
    (output / "a.json").write_text('unfinished')
    if seed == 1:
        raise ValueError("student bug")
    time.sleep(0.12)
""")
    base, data = tmp_path / "base", tmp_path / "data"
    base.mkdir()
    data.mkdir()
    limits = dict(wall_seconds=0.1, memory_mb=128, artifact_bytes=100000, gpu_memory_mb=0)
    # Resolve/import once before starting the short measured call.
    from coursekit.execution import resolve_device

    resolve_device("cpu")
    saved = adapt(solution, data, base, run_dir=tmp_path / "runs", limits=limits)
    assert json.loads((saved / "a.json").read_text())["version"] == 1
    assert json.loads((saved / "b.json").read_text())["version"] == 1
    record = json.loads((saved.parent / "training.json").read_text())
    assert record["status"] == "complete"
    assert record["adaptation"]["stop_reason"] == "time_limit"
    assert record["adaptation"]["checkpoint"]["index"] == 1
    with pytest.raises(ValueError, match="student bug"):
        adapt(solution, data, base, run_dir=tmp_path / "runs", limits=limits, seed=1)
    with pytest.raises(AdaptationTimeout, match="without a committed"):
        adapt(solution, data, base, run_dir=tmp_path / "runs", limits=limits, seed=2)


@pytest.mark.parametrize("import_seconds", [2, 10])
def test_local_adaptation_charges_numeric_setup_and_imports(
    queries, tmp_path, monkeypatch, import_seconds
):
    from coursekit import training
    from coursekit.labs import visual_queries_learning

    solution = tmp_path / "solution"
    shutil.copytree(Path(__file__).parents[1] / "labs/visual_queries/starter", solution)
    (solution / "train.py").write_text("""
import json

def adapt(data, initial, output, config, seed, *, device, limits, checkpoint, report):
    state = {"answer": 1, "vocab_size": 7, "episode_id": None,
             "remaining": limits["wall_seconds"]}
    (output / "state.json").write_text(json.dumps(state))
    checkpoint()
""")
    initial = tmp_path / "initial"
    initial.mkdir()
    clock = [0.0]
    open_data, load_module = visual_queries_learning.open_data, training.load_module

    def timed_data(root):
        clock[0] += 1
        return open_data(root)

    def timed_import(*args):
        clock[0] += import_seconds
        return load_module(*args)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(visual_queries_learning, "open_data", timed_data)
    monkeypatch.setattr(training, "load_module", timed_import)
    limits = dict(wall_seconds=10, memory_mb=128, artifact_bytes=100000, gpu_memory_mb=0)
    if import_seconds == 2:
        saved = adapt(solution, queries, initial, run_dir=tmp_path / "runs", limits=limits)
        assert json.loads((saved / "state.json").read_text())["remaining"] == 7
    else:
        with pytest.raises(AdaptationTimeout, match="without a committed"):
            adapt(solution, queries, initial, run_dir=tmp_path / "runs", limits=limits)
    record_path = next((tmp_path / "runs").glob("*/training.json"))
    record = json.loads(record_path.read_text())
    assert record["adaptation"]["seconds"] == 1 + import_seconds
    assert record["adaptation"]["configured_seconds"] == 10
    assert record["adaptation"]["budget_seconds"] == 10
    assert record["adaptation"]["stop_reason"] == (
        "completed" if import_seconds == 2 else "time_limit"
    )
    assert limits["wall_seconds"] == 10
    if import_seconds == 10:
        assert not (record_path.parent / "artifacts/state.json").exists()
