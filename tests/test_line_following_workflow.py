import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from coursekit import resources, submissions, training
from coursekit.common import read_toml
from coursekit.labs import get_lab
from coursekit.labs.line_following_course import END, GROUND, INK


@pytest.fixture
def project(tmp_path):
    original = Path(__file__).resolve().parents[1]
    shutil.copytree(original / "labs/line_following", tmp_path / "labs/line_following")
    specification = resources.manifest(tmp_path / "labs/line_following/resources.toml")
    resources.sync(tmp_path / "data/line_following", specification)
    return tmp_path


def test_bundled_resources_training_archive_and_local_evaluation(project):
    work = project / "work/line_following/starter"
    shutil.copytree(project / "labs/line_following/starter", work)
    training.train_local(work, project, seed=5)
    archive = submissions.pack(work, project / "dist")
    extracted = project / "unpacked"
    metadata = submissions.extract(archive, extracted)
    assert metadata["lab"] == "line_following"
    solution = training.load_solution(extracted)
    adapter = get_lab("line_following")
    result = adapter.evaluate(
        solution, project / "data/line_following", compiled=False, time_limit=0.002
    )
    assert result["episodes"] == 4
    assert result["score"] == result["completion_rate"] == 0
    assert result["timeout_rate"] == 1
    assert result["mean_success_time"] is None
    json.dumps(result, allow_nan=False)
    provenance = json.loads((extracted / "artifacts/provenance.json").read_text())
    assert provenance["lab"] == "line_following" and provenance["seed"] == 5
    # Bundled resources retain the normal byte-verification contract.
    spec = resources.manifest(project / "labs/line_following/resources.toml")
    assert (
        resources.ready(project / "data/line_following", spec)
        == provenance["resource_hash"]
    )
    # Moving a public checkout does not change the identity of its bundled bytes.
    moved = project / "another-checkout/labs/line_following"
    shutil.copytree(project / "labs/line_following", moved)
    moved_spec = resources.manifest(moved / "resources.toml")
    assert resources.resource_fingerprint(moved_spec) == resources.resource_fingerprint(
        spec
    )
    (project / "data/line_following/train.csv").write_text("corrupted")
    with pytest.raises(Exception, match="Resource changed"):
        resources.ready(project / "data/line_following", spec)
    resources.sync(project / "data/line_following", spec)
    assert (
        resources.ready(project / "data/line_following", spec)
        == provenance["resource_hash"]
    )


def test_baseline_batch_memory_reversal_and_debug(project):
    work = project / "work/line_following/baseline"
    shutil.copytree(project / "labs/line_following/baseline", work)
    training.train_local(work, project)
    solution = training.load_solution(work)
    scalar = training.load_solution(work)
    observation = {
        "time": torch.zeros(2),
        "rgb": torch.tensor(np.tile(GROUND, (2, 16, 1)), dtype=torch.float32),
        "gyro": torch.zeros((2, 3)),
        "accel": torch.zeros((2, 3)),
        "encoders": torch.zeros((2, 2)),
    }
    observation["rgb"][:, 7:9] = torch.tensor(INK, dtype=torch.float32)
    action = solution.act(observation, torch.ones(2, dtype=torch.bool))
    individual = scalar.act(
        {k: v[:1] for k, v in observation.items()}, torch.ones(1, dtype=torch.bool)
    )
    torch.testing.assert_close(action[:1], individual)
    assert (action > 0).all()
    observation["time"] += 0.02
    observation["rgb"][0, 7:9] = torch.tensor(END, dtype=torch.float32)
    solution.act(observation, torch.zeros(2, dtype=torch.bool))
    observation["rgb"][0] = torch.tensor(GROUND, dtype=torch.float32)
    for _ in range(3):
        observation["time"] += 0.02
        action = solution.act(observation, torch.zeros(2, dtype=torch.bool))
    assert (action[0] < 0).all() and (action[1] > 0).all()
    diagnostic = solution.debug(0)
    diagnostic["command"][:] = 123
    assert abs(solution.debug(0)["command"]).max() <= 1
    observation["rgb"][:, 7:9] = torch.tensor(INK, dtype=torch.float32)
    action = solution.act(observation, torch.tensor([False, True]))
    assert (action[0] < 0).all() and (action[1] > 0).all()
    assert read_toml(work / "config.toml")["throttle"] > 0


def test_local_evaluation_allows_online_gradient_adaptation(project):
    class Adaptive:
        def __init__(self):
            self.parameter = torch.nn.Parameter(torch.tensor(0.1))
            self.calls = 0

        def act(self, observation, reset):
            loss = (self.parameter - observation["gyro"][0, 0]).square()
            loss.backward()
            with torch.no_grad():
                self.parameter -= 0.01 * self.parameter.grad
                self.parameter.grad = None
            self.calls += 1
            return torch.zeros((len(reset), 2))

    solution = Adaptive()
    result = get_lab("line_following").evaluate(
        solution, project / "data/line_following", compiled=False, time_limit=0.001
    )
    assert solution.calls == result["episodes"] == 4
    assert solution.parameter.item() != pytest.approx(0.1)
