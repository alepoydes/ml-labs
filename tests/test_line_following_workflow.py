import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from coursekit import resources, submissions, training
from coursekit.common import read_toml
from coursekit.labs import get_lab
from coursekit.labs.line_following_cases import sample_case
from coursekit.labs.line_following_course import END, GROUND, INK, START


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
    events = []
    result = adapter.evaluate(
        solution,
        project / "data/line_following",
        time_limit=0.002,
        progress=events.append,
    )
    assert result["episodes"] == len(adapter.benchmark(project / "data/line_following"))
    assert result["score"] == result["completion_rate"] == 0
    assert result["timeout_rate"] == 1
    assert result["mean_success_time"] is None
    cases = [event["metrics"] for event in events if "metrics" in event]
    assert len(cases) == result["episodes"]
    assert all(case["travelled_distance"] >= 0 for case in cases)
    assert all(case["status"] == "timeout" for case in cases)
    assert result["mean_completion_fraction"] == pytest.approx(
        np.mean([case["progress"] / case["route_length"] for case in cases])
    )
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
    # Advance through the endpoint before reversing; colour alone is not the endpoint.
    for _ in range(30):
        observation["time"] += 0.02
        observation["encoders"][0] += 0.08 * 0.02 / 0.035
        action = solution.act(observation, torch.zeros(2, dtype=torch.bool))
    assert (action[0] < 0).all() and (action[1] > 0).all()
    diagnostic = solution.debug(0)
    diagnostic["command"][:] = 123
    assert abs(solution.debug(0)["command"]).max() <= 1
    observation["rgb"][:, 7:9] = torch.tensor(INK, dtype=torch.float32)
    action = solution.act(observation, torch.tensor([False, True]))
    assert (action[0] < 0).all() and (action[1] > 0).all()
    # On an incomplete return, blue uses the same physical-end reversal rule.
    observation["rgb"][0, 7:9] = torch.tensor(START, dtype=torch.float32)
    observation["time"] += 0.02
    solution.act(observation, torch.zeros(2, dtype=torch.bool))
    assert solution.debug(0)["drive_direction"] == -1
    observation["rgb"][0] = torch.tensor(GROUND, dtype=torch.float32)
    for _ in range(30):
        observation["time"] += 0.02
        observation["encoders"][0] -= 0.08 * 0.02 / 0.035
        action = solution.act(observation, torch.zeros(2, dtype=torch.bool))
    assert solution.debug(0)["drive_direction"] == 1
    assert abs(float(action[0, 0] - action[0, 1])) < 0.01
    assert read_toml(work / "config.toml")["speed"] > 0


def test_baseline_keeps_one_line_when_a_second_branch_appears(project):
    work = training.new_solution(project, "line_following", "baseline", "branches")
    training.train_local(work, project)
    solution = training.load_solution(work)
    observation = {
        "time": torch.zeros(2),
        "rgb": torch.tensor(np.tile(GROUND, (2, 16, 1)), dtype=torch.float32),
        "gyro": torch.zeros((2, 3)),
        "accel": torch.zeros((2, 3)),
        "encoders": torch.zeros((2, 2)),
    }
    observation["rgb"][:, 9:11] = torch.tensor(INK, dtype=torch.float32)
    solution.act(observation, torch.ones(2, dtype=torch.bool))
    original = solution.debug(0)["line_offset_m"]
    observation["rgb"][0, 1:3] = torch.tensor(INK, dtype=torch.float32)
    observation["time"] += 0.02
    solution.act(observation, torch.zeros(2, dtype=torch.bool))
    assert solution.debug(0)["line_offset_m"] == pytest.approx(original, abs=1e-6)
    assert solution.debug(1)["line_offset_m"] == pytest.approx(original, abs=1e-6)
    # A separate branch must not move the estimated line into the white gap.
    assert original > 0.01


def test_baseline_partial_reset_matches_fresh_controller(project):
    work = training.new_solution(project, "line_following", "baseline", "resets")
    training.train_local(work, project)
    batched = training.load_solution(work)
    single = [training.load_solution(work), training.load_solution(work)]
    rng = np.random.default_rng(123)
    for step in range(40):
        reset = torch.tensor([step in (0, 12), step in (0, 25)])
        observation = {
            "time": torch.tensor([step * 0.02, step * 0.02]),
            "rgb": torch.tensor(np.tile(GROUND, (2, 16, 1)), dtype=torch.float32),
            "gyro": torch.tensor(rng.normal(0, 0.05, (2, 3)), dtype=torch.float32),
            "accel": torch.zeros((2, 3)),
            "encoders": torch.full((2, 2), step * 0.05),
        }
        if step % 11 < 8:
            observation["rgb"][:, 6:8] = torch.tensor(INK, dtype=torch.float32)
        for i in range(2):
            if reset[i]:
                single[i] = training.load_solution(work)
        actual = batched.act(observation, reset)
        expected = torch.cat(
            [
                s.act({k: v[i : i + 1] for k, v in observation.items()}, reset[i : i + 1])
                for i, s in enumerate(single)
            ]
        )
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all() and actual.abs().max() <= 1


@pytest.mark.parametrize("paint", [INK, START, END], ids=["black", "blue", "red"])
def test_baseline_heading_estimate_while_turning_with_an_offset_line(project, paint):
    from coursekit.labs import line_following as physics
    from coursekit.labs.line_following_course import Course, optical

    work = training.new_solution(project, "line_following", "baseline", "heading")
    training.train_local(work, project)
    solution = training.load_solution(work)
    course = Course.polyline("straight", np.array([[-3.0, 0.0], [3.0, 0.0]]))
    q = physics.initial_state(physics.robot()).q
    q[1] = 0.03
    dt, speed, yaw_rate = 0.02, 0.035, -0.8
    for step in range(61):
        rotation = physics.rotation(q)
        observation = {
            "time": torch.tensor([step * dt]),
            "rgb": torch.from_numpy(optical(course, q[:3].numpy(), rotation.numpy()))[None],
            "gyro": (rotation.T @ q.new_tensor([0, 0, yaw_rate if step else 0])).float()[
                None
            ],
            "accel": torch.zeros((1, 3)),
            "encoders": torch.full((1, 2), step * dt * speed / 0.035),
        }
        ground = observation["rgb"].new_tensor(GROUND)
        # Repaint the same coverage fractions: colour must not freeze geometry.
        contrast = (np.asarray(paint) - GROUND) / (np.asarray(INK) - GROUND)
        observation["rgb"] = ground + (observation["rgb"] - ground) * ground.new_tensor(
            contrast
        )
        solution.act(observation, torch.tensor([step == 0]))
        if step == 60:
            assert solution.debug()["heading_error_rad"] == pytest.approx(
                -float(q[5]), abs=0.12
            )
        q[:2] += dt * speed * torch.stack((q[5].cos(), q[5].sin()))
        q[5] += dt * yaw_rate


@pytest.mark.parametrize("rounded", [False, True])
def test_baseline_recovers_corners_and_settles_with_varied_robots(project, rounded):
    from coursekit.labs.line_following_course import Course
    from coursekit.labs.line_following_env import Environment

    work = training.new_solution(project, "line_following", "baseline", "corners")
    training.train_local(work, project)
    solution = training.load_solution(work)
    if rounded:
        # Two cubic pieces round a right angle, with C2 joins to both straight legs.
        controls = np.array(
            [
                [[0, 0], [0.1, 0], [0.2, 0], [0.3, 0]],
                [[0.3, 0], [0.32, 0], [0.34, 0], [0.35, 0.01]],
                [[0.35, 0.01], [0.36, 0.02], [0.36, 0.04], [0.36, 0.06]],
                [[0.36, 0.06], [0.36, 0.16], [0.36, 0.26], [0.36, 0.36]],
            ]
        )
        course = Course("rounded", controls, np.array([0, 5, 6, 7, 12]))
        straight_x = 0.36
    else:
        course = Course.polyline("corner", [[0, 0], [0.5, 0], [0.5, 0.5]])
        straight_x = 0.5
    mirrored = Course("right turn", course.controls * [1, -1], course.knots, course.corners)
    seeds = list(range(8)) * 2
    # Use the public episode limit; retain the spatial settling requirements.
    env = Environment(
        [sample_case(c, seed) for c, seed in zip([course] * 8 + [mirrored] * 8, seeds)]
    )
    signs = np.repeat([1, -1], 8)
    reversed_on_loss = np.zeros(16, dtype=bool)
    encountered_loss = np.zeros(16, dtype=bool)
    sampled_straight = np.zeros(16, dtype=bool)
    worst_offset = np.zeros(16)
    worst_heading = np.zeros(16)
    with torch.no_grad():
        while not env.done.all():
            q = env.state.q.numpy().copy()
            action = solution.act(env.observe(), env.first)
            recovering = np.array([solution.debug(i)["recovering"] == 1 for i in range(16)])
            outward = env.leg == 0
            encountered_loss |= recovering & outward
            env.step(action)
            travel = env.state.q.numpy()[:, :2] - q[:, :2]
            forward = travel[:, 0] * np.cos(q[:, 5]) + travel[:, 1] * np.sin(q[:, 5])
            reversed_on_loss |= recovering & outward & (forward < -0.0001)
            # Measure the straight after the bend, before the endpoint marker.
            settled = outward & (q[:, 1] * signs > 0.20) & (q[:, 1] * signs < 0.24)
            sampled_straight |= settled
            worst_offset = np.maximum(
                worst_offset, np.where(settled, abs(q[:, 0] - straight_x), 0)
            )
            worst_heading = np.maximum(
                worst_heading, np.where(settled, abs(q[:, 5] - signs * np.pi / 2), 0)
            )
    assert (env.status == "finished").all(), env.info()
    np.testing.assert_allclose(env.info()["completion_fraction"], 1)
    assert sampled_straight.all()
    assert (worst_offset < 0.006).all(), worst_offset
    assert (worst_heading < 0.15).all(), worst_heading
    if not rounded:
        assert encountered_loss.any()
        assert reversed_on_loss[encountered_loss].all()


def test_baseline_completes_public_bend_without_sustained_steering_oscillation(project):
    from coursekit.labs.line_following_env import Environment
    from coursekit.labs.line_following_task import benchmark

    published = next(
        case
        for case in benchmark(project / "data/line_following", "train")
        if case.name == "train-04"
    )
    course = published.course
    cases = [sample_case(course, i) for i in range(8)] + [published]
    from dataclasses import replace

    for base in (cases[0], published):
        for sensor in range(5):
            deviations = tuple(
                std if sensor == j + 1 else 0.0 for j, std in enumerate(base.noise_std)
            )
            cases.append(replace(base, noise_std=deviations))
    work = training.new_solution(project, "line_following", "baseline", "public-bend")
    training.train_local(work, project)
    solution = training.load_solution(work)
    env = Environment(cases)
    traces = [[] for _ in cases]
    rotation = np.zeros(len(cases))
    last_progress, progress_time = env.progress.copy(), np.zeros(len(cases))
    longest_stall = np.zeros(len(cases))
    with torch.no_grad():
        while not env.done.all():
            q = env.state.q.numpy().copy()
            observation = env.observe()
            action = solution.act(observation, env.first)
            # The gently curving section after the first bend, on the outward leg.
            selected = (env.progress > 1.1) & (env.progress < 1.7) & (env.leg == 0)
            if selected.any():
                s = env.progress[selected]
                tangent = course.at(s, 1)
                tangent /= np.linalg.norm(tangent, axis=1)[:, None]
                normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
                offset = ((q[selected, :2] - course.at(s)) * normal).sum(axis=1)
                heading = q[selected, 5] - np.arctan2(tangent[:, 1], tangent[:, 0])
                heading = (heading + np.pi) % (2 * np.pi) - np.pi
                rate = observation["gyro"][selected, 2].numpy()
                for i, row in zip(np.flatnonzero(selected), zip(offset, heading, rate)):
                    traces[i].append(row)
            env.step(action)
            rotation += abs(env.state.q.numpy()[:, 5] - q[:, 5])
            clock = env.ticks / 1000
            longest_stall = np.maximum(longest_stall, clock - progress_time)
            advanced = env.progress - last_progress >= 0.01
            last_progress[advanced], progress_time[advanced] = (
                env.progress[advanced],
                clock[advanced],
            )
    assert (env.status == "finished").all(), env.info()
    np.testing.assert_allclose(env.info()["completion_fraction"], 1)
    # Whole outward and return manoeuvres: finishing must not hide repeated spins
    # or long stretches where visible paint no longer earns accepted progress.
    arc = np.linspace(0, course.length, 10001)
    ideal_rotation = 2 * np.trapezoid(abs(course.geometry(arc)[0]), arc)
    assert (rotation < 2.5 * ideal_rotation).all(), rotation
    assert (longest_stall < 30).all(), longest_stall
    for index, trace in enumerate(traces):
        assert len(trace) > 100
        offset, heading, rate = np.asarray(trace).T
        assert np.max(abs(offset)) < 0.006
        assert np.max(abs(heading)) < 0.10
        # Preserve the steering-rate bound on the original noisy robot family.
        # Noise ablations retain all spatial and whole-journey requirements above;
        # removing noise can change the frequency of optical quantisation chatter.
        if index < 9:
            assert rate.std() < 0.20


def test_local_evaluation_allows_online_gradient_adaptation(project):
    class Adaptive:
        def __init__(self):
            self.parameter = torch.nn.Parameter(torch.tensor(0.1))
            self.calls = 0

        def act(self, observation, reset):
            assert set(observation) == {"time", "rgb", "gyro", "accel", "encoders"}
            loss = (self.parameter - observation["gyro"][0, 0]).square()
            loss.backward()
            with torch.no_grad():
                self.parameter -= 0.01 * self.parameter.grad
                self.parameter.grad = None
            self.calls += 1
            return torch.zeros((len(reset), 2))

    solution = Adaptive()
    result = get_lab("line_following").evaluate(
        solution, project / "data/line_following", time_limit=0.001
    )
    assert (
        solution.calls
        == result["episodes"]
        == len(get_lab("line_following").benchmark(project / "data/line_following"))
    )
    assert solution.parameter.item() != pytest.approx(0.1)


@pytest.mark.parametrize("step", ["evaluate", "check"])
@pytest.mark.parametrize("fail", [False, True])
def test_robot_cli_saves_episode_tables_and_retains_incomplete_diagnostics(
    project, monkeypatch, capsys, step, fail
):
    from types import SimpleNamespace

    from coursekit import cli
    from coursekit.labs import line_following_task

    work = training.new_solution(project, "line_following", "starter", "diagnostics")
    if fail:
        (work / "solution.py").write_text(
            "import torch\nclass Solution:\n"
            " def __init__(self,*args): self.calls=0\n"
            " def act(self,observation,reset):\n"
            "  self.calls += 1\n"
            "  if self.calls == 2: raise RuntimeError('controller failed')\n"
            "  return torch.zeros((len(reset),2))\n"
        )
    training.train_local(work, project)
    path = submissions.pack(work, project / "dist") if step == "check" else work
    adapter = SimpleNamespace(
        evaluate=lambda *args, **kwargs: line_following_task.evaluate(
            *args, **kwargs, compiled=False, time_limit=0.001
        )
    )
    monkeypatch.setattr(cli, "get_lab", lambda name: adapter)
    monkeypatch.chdir(project)
    monkeypatch.setattr("sys.argv", ["coursekit", step, str(path)])
    output = project / "runs/line_following/diagnostics"
    (output / f"{step}.json").write_text('{"score": 100}')
    if fail:
        with pytest.raises(RuntimeError, match="controller failed"):
            cli.main()
        assert not (output / f"{step}.json").exists()
    else:
        cli.main()
        assert json.loads((output / f"{step}.json").read_text())["score"] == 0
    saved = json.loads((output / f"{step}.cases.json").read_text())
    assert saved["complete"] is not fail
    assert saved["error"] == (
        {"type": "RuntimeError", "message": "controller failed"} if fail else None
    )
    cases = saved["cases"]
    assert len(cases) == (2 if fail else 15)
    if fail:
        assert cases[-1]["status"] == "interrupted" and cases[-1]["score"] is None
    rows = list(csv.DictReader((output / f"{step}.cases.csv").open()))
    assert len(rows) == len(cases)
    for row, case in zip(rows, cases):
        assert row["status"] == case["status"]
        for field in ("progress", "travelled_distance", "completion_fraction"):
            assert float(row[field]) == case[field]
    text = capsys.readouterr().out
    assert "Accepted (m)" in text and "Travelled (m)" in text and "Time limit" in text
    if fail:
        assert "Interrupted; no result" in text and "no official aggregate score" in text
