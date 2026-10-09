from dataclasses import replace

import numpy as np
import pytest
import torch

from coursekit.common import CourseError
from coursekit.labs.line_following_cases import (
    case_identity,
    load_cases,
    sample_case,
    save_cases,
    save_split,
)
from coursekit.labs.line_following_curve import Course, save_courses
from coursekit.labs.line_following_env import Environment


def test_fixed_cases_round_trip_replay_and_batch_order(tmp_path):
    course = Course.polyline("line", [[0, 0], [1, 0]])
    cases = [sample_case(course, i, name=f"example-{i}") for i in range(2)]
    save_courses(tmp_path / "courses.npz", [course])
    save_cases(tmp_path / "cases.npz", cases)
    save_split(tmp_path / "train.csv", cases[::-1])
    loaded = load_cases(tmp_path, "train")
    assert [case_identity(c) for c in loaded] == [case_identity(c) for c in cases[::-1]]
    env = Environment(loaded)
    singleton = Environment([cases[1]])
    initial = env.observe()
    action = torch.full((2, 2), 0.12)
    for _ in range(3):
        env.step(action)
        singleton.step(action[:1])
        for name, value in env.observe().items():
            torch.testing.assert_close(
                value[0], singleton.observe()[name][0], rtol=0, atol=0
            )
    state = tuple(v.clone() for v in env.state)
    noise = env.observe()
    env.reset([True, False])
    for name, value in env.observe().items():
        torch.testing.assert_close(value[0], initial[name][0], rtol=0, atol=0)
        torch.testing.assert_close(value[1], noise[name][1], rtol=0, atol=0)
    for _ in range(3):
        env.step(action)
    for actual, expected in zip(env.state, state):
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for name, value in env.observe().items():
        torch.testing.assert_close(value[0], noise[name][0], rtol=0, atol=0)
    assert all(
        torch.equal(a, b[0])
        for a, b in zip(cases[1].initial, Environment([cases[1]]).state)
    )


def test_sensor_noise_is_separate_from_robot_and_independent_channels():
    course = Course.polyline("line", [[0, 0], [1, 0]])
    case = sample_case(course, 8)
    quiet = replace(case, noise_std=(0, 0, 0, 0))
    # Turning off RGB leaves the random sequence of every other sensor intact.
    no_rgb = replace(case, noise_std=(0, *case.noise_std[1:]))
    env = Environment([case, quiet, no_rgb])
    for _ in range(3):
        env.step(torch.full((3, 2), 0.08))
        for value in env.state:
            torch.testing.assert_close(value[0], value[1], atol=0, rtol=0)
        obs = env.observe()
        torch.testing.assert_close(obs["rgb"][1], obs["rgb"][2], atol=0, rtol=0)
        for name in ("gyro", "accel", "encoders"):
            torch.testing.assert_close(obs[name][0], obs[name][2], atol=0, rtol=0)
        assert not torch.equal(obs["rgb"][0], obs["rgb"][1])
    assert case_identity(case) != case_identity(quiet)


def test_case_loading_rejects_unknown_ids_invalid_models_and_rng(tmp_path):
    course = Course.polyline("line", [[0, 0], [1, 0]])
    case = sample_case(course, 1)
    save_courses(tmp_path / "courses.npz", [course])
    save_cases(tmp_path / "cases.npz", [case])
    (tmp_path / "train.csv").write_text("case\nmissing\n")
    with pytest.raises(CourseError, match="identifiers"):
        load_cases(tmp_path, "train")
    with pytest.raises(CourseError, match="noise"):
        replace(case, noise_state='{"bit_generator": "unknown"}')
    with np.load(tmp_path / "cases.npz") as f:
        arrays = {k: f[k] for k in f.files}
    arrays["robot_mass"][0, 0] = -1
    np.savez_compressed(tmp_path / "cases.npz", **arrays)
    with pytest.raises(CourseError, match="mass"):
        load_cases(tmp_path)
    arrays["robot_mass"][0] = case.robot.mass.numpy()
    arrays["version"] = np.array(True)
    np.savez_compressed(tmp_path / "cases.npz", **arrays)
    with pytest.raises(CourseError, match="format"):
        load_cases(tmp_path)
    with pytest.raises(CourseError, match="Case"):
        Environment([course])
