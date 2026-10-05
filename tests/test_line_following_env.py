import numpy as np
import pytest
import torch

from coursekit.common import CourseError
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import Course
from coursekit.labs.line_following_env import (
    Environment,
    validate_action,
    validate_observation,
)


def straight(length=1.0, name="straight"):
    return Course(name, np.array([[0.0, 0.0], [length, 0.0]]))


def test_batching_and_unrelated_resets_preserve_episode_conditions():
    single = Environment([straight()], [19], compiled=False, time_limit=0.003)
    batch = Environment(
        [straight(), straight(2.0)], [19, 71], compiled=False, time_limit=0.003
    )
    for name in single.observe():
        torch.testing.assert_close(single.observe()[name][0], batch.observe()[name][0])
    assert batch.reference[0] == single.reference[0]
    untouched = {name: value[0].clone() for name, value in batch.observe().items()}
    batch.reset([False, True], courses=[straight(3.0)], seeds=[101])
    assert batch.first.tolist() == [True, True]
    for name in untouched:
        torch.testing.assert_close(
            batch.observe()[name][0], untouched[name], rtol=0, atol=0
        )
    one = single.step(torch.zeros((1, 2), dtype=torch.float32))
    many = batch.step(torch.zeros((2, 2), dtype=torch.float32))
    for name in one[0]:
        torch.testing.assert_close(one[0][name][0], many[0][name][0], atol=2e-6, rtol=1e-6)
    for a, b in zip(single.state, batch.state):
        torch.testing.assert_close(a[0], b[0], atol=1e-12, rtol=1e-10)
    assert many[3].tolist() == [True, True]
    assert many[4]["status"].tolist() == ["timeout", "timeout"]
    assert many[4]["score"].tolist() == [0.0, 0.0]
    old = batch.observe()
    batch.reset([False, True])
    assert batch.first.tolist() == [False, True]
    for name in old:
        torch.testing.assert_close(batch.observe()[name][0], old[name][0], rtol=0, atol=0)


def test_observations_have_no_alias_to_environment_and_terminal_rows_freeze():
    env = Environment([straight()], [0], compiled=False, time_limit=0.002)
    obs = env.observe()
    validate_observation(obs, env.first)
    obs["encoders"].fill_(12345)
    assert (env.observe()["encoders"].abs() < 0.01).all()
    env.step(torch.zeros((1, 2)))
    state = tuple(v.clone() for v in env.state)
    obs = env.observe()
    _, reward, terminated, truncated, info = env.step(torch.ones((1, 2)))
    assert reward.tolist() == [0.0] and truncated.item() and not terminated.item()
    assert info["time"].tolist() == [0.002]
    for name in obs:
        torch.testing.assert_close(obs[name], env.observe()[name], rtol=0, atol=0)
    for before, after in zip(state, env.state):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_return_leg_allows_reverse_driving_without_rotating_body():
    env = Environment([straight(0.4)], [3], compiled=False, time_limit=1.0, noise=False)
    positions = [0.12, 0.24, 0.36, 0.24, 0.12, 0.0]

    def transport(state, action, model):
        # Prescribed poses exercise the judge/lifecycle, not motor feasibility.
        q = state.q.clone()
        q[0, 0] = positions[min(env.ticks[0], len(positions) - 1)]
        return state._replace(q=q), physics.Diagnostics(
            torch.zeros(1), torch.zeros((1, 3, 3))
        )

    env._advance = transport
    frames = []
    obs, reward, terminated, truncated, info = env.step(
        torch.zeros((1, 2)), on_step=lambda e: frames.append(e.info())
    )
    assert terminated.item() and not truncated.item()
    assert info["status"].tolist() == ["finished"]
    assert info["time"].tolist() == [0.006]
    assert info["outward_time"].tolist() == [0.003]
    assert info["score"].tolist() == [100.0]
    assert reward.item() > 0.8
    assert frames[2]["leg"].tolist() == [1]
    assert all(frame["q"][0, 5] == 0 for frame in frames)
    assert not env.first.item()
    assert obs["time"].item() == pytest.approx(0.006)


def test_fall_and_numerical_failure_remain_distinct():
    env = Environment([straight()], [0], compiled=False, time_limit=0.003)

    def fall(state, action, model):
        return state._replace(fallen=torch.ones(1, dtype=torch.bool)), physics.Diagnostics(
            torch.zeros(1), torch.zeros((1, 3, 3))
        )

    env._advance = fall
    _, _, terminated, truncated, info = env.step(torch.zeros((1, 2)))
    assert terminated.item() and not truncated.item()
    assert info["status"].tolist() == ["fallen"] and info["score"].tolist() == [0.0]
    env.reset()

    def invalid(state, action, model):
        return state, physics.Diagnostics(torch.ones(1), torch.zeros((1, 3, 3)))

    env._advance = invalid
    with pytest.raises(FloatingPointError, match="converge"):
        env.step(torch.zeros((1, 2)))
    assert env.status.tolist() == ["running"]


@pytest.mark.parametrize(
    "action",
    [
        torch.zeros(2),
        torch.zeros((1, 2), dtype=torch.float64),
        torch.full((1, 2), float("nan")),
        torch.full((1, 2), 1.01),
    ],
)
def test_invalid_motor_commands_are_rejected(action):
    with pytest.raises(CourseError, match="act"):
        validate_action(action, 1)


@pytest.mark.parametrize("seeds", [[1.5], [True], [-1], [2**64]])
def test_episode_seeds_cannot_silently_change(seeds):
    with pytest.raises(CourseError, match="seed"):
        Environment([straight()], seeds, compiled=False)


def test_reset_does_not_convert_numeric_values_to_boolean_masks():
    env = Environment([straight()], [0], compiled=False)
    with pytest.raises(CourseError, match="Boolean"):
        env.reset([1])
