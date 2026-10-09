import os

import numpy as np
import pytest
import torch

from coursekit.common import CourseError
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import Course
from coursekit.labs.line_following_env import (
    Environment,
    _inspect,
    validate_action,
    validate_observation,
)


def straight(length=1.0, name="straight"):
    return Course.polyline(name, np.array([[0.0, 0.0], [length, 0.0]]))


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


def test_accepted_distance_is_training_feedback_separate_from_observations():
    env = Environment([straight()], [13], noise=False, time_limit=0.08)
    initial = previous = env.info()["progress"]
    while not env.done.all():
        observation, reward, _, _, info = env.step(torch.full((1, 2), 0.2))
        validate_observation(observation, env.first)
        assert "progress" not in observation and "reward" not in observation
        np.testing.assert_allclose(
            reward.numpy(), (info["progress"] - previous) / info["route_length"]
        )
        assert (info["progress"] >= previous).all()
        previous = info["progress"]
    assert (previous > initial).all() and info["score"].tolist() == [0.0]
    info["progress"][:] = -1
    assert (env.info()["progress"] > initial).all()


def test_native_batches_resets_and_terminal_masks_match_eager():
    import subprocess
    import sys

    import numba

    from coursekit.labs.line_following_jit import CACHE_DIR

    original_numba_threads = numba.get_num_threads()
    for size, threads in ((1, 1), (2, 1), (8, 2), (1, 1)):
        torch.set_num_threads(threads)
        courses, seeds = [straight()] * size, list(range(size))
        native = Environment(courses, seeds)
        eager = Environment(courses, seeds, compiled=False)
        for phase in range(3):
            if phase == 1 and size > 1:
                native.status[0] = eager.status[0] = "timeout"
            if phase == 2:
                for env in (native, eager):
                    env.reset(seeds=[31 + i for i in range(size)])
            action = torch.full((size, 2), 0.1 * (phase + 1))
            a, b = native.step(action), eager.step(action)
            assert numba.get_num_threads() == original_numba_threads
            for observed, expected in zip(native.state, eager.state):
                torch.testing.assert_close(observed, expected, atol=1e-11, rtol=1e-9)
            for name in a[0]:
                torch.testing.assert_close(a[0][name], b[0][name], atol=2e-6, rtol=1e-6)
            np.testing.assert_allclose(a[4]["progress"], b[4]["progress"], atol=1e-12)
            np.testing.assert_allclose(
                a[4]["travelled_distance"], b[4]["travelled_distance"], atol=1e-12
            )
            np.testing.assert_allclose(
                a[4]["completion_fraction"], b[4]["completion_fraction"], atol=1e-12
            )
            np.testing.assert_array_equal(a[4]["status"], b[4]["status"])
        ticks = native.ticks.copy()
        distance = native.travelled_distance.copy()
        native.state.p[0, 0] = float("nan")
        with pytest.raises(FloatingPointError, match="Non-finite"):
            native.step(action)
        np.testing.assert_array_equal(native.ticks, ticks)
        np.testing.assert_array_equal(native.travelled_distance, distance)

    # A fresh interpreter must load usable artifacts without tracing or compiling.
    assert any(CACHE_DIR.rglob("*.nbc"))
    script = """
import numpy as np
import torch
from coursekit.labs import line_following_rollout as rollout
from coursekit.labs.line_following_course import Course
from coursekit.labs.line_following_env import Environment
torch.set_num_threads(1)
import numba
from llvmlite import binding as llvm
# Initialize Numba's runtime itself before forbidding compilation of engine code.
@numba.njit
def runtime_warmup(x):
    return x.copy()
runtime_warmup(np.zeros(1))
def forbidden(*args, **kwargs):
    raise AssertionError('A cached engine kernel was recompiled')
torch.compile = forbidden
llvm.ModulePassManager.run = forbidden
llvm.FunctionPassManager.run = forbidden
course = Course.polyline('cached', np.array([[0.,0.],[1.,0.]]))
warmed = None
for size, threads in ((1,1),(2,1),(8,2),(1,1)):
    torch.set_num_threads(threads)
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    env = Environment([course] * size, [51+i for i in range(size)])
    reference = Environment([course] * size, [51+i for i in range(size)], compiled=False)
    commands = torch.linspace(-.2, .3, 2 * size).reshape(2, size).T
    # Transposed, sliced and broadcast outputs are all valid controller commands.
    actions = (
        commands.contiguous(),
        commands,
        commands.repeat_interleave(2, dim=0)[::2],
        commands[:1].expand(size, -1),
    )
    for action in actions:
        env.step(action); reference.step(action)
        if warmed is None:
            warmed = tuple(rollout.block.signatures)
        assert tuple(rollout.block.signatures) == warmed, 'Controller layout required another kernel'
        for actual, expected in zip(env.state, reference.state):
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
        np.testing.assert_allclose(env.info()['travelled_distance'], reference.info()['travelled_distance'], atol=1e-12)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        # Cache configuration is fixed when the engine is imported. Share that
        # actual directory, including when the test runner supplied its own root.
        env={**os.environ, "NUMBA_CACHE_DIR": str(CACHE_DIR.parents[1])},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_return_leg_allows_reverse_driving_without_rotating_body():
    env = Environment([straight(0.4)], [3], compiled=False, time_limit=1.0, noise=False)
    positions = [0.12, 0.24, 0.36, 0.40, 0.24, 0.12, 0.0]

    def transport(state, action, model):
        # Prescribed poses exercise the judge/lifecycle, not motor feasibility.
        q = state.q.clone()
        q[0, 0] = positions[min(env.ticks[0], len(positions) - 1)]
        return _inspect(
            state._replace(q=q), physics.Diagnostics(torch.zeros(1), torch.zeros((1, 3, 3)))
        )

    env._advance = transport
    frames = []
    obs, reward, terminated, truncated, info = env.step(
        torch.zeros((1, 2)), on_step=lambda e: frames.append(e.info())
    )
    assert terminated.item() and not truncated.item()
    assert info["status"].tolist() == ["finished"]
    assert info["time"].tolist() == [0.007]
    assert info["outward_time"].tolist() == [0.004]
    assert info["score"].tolist() == [100.0]
    np.testing.assert_allclose(info["travelled_distance"], [0.8], atol=1e-12)
    np.testing.assert_allclose(info["progress"], [0.8], atol=1e-12)
    np.testing.assert_allclose(info["completion_fraction"], [1.0], atol=1e-12)
    assert reward.item() > 0.8
    # The chassis reaches the endpoint at x=.36, but the sensor has not reached it.
    assert frames[2]["leg"].tolist() == [0]
    assert frames[3]["leg"].tolist() == [1]
    assert all(frame["q"][0, 5] == 0 for frame in frames)
    assert not env.first.item()
    assert obs["time"].item() == pytest.approx(0.007)


def test_fall_and_numerical_failure_remain_distinct():
    env = Environment([straight()], [0], compiled=False, time_limit=0.003)

    def fall(state, action, model):
        return _inspect(
            state._replace(fallen=torch.ones(1, dtype=torch.bool)),
            physics.Diagnostics(torch.zeros(1), torch.zeros((1, 3, 3))),
        )

    env._advance = fall
    _, _, terminated, truncated, info = env.step(torch.zeros((1, 2)))
    assert terminated.item() and not truncated.item()
    assert info["status"].tolist() == ["fallen"] and info["score"].tolist() == [0.0]
    env.reset()

    def invalid(state, action, model):
        return _inspect(state, physics.Diagnostics(torch.ones(1), torch.zeros((1, 3, 3))))

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


def test_ground_path_counts_every_step_and_freezes_and_resets_independent_rows():
    env = Environment(
        [straight(), straight()], [7, 11], compiled=False, time_limit=0.004, noise=False
    )
    positions = np.array(
        [
            [[0.003, 0.004], [0.0, 0.006]],
            [[0.0, 0.0], [0.0, 0.006]],
            [[1.0, 1.0], [0.0, -0.002]],
            [[2.0, 2.0], [0.0, -0.002]],
        ]
    )
    calls = 0

    def move(state, action, model):
        nonlocal calls
        q = state.q.clone()
        q[:, :2] = torch.from_numpy(positions[calls])
        q[:, 2] += 0.01  # Vertical movement and wheel spin do not add ground distance.
        q[:, 5] += 0.1
        q[:, 6:] += 123
        fallen = state.fallen.clone()
        fallen[0] |= calls >= 1
        calls += 1
        return _inspect(
            state._replace(q=q, fallen=fallen),
            physics.Diagnostics(torch.zeros(2), torch.zeros((2, 3, 3))),
        )

    env._advance = move
    frames = []
    result = env.step(torch.zeros((2, 2)), on_step=lambda e: frames.append(e.info()))
    info = result[-1]
    assert info["status"].tolist() == ["fallen", "timeout"]
    assert info["score"].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(info["travelled_distance"], [0.010, 0.014], atol=1e-12)
    np.testing.assert_allclose(
        info["completion_fraction"], info["progress"] / info["route_length"]
    )
    np.testing.assert_allclose(
        [f["travelled_distance"][0] for f in frames], [0.005, 0.010, 0.010, 0.010]
    )
    env.step(torch.ones((2, 2)))
    np.testing.assert_array_equal(
        env.info()["travelled_distance"], info["travelled_distance"]
    )
    info["travelled_distance"][:] = 99  # Returned diagnostics never alias the counters.
    env.reset([True, False], seeds=[7])
    np.testing.assert_allclose(env.info()["travelled_distance"], [0.0, 0.014], atol=1e-12)
    env.reset([False, True])
    np.testing.assert_array_equal(env.info()["travelled_distance"], [0.0, 0.0])
