import json

import numpy as np
import pytest
import torch

from coursekit.common import CourseError
from coursekit.inspection import BlindjackGame, debug_snapshot, inspect_hand
from coursekit.labs.blindjack import reward


def test_controller_sensor_snapshots_transport_and_online_autograd():
    from coursekit.controller import Controller
    from coursekit.inspection import decode_debug, encode_debug

    controller = Controller()
    assert controller.debug() == {}
    rgb = torch.arange(96, dtype=torch.float32).reshape(2, 16, 3).requires_grad_()
    observation = {"rgb": rgb, "gyro": torch.ones(2, 3), "time": torch.tensor([0.0, 0.02])}
    controller.act(observation, torch.tensor([True, False]))
    observation["gyro"][0] = 7
    values = debug_snapshot(controller, 1)
    np.testing.assert_array_equal(values["sensor.red"], rgb[1, :, 0].detach())
    np.testing.assert_array_equal(debug_snapshot(controller)["sensor.gyro"], [1, 1, 1])
    (rgb**2).sum().backward()
    torch.testing.assert_close(rgb.grad, 2 * rgb)
    values["empty"] = np.empty((0, 7))
    restored = decode_debug(json.loads(json.dumps(encode_debug(values))))
    for key in values:
        np.testing.assert_array_equal(restored[key], values[key])
    for invalid in (
        {"x": {"array": [1], "shape": [1000000]}},
        {"x": float("nan")},
        {"x": {"array": [1, 2], "shape": [1, 2]}},
    ):
        with pytest.raises((CourseError, ValueError)):
            decode_debug(invalid)


def test_debug_history_retains_gaps_shapes_rows_and_seek_without_recopying():
    from coursekit.inspection import DebugHistory

    controls = []
    for i, value in enumerate(
        (np.array([1.0, 2.0]), None, np.array([3.0, 4.0, 5.0]), np.array([6.0, 7.0]))
    ):
        controls.append(
            {
                "observation": {"time": np.array([i * 0.02, i * 0.02])},
                "action": np.zeros((2, 2)),
                "debug": [
                    {"v": value} if value is not None else {},
                    {"v": np.array([-1.0, -2.0])},
                ],
            }
        )
    history = DebugHistory()
    history.bind(controls, 0)
    times, values = history.series("v", (2,), 3)
    np.testing.assert_allclose(times, [0, 0.02, 0.04, 0.06])
    np.testing.assert_allclose(values, [[1, 2], [np.nan, np.nan], [np.nan, np.nan], [6, 7]])
    assert np.shares_memory(values, history.series("v", (2,), 1)[1])
    controls.append(
        {
            "observation": {"time": np.array([0.08, 0.08])},
            "action": np.zeros((2, 2)),
            "debug": [{"v": np.array([8.0, 9.0])}, {}],
        }
    )
    np.testing.assert_array_equal(history.series("v", (2,), 4)[1][-1], [8, 9])
    history.bind(controls, 1)
    np.testing.assert_array_equal(history.series("v", (2,), 1)[1], [[-1, -2], [-1, -2]])
    history.bind(controls[:1], 0)
    np.testing.assert_array_equal(history.series("v", (2,), 0)[1], [[1, 2]])


def test_debug_snapshot_copies_arrays_detaches_graph_and_preserves_policy():
    class Policy:
        def __init__(self):
            self.value = torch.tensor([1.0, 2.0], requires_grad=True)
            self.array = np.array([[3.0, 4.0]])

        def debug(self, index=0):
            return {
                "activation": self.value * 2,
                "array": self.array,
                "mode": "following",
                "ready": True,
            }

    policy = Policy()
    snap = debug_snapshot(policy)
    snap["array"][:] = 0
    snap["activation"][:] = 0
    np.testing.assert_array_equal(policy.array, [[3.0, 4.0]])
    torch.testing.assert_close(policy.value, torch.tensor([1.0, 2.0]))
    assert policy.value.grad is None and snap["mode"] == "following"
    assert debug_snapshot(object()) == {}


def test_human_model_comparison_uses_same_hands_and_existing_reward(data):
    game = BlindjackGame(data)

    class Policy:
        def select(self, images):
            assert images.shape == (7, 128, 128, 3)
            images[:] = 0  # Inspection must not change the canonical images.
            return np.array([True, False, True, False, False, True, False])

    original = game.hand(0)
    decision = inspect_hand(Policy(), original)
    np.testing.assert_array_equal(original, game.hand(0))
    with pytest.raises(CourseError, match="Commit"):
        game.reveal(0)
    human = np.array([True, True, False, False, False, False, False])
    record = game.commit(0, human, decision, seconds=2.0)
    values = game.reveal(0)
    assert record["human_reward"] == reward(values, human)
    assert record["model_reward"] == reward(values, decision["mask"])
    human[:] = False
    assert game.records[0]["human"][:2] == [True, True]
    with pytest.raises(CourseError, match="already"):
        game.commit(0, human, decision)
    summary = game.summary()
    assert summary["hands"] == summary["paired_hands"] == 1
    assert (
        summary["human_score"]
        == summary["paired_human_score"]
        == record["human_reward"] * 100 / 21
    )
    json.dumps(record, allow_nan=False)


def test_comparison_means_use_only_paired_hands(data):
    game = BlindjackGame(data)
    mask = np.zeros(7, dtype=bool)
    game.commit(0, mask)
    game.commit(1, mask, {"mask": mask, "seconds": 0.0})
    assert game.summary() == {
        "hands": 2,
        "human_score": 0.0,
        "paired_hands": 1,
        "paired_human_score": 0.0,
        "model_score": 0.0,
    }
