import json

import numpy as np
import pytest
import torch

from coursekit.common import CourseError
from coursekit.inspection import BlindjackGame, debug_snapshot, inspect_hand
from coursekit.labs.blindjack import reward


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
