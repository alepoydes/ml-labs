import itertools

import numpy as np
import pytest

from coursekit.common import CourseError
from coursekit.images import ImageSet, write_images
from coursekit.labs.blindjack import (
    COMPOSITIONS,
    benchmark,
    reward,
    validate_images,
    validate_mask,
)


def test_all_compositions_and_blind_expectation():
    masks = np.array(list(itertools.product((False, True), repeat=7)))
    totals = COMPOSITIONS @ masks.T
    rewards = np.where(totals <= 21, totals, 0)
    assert len(COMPOSITIONS) == 330
    assert np.all(rewards.max(axis=1) == 21)
    by_count = [rewards[:, masks.sum(axis=1) == n].mean() for n in range(8)]
    assert np.argmax(by_count) == 3
    assert by_count[3] == pytest.approx(12.7151515151515)


def test_rewards_and_contracts():
    values = np.array([0, 1, 2, 3, 4, 5, 6])
    assert reward(values, np.zeros(7, bool)) == 0
    assert reward(values, np.ones(7, bool)) == 21
    mask = np.ones(7, bool)
    mask[0] = False
    assert reward(values, mask) == 21
    assert reward(np.arange(4, 11), np.ones(7, bool)) == 0
    for bad in ([True] * 7, np.ones(7, int), np.ones((7, 1), bool)):
        with pytest.raises(CourseError):
            validate_mask(bad)
    with pytest.raises(CourseError):
        validate_images(np.zeros((7, 128, 128, 3)))


def test_sheet_boundary_roundtrip(tmp_path):
    rng = np.random.default_rng(14)
    # Identifiable pixels at the start, row boundary, sheet boundary, and padding.
    images = np.full((1025, 128, 128, 3), 255, np.uint8)
    images[:, 0, 0] = rng.integers(0, 256, (1025, 3), dtype=np.uint8)
    write_images(tmp_path, images)
    stored = ImageSet(tmp_path, len(images))
    indices = np.array([0, 31, 32, 1023, 1024, 0])
    np.testing.assert_array_equal(stored[indices], images[indices])
    with pytest.raises(CourseError):
        stored[np.array([1025])]


def test_benchmark_has_distinct_values(data):
    images, labels, hands = benchmark(data)
    assert hands.shape == (330, 7)
    assert np.all(np.diff(np.sort(labels[hands], axis=1), axis=1) > 0)
    assert images[hands[0]].shape == (7, 128, 128, 3)
