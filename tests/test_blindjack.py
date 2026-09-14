import csv
import itertools

import numpy as np
import pytest

from coursekit.common import CourseError, write_json
from coursekit.images import ImageSet, write_images
from coursekit.labs.blindjack import (
    COMPOSITIONS,
    benchmark,
    load_split,
    reward,
    validate_images,
    validate_mask,
    write_labels,
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
    labels = np.arange(len(images)) % 11
    write_labels(tmp_path, labels)
    write_json(
        tmp_path / "dataset.json",
        {"tile_size": 128, "columns": 32, "counts": {".": len(images)}},
    )
    with (tmp_path / "labels.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(rows[i]["sheet"], rows[i]["row"], rows[i]["column"]) for i in indices] == [
        ("images-0000.png", "1", "1"),
        ("images-0000.png", "1", "32"),
        ("images-0000.png", "2", "1"),
        ("images-0000.png", "32", "32"),
        ("images-0001.png", "1", "1"),
        ("images-0000.png", "1", "1"),
    ]
    loaded, loaded_labels = load_split(tmp_path, ".")
    np.testing.assert_array_equal(loaded_labels, labels)
    np.testing.assert_array_equal(loaded[indices], images[indices])


@pytest.mark.parametrize(
    "damage", ["header", "location", "label", "count", "order", "duplicate"]
)
def test_csv_rejects_misaligned_or_invalid_labels(data, damage):
    path = data / "train/labels.csv"
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if damage == "header":
        rows[0][0] = "image"
    elif damage == "location":
        rows[1][1] = "0"
    elif damage == "label":
        rows[1][3] = "256"
    elif damage == "count":
        rows.pop()
    elif damage == "order":
        rows[1], rows[2] = rows[2], rows[1]
    else:
        rows[2] = rows[1]
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(CourseError):
        load_split(data, "train")


def test_benchmark_has_distinct_values(data):
    images, labels, hands = benchmark(data)
    assert hands.shape == (330, 7)
    assert np.all(np.diff(np.sort(labels[hands], axis=1), axis=1) > 0)
    assert images[hands[0]].shape == (7, 128, 128, 3)
