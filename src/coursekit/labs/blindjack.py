import csv
import itertools
import json
from pathlib import Path

import numpy as np

from coursekit.common import CourseError, file_hashes, fingerprint
from coursekit.images import ImageSet, image_location

COMPOSITIONS = np.array(list(itertools.combinations(range(11), 7)), dtype=np.int64)


def validate_images(images: np.ndarray) -> None:
    if (
        not isinstance(images, np.ndarray)
        or images.shape != (7, 128, 128, 3)
        or images.dtype != np.uint8
    ):
        raise CourseError("Expected uint8 RGB images with shape (7,128,128,3).")


def validate_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray) or mask.shape != (7,) or mask.dtype != np.bool_:
        raise CourseError("select() must return a Boolean NumPy array with shape (7,).")


def reward(values: np.ndarray, mask: np.ndarray) -> int:
    validate_mask(mask)
    total = int(np.asarray(values, dtype=np.int64)[mask].sum())
    return total if total <= 21 else 0


def make_hands(labels: np.ndarray, cycles: int, rng: np.random.Generator) -> np.ndarray:
    pools = [np.flatnonzero(labels == value) for value in range(11)]
    if any(len(pool) == 0 for pool in pools):
        raise CourseError("A benchmark requires all eleven classes.")
    compositions = np.tile(COMPOSITIONS, (cycles, 1))
    rng.shuffle(compositions)
    hands = np.array([[rng.choice(pools[v]) for v in values] for values in compositions])
    for hand in hands:
        rng.shuffle(hand)
    return hands


def write_labels(directory: Path, labels: np.ndarray) -> None:
    with (directory / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sheet", "row", "column", "label"))
        writer.writerows((*image_location(i), int(label)) for i, label in enumerate(labels))


def write_hands(directory: Path, hands: np.ndarray) -> None:
    with (directory / "hands.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("hand", "card", "sheet", "row", "column"))
        writer.writerows(
            (h + 1, c + 1, *image_location(int(index)))
            for h, hand in enumerate(hands)
            for c, index in enumerate(hand)
        )


def load_split(data: Path, split: str):
    meta = json.loads((data / "dataset.json").read_text())
    if meta["tile_size"] != 128 or meta["columns"] != 32:
        raise CourseError("Unsupported image sheet format.")
    labels = []
    values = {str(value) for value in range(11)}
    with (data / split / "labels.csv").open(newline="", encoding="utf-8") as stream:
        rows = csv.reader(stream)
        if next(rows, None) != ["sheet", "row", "column", "label"]:
            raise CourseError("Expected CSV header: sheet,row,column,label.")
        for i, row in enumerate(rows):
            if len(row) != 4 or row[:3] != list(map(str, image_location(i))):
                raise CourseError(
                    f"Invalid image location in {split}/labels.csv, line {i + 2}."
                )
            if row[3] not in values:
                raise CourseError(f"Invalid label in {split}/labels.csv, line {i + 2}.")
            labels.append(int(row[3]))
    labels = np.array(labels, dtype=np.uint8)
    if labels.shape != (meta["counts"][split],):
        raise CourseError("Invalid dataset label count.")
    return ImageSet(data / split, len(labels)), labels


def benchmark(data: Path, split: str = "validation"):
    images, labels = load_split(data, split)
    locations = {tuple(map(str, image_location(i))): i for i in range(len(labels))}
    indices = []
    with (data / split / "hands.csv").open(newline="", encoding="utf-8") as stream:
        rows = csv.reader(stream)
        if next(rows, None) != ["hand", "card", "sheet", "row", "column"]:
            raise CourseError("Expected CSV header: hand,card,sheet,row,column.")
        for i, row in enumerate(rows):
            if (
                len(row) != 5
                or row[:2] != [str(i // 7 + 1), str(i % 7 + 1)]
                or tuple(row[2:]) not in locations
            ):
                raise CourseError(f"Invalid card in {split}/hands.csv, line {i + 2}.")
            indices.append(locations[tuple(row[2:])])
    if not indices or len(indices) % 7:
        raise CourseError("Expected complete seven-card hands in hands.csv.")
    hands = np.array(indices, dtype=np.int64).reshape(-1, 7)
    if np.any(np.diff(np.sort(labels[hands], axis=1), axis=1) == 0):
        raise CourseError("Hand values must be distinct.")
    return images, labels, hands


def data_fingerprint(data: Path) -> str:
    return fingerprint(
        {k: v for k, v in file_hashes(data).items() if not k.startswith(".")}
    )


def evaluate(solution, data: Path, split: str = "validation", *, progress=None) -> dict:
    """Call select() for every hand and compute the score from validated masks."""
    images, labels, hands = benchmark(data, split)
    rewards, busts = [], 0
    for index, hand in enumerate(hands):
        mask = solution.select(images[hand])
        validate_mask(mask)
        values = labels[hand]
        busts += int(values[mask].sum() > 21)
        rewards.append(reward(values, mask))
        if progress:
            total = int(values[mask].sum())
            progress(
                {
                    "case": index + 1,
                    "cases": len(hands),
                    "metrics": {
                        "score": rewards[-1] * 100 / 21,
                        "reward": rewards[-1],
                        "total": total,
                        "selected_cards": int(mask.sum()),
                        "mask": mask.tolist(),
                        "status": "bust"
                        if total > 21
                        else "perfect"
                        if total == 21
                        else "played",
                    },
                }
            )
    return {
        "hands": len(hands),
        "mean_reward": float(np.mean(rewards)),
        "score": float(np.mean(rewards) * 100 / 21),
        "bust_rate": busts / len(hands),
        "perfect_rate": rewards.count(21) / len(hands),
    }
