import itertools
import json
from pathlib import Path

import numpy as np

from coursekit.common import CourseError, file_hashes, fingerprint
from coursekit.images import ImageSet

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


def load_split(data: Path, split: str):
    meta = json.loads((data / "dataset.json").read_text())
    if meta["tile_size"] != 128 or meta["columns"] != 32:
        raise CourseError("Unsupported image sheet format.")
    with np.load(data / split / "labels.npz", allow_pickle=False) as arrays:
        labels = arrays["labels"]
    if (
        labels.shape != (meta["counts"][split],)
        or labels.dtype.kind not in "iu"
        or np.any(labels > 10)
        or np.any(labels < 0)
    ):
        raise CourseError("Invalid dataset labels or count.")
    return ImageSet(data / split, len(labels)), labels


def benchmark(data: Path, split: str = "validation"):
    images, labels = load_split(data, split)
    with np.load(data / split / "hands.npz", allow_pickle=False) as arrays:
        hands = arrays["hands"]
    if (
        hands.ndim != 2
        or hands.shape[1] != 7
        or hands.dtype.kind not in "iu"
        or not len(hands)
    ):
        raise CourseError("Invalid benchmark hand array.")
    if np.any(hands < 0) or np.any(hands >= len(labels)):
        raise CourseError("Benchmark image index out of range.")
    if np.any(np.diff(np.sort(labels[hands], axis=1), axis=1) == 0):
        raise CourseError("Hand values must be distinct.")
    return images, labels, hands


def data_fingerprint(data: Path) -> str:
    return fingerprint(
        {k: v for k, v in file_hashes(data).items() if not k.startswith(".")}
    )


def evaluate(worker, data: Path, split: str = "validation") -> dict:
    images, labels, hands = benchmark(data, split)
    rewards, busts = [], 0
    for hand in hands:
        mask = worker.select(images[hand])
        values = labels[hand]
        busts += int(values[mask].sum() > 21)
        rewards.append(reward(values, mask))
    return {
        "hands": len(hands),
        "mean_reward": float(np.mean(rewards)),
        "score": float(np.mean(rewards) * 100 / 21),
        "bust_rate": busts / len(hands),
        "perfect_rate": rewards.count(21) / len(hands),
    }
