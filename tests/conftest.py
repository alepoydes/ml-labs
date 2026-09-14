import shutil
from pathlib import Path

import numpy as np
import pytest

from coursekit.common import write_json
from coursekit.images import write_images
from coursekit.labs.blindjack import make_hands, write_labels
from coursekit.training import train


@pytest.fixture
def solution(tmp_path):
    root = Path(__file__).resolve().parents[1]
    solution = tmp_path / "solution"
    shutil.copytree(root / "labs/blindjack/starter", solution)
    train(solution, tmp_path, "training-resource-fingerprint", 0)
    return solution


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    labels = np.arange(11, dtype=np.uint8)
    images = np.full((11, 128, 128, 3), 255, dtype=np.uint8)
    images[:, 30:90, 30:90, 0] = labels[:, None, None]
    for split in ("train", "validation", "hidden"):
        write_images(root / split, images)
        write_labels(root / split, labels)
        np.savez_compressed(
            root / split / "hands.npz",
            hands=make_hands(labels, 1, np.random.default_rng(7)),
        )
    write_json(
        root / "dataset.json",
        {
            "tile_size": 128,
            "columns": 32,
            "counts": {"train": 11, "validation": 11, "hidden": 11},
        },
    )
    return root
