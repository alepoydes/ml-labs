import shutil
from pathlib import Path

import numpy as np
import pytest

from coursekit.common import digest, write_json
from coursekit.images import write_images
from coursekit.labs.blindjack import make_hands, write_hands, write_labels
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
        write_hands(root / split, make_hands(labels, 1, np.random.default_rng(7)))
    write_json(
        root / "dataset.json",
        {
            "tile_size": 128,
            "columns": 32,
            "counts": {"train": 11, "validation": 11, "hidden": 11},
        },
    )
    return root


@pytest.fixture
def student_project(tmp_path, data):
    """A standalone public project with local, verified development resources."""
    import json

    from coursekit.resources import sync

    root = tmp_path / "student"
    lab = Path(__file__).resolve().parents[1] / "labs/blindjack"
    for template in ("starter", "baseline"):
        shutil.copytree(
            lab / template,
            root / "labs/blindjack" / template,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    # Enough examples for the baseline's stratified internal training holdout.
    labels = np.tile(np.arange(11, dtype=np.uint8), 20)
    images = np.full((len(labels), 128, 128, 3), 255, dtype=np.uint8)
    images[:, 30:90, 30:90, 0] = labels[:, None, None]
    write_images(data / "train", images)
    write_labels(data / "train", labels)
    (data / "train/hands.csv").unlink()
    write_json(
        data / "dataset.json",
        {
            "tile_size": 128,
            "columns": 32,
            "counts": {"train": len(labels), "validation": 11},
        },
    )
    entries = [
        {
            "path": path.relative_to(data).as_posix(),
            "url": path.as_uri(),
            "sha256": digest(path),
            "size": path.stat().st_size,
        }
        for path in sorted(data.rglob("*"))
        if path.is_file() and "hidden" not in path.parts
    ]
    spec = {"schema_version": 1, "files": entries}
    manifest = "schema_version = 1\n" + "".join(
        "\n[[files]]\n"
        + "".join(f"{key} = {json.dumps(value)}\n" for key, value in entry.items())
        for entry in entries
    )
    (root / "labs/blindjack/resources.toml").write_text(manifest)
    sync(root / "data/blindjack", spec)
    return root
