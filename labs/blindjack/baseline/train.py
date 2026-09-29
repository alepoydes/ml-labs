from pathlib import Path

import numpy as np
from lib.features import pixels
from lib.model import create_model, save_model

from coursekit.labs.blindjack import load_split


def train(data_dir: Path, output_dir: Path, config: dict, seed: int) -> dict:
    images, labels = load_split(data_dir, "train")
    features = pixels(images[np.arange(len(images))], config["resolution"])
    model = create_model(config, seed)
    model.fit(features, labels)
    save_model(output_dir, config["resolution"], model)
    return {
        "epoch": np.arange(1, model.n_iter_ + 1),
        "loss": np.asarray(model.loss_curve_),
        # This is the internal training holdout, not the public benchmark.
        "validation_accuracy": np.asarray(model.validation_scores_),
    }
