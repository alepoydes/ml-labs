from pathlib import Path

import joblib
import numpy as np
from lib.features import pixels
from sklearn.neural_network import MLPClassifier

from coursekit.labs.blindjack import load_split


def train(data_dir: Path, output_dir: Path, config: dict, seed: int) -> None:
    images, labels = load_split(data_dir, "train")
    features = pixels(images[np.arange(len(images))], config["resolution"])
    model = MLPClassifier(
        hidden_layer_sizes=(config["hidden_units"],),
        early_stopping=True,
        random_state=seed,
        max_iter=config["max_iter"],
        learning_rate_init=config["learning_rate"],
        n_iter_no_change=config.get("patience", 10),
        activation=config.get("activation", "relu"),
    )
    model.fit(features, labels)
    joblib.dump((config["resolution"], model), output_dir / "model.joblib")
