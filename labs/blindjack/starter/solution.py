from pathlib import Path

import numpy as np


class Solution:
    def __init__(self, artifact_dir: Path, config: dict, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.count = config["selected_cards"]

    def select(self, images: np.ndarray) -> np.ndarray:
        mask = np.zeros(7, dtype=bool)
        mask[self.rng.choice(7, self.count, replace=False)] = True
        return mask
