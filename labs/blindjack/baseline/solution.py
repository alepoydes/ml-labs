from pathlib import Path

import numpy as np
from lib.model import load_model, predict_classes


class Solution:
    def __init__(self, artifact_dir: Path, config: dict, seed: int) -> None:
        self.resolution, self.model = load_model(artifact_dir)

    def select(self, images: np.ndarray) -> np.ndarray:
        values = predict_classes(self.model, images, self.resolution)
        mask, remaining = np.zeros(7, dtype=bool), 21
        for i in np.argsort(-values.astype(int), kind="stable"):
            if values[i] <= remaining:
                mask[i] = True
                remaining -= int(values[i])
        return mask
