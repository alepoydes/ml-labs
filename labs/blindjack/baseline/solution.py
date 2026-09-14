from pathlib import Path

import joblib
import numpy as np
from lib.features import pixels


class Solution:
    def __init__(self, artifact_dir: Path, config: dict, seed: int) -> None:
        self.resolution, self.model = joblib.load(artifact_dir / "model.joblib")

    def select(self, images: np.ndarray) -> np.ndarray:
        values = self.model.classes_[
            self.model.predict_proba(pixels(images, self.resolution)).argmax(axis=1)
        ]
        mask, remaining = np.zeros(7, dtype=bool), 21
        for i in np.argsort(-values.astype(int), kind="stable"):
            if values[i] <= remaining:
                mask[i] = True
                remaining -= int(values[i])
        return mask
