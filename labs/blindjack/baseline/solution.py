from pathlib import Path

import numpy as np
from lib.model import load_model, predict_classes


class Solution:
    def __init__(self, artifact_dir: Path, config: dict, seed: int) -> None:
        self.resolution, self.model = load_model(artifact_dir)
        self.last = None

    def select(self, images: np.ndarray) -> np.ndarray:
        values = predict_classes(self.model, images, self.resolution)
        mask, remaining = np.zeros(7, dtype=bool), 21
        for i in np.argsort(-values.astype(int), kind="stable"):
            if values[i] <= remaining:
                mask[i] = True
                remaining -= int(values[i])
        self.last = (values.copy(), mask.copy())
        return mask

    def debug(self, index=0):
        if self.last is None:
            return {}
        values, mask = self.last
        return {
            "predicted_values": values.copy(),
            "selected": mask.copy(),
            "predicted_total": int(values[mask].sum()),
        }
