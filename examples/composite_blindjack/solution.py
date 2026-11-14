import numpy as np
import torch
from lib.model import encoder, features, head, load


class Solution:
    def __init__(self, artifact_dir, config, seed, *, device):
        self.config, self.device = config, device
        self.encoder = load(encoder(config), artifact_dir / "encoder.pt", device).eval()
        self.members = [
            load(head(config), artifact_dir / f"member_{i}.pt", device).eval()
            for i in (1, 2)
        ]

    def select(self, images):
        with torch.no_grad():
            x = self.encoder(features(images, self.config["resolution"], self.device))
            logits = torch.stack([model(x) for model in self.members]).mean(0)
            values = logits.argmax(1).cpu().numpy()
        # A deliberately simple policy; the example teaches component lifecycle.
        selected = np.zeros(7, dtype=bool)
        total = 0
        for index in np.argsort(values):
            if total + values[index] <= 21:
                selected[index] = True
                total += values[index]
        return selected
