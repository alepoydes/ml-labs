from pathlib import Path

import torch


class Solution:
    def __init__(self, artifact_dir: Path, config: dict, seed: int):
        pass

    def act(self, observation, reset):
        return torch.zeros((len(reset), 2), dtype=torch.float32, device=reset.device)
