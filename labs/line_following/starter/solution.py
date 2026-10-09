from pathlib import Path

import torch

from coursekit.controller import Controller


class Solution(Controller):
    def __init__(self, artifact_dir: Path, config: dict, seed: int):
        super().__init__()

    def act(self, observation, reset):
        super().act(observation, reset)
        return torch.zeros((len(reset), 2), dtype=torch.float32, device=reset.device)
