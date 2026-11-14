import json

import numpy as np


class Solution:
    def __init__(self, artifact_dir, config, seed, *, device):
        self.state = json.loads((artifact_dir / "state.json").read_text())

    def predict(self, images, question_tokens, question_mask):
        # Replace this with your model; rows are independent questions.
        return np.full(len(images), self.state["answer"], dtype=np.int64)
