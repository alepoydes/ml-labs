import torch
from lib.model import load, tensors


class Solution:
    def __init__(self, artifact_dir, config, seed, *, device):
        self.device = device
        self.model, self.episode_id = load(artifact_dir, device)
        self.model.eval()

    def predict(self, images, question_tokens, question_mask):
        with torch.inference_mode():
            logits = self.model(
                *tensors(images, question_tokens, question_mask, self.device)
            )
            # Retain the exact batched result; debug never repeats inference.
            self.last_logits = logits.detach()
            return logits.argmax(dim=1).cpu().numpy()

    def debug(self, index=0):
        if not hasattr(self, "last_logits"):
            return {}
        return {"answer_probabilities": self.last_logits[index].float().softmax(-1)}
