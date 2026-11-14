"""A spatial CNN and an ordered, fixed-length question MLP."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, max_tokens):
        super().__init__()
        self.vocab_size, self.max_tokens = vocab_size, max_tokens
        layers, channels = [], 3
        for width in (16, 32, 64, 64):
            layers.extend(
                [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            )
            channels = width
        self.image = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        # Flattening retains word order; averaging embeddings would discard it.
        self.question = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_tokens * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Linear(256, vocab_size)

    def forward(self, images, tokens, mask):
        # White background becomes zero; RGB ink and its absolute position remain.
        images = images.permute(0, 3, 1, 2).float().div(255).sub(1)
        visual = self.image(images)
        tokens = F.pad(tokens.masked_fill(~mask, 0), (0, self.max_tokens - tokens.shape[1]))
        question = self.question(self.embedding(tokens))
        logits = self.head(self.fusion(torch.cat((visual, question), dim=1)))
        logits[:, 0] = torch.finfo(logits.dtype).min  # Padding is never an answer.
        return logits


def tensors(images, tokens, mask, device):
    return tuple(
        torch.as_tensor(np.ascontiguousarray(x), device=device)
        for x in (images, tokens, mask)
    )


def expand(model, vocab_size):
    """Keep every old weight exactly, initializing only appended vocabulary rows."""
    if vocab_size == model.vocab_size:
        return model
    if vocab_size < model.vocab_size:
        raise ValueError("Adaptation cannot shrink the saved vocabulary.")
    expanded = Model(vocab_size, model.max_tokens).to(next(model.parameters()).device)
    state = expanded.state_dict()
    for name, value in model.state_dict().items():
        if name in {"embedding.weight", "head.weight", "head.bias"}:
            state[name][: model.vocab_size].copy_(value)
        else:
            state[name].copy_(value)
    expanded.load_state_dict(state)
    return expanded


def save(model, folder, episode_id):
    torch.save(
        dict(
            vocab_size=model.vocab_size,
            max_tokens=model.max_tokens,
            episode_id=episode_id,
            weights={k: v.detach().cpu() for k, v in model.state_dict().items()},
        ),
        folder / "model.pt",
    )


def load(folder, device):
    state = torch.load(folder / "model.pt", map_location=device, weights_only=True)
    model = Model(state["vocab_size"], state["max_tokens"]).to(device)
    model.load_state_dict(state["weights"])
    return model, state["episode_id"]
