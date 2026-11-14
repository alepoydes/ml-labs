import torch
from torch import nn


def features(images, resolution, device):
    x = torch.as_tensor(images.copy(), device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2) / 255
    return torch.nn.functional.adaptive_avg_pool2d(x, resolution).flatten(1)


def encoder(config):
    return nn.Sequential(
        nn.Linear(3 * config["resolution"] ** 2, config["width"]), nn.Tanh()
    )


def head(config):
    return nn.Linear(config["width"], 11)


def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device)
