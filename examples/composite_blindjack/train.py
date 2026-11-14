"""A small API example: one encoder, two independently fitted heads, mean logits."""

import shutil

import numpy as np
import torch
from lib.model import encoder, features, head, load
from torch import nn

from coursekit.labs.blindjack import load_split


def fit(model, x, target, loss_fn, steps, rate, name, report):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if report:
            report(
                {
                    "stage": name,
                    "step": step + 1,
                    "total": steps,
                    "metrics": {"loss": losses[-1]},
                }
            )
    return {"step": np.arange(1, steps + 1), "loss": np.array(losses)}


def train(
    data_dir, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
):
    inputs = inputs or {}
    if stage == "all" and inputs:
        if set(inputs) != {"encoder", "member_1", "member_2"}:
            raise ValueError("Assembly requires encoder, member_1 and member_2 snapshots")
        # A member carries its encoder, so incompatible feature spaces cannot be mixed.
        reference = (inputs["encoder"] / "encoder.pt").read_bytes()
        for member in ("member_1", "member_2"):
            if (inputs[member] / "encoder.pt").read_bytes() != reference:
                raise ValueError("All members must use the same encoder")
            shutil.copy2(inputs[member] / "head.pt", output_dir / f"{member}.pt")
        shutil.copy2(inputs["encoder"] / "encoder.pt", output_dir / "encoder.pt")
        return None
    if stage not in {"all", "encoder", "member"}:
        raise ValueError(f"Unknown stage: {stage}")
    if (stage == "member" and set(inputs) != {"encoder"}) or (stage != "member" and inputs):
        raise ValueError("Only member training accepts an encoder input")

    torch.manual_seed(seed)
    images, labels = load_split(data_dir, "train")
    x = features(images[np.arange(len(images))], config["resolution"], device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    representation = encoder(config).to(device)
    history = {}
    if stage == "member":
        load(representation, inputs["encoder"] / "encoder.pt", device)
        shutil.copy2(inputs["encoder"] / "encoder.pt", output_dir / "encoder.pt")
    else:
        decoder = nn.Linear(config["width"], x.shape[1]).to(device)
        history["encoder"] = fit(
            nn.Sequential(representation, decoder),
            x,
            x,
            nn.MSELoss(),
            config["encoder_steps"],
            config["learning_rate"],
            "encoder",
            report,
        )
        torch.save(representation.state_dict(), output_dir / "encoder.pt")
    if stage == "encoder":
        return history
    with torch.no_grad():
        encoded = representation(x)
    names = ("head",) if stage == "member" else ("member_1", "member_2")
    for name in names:
        predictor = head(config).to(device)
        history[name] = fit(
            predictor,
            encoded,
            y,
            nn.CrossEntropyLoss(),
            config["member_steps"],
            config["learning_rate"],
            name,
            report,
        )
        torch.save(predictor.state_dict(), output_dir / f"{name}.pt")
    return history


def adapt(data_dir, input_dir, output_dir, config, seed, *, device, limits, report=None):
    """Fine-tune both heads against the saved encoder; write a complete new snapshot."""
    torch.manual_seed(seed)
    images, labels = load_split(data_dir, "train")
    x = features(images[np.arange(len(images))], config["resolution"], device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    representation = encoder(config).to(device)
    load(representation, input_dir / "encoder.pt", device)
    with torch.no_grad():
        encoded = representation(x)
    shutil.copy2(input_dir / "encoder.pt", output_dir / "encoder.pt")
    history = {}
    for name in ("member_1", "member_2"):
        predictor = head(config).to(device)
        load(predictor, input_dir / f"{name}.pt", device)
        history[name] = fit(
            predictor,
            encoded,
            y,
            nn.CrossEntropyLoss(),
            config["member_steps"],
            config["learning_rate"],
            name,
            report,
        )
        torch.save(predictor.state_dict(), output_dir / f"{name}.pt")
    return history
