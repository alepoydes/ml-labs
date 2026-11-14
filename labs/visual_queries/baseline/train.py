"""Supervised learning and fresh-base adaptation, using only the supplied views."""

import time

import numpy as np
import torch
from lib.model import Model, expand, load, save, tensors
from torch.nn import functional as F

from coursekit.labs.visual_queries_data import batch, episode, load_split, metadata


def shuffled_rows(data, rows, rng):
    # Decode a sheet once, rather than once per randomly selected question.
    groups = {}
    for row in rows:
        sheet = data[0].positions[data[1][row]][0]
        groups.setdefault(sheet, []).append(row)
    keys = list(groups)
    return np.concatenate(
        [rng.permutation(groups[keys[i]]) for i in rng.permutation(len(keys))]
    )


def examples(data, rows, device):
    return tensors(*batch(data, rows), device), torch.as_tensor(
        data[3][rows], device=device
    )


def combine(inputs, targets, rehearsal, fraction, rng, length):
    if rehearsal is None:
        return inputs, targets
    old_inputs, old_targets = rehearsal
    rows = torch.as_tensor(
        rng.integers(len(old_targets), size=max(1, round(len(targets) * fraction))),
        device=targets.device,
    )
    images, tokens, mask = inputs
    old_images, old_tokens, old_mask = (x[rows] for x in old_inputs)
    return (
        torch.cat((images, old_images)),
        torch.cat(
            (
                F.pad(tokens, (0, length - tokens.shape[1])),
                F.pad(old_tokens, (0, length - old_tokens.shape[1])),
            )
        ),
        torch.cat(
            (
                F.pad(mask, (0, length - mask.shape[1])),
                F.pad(old_mask, (0, length - old_mask.shape[1])),
            )
        ),
    ), torch.cat((targets, old_targets[rows]))


def accuracy(model, data, rows, size, device):
    correct = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), size):
            inputs, targets = examples(data, rows[start : start + size], device)
            correct += int((model(*inputs).argmax(1) == targets).sum())
    return correct / len(rows)


def fit(
    model,
    data,
    rows,
    config,
    rng,
    device,
    report,
    *,
    validation=None,
    rehearsal=None,
    deadline=float("inf"),
):
    adapting = validation is None
    stage = "adapt" if adapting else "train"
    epochs = config["adapt_epochs" if adapting else "epochs"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["adapt_learning_rate" if adapting else "learning_rate"],
        weight_decay=config["weight_decay"],
    )
    history = dict(step=[], loss=[], accuracy=[])
    if validation is not None:
        history["validation_accuracy"] = []
    best, best_accuracy, stale = None, -1, 0
    for epoch_index in range(epochs):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        order = shuffled_rows(data, rows, rng)
        for start in range(0, len(order), config["batch_size"]):
            if time.monotonic() >= deadline:
                break
            inputs, targets = examples(
                data, order[start : start + config["batch_size"]], device
            )
            inputs, targets = combine(
                inputs,
                targets,
                rehearsal,
                config["rehearsal_fraction"],
                rng,
                model.max_tokens,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            correct += int((logits.detach().argmax(1) == targets).sum())
            count += len(targets)
        if not count:
            break
        metrics = dict(loss=loss_sum / count, accuracy=correct / count)
        if validation is not None:
            value = accuracy(model, data, validation, config["batch_size"], device)
            metrics["validation_accuracy"] = value
            if value > best_accuracy:
                best_accuracy, stale = value, 0
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history["step"].append(epoch_index + 1)
        for key, value in metrics.items():
            history[key].append(value)
        print(f"{stage} epoch {epoch_index + 1}/{epochs}: {metrics}", flush=True)
        if report:
            report(dict(stage=stage, step=epoch_index + 1, total=epochs, metrics=metrics))
        if validation is not None and stale >= config["patience"]:
            break
    if best is not None:
        model.load_state_dict(best)
    return {
        stage: {
            key: np.asarray(value, dtype=np.int64 if key == "step" else float)
            for key, value in history.items()
        }
    }


def train(
    data_dir, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
):
    if stage != "all" or inputs:
        raise ValueError(
            "This baseline trains one complete model without component inputs."
        )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data, meta = load_split(data_dir, "base/train"), metadata(data_dir)
    # Questions sharing an image stay on the same side of the internal holdout.
    images = rng.permutation(np.unique(data[1]))
    n = max(1, int(len(images) * config["validation_fraction"]))
    held_out = np.isin(data[1], images[:n])
    rows, validation = np.flatnonzero(~held_out), np.flatnonzero(held_out)
    if not len(rows) or not len(validation):
        raise ValueError("Training needs at least two distinct images.")
    model = Model(meta["base_vocab_size"], meta["max_question_tokens"]).to(device)
    history = fit(model, data, rows, config, rng, device, report, validation=validation)
    save(model, output_dir, None)
    return history


def adapt(data_dir, input_dir, output_dir, config, seed, *, device, limits, report=None):
    started = time.monotonic()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, _ = load(input_dir, device)
    adapting = (data_dir / "episode.json").exists()
    task = (
        episode(data_dir)
        if adapting
        else dict(vocab_size=metadata(data_dir)["base_vocab_size"], episode_id=None)
    )
    model = expand(model, task["vocab_size"])
    data = load_split(data_dir, "support" if adapting else "base/train")
    rehearsal = None
    if adapting and config["rehearsal_examples"] and config["rehearsal_fraction"]:
        base = load_split(data_dir, "rehearsal/base/train")
        rows = rng.choice(
            len(base[2]), min(config["rehearsal_examples"], len(base[2])), replace=False
        )
        rehearsal = examples(base, np.sort(rows), device)
    # Leave time for saving and worker overhead; the executor enforces the hard limit.
    deadline = started + 0.85 * (limits or {}).get("wall_seconds", 300)
    history = fit(
        model,
        data,
        np.arange(len(data[2])),
        config,
        rng,
        device,
        report,
        rehearsal=rehearsal,
        deadline=deadline,
    )
    save(model, output_dir, task["episode_id"])
    return history
