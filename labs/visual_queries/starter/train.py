"""The lifecycle example deliberately learns nothing."""

import json

import numpy as np

from coursekit.common import write_json
from coursekit.labs.visual_queries_data import batch


def train(
    datasets, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
):
    if stage != "all" or inputs:
        raise ValueError("This starter has one complete stage and no component inputs.")
    data = datasets.load_split("base/train")
    # These are the three arrays a learned model would consume.
    images, tokens, mask = batch(
        data, np.arange(min(config["example_batch_size"], len(data[2])))
    )
    count = len(images)
    write_json(
        output_dir / "state.json",
        dict(
            answer=config["constant_answer_id"],
            vocab_size=datasets.metadata["base_vocab_size"],
            episode_id=None,
        ),
    )
    if report:
        report(dict(stage="prepare", step=1, total=1, metrics={"examples": count}))
    return {"prepare": {"step": np.array([1]), "examples": np.array([count])}}


def adapt(
    datasets,
    input_dir,
    output_dir,
    config,
    seed,
    *,
    device,
    limits,
    checkpoint=None,
    report=None,
):
    state = json.loads((input_dir / "state.json").read_text())
    # Manual fine-tuning may use the public root; assessment provides one support view.
    adapting = datasets.episode["episode_id"] is not None
    task = datasets.episode
    support = datasets.load_split("support" if adapting else "base/train")
    state.update(vocab_size=task["vocab_size"], episode_id=task["episode_id"])
    write_json(output_dir / "state.json", state)
    history = {"adapt": {"step": np.array([1]), "examples": np.array([len(support[2])])}}
    # Acknowledgement means this complete state also survives a later time limit.
    if checkpoint:
        checkpoint(history)
    if report:
        report(dict(stage="adapt", step=1, total=1, metrics={"examples": len(support[2])}))
    return history
