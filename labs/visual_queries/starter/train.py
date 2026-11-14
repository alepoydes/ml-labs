"""The lifecycle example deliberately learns nothing."""

import json

import numpy as np

from coursekit.common import write_json
from coursekit.labs.visual_queries_data import (
    batch,
    episode,
    load_split,
    metadata,
    vocabulary,
)


def train(
    data_dir, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
):
    if stage != "all" or inputs:
        raise ValueError("This starter has one complete stage and no component inputs.")
    data = load_split(data_dir, "base/train")
    # These are the three arrays a learned model would consume.
    images, tokens, mask = batch(
        data, np.arange(min(config["example_batch_size"], len(data[2])))
    )
    count = len(images)
    write_json(
        output_dir / "state.json",
        dict(
            answer=vocabulary(data_dir).index("yes"),
            vocab_size=metadata(data_dir)["base_vocab_size"],
            episode_id=None,
        ),
    )
    if report:
        report(dict(stage="prepare", step=1, total=1, metrics={"examples": count}))
    return {"prepare": {"step": np.array([1]), "examples": np.array([count])}}


def adapt(data_dir, input_dir, output_dir, config, seed, *, device, limits, report=None):
    state = json.loads((input_dir / "state.json").read_text())
    # Manual fine-tuning may use the public root; assessment provides one support view.
    adapting = (data_dir / "episode.json").exists()
    task = (
        episode(data_dir)
        if adapting
        else dict(vocab_size=metadata(data_dir)["base_vocab_size"], episode_id=None)
    )
    support = load_split(data_dir, "support" if adapting else "base/train")
    state.update(vocab_size=task["vocab_size"], episode_id=task["episode_id"])
    write_json(output_dir / "state.json", state)
    if report:
        report(dict(stage="adapt", step=1, total=1, metrics={"examples": len(support[2])}))
    return {"adapt": {"step": np.array([1]), "examples": np.array([len(support[2])])}}
