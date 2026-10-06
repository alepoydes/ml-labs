"""Small GUI-independent inspection helpers shared by local graphical tools."""

import copy
import time

import numpy as np
import torch

from coursekit.common import CourseError
from coursekit.labs.blindjack import benchmark, reward, validate_mask


def debug_snapshot(solution, index=0):
    """Copy optional diagnostics; never retain mutable policy arrays or graphs."""
    if not hasattr(solution, "debug"):
        return {}
    return copy_debug(solution.debug(index))


def copy_debug(values):
    """Validate and copy live or recorded diagnostics using the same value contract."""
    if not isinstance(values, dict) or len(values) > 64:
        raise CourseError("debug() must return a dictionary with at most 64 entries.")
    copied = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name or len(name) > 128:
            raise CourseError("Debug field names must be nonempty short strings.")
        if isinstance(value, str):
            if len(value) > 4096:
                raise CourseError("Debug text exceeds 4096 characters.")
            copied[name] = value
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if (
            value.dtype.kind not in "biuf"
            or value.ndim > 2
            or value.size > 4096
            or not np.isfinite(value).all()
        ):
            raise CourseError(
                "Debug values must be finite scalars or small numeric vectors/matrices."
            )
        copied[name] = value.item() if value.ndim == 0 else value.copy()
    return copied


def inspect_hand(solution, images):
    started = time.perf_counter()
    mask = solution.select(images.copy())
    elapsed = time.perf_counter() - started
    validate_mask(mask)
    return {"mask": mask.copy(), "seconds": elapsed, "debug": debug_snapshot(solution)}


class BlindjackGame:
    """First committed human answers and paired model decisions on public hands."""

    def __init__(self, data):
        self.images, self.labels, self.hands = benchmark(data)
        self.records = {}

    def hand(self, index):
        if type(index) is not int or not 0 <= index < len(self.hands):
            raise CourseError("Hand index is out of range.")
        return self.images[self.hands[index]]

    def reveal(self, index):
        if index not in self.records:
            raise CourseError("Commit your answer before revealing card values.")
        return self.labels[self.hands[index]].copy()

    def commit(self, index, human, model=None, *, seconds=None):
        self.hand(index)
        if index in self.records:
            raise CourseError("This hand already has a committed answer.")
        validate_mask(human)
        if seconds is not None and (not np.isfinite(seconds) or seconds < 0):
            raise CourseError("Decision time must be finite and nonnegative.")
        if model is not None:
            validate_mask(model["mask"])
            if not np.isfinite(model["seconds"]) or model["seconds"] < 0:
                raise CourseError("Model decision time must be finite and nonnegative.")
        values = self.labels[self.hands[index]]
        record = {
            "hand": index,
            "human": human.tolist(),
            "human_reward": reward(values, human),
            "human_seconds": seconds,
            "model": model["mask"].tolist() if model is not None else None,
            "model_reward": reward(values, model["mask"]) if model is not None else None,
            "model_seconds": model["seconds"] if model is not None else None,
        }
        self.records[index] = record
        return copy.deepcopy(record)

    def summary(self):
        records = list(self.records.values())
        paired = [r for r in records if r["model"] is not None]

        def score(rows, player):
            return (
                float(np.mean([r[f"{player}_reward"] for r in rows]) * 100 / 21)
                if rows
                else None
            )

        return {
            "hands": len(records),
            "human_score": score(records, "human"),
            "paired_hands": len(paired),
            "paired_human_score": score(paired, "human"),
            "model_score": score(paired, "model"),
        }
