"""Batched inference and authenticated replay, independent of Qt and model architecture."""

import time
from pathlib import Path

import numpy as np

from coursekit import recording
from coursekit.common import CourseError, file_hashes, fingerprint
from coursekit.inspection import copy_debug, debug_snapshot
from coursekit.labs.visual_queries import (
    data_fingerprint,
    validate_answers,
    validate_inputs,
)
from coursekit.labs.visual_queries_data import (
    EXAMPLE_FIELDS,
    batch,
    load_split,
    metadata,
    rows,
    scope,
    vocabulary,
)


class Dataset:
    """Host-side text and tiles. Only arguments() output enters a model process."""

    def __init__(self, root, split, *, fingerprint=None):
        self.root, self.split = Path(root).resolve(), split
        self.meta = metadata(self.root)
        if self.meta["format_version"] != 2:
            raise CourseError("The inspector requires readable resource format 2.")
        self.loaded = load_split(self.root, split)
        self.examples = rows(self.root / split / "examples.csv", EXAMPLE_FIELDS)
        self.words = vocabulary(
            self.root, split.split("/")[1] if split.startswith("adaptation/") else None
        )
        self.fingerprint = fingerprint or data_fingerprint(self.root)
        self.settings = dict(
            vocab_size=scope(self.root, split),
            image_shape=self.meta["image_shape"],
            max_tokens=self.meta["max_question_tokens"],
            max_batch=self.meta["max_batch_size"],
        )

    def arguments(self, numbers):
        return batch(self.loaded, np.asarray(numbers, dtype=np.int64) - 1)

    def image(self, number):
        return self.loaded[0][np.array([self.loaded[1][number - 1]])][0]

    def new_run(self, identity=None):
        return dict(
            data=str(self.root),
            data_hash=self.fingerprint,
            split=self.split,
            model=identity or {},
            batches=[],
            complete=False,
        )


def probabilities(debug, size):
    value = debug.get("answer_probabilities")
    if value is None:
        return None
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (size,)
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or np.any(value > 1)
        or value[0] != 0
        or not np.isclose(value.sum(), 1, atol=1e-5, rtol=1e-5)
    ):
        raise CourseError(
            "answer_probabilities must be normalized over the active vocabulary, with zero padding probability."
        )
    return value


def predict_batch(solution, arguments, settings, *, diagnostics=True):
    validate_inputs(*arguments, **settings)
    started = time.perf_counter()
    answers = solution.predict(*arguments)
    validate_answers(answers, len(arguments[0]), settings["vocab_size"])
    result = dict(answers=answers.copy(), seconds=time.perf_counter() - started)
    if diagnostics:
        result["debug"] = []
        for index in range(len(answers)):
            try:
                value = debug_snapshot(solution, index)
                probabilities(value, settings["vocab_size"])
                result["debug"].append(value)
            except Exception as exc:
                result["debug_error"] = f"Row {index + 1}: {exc}"
                result["debug"].extend({} for _ in range(len(answers) - index))
                break
    return result


def new_record(**identity):
    return dict(lab="visual_queries", version=1, runs=[], complete=False, **identity)


def model_identity(solution, artifacts, device, seed):
    from coursekit.training import controller_identity

    value = controller_identity(solution) | dict(
        artifacts_hash=fingerprint(file_hashes(artifacts)),
        artifact_dir=str(Path(artifacts).resolve()),
        device=device,
        seed=seed,
    )
    value["version"] = fingerprint(
        {
            k: value[k]
            for k in ("source_hash", "config_hash", "artifacts_hash", "device", "seed")
        }
    )
    return value


def read_record(path, *, data_override=None):
    """Verify all sources before a GUI swaps its current session."""
    value = recording.load(path)
    if (
        not isinstance(value, dict)
        or value.get("lab") != "visual_queries"
        or type(value.get("version")) is not int
        or value["version"] != 1
        or type(value.get("complete")) is not bool
        or type(value.get("diagnostic", False)) is not bool
        or not isinstance(value.get("runs"), list)
    ):
        raise CourseError("Unsupported Visual Queries recording.")
    datasets, hashes = [], {}
    for run in value["runs"]:
        if (
            not isinstance(run, dict)
            or any(
                not isinstance(run.get(k), str) or not run[k]
                for k in ("data", "data_hash", "split")
            )
            or type(run.get("complete")) is not bool
            or not isinstance(run.get("batches"), list)
            or not isinstance(run.get("model"), dict)
            or not isinstance(run.get("dataset", ""), str)
            or run.get("error") is not None
            and not isinstance(run["error"], str)
        ):
            raise CourseError("Invalid recorded run metadata.")
        # These optional identity fields are displayed by both student and teacher
        # viewers. Validate them before either viewer replaces its current session.
        if any(
            key in run["model"] and not isinstance(run["model"][key], str)
            for key in ("name", "version", "artifacts_hash", "archive_sha256", "source")
        ):
            raise CourseError("Invalid recorded model identity.")
        root = Path(data_override or run["data"]).resolve()
        if root not in hashes:
            hashes[root] = data_fingerprint(root)
        if hashes[root] != run["data_hash"]:
            raise CourseError(
                "The recorded dataset is missing or changed. Select the original corpus."
            )
        data = Dataset(root, run["split"], fingerprint=hashes[root])
        seen = set()
        for item in run["batches"]:
            if not isinstance(item, dict) or not {"rows", "answers"} <= item.keys():
                raise CourseError("Invalid recorded batch.")
            numbers = item["rows"]
            if (
                not isinstance(numbers, np.ndarray)
                or numbers.dtype != np.int64
                or numbers.ndim != 1
                or not 1 <= len(numbers) <= data.settings["max_batch"]
                or np.any(numbers < 1)
                or np.any(numbers > len(data.examples))
            ):
                raise CourseError("Invalid recorded question rows.")
            if len(set(numbers)) != len(numbers) or seen.intersection(numbers):
                raise CourseError("Repeated recorded question rows.")
            seen.update(numbers)
            validate_answers(item["answers"], len(numbers), data.settings["vocab_size"])
            if "debug" in item:
                if not isinstance(item["debug"], list) or len(item["debug"]) != len(
                    numbers
                ):
                    raise CourseError("Diagnostics must match every recorded batch row.")
                item["debug"] = [copy_debug(v) for v in item["debug"]]
                for debug in item["debug"]:
                    probabilities(debug, data.settings["vocab_size"])
        datasets.append(data)
    if not datasets:
        raise CourseError("This recording has no prediction runs.")
    return value, datasets
