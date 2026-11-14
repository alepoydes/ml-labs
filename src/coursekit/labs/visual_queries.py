"""One-token prediction and independent adaptation for Visual Queries."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from coursekit.common import CourseError, file_hashes, fingerprint, write_json
from coursekit.labs.visual_queries_data import (
    batch,
    episode,
    load_split,
    metadata,
    rows,
    scope,
    vocabulary,
)


def validate_inputs(
    images, tokens, mask, *, vocab_size, image_shape, max_tokens=96, max_batch=16
):
    if (
        not isinstance(images, np.ndarray)
        or images.dtype != np.uint8
        or images.ndim != 4
        or images.shape[1:] != tuple(image_shape)
        or not 1 <= len(images) <= max_batch
    ):
        raise CourseError("Expected a bounded uint8 RGB image batch.")
    if (
        not isinstance(tokens, np.ndarray)
        or tokens.dtype != np.int64
        or tokens.ndim != 2
        or tokens.shape[0] != len(images)
        or not 1 <= tokens.shape[1] <= max_tokens
    ):
        raise CourseError("Expected int64 question tokens with shape (B,L).")
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.shape != tokens.shape
        or not mask[:, 0].all()
        or np.any(mask[:, 1:] & ~mask[:, :-1])
    ):
        raise CourseError("Question masks must be nonempty true prefixes.")
    if (
        np.any(tokens[~mask] != 0)
        or np.any(tokens[mask] < 1)
        or np.any(tokens[mask] >= vocab_size)
    ):
        raise CourseError(
            "Question token outside the active vocabulary or invalid padding."
        )


def validate_answers(answers, count, vocab_size):
    if (
        not isinstance(answers, np.ndarray)
        or answers.dtype != np.int64
        or answers.shape != (count,)
        or np.any(answers < 1)
        or np.any(answers >= vocab_size)
    ):
        raise CourseError(
            "predict() must return int64 (B,) non-padding IDs in the active vocabulary."
        )


def data_fingerprint(root):
    return fingerprint(
        {k: v for k, v in file_hashes(Path(root)).items() if not k.startswith(".")}
    )


def evaluate(solution, data, split="base/validation", *, progress=None, batch_size=16):
    data = Path(data)
    meta, size = metadata(data), scope(data, split)
    loaded = load_split(data, split)
    answers = loaded[3]
    if answers is None or not len(answers):
        raise CourseError("Evaluation needs a nonempty labeled question split.")
    count = len(answers)
    correct = 0
    levels = {}
    language = {name: dict(correct=0, questions=0) for name in ("known", "new")}
    # Only an instructor dataset has this sidecar; it is never a worker mount.
    judge_path = data / "judge.json"
    judge = json.loads(judge_path.read_text()).get(split, {}) if judge_path.exists() else {}
    ids = [
        r["question_id"]
        for r in rows(
            data / split / "questions.csv", ["question_id", "image_id", "token_ids"]
        )
    ]
    step = min(batch_size, meta["max_batch_size"])
    for start in range(0, count, step):
        selected = np.arange(start, min(start + step, count))
        arguments = batch(loaded, selected)
        validate_inputs(
            *arguments,
            vocab_size=size,
            image_shape=meta["image_shape"],
            max_tokens=meta["max_question_tokens"],
            max_batch=meta["max_batch_size"],
        )
        predicted = solution.predict(*arguments)
        validate_answers(predicted, len(selected), size)
        right = predicted == answers[selected]
        correct += int(right.sum())
        # New answer concepts count too, even with an otherwise familiar question.
        novel = np.array(
            [
                np.any(loaded[2][i] >= meta["base_vocab_size"])
                or answers[i] >= meta["base_vocab_size"]
                or judge.get(ids[i], {}).get("language") == "new"
                for i in selected
            ],
            dtype=bool,
        )
        for name, selected_group in (("known", ~novel), ("new", novel)):
            language[name]["questions"] += int(selected_group.sum())
            language[name]["correct"] += int(right[selected_group].sum())
        for index, value in zip(selected, right):
            if ids[index] in judge:
                level = str(judge[ids[index]]["level"])
                record = levels.setdefault(level, dict(correct=0, questions=0))
                record["correct"] += int(value)
                record["questions"] += 1
        if progress:
            progress(dict(completed=int(selected[-1]) + 1, total=count, correct=correct))
    return dict(
        score=100 * correct / count,
        correct=correct,
        questions=count,
        levels=levels,
        language=language,
    )


def adaptation_view(root, name, destination, *, rehearsal=None):
    """Copy only explicitly authorized examples; never copy a containing mixed tree."""
    root, destination = Path(root), Path(destination)
    rehearsal = Path(rehearsal) if rehearsal is not None else root
    value = episode(root, name)
    if vocabulary(root) != vocabulary(rehearsal):
        raise CourseError("Rehearsal and evaluation vocabulary differ.")
    destination.mkdir(parents=True)
    shutil.copyfile(root / "vocabulary.csv", destination / "vocabulary.csv")
    write_json(destination / "episode.json", value)
    permitted = [
        (root, f"adaptation/{name}/support", "support"),
        (rehearsal, "base/train", "rehearsal/base/train"),
    ]
    for source, split, target in permitted:
        load_split(source, split)
        directory = destination / target
        directory.mkdir(parents=True)
        for filename in ("images.csv", "questions.csv", "answers.csv"):
            shutil.copyfile(source / split / filename, directory / filename)
        for filename in {
            r["sheet"]
            for r in rows(
                source / split / "images.csv", ["image_id", "sheet", "row", "column"]
            )
        }:
            (directory / filename).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / split / filename, directory / filename)
    # The glyph pool is public supervised training data, in its own canonical sheets.
    from coursekit.labs.visual_queries_data import load_glyphs

    glyphs, _ = load_glyphs(rehearsal)
    folder = destination / "rehearsal/glyphs"
    folder.mkdir(parents=True)
    for filename in (
        "metadata.json",
        "images.csv",
        "labels.csv",
        *{r["sheet"] for r in glyphs.index},
    ):
        (folder / filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rehearsal / "glyphs" / filename, folder / filename)
    meta = metadata(root) | dict(splits=["support", "rehearsal/base/train"], episodes=[])
    write_json(destination / "dataset.json", meta)
    # Rehearsal also forms a loadable resource root for the same public loader.
    shutil.copyfile(rehearsal / "vocabulary.csv", destination / "rehearsal/vocabulary.csv")
    write_json(
        destination / "rehearsal/dataset.json",
        metadata(rehearsal) | dict(splits=["base/train"], episodes=[]),
    )
    return destination


def run_benchmark(data, run_phase, *, progress=None):
    """run_phase(split, episode_id, progress) owns fresh loading and adaptation."""
    data = Path(data)
    meta = metadata(data)
    phases = [("base/validation", None)] + [
        (f"adaptation/{name}/query", name) for name in meta["episodes"]
    ]
    if not meta["episodes"]:
        raise CourseError("A complete benchmark must include adaptation episodes.")
    counts = [len(load_split(data, split)[2]) for split, _ in phases]
    total, completed, correct = sum(counts), 0, 0
    details, levels, language = [], {}, {}
    for (split, name), count in zip(phases, counts):

        def report(event):
            if progress:
                progress(
                    dict(
                        phase="prediction",
                        episode=name,
                        completed=completed + event["completed"],
                        total=total,
                    )
                )

        result = run_phase(split, name, report)
        if result["questions"] != count:
            raise CourseError("Incomplete evaluation phase.")
        completed += count
        correct += result["correct"]
        details.append(dict(episode=name, **result))
        for field, totals in (("levels", levels), ("language", language)):
            for group, record in result.get(field, {}).items():
                target = totals.setdefault(group, dict(correct=0, questions=0))
                for key in target:
                    target[key] += record[key]
    if (data / "judge.json").exists() and {
        level: value["questions"] for level, value in levels.items()
    } != {str(level): meta["scored_questions_per_level"] for level in range(1, 6)}:
        raise CourseError("The scored difficulty quotas are incomplete or unequal.")
    return dict(
        score=100 * correct / total,
        correct=correct,
        questions=total,
        accuracy=correct / total,
        phases=details,
        levels=levels,
        language=language,
    )


def evaluate_solution(solution, data, *, seed=0, device="cpu", run_dir=None, progress=None):
    from coursekit.training import adapt_local, load_solution

    solution, data = Path(solution), Path(data)
    meta = metadata(data)
    run_dir = Path(run_dir) if run_dir else solution.parent / "adaptation-runs"

    def run(split, name, report):
        artifacts = solution / "artifacts"
        if name is not None:
            with tempfile.TemporaryDirectory(prefix="course-support-") as temporary:
                view = adaptation_view(data, name, Path(temporary) / "data")
                artifacts = adapt_local(
                    solution,
                    view,
                    artifacts,
                    run_dir=run_dir,
                    seed=seed,
                    device=device,
                    limits=meta["adaptation_limits"],
                    report=(
                        lambda e: progress(
                            dict(phase="adaptation", episode=name, training=e)
                        )
                    )
                    if progress
                    else None,
                )
        instance = load_solution(solution, seed, artifact_dir=artifacts, device=device)
        return evaluate(instance, data, split, progress=report)

    return run_benchmark(data, run, progress=progress)
