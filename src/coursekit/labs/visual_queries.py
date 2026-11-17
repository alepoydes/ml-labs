"""One-token prediction and independent adaptation for Visual Queries."""

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from coursekit.common import CourseError, file_hashes, fingerprint, write_json
from coursekit.labs.visual_queries_data import (
    batch,
    episode,
    episode_label,
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


def evaluate(
    solution,
    data,
    split="base/validation",
    *,
    progress=None,
    batch_size=16,
    on_batch=None,
    indices=None,
    vocab_size=None,
):
    data = Path(data)
    meta, size = metadata(data), vocab_size or scope(data, split)
    loaded = load_split(data, split)
    answers = loaded[3]
    if answers is None or not len(answers):
        raise CourseError("Evaluation needs a nonempty labeled question split.")
    selection = np.arange(len(answers)) if indices is None else np.asarray(indices)
    if (
        selection.ndim != 1
        or selection.dtype.kind not in "iu"
        or not len(selection)
        or np.any(selection < 0)
        or np.any(selection >= len(answers))
        or len(np.unique(selection)) != len(selection)
    ):
        raise CourseError("Choose distinct valid evaluation rows.")
    count = len(selection)
    correct = 0
    levels = {}
    language = {name: dict(correct=0, questions=0) for name in ("known", "new")}
    # Only an instructor dataset has this sidecar; it is never a worker mount.
    judge_path = data / "judge.json"
    judge = json.loads(judge_path.read_text()).get(split, {}) if judge_path.exists() else {}
    ids = (
        [str(i + 1) for i in range(len(answers))]
        if meta["format_version"] == 2
        else [
            r["question_id"]
            for r in rows(
                data / split / "questions.csv", ["question_id", "image_id", "token_ids"]
            )
        ]
    )
    step = min(batch_size, meta["max_batch_size"])
    for start in range(0, count, step):
        selected = selection[start : start + step]
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
        if on_batch is not None:
            on_batch(dict(rows=selected.copy() + 1, answers=predicted.copy()))
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
            progress(dict(completed=min(start + step, count), total=count, correct=correct))
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
    modern = metadata(root)["format_version"] == 2
    for source, split, target in permitted:
        load_split(source, split)
        directory = destination / target
        directory.mkdir(parents=True)
        for filename in (
            ("examples.csv",) if modern else ("images.csv", "questions.csv", "answers.csv")
        ):
            shutil.copyfile(source / split / filename, directory / filename)
        for filename in {r["sheet"] for r in load_split(source, split)[0].index}:
            (directory / filename).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / split / filename, directory / filename)
    if modern:
        shutil.copyfile(
            root / "adaptation" / name / "vocabulary.csv",
            destination / "support/vocabulary.csv",
        )
    # The glyph pool is public supervised training data, in its own canonical sheets.
    from coursekit.labs.visual_queries_data import load_glyphs

    glyphs, _ = load_glyphs(rehearsal)
    folder = destination / "rehearsal/glyphs"
    folder.mkdir(parents=True)
    for filename in (
        "metadata.json",
        *(("examples.csv",) if modern else ("images.csv", "labels.csv")),
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


def retention_rows(data):
    """One-based published assignments; each base control is scored exactly once."""
    meta = metadata(data)
    version = meta.get("assessment_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise CourseError("Unsupported Visual Queries assessment schedule.")
    if version == 1:
        return {}
    assignments = meta.get("retention_rows", {})
    count = len(load_split(data, "base/validation")[2])
    if (
        not isinstance(assignments, dict)
        or set(assignments) != set(meta["episodes"])
        or any(not isinstance(v, list) or not v for v in assignments.values())
    ):
        raise CourseError("Missing post-adaptation base controls.")
    flat = [i for values in assignments.values() for i in values]
    if any(type(i) is not int for i in flat) or sorted(flat) != list(range(1, count + 1)):
        raise CourseError("Base controls must partition every row exactly once.")
    return {name: np.asarray(values, np.int64) - 1 for name, values in assignments.items()}


def combine_results(parts):
    result = dict(correct=0, questions=0, levels={}, language={})
    for part in parts:
        for field in ("correct", "questions"):
            result[field] += part[field]
        for field in ("levels", "language"):
            for name, counts in part.get(field, {}).items():
                target = result[field].setdefault(name, dict(correct=0, questions=0))
                for key in target:
                    target[key] += counts[key]
    result["score"] = 100 * result["correct"] / result["questions"]
    return result


def evaluate_adapted(predict, data, name, progress=None):
    """Use one saved adapted state for queries and its assigned old controls."""
    controls = retention_rows(data)
    split = f"adaptation/{name}/query"
    if not controls:
        return predict(split, progress=progress)
    size = episode(data, name)["vocab_size"]
    count = len(load_split(data, split)[2])
    total = count + len(controls[name])

    def report(offset):
        if progress is None:
            return None
        return lambda event: progress(
            dict(completed=offset + event["completed"], total=total)
        )

    query = predict(split, episode_id=name, progress=report(0))
    old = predict(
        "base/validation",
        indices=controls[name],
        episode_id=name,
        vocab_size=size,
        progress=report(count),
    )
    result = combine_results([query, old]) | dict(query=query, retention=old)
    return result


def run_benchmark(data, run_phase, *, progress=None):
    """run_phase(split, episode_id, progress) owns fresh loading and adaptation."""
    data = Path(data)
    meta = metadata(data)
    phases = [("base/validation", None)] + [
        (f"adaptation/{name}/query", name) for name in meta["episodes"]
    ]
    if not meta["episodes"]:
        raise CourseError("A complete benchmark must include adaptation episodes.")
    controls = retention_rows(data)
    counts = [
        len(load_split(data, split)[2]) + (len(controls[name]) if name in controls else 0)
        for split, name in phases
    ]
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
        scored = name is not None or not controls
        if scored:
            correct += result["correct"]
        details.append(dict(episode=name, scored=scored, **result))
        if not scored:
            continue
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
        score=100 * correct / sum(r["questions"] for r in details if r["scored"]),
        correct=correct,
        questions=sum(r["questions"] for r in details if r["scored"]),
        accuracy=correct / sum(r["questions"] for r in details if r["scored"]),
        assessment_version=meta.get("assessment_version", 1),
        phases=details,
        levels=levels,
        language=language,
    )


def adaptation_display(value):
    if not value:
        return dict(time="-", stop="-", model="-")
    saved = value.get("checkpoint")
    return dict(
        time=f"{value['seconds']:.2f} / {value['budget_seconds']:.2f}",
        stop=value["stop_reason"].replace("_", " "),
        model=f"Checkpoint {saved['index']} at {saved['seconds']:.2f} s"
        if saved
        else "Final",
    )


def adaptation_rows(result):
    return [
        dict(episode=f"Adaptation {index:02d}", **adaptation_display(phase["adaptation"]))
        for index, phase in enumerate(
            (p for p in result.get("phases", []) if p.get("episode") is not None), 1
        )
        if phase.get("adaptation")
    ]


def episode_results(result):
    """Human per-episode counts, without counting the combined row twice."""
    rows = []
    for index, phase in enumerate(
        (p for p in result.get("phases", []) if p.get("episode") is not None), 1
    ):
        if "retention" not in phase:
            continue
        values = {}
        for name, metrics in (
            ("old", phase["retention"]),
            ("new", phase["query"]),
            ("total", phase),
        ):
            values[name] = (
                f"{metrics['correct']}/{metrics['questions']} ({100 * metrics['correct'] / metrics['questions']:.1f}%)"
            )
        rows.append(dict(episode=f"Adaptation {index:02d}", **values))
    return rows


def result_table(result, meta):
    """Summarize a complete benchmark without averaging episode percentages."""

    def language(value):
        n = value.get("questions", 0)
        return f"{value['correct']}/{n} ({100 * value['correct'] / n:.1f}%)" if n else "-"

    table = [["Part", "Correct / total", "Accuracy", "Known language", "New language"]]
    timed = any(p.get("adaptation") for p in result["phases"])
    if timed:
        table[0] += ["Adaptation s / budget", "Stop", "Evaluated model"]
    phases = []
    for p in result["phases"]:
        label = episode_label(meta["episodes"], p["episode"])
        if p.get("scored") is False:
            label += " (diagnostic)"
        phases.append((label, p))
        for key, suffix in (("retention", "old controls"), ("query", "episode queries")):
            if key in p:
                phases.append((f"  {suffix}", p[key]))
    for label, row in [*phases, ("Total", result)]:
        groups = row.get("language", {})
        table.append(
            [
                label,
                f"{row['correct']}/{row['questions']}",
                f"{100 * row['correct'] / row['questions']:.2f}%",
                language(groups.get("known", {})),
                language(groups.get("new", {})),
            ]
        )
        if timed:
            table[-1] += list(adaptation_display(row.get("adaptation")).values())
    widths = [max(map(len, values)) for values in zip(*table)]
    return "\n".join(" | ".join(v.ljust(w) for v, w in zip(row, widths)) for row in table)


def evaluate_solution(solution, data, *, seed=0, device="cpu", run_dir=None, progress=None):
    from coursekit import recording
    from coursekit.labs.visual_queries_inspection import model_identity, new_record
    from coursekit.training import adapt_local, load_solution

    solution, data = Path(solution), Path(data)
    meta = metadata(data)
    run_dir = Path(run_dir) if run_dir else solution.parent / "adaptation-runs"
    path = run_dir.resolve() / f"predictions-{uuid.uuid4().hex}.npz"
    saved = new_record(diagnostic=False)
    resource_hash = data_fingerprint(data)

    def run(split, name, report):
        artifacts = solution / "artifacts"
        adaptation = None
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
                adaptation = json.loads(
                    (artifacts.parent / "training.json").read_text()
                ).get("adaptation")
        identity = model_identity(solution, artifacts, device, seed)
        instance = load_solution(solution, seed, artifact_dir=artifacts, device=device)
        if model_identity(solution, artifacts, device, seed) != identity:
            raise CourseError("Solution changed while loading; restart evaluation.")

        def predict(
            split, *, indices=None, episode_id=None, vocab_size=None, progress=None
        ):
            current = dict(
                data=str(data.resolve()),
                data_hash=resource_hash,
                split=split,
                episode=episode_id,
                model=identity,
                batches=[],
                complete=False,
            )
            saved["runs"].append(current)
            try:
                result = evaluate(
                    instance,
                    data,
                    split,
                    progress=progress,
                    indices=indices,
                    vocab_size=vocab_size,
                    on_batch=current["batches"].append,
                )
                current["complete"] = True
                return result
            finally:
                recording.save(path, saved)

        result = (
            predict(split, progress=report)
            if name is None
            else evaluate_adapted(predict, data, name, report)
        )
        if adaptation is not None:
            result["adaptation"] = adaptation
        return result

    try:
        result = run_benchmark(data, run, progress=progress)
        saved["complete"] = True
        return result | dict(recording=str(path))
    finally:
        recording.save(path, saved)
