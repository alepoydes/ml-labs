"""Versioned training snapshots; student functions own the models and stages."""

import copy
import json
import re
import shutil
import stat
import time
import uuid
from pathlib import Path

import numpy as np

from coursekit.common import (
    CourseError,
    digest,
    file_hashes,
    fingerprint,
    read_toml,
    write_json,
)
from coursekit.execution import resolve_device, runtime_identity, solution_lock
from coursekit.submissions import metadata

NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")


def checked_files(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise CourseError(f"Expected an artifact directory: {directory}")
    for path in directory.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise CourseError(f"Non-regular artifact: {path}")
    return file_hashes(directory)


def reporter(callback):
    steps = {}
    last = {}

    def emit(event):
        if not isinstance(event, dict) or set(event) - {
            "stage",
            "step",
            "total",
            "metrics",
        }:
            raise CourseError("Invalid training progress fields.")
        name, step, total = event.get("stage"), event.get("step"), event.get("total")
        if (
            not isinstance(name, str)
            or not NAME.fullmatch(name)
            or type(step) is not int
            or step < steps.get(name, 0)
            or (total is not None and (type(total) is not int or total < max(1, step)))
        ):
            raise CourseError("Invalid training progress counters.")
        metrics = event.get("metrics", {})
        if (
            not isinstance(metrics, dict)
            or len(metrics) > 32
            or any(
                not isinstance(k, str)
                or not NAME.fullmatch(k)
                or type(v) not in (float, int)
                or not np.isfinite(v)
                for k, v in metrics.items()
            )
        ):
            raise CourseError("Invalid training progress metrics.")
        if len(json.dumps(event)) > 8192:
            raise CourseError("Training progress exceeds 8 KiB.")
        steps[name] = step
        now = time.monotonic()
        if callback is not None and (now - last.get(name, -1) >= 0.1 or step == total):
            last[name] = now
            callback(event.copy())

    return emit


def save_history(directory, history):
    history = {} if history is None else history
    if not isinstance(history, dict):
        raise CourseError("Expected histories grouped by stage.")
    hashes = {}
    for name, columns in history.items():
        if (
            not isinstance(name, str)
            or not NAME.fullmatch(name)
            or not isinstance(columns, dict)
        ):
            raise CourseError("Invalid history group.")
        step = columns.get("step")
        if (
            not isinstance(step, np.ndarray)
            or step.ndim != 1
            or step.dtype.kind not in "iu"
            or np.any(step < 0)
            or np.any(step[1:] <= step[:-1])
        ):
            raise CourseError("History steps must be increasing nonnegative integers.")
        if any(
            not isinstance(k, str)
            or not NAME.fullmatch(k)
            or not isinstance(v, np.ndarray)
            or v.ndim != 1
            or len(v) != len(step)
            or v.dtype.kind not in "biuf"
            or not np.isfinite(v).all()
            for k, v in columns.items()
        ):
            raise CourseError(
                "History columns must be equally sized finite numeric vectors."
            )
        path = directory / f"{name}.npz"
        np.savez_compressed(path, **columns)
        hashes[path.name] = digest(path)
    return hashes


def load_history(run_dir):
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "training.json").read_text())
    if record["status"] != "complete":
        raise CourseError("Training did not complete.")
    if checked_files(run_dir / "artifacts") != record["artifacts"]:
        raise CourseError("Saved training artifacts changed.")
    result = {}
    for name, expected in record["histories"].items():
        if Path(name).name != name or not name.endswith(".npz"):
            raise CourseError("Invalid history path.")
        path = run_dir / "history" / name
        if digest(path) != expected:
            raise CourseError("Saved history changed.")
        with np.load(path, allow_pickle=False) as arrays:
            result[Path(name).stem] = dict(arrays)
    return result, record


def install(solution, artifacts):
    """Readers and installation share a lock; failed replacement restores the old state."""
    candidate = solution / f".artifacts-{uuid.uuid4().hex}"
    backup = solution / f".previous-{uuid.uuid4().hex}"
    try:
        shutil.copytree(artifacts, candidate)
        with solution_lock(solution):
            current = solution / "artifacts"
            try:
                if current.exists():
                    current.rename(backup)
                candidate.rename(current)
            except BaseException:
                if backup.exists():
                    backup.rename(current)
                raise
    finally:
        shutil.rmtree(candidate, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)


def run(
    solution,
    data,
    resource_hash,
    seed,
    *,
    run_dir=None,
    device="cpu",
    stage="all",
    inputs=None,
    report=None,
    operation="train",
    limits=None,
):
    from coursekit.training import load_module, source_hash

    solution, data = Path(solution).resolve(), Path(data).resolve()
    info = metadata((solution / "submission.toml").read_bytes())
    if info["api_version"] not in (2, 3):
        raise CourseError("This operation requires API v2 or v3.")
    if stage not in ["all", *info.get("training", {}).get("stages", [])]:
        raise CourseError(f"Unknown training stage: {stage}")
    device = resolve_device(device)
    run_dir = Path(run_dir or solution.parent / "runs" / solution.name) / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    output, histories = run_dir / "artifacts", run_dir / "history"
    output.mkdir()
    histories.mkdir()
    config = read_toml(solution / "config.toml")
    record = dict(
        format_version=2,
        status="running",
        operation=operation,
        stage=stage,
        source_hash=source_hash(solution),
        config_hash=digest(solution / "config.toml"),
        metadata_hash=digest(solution / "submission.toml"),
        config=copy.deepcopy(config),
        resource_hash=resource_hash,
        seed=seed,
        runtime=runtime_identity(device),
        limits=copy.deepcopy(limits),
        lab=info["lab"],
        api_version=info["api_version"],
        inputs={},
    )
    started = time.monotonic()
    write_json(run_dir / "training.json", record)
    try:
        # Pin copies so a mutable working variant cannot change an input mid-call.
        pinned = {}
        for name, path in (inputs or {}).items():
            if not isinstance(name, str) or not NAME.fullmatch(name):
                raise CourseError("Invalid input name.")
            path = Path(path)
            before = checked_files(path)
            target = run_dir / "inputs" / name
            shutil.copytree(path, target)
            if checked_files(path) != before or checked_files(target) != before:
                raise CourseError("Input artifacts changed while snapshotting.")
            record["inputs"][name] = before
            pinned[name] = target.resolve()
        if operation == "adapt":
            from coursekit.training_checkpoints import (
                AdaptationTimeout,
                Checkpoints,
                call_adapt,
                checkpoint_writer,
            )

            if set(pinned) != {"base"}:
                raise CourseError("Adaptation requires one complete base snapshot.")
            # Trusted input copying is setup; numeric access and student imports are timed.
            checkpoints = Checkpoints(run_dir / "checkpoints", limits)
        if info["api_version"] == 3:
            from coursekit.labs.visual_queries_learning import open_data

            data = open_data(data)
        module = load_module(solution, "train")
        kwargs = dict(device=device, report=reporter(report))
        if operation == "adapt":
            checkpoint = checkpoint_writer(
                output, histories, info["api_version"], limits, checkpoints
            )
            try:
                effective = copy.deepcopy(limits)
                effective["wall_seconds"] = max(
                    0.0, checkpoints.deadline - time.monotonic()
                )
                checkpoints.check_time()
                history = call_adapt(
                    module.adapt,
                    data,
                    pinned["base"],
                    output,
                    config,
                    seed,
                    limits=effective,
                    checkpoint=checkpoint,
                    **kwargs,
                )
            except AdaptationTimeout:
                if time.monotonic() < checkpoints.deadline:
                    raise
            reason = (
                "time_limit" if time.monotonic() >= checkpoints.deadline else "completed"
            )
            record["adaptation"] = checkpoints.result(reason=reason) | dict(
                configured_seconds=limits["wall_seconds"]
            )
            if reason == "time_limit":
                if checkpoints.last is None:
                    raise AdaptationTimeout(
                        "Time limit expired without a committed checkpoint."
                    )
                for target in (output, histories):
                    shutil.rmtree(target)
                    shutil.copytree(checkpoints.last / target.name, target)
                history = {}
                for path in histories.glob("*.npz"):
                    with np.load(path, allow_pickle=False) as arrays:
                        history[path.stem] = dict(arrays)
        else:
            history = module.train(
                data, output, config, seed, inputs=pinned or None, stage=stage, **kwargs
            )
        if (
            record["source_hash"] != source_hash(solution)
            or record["config_hash"] != digest(solution / "config.toml")
            or record["metadata_hash"] != digest(solution / "submission.toml")
        ):
            raise CourseError("Source or configuration changed during training.")
        for name, path in pinned.items():
            if checked_files(path) != record["inputs"][name]:
                raise CourseError("Training modified its input snapshot.")
        record["histories"] = save_history(histories, history)
        record["status"] = "complete"
        record["elapsed_seconds"] = time.monotonic() - started
        artifacts = checked_files(output)
        if (
            limits
            and sum((output / name).stat().st_size for name in artifacts)
            > limits["artifact_bytes"]
        ):
            raise CourseError("Adaptation output exceeds the artifact limit.")
        if "provenance.json" in artifacts:
            raise CourseError("provenance.json is owned by the training runner.")
        provenance = record | {
            "status": "complete" if stage == "all" else "intermediate",
            "artifacts": artifacts,
        }
        write_json(output / "provenance.json", provenance)
        record["artifacts"] = checked_files(output)
        write_json(run_dir / "training.json", record)
        if operation == "train" and stage == "all":
            install(solution, output)
        print(f"Training snapshot: {run_dir}")
        return output
    except BaseException as exc:
        (output / "provenance.json").unlink(missing_ok=True)
        write_json(
            run_dir / "training.json", record | {"status": "failed", "error": str(exc)}
        )
        raise


def adapt(solution, data, input_dir, *, run_dir, device="cpu", seed=0, limits, report=None):
    """Practice adaptation in this Python process; RAM and time are not sandboxed."""
    for name in ("wall_seconds", "memory_mb", "artifact_bytes"):
        if (
            type(limits.get(name)) not in (float, int)
            or not np.isfinite(limits[name])
            or limits[name] <= 0
        ):
            raise CourseError(f"Invalid adaptation limit: {name}")
    if type(limits.get("gpu_memory_mb")) is not int or limits["gpu_memory_mb"] < 0:
        raise CourseError("Invalid adaptation limit: gpu_memory_mb")
    return run(
        solution,
        data,
        fingerprint(checked_files(data)),
        seed,
        run_dir=run_dir,
        device=device,
        inputs={"base": input_dir},
        operation="adapt",
        limits=limits,
        report=report,
    )
