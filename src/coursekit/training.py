import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
from filelock import FileLock

from coursekit import resources
from coursekit.common import (
    CourseError,
    digest,
    file_hashes,
    fingerprint,
    read_toml,
    write_json,
)
from coursekit.submissions import metadata


def load_module(solution: Path, name: str):
    # A notebook can load another variant or rerun after editing its helpers.
    for key in list(sys.modules):
        if key in {"train", "solution", "lib"} or key.startswith("lib."):
            del sys.modules[key]
    importlib.invalidate_caches()
    if str(solution) in sys.path:
        sys.path.remove(str(solution))
    sys.path.insert(0, str(solution))
    spec = importlib.util.spec_from_file_location(name, solution / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_hash(solution: Path) -> str:
    paths = [solution / "train.py", solution / "solution.py"]
    paths += sorted((solution / "lib").rglob("*.py"))
    return fingerprint({p.relative_to(solution).as_posix(): digest(p) for p in paths})


def check_provenance(solution: Path) -> dict:
    path = solution / "artifacts/provenance.json"
    if not path.is_file():
        raise CourseError("Missing training provenance; run step_2_train.")
    value = json.loads(path.read_text())
    if (
        value.get("status") != "complete"
        or value.get("source_hash") != source_hash(solution)
        or value.get("config_hash") != digest(solution / "config.toml")
    ):
        raise CourseError(
            "Training is missing or source/configuration changed; rerun step_2_train."
        )
    return value


def train(
    solution: Path,
    data: Path,
    resource_hash: str,
    seed: int,
    *,
    run_dir: Path | None = None,
) -> None:
    solution = solution.resolve()
    info = metadata((solution / "submission.toml").read_bytes())
    output = solution / "artifacts"
    output.mkdir(exist_ok=True)
    provenance = output / "provenance.json"
    provenance.unlink(missing_ok=True)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "training.json").unlink(missing_ok=True)
    sources, config_hash = source_hash(solution), digest(solution / "config.toml")
    config = read_toml(solution / "config.toml")
    module = load_module(solution, "train")
    history = module.train(data.resolve(), output, config, seed)
    if sources != source_hash(solution) or config_hash != digest(solution / "config.toml"):
        raise CourseError(
            "Source or configuration changed during training; rerun training."
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "scipy",
            "scikit-learn",
            "joblib",
            "pillow",
            "torch",
            "coursekit",
        )
    }
    record = {
        "status": "complete",
        "lab": info["lab"],
        "api_version": info["api_version"],
        "resource_hash": resource_hash,
        "source_hash": sources,
        "config_hash": config_hash,
        "seed": seed,
        "python": platform.python_version(),
        "libraries": versions,
    }
    if run_dir is not None:
        history = {} if history is None else history
        if not isinstance(history, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, np.ndarray)
            or value.dtype.kind not in "biufc"
            or not np.isfinite(value).all()
            for key, value in history.items()
        ):
            raise CourseError("Training history must contain finite numeric NumPy arrays.")
        np.savez_compressed(run_dir / "training.npz", **history)
        write_json(
            run_dir / "training.json",
            {
                "provenance": record,
                "config": config,
                "artifacts": file_hashes(output),
                "history_hash": digest(run_dir / "training.npz"),
            },
        )
        print(f"Training history: {run_dir / 'training.npz'}")
    write_json(provenance, record)
    print(f"Training complete: {output}")


def train_local(solution: Path, project: Path, seed: int = 0) -> None:
    """The shared CLI/notebook entry point for training your own working copy."""
    info = metadata((solution / "submission.toml").read_bytes())
    data = project / "data" / info["lab"]
    spec = resources.manifest(project / "labs" / info["lab"] / "resources.toml")
    with FileLock(str(data / ".update.lock"), timeout=0):
        resource_hash = resources.ready(data, spec)
        train(
            solution,
            data,
            resource_hash,
            seed,
            run_dir=project / "runs" / info["lab"] / info["variant"],
        )


def load_training(solution: Path, run_dir: Path):
    """Read saved diagnostics, checking that they describe the current model."""
    if not (run_dir / "training.json").is_file():
        raise CourseError(
            "No saved training result. Run step_2_train or uncomment train_local(...) "
            "in the notebook's training cell."
        )
    saved = json.loads((run_dir / "training.json").read_text())
    artifacts = file_hashes(solution / "artifacts")
    artifacts.pop("provenance.json", None)
    if (
        saved["provenance"] != check_provenance(solution)
        or saved["artifacts"] != artifacts
        or saved["history_hash"] != digest(run_dir / "training.npz")
    ):
        raise CourseError("Saved training history/model changed; rerun step_2_train.")
    with np.load(run_dir / "training.npz", allow_pickle=False) as source:
        history = dict(source)
    return history, saved
