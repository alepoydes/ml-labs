import importlib.machinery
import importlib.metadata
import importlib.util
import json
import platform
import shutil
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
from coursekit.labs import require_lab
from coursekit.submissions import metadata


def new_solution(project: Path, lab: str, template="starter", variant=None) -> Path:
    """Copy a template for CLI or GUI use; never overwrite a working solution."""
    require_lab(lab)
    if template not in {"starter", "baseline"}:
        raise CourseError("Choose the starter or baseline template.")
    source = project / "labs" / lab / template
    variant = template if variant is None else variant
    raw = (
        (source / "submission.toml")
        .read_text()
        .replace('variant = "template"', f"variant = {json.dumps(variant)}")
    )
    metadata(raw.encode())
    destination = project / "work" / lab / variant
    if destination.exists():
        raise CourseError(f"Solution already exists: {destination}")
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (destination / "submission.toml").write_text(raw)
    return destination


class _SourceLoader(importlib.machinery.SourceFileLoader):
    """Read current source even if stale bytecode exists on a read-only mount."""

    def get_code(self, fullname):
        return self.source_to_code(self.get_data(self.path), self.path)


def load_module(solution: Path, name: str):
    """Import one of your working copy's Python files in the current process."""
    solution = solution.resolve()
    if not (solution / f"{name}.py").is_file():
        raise CourseError(
            f"Missing {name}.py in {solution}. Choose a working solution or create one with step_1_new_solution."
        )
    # A notebook can load another variant or rerun after editing its helpers.
    for key in list(sys.modules):
        if key in {"train", "solution", "lib"} or key.startswith("lib."):
            del sys.modules[key]
    # Timestamp-based bytecode can miss same-size edits within one second.
    # A source-only finder also covers imports performed later by the controller;
    # it never modifies the submission and leaves external library loaders alone.
    directories = {solution, *(p.parent for p in (solution / "lib").rglob("*.py"))}
    for directory in directories:
        sys.path_importer_cache[str(directory)] = importlib.machinery.FileFinder(
            str(directory), (_SourceLoader, importlib.machinery.SOURCE_SUFFIXES)
        )
    importlib.invalidate_caches()
    if str(solution) in sys.path:
        sys.path.remove(str(solution))
    sys.path.insert(0, str(solution))
    path = solution / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        name, path, loader=_SourceLoader(name, str(path))
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_solution(solution: Path, seed: int = 0):
    """Create your solution directly, with its saved artifacts and configuration."""
    solution = solution.resolve()
    module = load_module(solution, "solution")
    return module.Solution(
        solution / "artifacts", read_toml(solution / "config.toml"), seed
    )


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
    """Read saved diagnostics; source edits do not prevent exploring a saved model."""
    if not (run_dir / "training.json").is_file():
        raise CourseError(
            "No saved training result. Run step_2_train or uncomment train_local(...) "
            "in the notebook's training cell."
        )
    saved = json.loads((run_dir / "training.json").read_text())
    artifacts = file_hashes(solution / "artifacts")
    artifacts.pop("provenance.json", None)
    if (
        saved["provenance"]
        != json.loads((solution / "artifacts/provenance.json").read_text())
        or saved["artifacts"] != artifacts
        or saved["history_hash"] != digest(run_dir / "training.npz")
    ):
        raise CourseError("Saved training history/model changed; rerun step_2_train.")
    if saved["provenance"]["source_hash"] != source_hash(solution) or saved["provenance"][
        "config_hash"
    ] != digest(solution / "config.toml"):
        print(
            "Source or configuration changed since training. Analysis uses the saved "
            "model and training settings; retrain before packaging."
        )
    with np.load(run_dir / "training.npz", allow_pickle=False) as source:
        history = dict(source)
    return history, saved
