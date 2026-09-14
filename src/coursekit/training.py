import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

from coursekit.common import CourseError, digest, fingerprint, read_toml, write_json
from coursekit.submissions import metadata


def load_module(solution: Path, name: str):
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


def train(solution: Path, data: Path, resource_hash: str, seed: int) -> None:
    solution = solution.resolve()
    info = metadata((solution / "submission.toml").read_bytes())
    output = solution / "artifacts"
    output.mkdir(exist_ok=True)
    provenance = output / "provenance.json"
    provenance.unlink(missing_ok=True)
    sources, config_hash = source_hash(solution), digest(solution / "config.toml")
    module = load_module(solution, "train")
    module.train(data.resolve(), output, read_toml(solution / "config.toml"), seed)
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
    write_json(
        provenance,
        {
            "status": "complete",
            "lab": info["lab"],
            "api_version": info["api_version"],
            "resource_hash": resource_hash,
            "source_hash": sources,
            "config_hash": config_hash,
            "seed": seed,
            "python": platform.python_version(),
            "libraries": versions,
        },
    )
    print(f"Training complete: {output}")
