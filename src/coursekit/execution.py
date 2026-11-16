"""Explicit execution devices and a shared local artifact transaction lock."""

import platform
import tempfile
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from coursekit.common import CourseError, fingerprint


def resolve_device(device):
    if device == "cpu":
        return device
    import torch

    if not isinstance(device, str) or not device.startswith("cuda"):
        raise CourseError("Choose cpu or cuda[:index].")
    try:
        selected = torch.device(device)
        if selected.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this environment")
        index = selected.index if selected.index is not None else 0
        # An actual operation detects driver/build incompatibility, not only visibility.
        torch.ones(1, device=f"cuda:{index}").sum().item()
        return f"cuda:{index}"
    except (ValueError, RuntimeError, AssertionError) as exc:
        raise CourseError(f"Cannot use {device}: {exc}") from exc


def runtime_identity(device):
    import torch

    result = {"device": device, "torch": torch.__version__, "cuda": torch.version.cuda}
    result["python"] = platform.python_version()
    result["libraries"] = {
        name: version(name) for name in ("numpy", "scipy", "scikit-learn", "numba")
    }
    if device != "cpu":
        props = torch.cuda.get_device_properties(device)
        result.update(
            name=props.name,
            memory_bytes=props.total_memory,
            capability=[props.major, props.minor],
        )
    return result


@lru_cache(maxsize=128)
def _lock(path):
    return FileLock(
        str(Path(tempfile.gettempdir()) / f"coursekit-{fingerprint(path)}.lock")
    )


def solution_lock(solution):
    # Lock outside the solution: loading read-only submissions must not modify them.
    return _lock(str(Path(solution).resolve()))
