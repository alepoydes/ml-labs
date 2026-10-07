"""Local, guarded CPU kernels for the fixed-shape robot environment."""

import hashlib
import inspect
import os
import platform
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock


@lru_cache(maxsize=1)
def _revision():
    identity = (
        sys.version,
        torch.__version__,
        torch.__config__.show(),
        platform.system(),
        platform.machine(),
        torch.backends.cpu.get_cpu_capability(),
    )
    digest = hashlib.sha256(repr(identity).encode())
    for name in ("line_following.py", "line_following_env.py", "line_following_cache.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def ahead(fn, size):
    """Reuse a public AOT artifact, including guards, across local processes."""
    source = inspect.unwrap(fn)
    root = (
        Path(
            os.environ.get("XDG_CACHE_HOME")
            or os.environ.get("LOCALAPPDATA")
            or Path.home() / ".cache"
        )
        / "coursekit"
    )
    root = Path(os.environ.get("COURSEKIT_COMPILE_CACHE", root))
    path = root / _revision() / f"{source.__name__}-{size}-{torch.get_num_threads()}.pt"
    compiled = None

    def call(*args):
        nonlocal compiled
        if compiled is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock"):
                if path.is_file():
                    try:
                        with path.open("rb") as stream:
                            loaded = torch.compiler.load_compiled_function(
                                stream, f_globals=source.__globals__
                            )
                        result = loaded(*args)  # Validate guards before accepting a cache.
                    except Exception as exc:
                        warnings.warn(f"Rebuilding robot kernel cache: {exc}", stacklevel=2)
                    else:
                        compiled = loaded
                        return result
                compiled = fn.aot_compile((args, {}))
                temporary = path.with_suffix(".tmp")
                try:
                    compiled.save_compiled_function(temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        return compiled(*args)

    return call
