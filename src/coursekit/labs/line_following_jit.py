"""Configure the public Numba cache before constructing robot dispatchers.

Numba validates CPU compatibility and writes cache files atomically. Its cache
does not track changes in imported helpers, so the directory also identifies
the complete public engine. No student code or models enter this directory.
"""

import hashlib
import os
import sys
from pathlib import Path

import llvmlite
import numba
import numpy
import scipy

_sources = (
    "line_following_jit.py",
    "line_following_numba.py",
    "line_following_geometry.py",
    "line_following_rollout.py",
    "line_following.py",
    "line_following_course.py",
    "line_following_curve.py",
)
_digest = hashlib.sha256(
    repr(
        (
            sys.version,
            numba.__version__,
            llvmlite.__version__,
            numpy.__version__,
            scipy.__version__,
        )
    ).encode()
)
for _name in _sources:
    _digest.update(Path(__file__).with_name(_name).read_bytes())
_root = Path(
    os.environ.get("NUMBA_CACHE_DIR")
    or os.environ.get("COURSEKIT_COMPILE_CACHE")
    or Path(
        os.environ.get("XDG_CACHE_HOME")
        or os.environ.get("LOCALAPPDATA")
        or Path.home() / ".cache"
    )
    / "coursekit"
)
CACHE_DIR = _root / "numba" / _digest.hexdigest()


def njit(**options):
    """Construct a dispatcher with a versioned cache, without silent fallback.

    Numba normally falls back to a source-adjacent cache when its configured
    directory is unwritable. Such a fallback would bypass helper invalidation.
    Scope the documented cache settings to our dispatchers; student decorators
    keep their own cache configuration. The dispatcher retains its chosen path.
    """

    def decorate(function):
        previous = numba.config.CACHE_DIR, numba.config.CACHE_LOCATOR_CLASSES
        numba.config.CACHE_DIR = str(CACHE_DIR)
        numba.config.CACHE_LOCATOR_CLASSES = "UserProvidedCacheLocator"
        try:
            return numba.njit(**options)(function)
        finally:
            numba.config.CACHE_DIR, numba.config.CACHE_LOCATOR_CLASSES = previous

    return decorate
