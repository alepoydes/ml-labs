"""Portable local inspection records: JSON structure and compressed numeric arrays."""

import json
from pathlib import Path

import numpy as np

from coursekit.common import CourseError


def robot_metadata():
    """Identify the geometry and coverage rules used by both robot recorders."""
    from coursekit.labs import line_following_course as course

    return {
        "lab": "line_following",
        "version": 3,
        "coverage_version": course.COVERAGE_VERSION,
        "geometry": {
            "sensor_width": course.SENSOR_WIDTH,
            "sensor_depth": course.SENSOR_DEPTH,
            "line_width": course.LINE_WIDTH,
            "marker_length": course.MARKER_LENGTH,
            "ground": course.GROUND.tolist(),
            "ink": course.INK.tolist(),
            "start": course.START.tolist(),
            "end": course.END.tolist(),
        },
    }


def save(path, record):
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufU":
                raise CourseError("Record arrays must contain numbers or strings.")
            name = f"array_{len(arrays)}"
            arrays[name] = value
            return ["array", name]
        if isinstance(value, dict):
            return ["dict", {key: encode(item) for key, item in value.items()}]
        if isinstance(value, (list, tuple)):
            return ["list", [encode(item) for item in value]]
        if isinstance(value, np.generic):
            value = value.item()
        return ["value", value]

    tree = encode(record)
    arrays["metadata"] = np.array(json.dumps(tree, allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as arrays:
        used = {"metadata"}

        def decode(node):
            kind, value = node
            if kind == "array":
                if value == "metadata" or value in used:
                    raise CourseError("Invalid recording array reference.")
                used.add(value)
                array = arrays[value]
                if array.dtype.kind not in "biufU":
                    raise CourseError("Invalid recording array type.")
                return array.copy()
            if kind == "dict":
                return {key: decode(item) for key, item in value.items()}
            if kind == "list":
                return [decode(item) for item in value]
            if kind == "value" and (value is None or isinstance(value, (str, int, float))):
                return value
            raise CourseError("Invalid recording structure.")

        result = decode(json.loads(str(arrays["metadata"])))
        if used != set(arrays.files):
            raise CourseError("Unexpected recording arrays.")
        return result
