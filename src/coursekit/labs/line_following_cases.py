"""Fixed experiments: spline, physical robot, initial state and sensor noise.

Randomisation belongs to sample_case(), used when preparing additional examples.
An Environment only copies a Case; restarting it never samples another robot.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from coursekit.common import CourseError, fingerprint
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_curve import Course, iter_courses

CASE_VERSION = 1
SENSOR_NAMES = ("rgb", "gyro", "accel", "encoders")
NOISE_STD = (0.008, 0.004, 0.025, 0.0005)


@dataclass(frozen=True)
class Case:
    name: str
    course: Course
    robot: physics.Robot
    initial: physics.State
    noise_state: str  # JSON PCG64 state immediately before the first observation.
    noise_std: tuple = NOISE_STD

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > 128:
            raise CourseError("A case needs a nonempty name of at most 128 characters.")
        if not isinstance(self.course, Course) or self.course.length < 0.3:
            raise CourseError("A case needs a course at least 0.3 m long.")
        if not isinstance(self.robot, physics.Robot) or not isinstance(
            self.initial, physics.State
        ):
            raise CourseError("A case needs Robot and State values.")
        for name, value, template in zip(self.robot._fields, self.robot, physics.robot()):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != template.shape
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
                or name != "offset"
                and (value < 0).any()
                or name
                in {"mass", "inertia", "radius", "stiffness", "stall", "speed", "gravity"}
                and (value <= 0).any()
            ):
                raise CourseError(f"Invalid case robot field {name}.")
        for name, shape in (("q", (8,)), ("p", (8,)), ("spin", (3,))):
            value = getattr(self.initial, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
            ):
                raise CourseError(f"Invalid initial state field {name}.")
        if (
            not isinstance(self.initial.fallen, torch.Tensor)
            or self.initial.fallen.shape != ()
            or self.initial.fallen.device.type != "cpu"
            or self.initial.fallen.dtype != torch.bool
            or self.initial.fallen
        ):
            raise CourseError("A case must start with a nonfallen robot.")
        std = np.asarray(self.noise_std)
        if (
            std.shape != (4,)
            or std.dtype.kind not in "fi"
            or not np.isfinite(std).all()
            or (std < 0).any()
        ):
            raise CourseError("Expected four finite, nonnegative noise deviations.")
        self.noise_generator()  # Reject malformed RNG descriptions at the boundary.

    def noise_generator(self):
        try:
            state = json.loads(self.noise_state)
            if state["bit_generator"] != "PCG64":
                raise ValueError("Expected PCG64")
            generator = np.random.Generator(np.random.PCG64(0))
            generator.bit_generator.state = state
            return generator
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise CourseError("Invalid recorded sensor-noise state.") from exc


def sample_robot(rng: np.random.Generator) -> physics.Robot:
    """Public parameter family; called only while constructing new examples."""
    model = physics.robot()
    scale = torch.from_numpy(rng.uniform(0.85, 1.15, 4))
    height = rng.uniform(0.9, 1.1)
    inertia = model.inertia * scale[:, None]
    inertia[0] *= height**2
    offset = model.offset.clone()
    offset[0, 2] *= height
    return model._replace(
        mass=model.mass * scale,
        inertia=inertia,
        offset=offset,
        friction=torch.from_numpy(
            np.r_[rng.uniform(0.65, 0.95, 2), rng.uniform(0.45, 0.75)]
        ),
        stall=torch.from_numpy(rng.uniform(0.065, 0.095, 2)),
        speed=torch.from_numpy(rng.uniform(38.0, 52.0, 2)),
        ball_drag=torch.tensor(rng.uniform(0.00015, 0.00025), dtype=torch.float64),
    )


def sample_case(course: Course, seed: int, *, name=None) -> Case:
    """Prepare an explicit extra training example; seed is not an episode input.

    The draw order preserves the original pilot's episode-zero robots and noise.
    Serialised cases store the resulting numbers and RNG state, not this recipe.
    """
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed < 2**63
    ):
        raise CourseError("The generation seed must be a nonnegative 63-bit integer.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    model = sample_robot(rng)
    state = physics.initial_state(model)
    state.q[:2] = torch.from_numpy(course.at(0.0).copy())
    direction = course.at(0.0, derivative=1)
    state.q[5] = float(np.arctan2(direction[1], direction[0]))
    return Case(
        name or course.name, course, model, state, json.dumps(rng.bit_generator.state)
    )


def case_identity(case: Case):
    return fingerprint(
        {
            "version": CASE_VERSION,
            "name": case.name,
            "course": {
                k: getattr(case.course, k).tolist()
                for k in ("controls", "knots", "corners")
            },
            "robot": {k: v.tolist() for k, v in case.robot._asdict().items()},
            "initial": {k: v.tolist() for k, v in case.initial._asdict().items()},
            "noise_state": json.loads(case.noise_state),
            "noise_std": list(case.noise_std),
        }
    )


def save_cases(path: Path, cases: list[Case]):
    if not cases or len({c.name for c in cases}) != len(cases):
        raise CourseError("Expected nonempty cases with unique names.")
    arrays = {
        "version": np.array(CASE_VERSION),
        "names": np.array([c.name for c in cases]),
        "courses": np.array([c.course.name for c in cases]),
        "noise_state": np.array([c.noise_state for c in cases]),
        "noise_std": np.array([c.noise_std for c in cases], dtype=np.float64),
    }
    for group in ("robot", "initial"):
        for name in getattr(cases[0], group)._fields:
            arrays[f"{group}_{name}"] = np.stack(
                [getattr(getattr(c, group), name).numpy() for c in cases]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_split(path: Path, cases: list[Case]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case"])
        writer.writerows([case.name] for case in cases)


def load_cases(data: Path, split=None) -> list[Case]:
    return list(iter_cases(data, split))


def iter_cases(data: Path, split=None):
    """Read the documented NPZ/CSV schema; build geometry incrementally for GUIs."""
    expected = {"version", "names", "courses", "noise_state", "noise_std"}
    expected |= {f"robot_{k}" for k in physics.Robot._fields}
    expected |= {f"initial_{k}" for k in physics.State._fields}
    with np.load(data / "cases.npz", allow_pickle=False) as archive:
        if (
            set(archive.files) != expected
            or archive["version"].shape != ()
            or archive["version"].dtype.kind not in "iu"
            or archive["version"].item() != CASE_VERSION
        ):
            raise CourseError("Expected fixed-case format 1; prepare updated resources.")
        arrays = {k: archive[k] for k in expected - {"version"}}
    names = arrays["names"]
    if (
        names.ndim != 1
        or not len(names)
        or names.dtype.kind != "U"
        or len(set(names)) != len(names)
    ):
        raise CourseError("Invalid case identifiers.")
    for key in ("courses", "noise_state"):
        if arrays[key].shape != names.shape or arrays[key].dtype.kind != "U":
            raise CourseError(f"Invalid case {key}.")
    templates = {f"robot_{k}": v for k, v in physics.robot()._asdict().items()}
    templates.update(
        initial_q=torch.zeros(8),
        initial_p=torch.zeros(8),
        initial_spin=torch.zeros(3),
        initial_fallen=torch.tensor(False),
    )
    for key, template in templates.items():
        value = arrays[key]
        kind = "b" if key == "initial_fallen" else "f"
        if (
            value.shape != (len(names), *template.shape)
            or value.dtype.kind != kind
            or not np.isfinite(value).all()
        ):
            raise CourseError(f"Invalid case array {key}.")
    if (
        arrays["noise_std"].shape != (len(names), 4)
        or arrays["noise_std"].dtype.kind != "f"
    ):
        raise CourseError("Invalid sensor-noise array.")
    selection = names.tolist()
    if split is not None:
        if split not in {"train", "validation", "hidden", "diagnostic"}:
            raise CourseError("Unknown episode split.")
        with (data / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        if not rows or rows[0] != ["case"] or any(len(r) != 1 for r in rows[1:]):
            raise CourseError("Expected episode CSV header: case.")
        selection = [r[0] for r in rows[1:]]
        if (
            not selection
            or len(set(selection)) != len(selection)
            or not set(selection) <= set(names)
        ):
            raise CourseError("Expected distinct, known case identifiers.")
    indices = {name: i for i, name in enumerate(names)}
    geometry, source = {}, iter_courses(data / "courses.npz")
    for name in selection:
        i = indices[name]
        course_name = str(arrays["courses"][i])
        while course_name not in geometry:
            course = next(source, None)
            if course is None:
                raise CourseError(f"Unknown course in case {name}.")
            geometry[course.name] = course
        yield Case(
            name,
            geometry[course_name],
            physics.Robot(
                *(
                    torch.from_numpy(np.array(arrays[f"robot_{k}"][i], dtype=np.float64))
                    for k in physics.Robot._fields
                )
            ),
            physics.State(
                *(
                    torch.from_numpy(
                        np.array(
                            arrays[f"initial_{k}"][i],
                            dtype=bool if k == "fallen" else np.float64,
                        )
                    )
                    for k in physics.State._fields
                )
            ),
            str(arrays["noise_state"][i]),
            tuple(arrays["noise_std"][i]),
        )
