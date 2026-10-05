"""Ordered polyline courses, optical sampling, and conservative swept coverage.

Coordinates are metres. A course has one ordered centreline, including at crossings.
Smooth curves are supplied as finely sampled polylines; genuine corners are marked
separately for the reference-time calculation. Painting and judging use these same
segments. The judge never replaces a route coordinate with a global nearest point.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from coursekit.common import CourseError

LINE_WIDTH = 0.016
SENSOR_WIDTH = 0.128
SENSOR_DEPTH = 0.008
FOOTPRINT = np.array([0.070, 0.064])  # Body-frame half length and half width.
MARKER_LENGTH = 0.12
GROUND = np.array([0.94, 0.94, 0.94])
INK = np.array([0.06, 0.06, 0.06])
START = np.array([0.08, 0.25, 0.90])
END = np.array([0.90, 0.12, 0.08])


@dataclass(frozen=True)
class Course:
    name: str
    points: np.ndarray
    corners: np.ndarray | None = None
    arc: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        points = np.array(self.points, dtype=np.float64, copy=True)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 2 <= len(points) <= 8192
            or not np.isfinite(points).all()
            or not isinstance(self.name, str)
            or not self.name
        ):
            raise CourseError("Expected a named finite course with 2..8192 XY points.")
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if (lengths < 1e-7).any():
            raise CourseError("Consecutive course points must be distinct.")
        corners = (
            np.zeros(len(points), dtype=bool)
            if self.corners is None
            else np.array(self.corners, dtype=bool, copy=True)
        )
        if corners.shape != (len(points),) or corners[[0, -1]].any():
            raise CourseError("Corner flags must identify interior course vertices.")
        arc = np.r_[0.0, np.cumsum(lengths)]
        for name, value in (("points", points), ("corners", corners), ("arc", arc)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def length(self):
        return float(self.arc[-1])

    def at(self, distance):
        distance = np.clip(distance, 0, self.length)
        index = np.minimum(
            np.searchsorted(self.arc, distance, side="right") - 1, len(self.points) - 2
        )
        fraction = (distance - self.arc[index]) / np.diff(self.arc)[index]
        return (
            self.points[index] + fraction[..., None] * np.diff(self.points, axis=0)[index]
        )

    def reversed(self):
        return Course(self.name, self.points[::-1], self.corners[::-1])


def save_courses(path: Path, courses: list[Course]):
    if not courses or len({c.name for c in courses}) != len(courses):
        raise CourseError("Expected courses with unique names.")
    counts = np.array([len(c.points) for c in courses], dtype=np.int64)
    points = np.zeros((len(courses), int(counts.max()), 2))
    corners = np.zeros(points.shape[:2], dtype=bool)
    for row, course in enumerate(courses):
        points[row, : counts[row]] = course.points
        corners[row, : counts[row]] = course.corners
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=points,
        corners=corners,
        counts=counts,
        names=np.array([c.name for c in courses]),
    )


def load_courses(path: Path) -> list[Course]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"points", "corners", "counts", "names"}:
            raise CourseError("Unexpected course archive arrays.")
        points, corners, counts, names = (
            data[k] for k in ("points", "corners", "counts", "names")
        )
    if (
        points.ndim != 3
        or points.shape[2] != 2
        or corners.shape != points.shape[:2]
        or corners.dtype != np.bool_
        or counts.shape != (len(points),)
        or counts.dtype.kind not in "iu"
        or names.shape != counts.shape
        or names.dtype.kind != "U"
        or not len(points)
        or (counts < 2).any()
        or (counts > points.shape[1]).any()
        or not np.isfinite(points).all()
        or len(set(names)) != len(names)
    ):
        raise CourseError("Invalid course archive dimensions or values.")
    return [
        Course(str(name), row[:n], corner[:n])
        for name, row, corner, n in zip(names, points, corners, counts)
    ]


def colour(course: Course, points: np.ndarray) -> np.ndarray:
    """Paint a finite-width line and endpoint sections; average in the sensor."""
    shape = points.shape[:-1]
    points = np.asarray(points).reshape(-1, 2)
    starts, ends = course.points[:-1], course.points[1:]
    near = (
        (np.maximum(starts, ends) >= points.min(axis=0) - LINE_WIDTH / 2)
        & (np.minimum(starts, ends) <= points.max(axis=0) + LINE_WIDTH / 2)
    ).all(axis=1)
    result = np.broadcast_to(GROUND, (len(points), 3)).copy()
    if not near.any():
        return result.reshape(*shape, 3)
    delta = (ends - starts)[near]
    offset = points[:, None] - starts[near]
    fraction = np.clip(
        np.einsum("pni,ni->pn", offset, delta) / np.einsum("ni,ni->n", delta, delta), 0, 1
    )
    distance = np.linalg.norm(offset - fraction[..., None] * delta, axis=-1)
    index = distance.argmin(axis=1)
    rows = np.arange(len(points))
    painted = distance[rows, index] <= LINE_WIDTH / 2
    arc = (
        course.arc[:-1][near][index]
        + fraction[rows, index] * np.diff(course.arc)[near][index]
    )
    result[painted] = INK
    result[painted & (arc < MARKER_LENGTH)] = START
    result[painted & (arc > course.length - MARKER_LENGTH)] = END
    return result.reshape(*shape, 3)


def optical(course: Course, centre: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # 4 x 4 midpoint quadrature per sector, all in one transverse body-fixed row.
    offsets = (np.arange(4) + 0.5) / 4 - 0.5
    x = np.broadcast_to(offsets[None, :, None] * SENSOR_DEPTH, (16, 4, 4))
    y = (np.arange(16)[:, None, None] + 0.5 + offsets[None, None, :]) / 16 - 0.5
    local = np.stack((x, np.broadcast_to(y * SENSOR_WIDTH, x.shape)), axis=-1)
    world = centre[:2] + local @ rotation[:2, :2].T
    return colour(course, world).mean(axis=(1, 2)).astype(np.float32)


def _clip(course: Course, centre, axes, half, travel) -> np.ndarray:
    """Arc intervals inside a parallelogram translated along a straight segment.

    In local coordinates its sweep is a box plus a segment (a zonotope). Its
    bounding halfplanes have the two box normals and the travel normal.
    """
    inverse = np.linalg.inv(axes)
    motion = inverse @ travel
    normal = np.array([-motion[1], motion[0]]) / max(np.linalg.norm(motion), 1e-30)
    normals = np.array([[1.0, 0.0], [0.0, 1.0], normal])
    bounds = np.array(
        [half[0] + abs(motion[0]) / 2, half[1] + abs(motion[1]) / 2, np.abs(normal) @ half]
    )
    local = (course.points - centre - travel / 2) @ inverse.T
    a, d = local[:-1] @ normals.T, np.diff(local, axis=0) @ normals.T
    nonzero = np.abs(d) > 1e-15
    divisor = np.where(nonzero, d, 1)
    first, second = (-bounds - a) / divisor, (bounds - a) / divisor
    lower = np.maximum(0, np.where(nonzero, np.minimum(first, second), -np.inf).max(axis=1))
    upper = np.minimum(1, np.where(nonzero, np.maximum(first, second), np.inf).min(axis=1))
    valid = (lower <= upper) & ((np.abs(a) <= bounds) | nonzero).all(axis=1)
    length = np.diff(course.arc)
    return np.column_stack(
        (course.arc[:-1] + lower * length, course.arc[:-1] + upper * length)
    )[valid]


def swept_intervals(course: Course, q0, q1, r0, r1) -> np.ndarray:
    """Certified inner coverage of one step; inputs use the dynamics pose map.

    Axle position and Euler coordinates interpolate linearly during a step.
    Rotation changes any footprint point by at most radius * sum(abs(dangles)).
    Eroding the initial projected footprint by this bound leaves a fixed shape
    inside every translated intermediate footprint. Its translational sweep is
    exact. Include both endpoint footprints, which require no erosion.
    """
    translation = np.asarray(q1[:2]) - q0[:2]
    axes = r0[:2, :2]
    delta = np.linalg.norm(FOOTPRINT) * np.abs(q1[3:6] - q0[3:6]).sum()
    half = FOOTPRINT - delta * np.linalg.norm(np.linalg.inv(axes), axis=1)
    intervals = [
        _clip(course, q0[:2], axes, FOOTPRINT, np.zeros(2)),
        _clip(course, q1[:2], r1[:2, :2], FOOTPRINT, np.zeros(2)),
    ]
    if (half > 0).all():
        intervals.append(_clip(course, q0[:2], axes, half, translation))
    return np.concatenate(intervals)


def advance(course: Course, frontier: float, q0, q1, r0, r1) -> float:
    """Extend only the connected covered interval beginning at the old frontier."""
    intervals = swept_intervals(course, q0, q1, r0, r1)
    if not len(intervals):
        return frontier
    intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
    # Only roundoff-sized gaps may be joined, never gaps on the scale of the line.
    tolerance = 32 * np.finfo(np.float64).eps * max(1, course.length)
    for begin, end in intervals:
        if begin > frontier + tolerance:
            break
        frontier = max(frontier, float(end))
    return min(frontier, course.length)


def reference_time(course: Course, model) -> float:
    """Optimistic centreline speed profile for out-and-back travel in seconds.

    Independent traction/drive/yaw envelopes omit sensing and balancing costs.
    The far endpoint has zero translational speed for direction reversal; body
    rotation there is unnecessary. Explicit corners stop and turn in place.
    """
    mass = float(model.mass.sum())
    radius = model.radius[:2].detach().cpu().numpy()
    torque = model.stall.detach().cpu().numpy()
    vmax = float(np.min(radius * model.speed.detach().cpu().numpy()))
    traction = float(model.friction[:2].min() * model.gravity)
    acceleration = min(traction, float((torque / radius).sum()) / mass)
    half_track = float(model.offset[1:3, 1].abs().mean())
    inertia = float(
        model.inertia[:3, 2].sum()
        + (model.mass * model.offset[:, :2].square().sum(dim=1)).sum()
    )
    alpha = (
        min(float((torque / radius).sum()) * half_track, mass * traction * half_track)
        / inertia
    )
    omega = vmax / half_track
    counts = np.maximum(2, np.ceil(np.diff(course.arc) / 0.02).astype(int))
    distance = np.concatenate(
        [
            np.linspace(a, b, n, endpoint=False)
            for a, b, n in zip(course.arc[:-1], course.arc[1:], counts)
        ]
        + [np.array([course.length])]
    )
    points = course.at(distance)
    lengths = np.diff(distance)
    direction = np.arctan2(np.diff(points, axis=0)[:, 1], np.diff(points, axis=0)[:, 0])
    angles = np.arctan2(np.sin(np.diff(direction)), np.cos(np.diff(direction)))
    curvature = np.r_[0.0, angles / ((lengths[:-1] + lengths[1:]) / 2), 0.0]
    corner_index = np.cumsum(counts)[:-1][course.corners[1:-1]]
    curvature[corner_index] = 0
    cap = np.minimum(
        vmax / (1 + half_track * np.abs(curvature)),
        np.sqrt(traction / np.maximum(np.abs(curvature), 1e-12)),
    )
    # Rotation acceleration includes the demand from changing curvature at speed.
    change = np.abs(np.gradient(curvature, distance))
    cap = np.minimum(cap, np.sqrt(alpha / np.maximum(change, 1e-12)))
    cap[corner_index] = 0
    turning = 0.0
    for angle in np.abs(angles[corner_index - 1]):
        angle = min(angle, abs(np.pi - angle))  # Either drive direction is legal.
        turning += (
            2 * np.sqrt(angle / alpha)
            if angle <= omega**2 / alpha
            else angle / omega + omega / alpha
        )

    def leg(reverse, stop):
        speed = cap[::-1].copy() if reverse else cap.copy()
        ds = lengths[::-1] if reverse else lengths
        k = curvature[::-1] if reverse else curvature
        speed[0] = 0
        if stop:
            speed[-1] = 0
        for i, length in enumerate(ds):
            a = min(acceleration, alpha / max(abs(k[i]), abs(k[i + 1]), 1e-12))
            speed[i + 1] = min(speed[i + 1], np.sqrt(speed[i] ** 2 + 2 * a * length))
        for i in range(len(ds) - 1, -1, -1):
            a = min(acceleration, alpha / max(abs(k[i]), abs(k[i + 1]), 1e-12))
            speed[i] = min(speed[i], np.sqrt(speed[i + 1] ** 2 + 2 * a * ds[i]))
        return float(np.sum(2 * ds / (speed[:-1] + speed[1:])))

    return leg(False, True) + leg(True, False) + 2 * turning
