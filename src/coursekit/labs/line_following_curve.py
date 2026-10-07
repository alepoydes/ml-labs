"""Cubic Bezier courses in metres, with analytic derivatives and arc quadrature.

Control points and parameter knots are authoritative. Cached chords carry a
uniform position-error bound; they never replace the underlying curve.
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from coursekit.common import CourseError

GEOMETRY_TOLERANCE = 1e-5
FORMAT_VERSION = 2
_GAUSS = {n: ((x + 1) / 2, w / 2) for n in (8, 16) for x, w in [leggauss(n)]}


def bezier(control, t, derivative=0):
    """Evaluate a cubic or its local-parameter derivative by de Casteljau."""
    control = np.asarray(control)
    for degree in range(3, 3 - derivative, -1):
        control = degree * np.diff(control, axis=-2)
    t = np.asarray(t)[..., None, None]
    for _ in range(3 - derivative):
        control = (1 - t) * control[..., :-1, :] + t * control[..., 1:, :]
    return control[..., 0, :]


def split(control):
    a = (control[..., :-1, :] + control[..., 1:, :]) / 2
    b = (a[..., :-1, :] + a[..., 1:, :]) / 2
    middle = b.mean(axis=-2)
    return (
        np.stack((control[..., 0, :], a[..., 0, :], b[..., 0, :], middle), axis=-2),
        np.stack((middle, b[..., 1, :], a[..., 2, :], control[..., 3, :]), axis=-2),
    )


def integral(control, t=1.0, order=16):
    nodes, weights = _GAUSS[order]
    t = np.asarray(t)
    speed = np.linalg.norm(
        bezier(control[..., None, :, :], t[..., None] * nodes, 1), axis=-1
    )
    return t * (speed @ weights)


def curvature(control, t):
    v, a, j = (bezier(control, t, d) for d in (1, 2, 3))
    square = (v * v).sum(axis=-1)
    cross = v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]
    curvature = cross / square**1.5
    change = (v[..., 0] * j[..., 1] - v[..., 1] * j[..., 0]) / square**2
    change -= 3 * cross * (v * a).sum(axis=-1) / square**3
    return curvature, change


@dataclass(frozen=True, eq=False)
class Course:
    name: str
    controls: np.ndarray
    knots: np.ndarray | None = None
    corners: np.ndarray | None = None
    arc: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        controls = np.array(self.controls, dtype=np.float64, copy=True)
        if (
            controls.ndim != 3
            or controls.shape[1:] != (4, 2)
            or not 1 <= len(controls) <= 2048
            or not np.isfinite(controls).all()
            or not isinstance(self.name, str)
            or not self.name
        ):
            raise CourseError("Expected a named spline with 1..2048 cubic XY segments.")
        n = len(controls)
        knots = np.array(
            np.arange(n + 1) if self.knots is None else self.knots, dtype=float
        )
        corners = np.array(
            np.zeros(n + 1, dtype=bool) if self.corners is None else self.corners,
            copy=True,
        )
        if (
            knots.shape != (n + 1,)
            or not np.isfinite(knots).all()
            or knots[0] != 0
            or (np.diff(knots) <= 0).any()
            or corners.shape != (n + 1,)
            or corners.dtype != np.bool_
            or corners[[0, -1]].any()
        ):
            raise CourseError("Invalid spline parameter knots or interior corner flags.")
        spans = np.diff(knots)
        for order in range(3):
            end = bezier(controls[:-1], 1.0, order) / spans[:-1, None] ** order
            start = bezier(controls[1:], 0.0, order) / spans[1:, None] ** order
            smooth = np.ones(n - 1, dtype=bool) if order == 0 else ~corners[1:-1]
            if not np.allclose(end[smooth], start[smooth], rtol=1e-8, atol=1e-9):
                raise CourseError(
                    "Spline joins must be continuous, and C2 outside marked corners."
                )
        # The squared speed is quartic. Check every stationary value, not a sample grid.
        for control, span in zip(controls, spans):
            v = 3 * np.diff(control, axis=0)
            power = np.array([v[0], 2 * (v[1] - v[0]), v[2] - 2 * v[1] + v[0]])
            square = sum(np.convolve(p, p) for p in power.T)
            roots = np.polynomial.polynomial.polyroots(np.arange(1, 5) * square[1:])
            t = np.r_[
                0.0,
                1.0,
                roots.real[(abs(roots.imag) < 1e-9) & (roots.real > 0) & (roots.real < 1)],
            ]
            if np.min(np.linalg.norm(bezier(control, t, 1), axis=-1)) / span < 1e-8:
                raise CourseError("Spline tangents must not vanish.")
        for name, value in (("controls", controls), ("knots", knots), ("corners", corners)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        self._prepare()

    @classmethod
    def polyline(cls, name, points, corners=None):
        """Explicit straight-segment fixture; bends must be marked as corners."""
        points = np.asarray(points, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
        ):
            raise CourseError("Polyline vertices must be distinct XY points.")
        delta = np.diff(points, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        if (lengths < 1e-7).any():
            raise CourseError("Polyline vertices must be distinct XY points.")
        if corners is None:
            tangent = delta / lengths[:, None]
            corners = np.r_[
                False, np.linalg.norm(np.diff(tangent, axis=0), axis=1) > 1e-9, False
            ]
        controls = points[:-1, None] + np.arange(4)[None, :, None] / 3 * delta[:, None]
        return cls(
            name, controls, np.r_[0.0, np.cumsum(lengths)], np.asarray(corners, dtype=bool)
        )

    def _prepare(self):
        leaves = []
        pending = [(i, c, 0.0, 1.0, 0) for i, c in reversed(list(enumerate(self.controls)))]
        while pending:
            segment, c, a, b, depth = pending.pop()
            chord = c[0] + np.arange(4)[:, None] / 3 * (c[-1] - c[0])
            error = float(np.linalg.norm(c - chord, axis=1).max())
            length = float(integral(c))
            if error <= GEOMETRY_TOLERANCE and abs(
                length - integral(c, order=8)
            ) <= 1e-12 * max(1, length):
                leaves.append((segment, c, a, b, error, length))
            else:
                if depth >= 24 or len(leaves) + len(pending) > 262144:
                    raise CourseError(
                        "Spline subdivision did not meet geometry/quadrature tolerances."
                    )
                left, right = split(c)
                middle = (a + b) / 2
                pending.extend(
                    (
                        (segment, right, middle, b, depth + 1),
                        (segment, left, a, middle, depth + 1),
                    )
                )
        segment, control, a, b, error, length = map(np.array, zip(*leaves))
        values = {
            "_segments": segment,
            "_leaves": control,
            "_lower": control.min(axis=1),
            "_upper": control.max(axis=1),
            "_a": a,
            "_b": b,
            "_error": error,
            "_arc": np.r_[0.0, np.cumsum(length)],
        }
        # Use the same cumulative sums at shared boundaries to avoid coordinate gaps.
        boundaries = np.r_[np.flatnonzero(np.r_[True, np.diff(segment) != 0]), len(segment)]
        values["arc"] = values["_arc"][boundaries]
        for name, value in values.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def length(self):
        return float(self.arc[-1])

    @property
    def bounds(self):
        return self.controls.min(axis=(0, 1)), self.controls.max(axis=(0, 1))

    def _locate(self, distance):
        distance = np.clip(distance, 0, self.length)
        leaf = np.minimum(
            np.searchsorted(self._arc, distance, side="right") - 1, len(self._leaves) - 1
        )
        target = distance - self._arc[leaf]
        t = target / np.diff(self._arc)[leaf]
        lower, upper = np.zeros_like(t), np.ones_like(t)
        for _ in range(32):
            residual = integral(self._leaves[leaf], t) - target
            done = abs(residual) <= 1e-12
            if np.all(done):
                break
            lower = np.where(residual < 0, t, lower)
            upper = np.where(residual > 0, t, upper)
            speed = np.linalg.norm(bezier(self._leaves[leaf], t, 1), axis=-1)
            trial = t - residual / speed
            trial = np.where((trial > lower) & (trial < upper), trial, (lower + upper) / 2)
            t = np.where(done, t, trial)
        else:
            raise CourseError("Spline arc inversion did not converge.")
        return self._segments[leaf], self._a[leaf] + t * (self._b[leaf] - self._a[leaf])

    def at(self, distance, derivative=0):
        segment, t = self._locate(distance)
        return (
            bezier(self.controls[segment], t, derivative)
            / np.diff(self.knots)[segment][..., None] ** derivative
        )

    def geometry(self, distance):
        segment, t = self._locate(distance)
        return curvature(self.controls[segment], t)

    def mesh(self):
        return np.concatenate((self._leaves[:, 0], self._leaves[-1:, -1])), self._arc

    def distance_on_leaf(self, leaf, t):
        leaf, t = np.asarray(leaf), np.asarray(t)
        value = self._arc[leaf] + np.diff(self._arc)[leaf] * t
        # Affine spans integrate exactly; only curved spans need quadrature here.
        curved = self._error[leaf] > 32 * np.finfo(float).eps
        if np.any(curved):
            value = np.array(value, copy=True)
            value[curved] = self._arc[leaf[curved]] + integral(
                self._leaves[leaf[curved]], t[curved]
            )
        return np.where(
            t == 0, self._arc[leaf], np.where(t == 1, self._arc[leaf + 1], value)
        )

    def reversed(self):
        return Course(
            self.name,
            self.controls[::-1, ::-1],
            self.knots[-1] - self.knots[::-1],
            self.corners[::-1],
        )

    def profile(self, spacing):
        if not np.isfinite(spacing) or spacing <= 0:
            raise CourseError("Reference spacing must be positive and finite.")
        boundaries = np.r_[0.0, self.arc[self.corners], self.length]
        counts = np.maximum(2, np.ceil(np.diff(boundaries) / spacing).astype(int))
        distance = np.concatenate(
            [
                np.linspace(a, b, n, endpoint=False)
                for a, b, n in zip(boundaries[:-1], boundaries[1:], counts)
            ]
            + [np.array([self.length])]
        )
        # Include actual changes of cubic, not artificial exact subdivisions.
        third = (
            bezier(self.controls, 0.0, 3)
            * (self.knots[-1] / self.length / np.diff(self.knots))[:, None] ** 3
        )
        joins = (
            np.flatnonzero(
                ~np.isclose(third[:-1], third[1:], rtol=1e-8, atol=1e-8).all(axis=1)
            )
            + 1
        )
        distance = np.unique(np.r_[distance, self.arc[joins]])
        k, change = self.geometry(distance)
        left_k, left_change = curvature(self.controls[joins - 1], 1.0)
        right_k, right_change = curvature(self.controls[joins], 0.0)
        at_join = np.searchsorted(distance, self.arc[joins])
        k[at_join] = np.maximum(abs(left_k), abs(right_k))
        change[at_join] = np.maximum(abs(left_change), abs(right_change))
        indices = np.searchsorted(distance, self.arc[self.corners])
        return distance, k, change, indices

    @cached_property
    def reference_profile(self):
        return self.profile(0.001)


def save_courses(path: Path, courses: list[Course]):
    if not courses or len({c.name for c in courses}) != len(courses):
        raise CourseError("Expected courses with unique names.")
    counts = np.array([len(c.controls) for c in courses], dtype=np.int64)
    controls = np.zeros((len(courses), int(counts.max()), 4, 2))
    knots = np.zeros((len(courses), int(counts.max()) + 1))
    corners = np.zeros(knots.shape, dtype=bool)
    for i, course in enumerate(courses):
        controls[i, : counts[i]] = course.controls
        knots[i, : counts[i] + 1] = course.knots
        corners[i, : counts[i] + 1] = course.corners
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=FORMAT_VERSION,
        controls=controls,
        knots=knots,
        corners=corners,
        counts=counts,
        names=np.array([c.name for c in courses]),
    )


def load_courses(path: Path) -> list[Course]:
    return list(iter_courses(path))


def iter_courses(path: Path):
    """Validate the archive, then build course geometry one course at a time."""
    with np.load(path, allow_pickle=False) as data:
        if (
            set(data.files)
            != {"version", "controls", "knots", "corners", "counts", "names"}
            or data["version"].shape != ()
            or data["version"].item() != FORMAT_VERSION
        ):
            raise CourseError(
                "Expected spline course format 2; prepare updated course resources."
            )
        controls, knots, corners, counts, names = (
            data[k] for k in ("controls", "knots", "corners", "counts", "names")
        )
    if (
        controls.ndim != 4
        or controls.shape[2:] != (4, 2)
        or knots.shape != (len(controls), controls.shape[1] + 1)
        or corners.shape != knots.shape
        or corners.dtype != np.bool_
        or counts.shape != (len(controls),)
        or counts.dtype.kind not in "iu"
        or names.shape != counts.shape
        or names.dtype.kind != "U"
        or not len(controls)
        or (counts < 1).any()
        or (counts > controls.shape[1]).any()
        or not np.isfinite(controls).all()
        or not np.isfinite(knots).all()
        or len(set(names)) != len(names)
    ):
        raise CourseError("Invalid spline course archive dimensions or values.")
    for name, c, k, f, n in zip(names, controls, knots, corners, counts):
        yield Course(str(name), c[:n], k[: n + 1], f[: n + 1])
