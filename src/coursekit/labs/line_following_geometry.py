"""Native spline queries over the same certified leaves used by Course.

Loading and adaptive subdivision stay outside the time loop. Chords are used
only with their position-error bounds; optical distances and arc coordinates
are evaluated on the cubic itself.
"""

from typing import NamedTuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from coursekit.labs.line_following_course import (
    END,
    GROUND,
    INK,
    LINE_WIDTH,
    MARKER_LENGTH,
    SENSOR_DEPTH,
    SENSOR_WIDTH,
    START,
)
from coursekit.labs.line_following_jit import njit


class Geometry(NamedTuple):
    controls: np.ndarray
    error: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    arc: np.ndarray


class Courses(NamedTuple):
    geometry: Geometry
    starts: np.ndarray
    stops: np.ndarray
    arc_starts: np.ndarray


def prepare(course):
    return Geometry(
        course._leaves, course._error, course._lower, course._upper, course._arc
    )


def pack(courses):
    """Store each distinct course once; row offsets keep batch length dynamic."""
    unique, indices, positions = [], [], {}
    for course in courses:
        if id(course) not in positions:
            positions[id(course)] = len(unique)
            unique.append(prepare(course))
        indices.append(positions[id(course)])
    indices = np.array(indices)
    bounds = np.r_[0, np.cumsum([len(g.controls) for g in unique])]
    data = Geometry(*(np.concatenate([g[i] for g in unique]) for i in range(5)))
    for value in data:
        value.setflags(write=False)
    return Courses(data, bounds[indices], bounds[indices + 1], bounds[indices] + indices)


@njit(cache=True)
def row(courses, index):
    g = courses.geometry
    a, b, s = courses.starts[index], courses.stops[index], courses.arc_starts[index]
    return Geometry(
        g.controls[a:b], g.error[a:b], g.lower[a:b], g.upper[a:b], g.arc[s : s + b - a + 1]
    )


_nodes, _weights = leggauss(16)
NODES, WEIGHTS = (_nodes + 1) / 2, _weights / 2


@njit(cache=True, inline="always")
def cubic_coordinate(c, t, axis, order):
    """Evaluate one coordinate without allocating a two-element array."""
    u = 1 - t
    p0, p1, p2, p3 = c[0, axis], c[1, axis], c[2, axis], c[3, axis]
    if order == 1:
        return 3 * (u * u * (p1 - p0) + 2 * u * t * (p2 - p1) + t * t * (p3 - p2))
    if order == 2:
        return 6 * (u * (p2 - 2 * p1 + p0) + t * (p3 - 2 * p2 + p1))
    return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3


@njit(cache=True)
def cubic(c, t, order=0):
    """Cubic Bezier value, tangent or second derivative, in local parameter."""
    return cubic_coordinate(c, t, 0, order), cubic_coordinate(c, t, 1, order)


@njit(cache=True)
def distance_on_leaf(g, leaf, t):
    # Shared endpoints must use identical cumulative sums to avoid artificial gaps.
    if t == 0.0:
        return g.arc[leaf]
    if t == 1.0:
        return g.arc[leaf + 1]
    length = (g.arc[leaf + 1] - g.arc[leaf]) * t
    if g.error[leaf] > 32 * np.finfo(np.float64).eps:
        length = 0.0
        for j in range(16):
            v = cubic(g.controls[leaf], t * NODES[j], 1)
            length += t * WEIGHTS[j] * np.hypot(v[0], v[1])
    return g.arc[leaf] + length


@njit(cache=True)
def inverse2(a):
    determinant = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    return np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]]) / determinant


@njit(cache=True)
def sweep_planes(q0, q1, r0, r1):
    """Three inner sweeps: each endpoint row, and a rotation-eroded translation.

    A sensor point moves by at most radius * sum(abs(dangles)) under rotation.
    Eroding the moving row by that amount certifies coverage throughout the step.
    Each translated rectangle is an intersection of three pairs of halfplanes.
    """
    normals = np.empty((3, 3, 2))
    bounds = np.empty((3, 3))
    centres = np.empty((3, 2))
    radius = np.hypot(SENSOR_DEPTH, SENSOR_WIDTH) / 2
    erosion = radius * np.sum(np.abs(q1[3:6] - q0[3:6]))
    for k in range(3):
        inv = inverse2(r1[:2, :2] if k == 1 else r0[:2, :2])
        travel = q1[:2] - q0[:2] if k == 2 else np.zeros(2)
        centres[k] = (q1[:2] if k == 1 else q0[:2]) + travel / 2
        half = np.array([SENSOR_DEPTH / 2, SENSOR_WIDTH / 2])
        if k == 2:
            for axis in range(2):
                half[axis] -= erosion * np.linalg.norm(inv[axis])
        motion = inv @ travel
        normal = np.array([-motion[1], motion[0]]) / max(np.linalg.norm(motion), 1e-30)
        normals[k, :2] = inv
        normals[k, 2] = normal @ inv
        bounds[k, :2] = half + np.abs(motion) / 2
        bounds[k, 2] = np.dot(np.abs(normal), half)
        if np.min(half) <= 0:
            bounds[k] = -1.0  # Empty certified inner shape.
    return centres, normals, bounds


@njit(cache=True)
def clipped_leaf(g, leaf, centre, normals, bounds):
    """Parameter interval certified inside one sweep, or (1, 0) when empty."""
    a = g.controls[leaf, 0] - centre
    d = g.controls[leaf, 3] - g.controls[leaf, 0]
    lower, upper = 0.0, 1.0
    for axis in range(3):
        bound = bounds[axis] - g.error[leaf] * np.linalg.norm(normals[axis])
        if bound < 0:
            return 1.0, 0.0
        origin, direction = np.dot(a, normals[axis]), np.dot(d, normals[axis])
        if abs(direction) <= 1e-15:
            if abs(origin) > bound:
                return 1.0, 0.0
        else:
            first, second = (-bound - origin) / direction, (bound - origin) / direction
            lower, upper = max(lower, min(first, second)), min(upper, max(first, second))
    return lower, upper


@njit(cache=True)
def advance(g, frontier, q0, q1, r0, r1):
    """Follow only the covered component connected to the existing frontier.

    Leaves are in route order. Once a gap is found no later leaf can advance the
    route, even at a crossing. This avoids scanning/sorting the entire course.
    """
    centres, normals, bounds = sweep_planes(q0, q1, r0, r1)
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, g.arc[-1])
    leaf = max(0, np.searchsorted(g.arc, frontier, side="left") - 1)
    for i in range(leaf, len(g.controls)):
        intervals = np.empty((3, 2))
        for k in range(3):
            lo, hi = clipped_leaf(g, i, centres[k], normals[k], bounds[k])
            intervals[k, 0] = distance_on_leaf(g, i, lo) if lo <= hi else np.inf
            intervals[k, 1] = distance_on_leaf(g, i, hi) if lo <= hi else -np.inf
        # Three intervals need at most three passes to form their connected union.
        for _ in range(3):
            for k in range(3):
                if intervals[k, 0] <= frontier + tolerance:
                    frontier = max(frontier, intervals[k, 1])
        if frontier < g.arc[i + 1] - tolerance:
            break
    return min(frontier, g.arc[-1])


@njit(cache=True)
def nearest_on_leaf(c, x, y):
    """Safeguarded Newton projection, with the same chord and endpoint fallbacks.

    Scalar coordinates matter here: this loop is repeated for every quadrature
    point and nearby leaf. Allocating tiny vectors would dominate sensing time.
    """
    dx, dy = c[3, 0] - c[0, 0], c[3, 1] - c[0, 1]
    initial = min(
        1.0, max(0.0, ((x - c[0, 0]) * dx + (y - c[0, 1]) * dy) / (dx * dx + dy * dy))
    )
    t = initial
    px, py = cubic(c, t)
    initial_distance = np.hypot(px - x, py - y)
    for _ in range(4):
        px, py = cubic(c, t)
        vx, vy = cubic(c, t, 1)
        ax, ay = cubic(c, t, 2)
        hessian = max(vx * vx + vy * vy + (px - x) * ax + (py - y) * ay, 1e-30)
        t = min(1.0, max(0.0, t - ((px - x) * vx + (py - y) * vy) / hessian))
    px, py = cubic(c, t)
    distance = np.hypot(px - x, py - y)
    if initial_distance < distance:
        t, distance = initial, initial_distance
    for endpoint in range(2):
        d = np.hypot(c[3 * endpoint, 0] - x, c[3 * endpoint, 1] - y)
        if d < distance:
            t, distance = float(endpoint), d
    return distance, t


@njit(cache=True)
def optical(g, q, r):
    """16 RGB sectors, each with the existing 4x4 midpoint quadrature."""
    result = np.zeros((16, 3))
    points = np.empty((256, 2))
    for sector in range(16):
        for ix in range(4):
            for iy in range(4):
                local_x = ((ix + 0.5) / 4 - 0.5) * SENSOR_DEPTH
                local_y = ((sector + (iy + 0.5) / 4) / 16 - 0.5) * SENSOR_WIDTH
                for axis in range(2):
                    points[16 * sector + 4 * ix + iy, axis] = (
                        q[axis] + r[axis, 0] * local_x + r[axis, 1] * local_y
                    )
    low, high = np.empty(2), np.empty(2)
    for axis in range(2):
        low[axis] = np.min(points[:, axis]) - LINE_WIDTH / 2
        high[axis] = np.max(points[:, axis]) + LINE_WIDTH / 2
    # The same coarse selection as the NumPy optical reference, in route order.
    candidates = np.empty(len(g.controls), dtype=np.int64)
    count = 0
    for leaf in range(len(g.controls)):
        if (
            g.upper[leaf, 0] >= low[0]
            and g.upper[leaf, 1] >= low[1]
            and g.lower[leaf, 0] <= high[0]
            and g.lower[leaf, 1] <= high[1]
        ):
            candidates[count] = leaf
            count += 1
    for j in range(256):
        best, nearest, fraction = np.inf, 0, 0.0
        for k in range(count):
            leaf = candidates[k]
            distance, t = nearest_on_leaf(g.controls[leaf], points[j, 0], points[j, 1])
            if distance < best:
                best, nearest, fraction = distance, leaf, t
        colour = GROUND
        if best <= LINE_WIDTH / 2:
            arc = distance_on_leaf(g, nearest, fraction)
            colour = INK
            if arc < MARKER_LENGTH:
                colour = START
            if arc > g.arc[-1] - MARKER_LENGTH:
                colour = END
        for channel in range(3):
            result[j // 16, channel] += colour[channel] / 16
    return result.astype(np.float32)
