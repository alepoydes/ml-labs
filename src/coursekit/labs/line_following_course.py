"""Spline optics, certified swept coverage, and mechanical reference timing."""

import numpy as np

from coursekit.labs.line_following_curve import (
    Course as Course,
)
from coursekit.labs.line_following_curve import (
    bezier,
)
from coursekit.labs.line_following_curve import (
    load_courses as load_courses,
)
from coursekit.labs.line_following_curve import (
    save_courses as save_courses,
)

SENSOR_WIDTH = 0.094
SENSOR_DEPTH = 0.008
LINE_WIDTH = SENSOR_WIDTH / 8
SENSOR_HALF = np.array([SENSOR_DEPTH, SENSOR_WIDTH]) / 2
COVERAGE_VERSION = 1  # Centreline coverage by the certified sensor-row sweep.
MARKER_LENGTH = 0.12
GROUND = np.array([0.94, 0.94, 0.94])
INK = np.array([0.06, 0.06, 0.06])
START = np.array([0.08, 0.25, 0.90])
END = np.array([0.90, 0.12, 0.08])


def colour(course: Course, points: np.ndarray) -> np.ndarray:
    """Paint a finite-width line and endpoint sections; average in the sensor."""
    shape = points.shape[:-1]
    points = np.asarray(points).reshape(-1, 2)
    controls = course._leaves
    near = (
        (course._upper >= points.min(axis=0) - LINE_WIDTH / 2)
        & (course._lower <= points.max(axis=0) + LINE_WIDTH / 2)
    ).all(axis=1)
    result = np.broadcast_to(GROUND, (len(points), 3)).copy()
    if not near.any():
        return result.reshape(*shape, 3)
    leaves = np.flatnonzero(near)
    controls = controls[near]
    delta = controls[:, -1] - controls[:, 0]
    offset = points[:, None] - controls[:, 0]
    fraction = np.clip(
        np.einsum("pni,ni->pn", offset, delta) / (delta * delta).sum(axis=1), 0, 1
    )
    initial = fraction.copy()
    initial_distance = np.linalg.norm(bezier(controls, initial) - points[:, None], axis=-1)
    # Chords initialize local projection; distances and marker positions use the spline.
    for _ in range(4):
        offset = bezier(controls, fraction) - points[:, None]
        v, a = bezier(controls, fraction, 1), bezier(controls, fraction, 2)
        hessian = (v * v + offset * a).sum(axis=-1)
        fraction = np.clip(
            fraction - (offset * v).sum(axis=-1) / np.maximum(hessian, 1e-30), 0, 1
        )
    distance = np.linalg.norm(bezier(controls, fraction) - points[:, None], axis=-1)
    fraction = np.where(initial_distance < distance, initial, fraction)
    distance = np.minimum(distance, initial_distance)
    # Include endpoints even if a projection step left its local minimum.
    for t in (0, 1):
        endpoint = np.linalg.norm(controls[:, 3 * t] - points[:, None], axis=-1)
        better = endpoint < distance
        fraction = np.where(better, t, fraction)
        distance = np.minimum(distance, endpoint)
    index = distance.argmin(axis=1)
    rows = np.arange(len(points))
    painted = distance[rows, index] <= LINE_WIDTH / 2
    arc = course.distance_on_leaf(leaves[index], fraction[rows, index])
    result[painted] = INK
    result[painted & (arc < MARKER_LENGTH)] = START
    result[painted & (arc > course.length - MARKER_LENGTH)] = END
    return result.reshape(*shape, 3)


def sensor_areas(centre: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Ground-projected corners of the 16 rectangles, in the body-fixed axle row."""
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    local = np.broadcast_to(corners * [SENSOR_DEPTH, SENSOR_WIDTH / 16], (16, 4, 2)).copy()
    local[:, :, 1] += ((np.arange(16) + 0.5) / 16 - 0.5)[:, None] * SENSOR_WIDTH
    return centre[:2] + local @ rotation[:2, :2].T


def optical(course: Course, centre: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # 4 x 4 midpoint quadrature in exactly the same rectangles shown by the viewer.
    areas = sensor_areas(centre, rotation)
    offsets = (np.arange(4) + 0.5) / 4
    world = (
        areas[:, None, None, 0]
        + offsets[None, :, None, None] * (areas[:, 1] - areas[:, 0])[:, None, None]
        + offsets[None, None, :, None] * (areas[:, 3] - areas[:, 0])[:, None, None]
    )
    return colour(course, world).mean(axis=(1, 2)).astype(np.float32)


def _clip(course: Course, centre, axes, inverse, half, travel) -> np.ndarray:
    """Clip all swept parallelograms together against the same spline leaves.

    Each sweep is a box plus a segment. Erode its bounding halfplanes by the
    Bezier/chord error, so the accepted arc intervals certify the actual curve.
    """
    middle = centre + travel / 2
    extent = (np.abs(axes) @ half[..., None])[..., 0] + np.abs(travel) / 2
    near = (
        (course._upper[None] >= (middle - extent)[:, None])
        & (course._lower[None] <= (middle + extent)[:, None])
    ).all(axis=-1)
    selection = np.flatnonzero(near.any(axis=0))
    if not len(selection):
        return np.empty((0, 2))
    motion = (inverse @ travel[..., None])[..., 0]
    normal = np.stack((-motion[:, 1], motion[:, 0]), axis=-1)
    normal /= np.maximum(np.linalg.norm(motion, axis=-1, keepdims=True), 1e-30)
    normals = np.concatenate(
        (np.broadcast_to(np.eye(2), (len(half), 2, 2)), normal[:, None]), axis=1
    )
    bounds = np.column_stack(
        (half + np.abs(motion) / 2, (np.abs(normal) * half).sum(axis=-1))
    )
    world_normals = normals @ inverse
    bounds = (
        bounds[:, None]
        - course._error[selection][None, :, None]
        * np.linalg.norm(world_normals, axis=-1)[:, None]
    )
    controls = course._leaves[selection]
    a = np.einsum("kni,kji->knj", controls[None, :, 0] - middle[:, None], world_normals)
    d = np.einsum("ni,kji->knj", controls[:, -1] - controls[:, 0], world_normals)
    nonzero = np.abs(d) > 1e-15
    divisor = np.where(nonzero, d, 1)
    first, second = (-bounds - a) / divisor, (bounds - a) / divisor
    lower = np.maximum(
        0, np.where(nonzero, np.minimum(first, second), -np.inf).max(axis=-1)
    )
    upper = np.minimum(1, np.where(nonzero, np.maximum(first, second), np.inf).min(axis=-1))
    valid = (
        near[:, selection]
        & (lower <= upper)
        & ((np.abs(a) <= bounds) | nonzero).all(axis=-1)
        & (bounds >= 0).all(axis=-1)
    )
    shape, index = np.nonzero(valid)
    if not len(index):
        return np.empty((0, 2))
    leaves, lower, upper = selection[index], lower[valid], upper[valid]
    gaps = (shape[1:] != shape[:-1]) | (leaves[1:] + lower[1:] > leaves[:-1] + upper[:-1])
    starts, ends = np.r_[True, gaps], np.r_[gaps, True]
    return course.distance_on_leaf(
        np.stack((leaves[starts], leaves[ends])),
        np.stack((lower[starts], upper[ends])),
    ).T


def swept_intervals(course: Course, q0, q1, r0, r1) -> np.ndarray:
    """Certified inner sweep of the sensor row; inputs use the dynamics pose map.

    Axle position and Euler coordinates interpolate linearly during a step.
    Rotation changes any sensor point by at most radius * sum(abs(dangles)).
    Eroding the initial projected row by this bound leaves a fixed shape
    inside every translated intermediate row. Its translational sweep is
    exact. Include both endpoint rows, which require no erosion.
    """
    axes = np.array([r0[:2, :2], r1[:2, :2], r0[:2, :2]])
    inverse = np.linalg.inv(axes)
    delta = np.linalg.norm(SENSOR_HALF) * np.abs(q1[3:6] - q0[3:6]).sum()
    half = np.broadcast_to(SENSOR_HALF, (3, 2)).copy()
    half[2] -= delta * np.linalg.norm(inverse[2], axis=1)
    centre = np.array([q0[:2], q1[:2], q0[:2]])
    travel = np.array([[0.0, 0.0], [0.0, 0.0], np.asarray(q1[:2]) - q0[:2]])
    valid = (half > 0).all(axis=-1)
    return _clip(
        course, centre[valid], axes[valid], inverse[valid], half[valid], travel[valid]
    )


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


def reference_time(course: Course, model, *, spacing=None) -> float:
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
    distance, curvature, change, corner_index = (
        course.reference_profile if spacing is None else course.profile(spacing)
    )
    lengths = np.diff(distance)
    cap = np.minimum(
        vmax / (1 + half_track * np.abs(curvature)),
        np.sqrt(traction / np.maximum(np.abs(curvature), 1e-12)),
    )
    # Rotation acceleration includes the demand from changing curvature at speed.
    cap = np.minimum(cap, np.sqrt(alpha / np.maximum(np.abs(change), 1e-12)))
    cap[corner_index] = 0
    turning = 0.0
    joins = np.flatnonzero(course.corners)
    incoming = bezier(course.controls[joins - 1], 1.0, 1)
    outgoing = bezier(course.controls[joins], 0.0, 1)
    angles = np.arctan2(
        incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0],
        (incoming * outgoing).sum(axis=1),
    )
    for angle in np.abs(angles):
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
        acceleration_cap = np.minimum(
            acceleration, alpha / np.maximum(np.maximum(abs(k[:-1]), abs(k[1:])), 1e-12)
        )
        increments = 2 * acceleration_cap * ds
        square = speed**2
        # v[i+1]^2 <= v[i]^2 + 2*a[i]*ds[i]; prefix minima implement each pass.
        budget = np.r_[0.0, np.cumsum(increments)]
        square = budget + np.minimum.accumulate(square - budget)
        budget = np.r_[0.0, np.cumsum(increments[::-1])]
        square = (budget + np.minimum.accumulate(square[::-1] - budget))[::-1]
        speed = np.sqrt(np.maximum(square, 0))
        speed[0] = 0
        if stop:
            speed[-1] = 0
        return float(np.sum(2 * ds / (speed[:-1] + speed[1:])))

    return leg(False, True) + leg(True, False) + 2 * turning
