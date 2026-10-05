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

LINE_WIDTH = 0.016
SENSOR_WIDTH = 0.128
SENSOR_DEPTH = 0.008
FOOTPRINT = np.array([0.070, 0.064])
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
    middle = centre + travel / 2
    extent = np.abs(axes) @ half + np.abs(travel) / 2
    selection = np.flatnonzero(
        ((course._upper >= middle - extent) & (course._lower <= middle + extent)).all(
            axis=1
        )
    )
    if not len(selection):
        return np.empty((0, 2))
    inverse = np.linalg.inv(axes)
    motion = inverse @ travel
    normal = np.array([-motion[1], motion[0]]) / max(np.linalg.norm(motion), 1e-30)
    normals = np.array([[1.0, 0.0], [0.0, 1.0], normal])
    bounds = np.array(
        [half[0] + abs(motion[0]) / 2, half[1] + abs(motion[1]) / 2, np.abs(normal) @ half]
    )
    # A Bezier point differs from its chord at the same parameter by <= error.
    # Erode every halfplane by that bound: accepted intervals certify the curve.
    controls = course._leaves[selection]
    world_normals = normals @ inverse
    bounds = bounds[None, :] - course._error[selection, None] * np.linalg.norm(
        world_normals, axis=1
    )
    local = (controls[:, [0, -1]] - centre - travel / 2) @ inverse.T
    a, d = local[:, 0] @ normals.T, (local[:, 1] - local[:, 0]) @ normals.T
    nonzero = np.abs(d) > 1e-15
    divisor = np.where(nonzero, d, 1)
    first, second = (-bounds - a) / divisor, (bounds - a) / divisor
    lower = np.maximum(0, np.where(nonzero, np.minimum(first, second), -np.inf).max(axis=1))
    upper = np.minimum(1, np.where(nonzero, np.maximum(first, second), np.inf).min(axis=1))
    valid = (
        (lower <= upper)
        & ((np.abs(a) <= bounds) | nonzero).all(axis=1)
        & (bounds >= 0).all(axis=1)
    )
    leaves = selection[valid]
    if not len(leaves):
        return np.empty((0, 2))
    lower, upper = lower[valid], upper[valid]
    # Merge adjacent covered parameter intervals before doing arc quadrature.
    gaps = leaves[1:] + lower[1:] > leaves[:-1] + upper[:-1]
    starts, ends = np.r_[True, gaps], np.r_[gaps, True]
    return np.column_stack(
        (
            course.distance_on_leaf(leaves[starts], lower[starts]),
            course.distance_on_leaf(leaves[ends], upper[ends]),
        )
    )


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
