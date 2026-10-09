"""Native blocks of checked physics and route judging; no policy calls inside.

The outer row loop may run in parallel. Each row has its own state, course and
workspace; time remains sequential within that row. Recorded frames are optional.
"""

import numpy as np
from numba import prange

from coursekit.labs import line_following_geometry as geometry
from coursekit.labs import line_following_numba as physics
from coursekit.labs.line_following import Robot
from coursekit.labs.line_following_jit import njit

RUNNING, FINISHED, FALLEN, TIMEOUT = range(4)
TICKS, LEG, FRONTIER, OUTWARD_TIME, TRAVELLED, STATUS = range(6)


@njit(cache=True)
def model_row(model, i):
    return Robot(
        model.mass[i],
        model.inertia[i],
        model.offset[i],
        model.radius[i],
        model.friction[i],
        model.stiffness[i],
        model.compression[i],
        model.ball_drag[i],
        model.stall[i],
        model.speed[i],
        model.gravity[i],
    )


@njit(cache=True)
def advance_row(
    q, p, spin, fallen, model, course, reverse, action, counters, limit, steps, history, dt
):
    """Update one row in place, stopping at the first physical terminal step.

    counters = ticks, leg, accepted frontier, outward time, travelled path, status.
    The route check follows the physics check: a numerical failure cannot score.
    """
    r0 = physics.rotation(q)
    for i in range(steps):
        if counters[STATUS] != RUNNING:
            break
        try:
            q1, p1, spin1, fell, _, _ = physics.step(q, p, spin, action, model, dt)
        except Exception:
            return i, 1
        r1 = physics.rotation(q1)
        counters[TICKS] += 1
        counters[TRAVELLED] += np.hypot(q1[0] - q[0], q1[1] - q[1])
        if fell:
            counters[STATUS] = FALLEN
        else:
            g = course if counters[LEG] == 0 else reverse
            counters[FRONTIER] = geometry.advance(g, counters[FRONTIER], q, q1, r0, r1)
            if counters[FRONTIER] >= g.arc[-1]:
                if counters[LEG] == 0:
                    counters[LEG] = 1
                    counters[OUTWARD_TIME] = counters[TICKS] * dt
                    counters[FRONTIER] = geometry.advance(reverse, 0.0, q1, q1, r1, r1)
                else:
                    counters[STATUS] = FINISHED
            if counters[STATUS] == RUNNING and counters[TICKS] >= limit:
                counters[STATUS] = TIMEOUT
        q[:], p[:], spin[:] = q1, p1, spin1
        fallen[0] = fell
        r0 = r1
        if len(history):
            history[i, :8], history[i, 8:16], history[i, 16:19] = q, p, spin
            history[i, 19] = fell
            history[i, 20:] = counters
    return int(counters[TICKS]), 0


@njit(cache=True, nogil=True)
def serial_block(
    q,
    p,
    spin,
    fallen,
    models,
    courses,
    reverse,
    action,
    counters,
    limit,
    steps,
    history,
    errors,
    dt,
):
    for row in range(len(q)):
        _, errors[row] = advance_row(
            q[row],
            p[row],
            spin[row],
            fallen[row : row + 1],
            model_row(models, row),
            geometry.row(courses, row),
            geometry.row(reverse, row),
            action[row],
            counters[row],
            limit,
            steps,
            history[row],
            dt,
        )


@njit(cache=True, nogil=True, parallel=True)
def block(
    q,
    p,
    spin,
    fallen,
    models,
    courses,
    reverse,
    action,
    counters,
    limit,
    steps,
    history,
    errors,
    dt,
):
    if len(q) < 8:
        serial_block(
            q,
            p,
            spin,
            fallen,
            models,
            courses,
            reverse,
            action,
            counters,
            limit,
            steps,
            history,
            errors,
            dt,
        )
        return
    # Unpack before prange: its native gufunc arguments must be arrays/scalars,
    # rather than nested tuples of arrays. The physical parameter names remain explicit.
    (
        mass,
        inertia,
        offset,
        radius,
        friction,
        stiffness,
        compression,
        drag,
        stall,
        speed,
        gravity,
    ) = models
    fc, fe, fl, fu, fa = courses.geometry
    fs, ft, fas = courses.starts, courses.stops, courses.arc_starts
    rc, re, rl, ru, ra = reverse.geometry
    rs, rt, ras = reverse.starts, reverse.stops, reverse.arc_starts
    for row in prange(len(q)):
        model = Robot(
            mass[row],
            inertia[row],
            offset[row],
            radius[row],
            friction[row],
            stiffness[row],
            compression[row],
            drag[row],
            stall[row],
            speed[row],
            gravity[row],
        )
        a, b, s = fs[row], ft[row], fas[row]
        forward = geometry.Geometry(
            fc[a:b], fe[a:b], fl[a:b], fu[a:b], fa[s : s + b - a + 1]
        )
        a, b, s = rs[row], rt[row], ras[row]
        backward = geometry.Geometry(
            rc[a:b], re[a:b], rl[a:b], ru[a:b], ra[s : s + b - a + 1]
        )
        _, errors[row] = advance_row(
            q[row],
            p[row],
            spin[row],
            fallen[row : row + 1],
            model,
            forward,
            backward,
            action[row],
            counters[row],
            limit,
            steps,
            history[row],
            dt,
        )


@njit(cache=True)
def kinematics(q, p, models):
    velocity = np.empty_like(q)
    angular, rotations = np.empty((len(q), 3)), np.empty((len(q), 3, 3))
    for i in range(len(q)):
        model = model_row(models, i)
        r, e, _, _, _, linear = physics.kinematics(q[i], model)
        velocity[i] = np.linalg.solve(physics.inertia(linear, e, model), p[i])
        angular[i], rotations[i] = r @ e @ velocity[i, 3:6], r
    return velocity, angular, rotations
