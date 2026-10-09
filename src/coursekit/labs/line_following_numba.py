"""Analytic CPU mechanics for the robot in line_following.py.

The coordinates, energies and splitting are identical to the autodiff example.
Arrays are float64 in SI units. Small native functions keep the derivation visible;
the environment calls them from a compiled loop, without Python between steps.
"""

import numpy as np

from coursekit.labs.line_following_jit import njit


@njit(cache=True)
def cross_matrix(a):
    """The matrix [a]x such that [a]x b = a cross b."""
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


@njit(cache=True)
def rotation(q):
    r, p, y = q[3], q[4], q[5]
    cr, cp, cy = np.cos(r), np.cos(p), np.cos(y)
    sr, sp, sy = np.sin(r), np.sin(p), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@njit(cache=True)
def angular_map(q):
    """Body angular velocity = E(roll, pitch) @ Euler-angle rates."""
    cr, sr = np.cos(q[3]), np.sin(q[3])
    cp, sp = np.cos(q[4]), np.sin(q[4])
    e = np.array([[1.0, 0.0, -sp], [0.0, cr, sr * cp], [0.0, -sr, cr * cp]])
    de = np.zeros((3, 3, 3))  # derivative index, velocity axis, angle-rate index
    de[0, 1, 1], de[0, 1, 2] = -sr, cr * cp
    de[0, 2, 1], de[0, 2, 2] = -cr, -sr * cp
    de[1, 0, 2], de[1, 1, 2], de[1, 2, 2] = -cp, -sr * sp, -cr * sp
    return e, de


@njit(cache=True)
def kinematics(q, model):
    """Centre velocity Jacobians and body/world angular-velocity maps."""
    r = rotation(q)
    e, de = angular_map(q)
    w = r @ e
    offsets = model.offset @ r.T
    linear = np.zeros((4, 3, 8))
    for body in range(4):
        linear[body, :, :3] = np.eye(3)
        linear[body, :, 3:6] = -cross_matrix(offsets[body]) @ w
    return r, e, de, w, offsets, linear


@njit(cache=True)
def inertia(linear, e, model):
    """Assemble M from centre and angular velocities, T = v.T M v / 2."""
    mass = np.zeros((8, 8))
    for body in range(4):
        mass += model.mass[body] * (linear[body].T @ linear[body])
    for body in range(3):
        angular = np.zeros((3, 8))
        angular[:, 3:6] = e
        if body:
            angular[1, 5 + body] = 1.0  # wheel rotation relative to the body
        mass += angular.T @ (model.inertia[body, :, None] * angular)
    return mass


@njit(cache=True)
def mass_matrix(q, model):
    _, e, _, _, _, linear = kinematics(q, model)
    return inertia(linear, e, model)


@njit(cache=True)
def contacts(q, r, w, offsets, linear, model):
    """Gap and material-velocity maps for the two tyres and the free sphere.

    The rotating material has velocity v_c + omega cross arm. Differentiating
    just the lowest geometric point would lose wheel spin and therefore slip.
    """
    axle = r[:, 1].copy()
    down = axle[2] * axle
    down[2] -= 1.0
    down /= np.linalg.norm(down)
    arms = np.zeros((3, 3))
    arms[0], arms[1] = model.radius[0] * down, model.radius[1] * down
    arms[2, 2] = -model.radius[2]
    gap = np.empty(3)
    jac = np.zeros((3, 3, 11))
    for i in range(3):
        gap[i] = q[2] + offsets[i + 1, 2] + arms[i, 2]
        jac[i, :, :8] = linear[i + 1]
        angular = np.zeros((3, 11))
        if i < 2:
            angular[:, 3:6] = w
            angular[:, 6 + i] = axle
        else:
            angular[:, 8:] = np.eye(3)
        jac[i] -= cross_matrix(arms[i]) @ angular
    bearing = np.zeros((3, 11))
    bearing[:, 3:6], bearing[:, 8:] = -w, np.eye(3)
    return gap, jac, bearing


@njit(cache=True, inline="always")
def cross(a, b):
    return np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )


@njit(cache=True)
def lagrangian_derivatives(q, v, model):
    """Differentiate the four bodies' energies directly, without assembling dM.

    For a centre offset c in world coordinates, u = x_dot + omega cross c.
    Thus dT = sum(m u dot du + I omega_body dot d(omega_body)). The velocity
    derivatives give momenta; angle derivatives give L_q. This contraction
    avoids constructing a third-order mass derivative at each Broyden iteration.
    """
    r = rotation(q)
    e, de = angular_map(q)
    w = r @ e
    offsets = model.offset @ r.T
    local_omega = e @ v[3:6]
    omega = r @ local_omega
    local_derivative, world_derivative = np.empty((3, 3)), np.empty((3, 3))
    for a in range(3):
        local_derivative[a] = de[a] @ v[3:6]
        world_derivative[a] = cross(w[:, a], omega) + r @ local_derivative[a]
    gradient, momentum = np.zeros(8), np.zeros(8)
    torque_momentum = np.zeros(3)
    for body in range(4):
        c = offsets[body]
        u = v[:3] + cross(omega, c)
        momentum[:3] += model.mass[body] * u
        torque_momentum += model.mass[body] * cross(c, u)
        gradient[2] -= model.gravity * model.mass[body]
        for a in range(3):
            dc = cross(w[:, a], c)
            du = cross(world_derivative[a], c) + cross(omega, dc)
            gradient[3 + a] += model.mass[body] * (np.dot(u, du) - model.gravity * dc[2])
    momentum[3:6] = w.T @ torque_momentum
    for body in range(3):
        angular = local_omega.copy()
        if body:
            angular[1] += v[5 + body]
        angular_momentum = model.inertia[body] * angular
        momentum[3:6] += e.T @ angular_momentum
        if body:
            momentum[5 + body] = angular_momentum[1]
        for a in range(3):
            gradient[3 + a] += np.dot(angular_momentum, local_derivative[a])
    axle = r[:, 1].copy()
    # A thin tyre's lowest height is centre_z - radius*sqrt(1-axle_z**2).
    side = np.sqrt(1.0 - axle[2] ** 2)
    for i in range(3):
        gap = q[2] + offsets[i + 1, 2] - model.radius[i] * (side if i < 2 else 1.0)
        force = model.stiffness[i] * max(-gap, 0.0)
        gradient[2] += force
        for a in range(3):
            dg = cross(w[:, a], offsets[i + 1])[2]
            if i < 2:
                dg += model.radius[i] * axle[2] * cross(w[:, a], axle)[2] / side
            gradient[3 + a] += force * dg
    return gradient, momentum


@njit(cache=True)
def discrete_momenta(q0, q1, model, dt):
    force, momentum = lagrangian_derivatives((q0 + q1) / 2, (q1 - q0) / dt, model)
    return dt / 2 * force - momentum, dt / 2 * force + momentum


@njit(cache=True)
def conservative_step(q0, p0, model, dt, iterations=5):
    """Midpoint variational step; inverse Broyden avoids coordinate Hessians."""
    inverse = np.linalg.solve(mass_matrix(q0, model), -dt * np.eye(8))
    q1 = q0 - inverse @ p0
    left, p1 = discrete_momenta(q0, q1, model, dt)
    residual = p0 + left
    for _ in range(iterations):
        delta = -inverse @ residual
        next_q = q1 + delta
        left, p1 = discrete_momenta(q0, next_q, model, dt)
        next_residual = p0 + left
        threshold = (64 * np.finfo(np.float64).eps * (1 + np.linalg.norm(q1))) ** 2
        change = inverse @ (next_residual - residual)
        denominator = np.dot(delta, change)
        if abs(denominator) > threshold:
            inverse += np.outer(delta - change, delta @ inverse) / denominator
        q1, residual = next_q, next_residual
    return q1, p1, np.linalg.norm(residual)


@njit(cache=True)
def resistance(q, total, action, model, dt, sweeps=16):
    """Implicit bearing/compression losses and isotropic Coulomb friction.

    Project impulses onto nonnegative normal loads and tangential disks. The
    latter share a traction budget between lateral and longitudinal slip.
    """
    r, e, _, w, offsets, linear = kinematics(q, model)
    matrix = np.zeros((11, 11))
    matrix[:8, :8] = inertia(linear, e, model)
    matrix[8:, 8:] = model.inertia[3, 0] * np.eye(3)
    gap, jac, bearing = contacts(q, r, w, offsets, linear, model)
    normal_jac = jac[:, 2, :].copy()
    tangent_jac = jac[:, :2, :].copy().reshape(6, 11)
    rhs = np.empty((11, 10))
    rhs[:, 0], rhs[:, 1:4], rhs[:, 4:] = total, normal_jac.T, tangent_jac.T
    response = np.linalg.solve(matrix, rhs)
    speeds = response[:, 0].copy()
    driven = total.copy()
    for i in range(2):
        torque = model.stall[i] * min(
            1.0, max(-1.0, action[i] - speeds[6 + i] / model.speed[i])
        )
        driven[6 + i] += dt * torque
    speeds = np.linalg.solve(matrix + dt * model.ball_drag * (bearing.T @ bearing), driven)
    total = matrix @ speeds

    normal_response = response[:, 1:4].copy()
    mobility = normal_jac @ normal_response
    rate = 1.0 / np.max(np.sum(np.abs(mobility), axis=1))
    damping = dt * model.compression * (gap <= 0)
    normal = np.zeros(3)
    normal_speed = normal_jac @ speeds
    for _ in range(sweeps):
        trial = normal - rate * (normal_speed + mobility @ normal)
        normal = np.maximum(0.0, trial * damping / (damping + rate))
    speeds += normal_response @ normal

    tangent_response = response[:, 4:].copy()
    mobility = tangent_jac @ tangent_response
    scale = np.empty(6)
    bound = np.empty(3)
    for i in range(3):
        scale[2 * i : 2 * i + 2] = 1.0 / (
            mobility[2 * i, 2 * i] + mobility[2 * i + 1, 2 * i + 1]
        )
        bound[i] = model.friction[i] * (
            normal[i] + dt * model.stiffness[i] * max(-gap[i], 0.0)
        )
    root = np.sqrt(scale)
    rate = 1.0 / np.max(np.sum(np.abs(root[:, None] * mobility * root[None, :]), axis=1))
    tangent = np.zeros(6)
    tangent_speed = tangent_jac @ speeds
    for _ in range(sweeps):
        trial = tangent - rate * scale * (tangent_speed + mobility @ tangent)
        for i in range(3):
            norm = max(np.hypot(trial[2 * i], trial[2 * i + 1]), 1e-30)
            tangent[2 * i : 2 * i + 2] = trial[2 * i : 2 * i + 2] * min(
                1.0, bound[i] / norm
            )
    total += normal_jac.T @ normal + tangent_jac.T @ tangent
    impulse = np.empty((3, 3))
    impulse[:, :2], impulse[:, 2] = tangent.reshape(3, 2), normal
    return total, impulse


@njit(cache=True)
def step(q, p, spin, action, model, dt=0.001):
    """One checked Strang-split step. A failed solve never becomes a fall."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("The physical time step must be positive and finite.")
    if not (np.isfinite(q).all() and np.isfinite(p).all() and np.isfinite(spin).all()):
        raise FloatingPointError("Non-finite robot dynamics.")
    total = np.concatenate((p, spin))
    total, impulse0 = resistance(q, total, action, model, dt / 2)
    q1, p1, residual = conservative_step(q, total[:8], model, dt)
    total[:8] = p1
    total, impulse1 = resistance(q1, total, action, model, dt / 2)
    if not (np.isfinite(q1).all() and np.isfinite(total).all() and np.isfinite(residual)):
        raise FloatingPointError("Non-finite robot dynamics.")
    if residual > 1e-9:
        raise FloatingPointError("The variational step did not converge; reduce dt.")
    fallen = rotation(q1)[2, 2] <= np.cos(0.9)
    return q1, total[:8].copy(), total[8:].copy(), fallen, residual, impulse0 + impulse1
