"""Spatial robot mechanics in SI units, derived from energy with torch.func.

q = (axle x, y, z, roll, pitch, yaw, left angle, right angle).
p is its canonical momentum; spin is the sphere's world angular momentum.
The homogeneous sphere needs no orientation state. Body angles form a local
chart: an episode ends well before roll or pitch can reach its singularities.
"""

from functools import partial
from typing import NamedTuple

import torch
from torch import Tensor
from torch.func import grad, jacrev, jvp, vmap


class Robot(NamedTuple):
    mass: Tensor  # Body, left wheel, right wheel, sphere.
    inertia: Tensor  # Principal moments at each centre of mass; wheel axis is y.
    offset: Tensor  # Centres relative to the axle midpoint, in the body frame.
    radius: Tensor  # Left wheel, right wheel, sphere.
    friction: Tensor
    stiffness: Tensor  # Normal contact stiffness, N/m.
    compression: Tensor  # Normal damping during compression, N s/m.
    ball_drag: Tensor  # Viscous rotational damping, N m s.
    stall: Tensor  # Current-limited motor torque, N m.
    speed: Tensor  # Motor no-load angular speed, rad/s.
    gravity: Tensor


class State(NamedTuple):
    q: Tensor
    p: Tensor
    spin: Tensor
    fallen: Tensor


class Diagnostics(NamedTuple):
    residual: Tensor
    impulse: Tensor  # Integrated world contact resistance, left/right/sphere.


def robot(*, dtype=torch.float64, device=None) -> Robot:
    """Illustrative physical parameters, not a calibrated assessment profile."""

    def t(value):
        return torch.tensor(value, dtype=dtype, device=device)

    masses = t([0.65, 0.06, 0.06, 0.02])
    radius = t([0.035, 0.035, 0.014])
    wheel_axis = 0.5 * masses[1:3] * radius[:2].square()
    wheel_side = masses[1:3] * (3 * radius[:2].square() + 0.016**2) / 12
    sphere = 0.4 * masses[3] * radius[2].square()
    inertia = torch.cat(
        (
            t([[0.003, 0.0025, 0.0018]]),
            torch.stack((wheel_side, wheel_axis, wheel_side), dim=-1),
            sphere.expand(1, 3),
        )
    )
    return Robot(
        masses,
        inertia,
        t([[0, 0, 0.10], [0, 0.055, 0], [0, -0.055, 0], [-0.085, 0, 0]]),
        radius,
        t([0.8, 0.8, 0.6]),
        t([12000, 12000, 8000]),
        t([30, 30, 8]),
        t(0.0002),
        t([0.08, 0.08]),
        t([45, 45]),
        t(9.81),
    )


def rotation(q: Tensor) -> Tensor:
    """Body-to-world rotation Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = q[3:6].unbind()
    cr, cp, cy = r.cos(), p.cos(), y.cos()
    sr, sp, sy = r.sin(), p.sin(), y.sin()
    return torch.stack(
        (
            torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr)),
            torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr)),
            torch.stack((-sp, cp * sr, cp * cr)),
        )
    )


def centres(q: Tensor, model: Robot) -> Tensor:
    return q[:3] + model.offset @ rotation(q).T


def motion(q: Tensor, v: Tensor, model: Robot):
    """Differentiate the pose map to obtain centre and angular velocities."""
    r, dr = jvp(rotation, (q,), (v,))
    skew = dr @ r.T
    omega = torch.stack((skew[2, 1], skew[0, 2], skew[1, 0]))
    linear = v[:3] + model.offset @ dr.T
    return linear, omega, r


def contact_geometry(q: Tensor, model: Robot):
    """Lowest points of two thin circular tyres and a sphere above a plane."""
    r = rotation(q)
    up = q.new_tensor([0, 0, 1])
    axle = r[:, 1]
    down = -(up - axle[2] * axle)
    down = down / torch.linalg.vector_norm(down)
    arms = torch.cat((model.radius[:2, None] * down, -model.radius[2:] * up[None]))
    points = centres(q, model)[1:] + arms
    return points[:, 2], arms


def potential(q: Tensor, model: Robot) -> Tensor:
    gap, _ = contact_geometry(q, model)
    elastic = 0.5 * (model.stiffness * torch.relu(-gap).square()).sum()
    return model.gravity * (model.mass * centres(q, model)[:, 2]).sum() + elastic


def weighted_velocity(q: Tensor, v: Tensor, model: Robot) -> Tensor:
    """Scale linear/angular velocities so their squared norm is twice kinetic energy."""
    linear, omega, r = motion(q, v, model)
    local = r.T @ omega
    wheel = local[None] + torch.stack((v[6:] * 0, v[6:], v[6:] * 0), dim=-1)
    return torch.cat(
        (
            (model.mass[:, None].sqrt() * linear).flatten(),
            model.inertia[0].sqrt() * local,
            (model.inertia[1:3].sqrt() * wheel).flatten(),
        )
    )


def lagrangian(q: Tensor, v: Tensor, model: Robot) -> Tensor:
    # Sphere rotation is independent and isotropic; spin stores its momentum.
    return 0.5 * weighted_velocity(q, v, model).square().sum() - potential(q, model)


momentum = grad(lagrangian, argnums=1)


def mass_matrix(q: Tensor, v: Tensor, model: Robot) -> Tensor:
    # weighted_velocity is linear in v: its values on the basis are J's columns.
    basis = torch.eye(8, dtype=q.dtype, device=q.device)
    rows = vmap(weighted_velocity, in_dims=(None, 0, None))(q, basis, model)
    return rows @ rows.T


def velocity(state: State, model: Robot) -> Tensor:
    matrix = mass_matrix(state.q, torch.zeros_like(state.q), model)
    return torch.linalg.solve(matrix, state.p)


def energy(state: State, model: Robot) -> Tensor:
    v = velocity(state, model)
    return (
        0.5 * state.p.dot(v)
        + potential(state.q, model)
        + 0.5 * state.spin.square().sum() / model.inertia[3, 0]
    )


def initial_state(model: Robot, *, supported=True) -> State:
    q = model.mass.new_zeros(8)
    q[2] = model.radius[0]
    if supported:
        q[4] = torch.asin((model.radius[2] - model.radius[0]) / -model.offset[3, 0])
    return State(
        q, torch.zeros_like(q), q.new_zeros(3), q.new_tensor(False, dtype=torch.bool)
    )


def discrete_lagrangian(q0: Tensor, q1: Tensor, model: Robot, dt: float) -> Tensor:
    return dt * lagrangian((q0 + q1) / 2, (q1 - q0) / dt, model)


left_momentum = grad(discrete_lagrangian, argnums=0)
right_momentum = grad(discrete_lagrangian, argnums=1)


def conservative_step(state: State, model: Robot, dt: float, iterations=5):
    """Midpoint variational step with Broyden updates and a checked residual."""
    q0, p0, spin, fallen = state
    mass = mass_matrix(q0, torch.zeros_like(q0), model)
    # Initial inverse Jacobian: H = -h M(q0)^(-1).
    inverse = torch.linalg.solve(mass, -dt * torch.eye(8, dtype=q0.dtype, device=q0.device))
    q1 = q0 - inverse @ p0
    residual = p0 + left_momentum(q0, q1, model, dt)
    for _ in range(iterations):
        delta = -inverse @ residual
        next_q = q1 + delta
        next_residual = p0 + left_momentum(q0, next_q, model, dt)
        # A secant at floating-point resolution cannot improve the matrix.
        threshold = (
            64 * torch.finfo(q0.dtype).eps * (1 + torch.linalg.vector_norm(q1))
        ) ** 2
        change = inverse @ (next_residual - residual)
        denominator = delta.dot(change)
        valid = denominator.abs() > threshold
        update = torch.outer(delta - change, delta @ inverse)
        inverse = inverse + torch.where(
            valid, update / torch.where(valid, denominator, 1), 0
        )
        q1, residual = next_q, next_residual
    p1 = right_momentum(q0, q1, model, dt)
    return State(q1, p1, spin, fallen), torch.linalg.vector_norm(residual)


def contact_velocity(q: Tensor, speeds: Tensor, model: Robot) -> Tensor:
    """Material velocity at contact, including tyre and sphere rotation."""
    linear, omega, r = motion(q, speeds[:8], model)
    _, arms = contact_geometry(q, model)
    wheel = omega[None] + speeds[6:8, None] * r[:, 1]
    angular = torch.cat((wheel, speeds[8:][None]))
    return linear[1:] + torch.linalg.cross(angular, arms)


def relative_spin(q: Tensor, speeds: Tensor, model: Robot) -> Tensor:
    return speeds[8:] - motion(q, speeds[:8], model)[1]


contact_jacobian = jacrev(contact_velocity, argnums=1)
spin_jacobian = jacrev(relative_spin, argnums=1)


def resistance(
    q: Tensor, total: Tensor, action: Tensor, model: Robot, dt: float, sweeps=16
):
    """Implicit bearing damping, compression damping, and Coulomb friction.

    Contact impulses minimise kinetic energy at fixed configuration. Tangential
    impulses share a disk bound, so both sliding directions use the same traction.
    """
    mass = mass_matrix(q, torch.zeros_like(q), model)
    matrix = torch.block_diag(
        mass, model.inertia[3, 0] * torch.eye(3, dtype=q.dtype, device=q.device)
    )
    zero = q.new_zeros(11)
    b = spin_jacobian(q, zero, model)
    jac = contact_jacobian(q, zero, model)
    normal_jac = jac[:, 2, :]
    tangent_jac = jac[:, :2, :].reshape(6, 11)
    # One factorization supplies velocity and both contact responses.
    response = torch.linalg.solve(
        matrix, torch.cat((total[:, None], normal_jac.T, tangent_jac.T), dim=1)
    )
    speeds = response[:, 0]
    torque = model.stall * (action.clamp(-1, 1) - speeds[6:8] / model.speed).clamp(-1, 1)
    drive = torch.cat((q.new_zeros(6), torque, q.new_zeros(3)))
    total = total + dt * drive
    speeds = torch.linalg.solve(matrix + dt * model.ball_drag * (b.T @ b), total)
    total = matrix @ speeds

    gap, _ = contact_geometry(q, model)
    active = gap <= 0
    normal = -model.stiffness * gap.clamp_max(0)
    normal_response = response[:, 1:4]
    normal_mobility = normal_jac @ normal_response
    rate = normal_mobility.abs().sum(-1).max().reciprocal()
    damping = dt * model.compression * active
    normal_impulse = q.new_zeros(3)
    normal_speed = normal_jac @ speeds
    for _ in range(sweeps):
        trial = normal_impulse - rate * (normal_speed + normal_mobility @ normal_impulse)
        normal_impulse = (trial * damping / (damping + rate)).clamp_min(0)
    speeds = speeds + normal_response @ normal_impulse

    tangent_response = response[:, 4:]
    tangent_mobility = tangent_jac @ tangent_response
    # One scalar per contact keeps the friction projection a Euclidean disk.
    scale = tangent_mobility.diagonal().reshape(3, 2).sum(-1).reciprocal()
    scale = scale.repeat_interleave(2)
    root = scale.sqrt()
    rate = (root[:, None] * tangent_mobility * root[None]).abs().sum(-1).max().reciprocal()
    bound = model.friction * (normal_impulse + dt * normal)
    tangent_impulse = q.new_zeros(6)
    tangent_speed = tangent_jac @ speeds
    for _ in range(sweeps):
        trial = tangent_impulse - rate * scale * (
            tangent_speed + tangent_mobility @ tangent_impulse
        )
        trial = trial.reshape(3, 2)
        norm = torch.linalg.vector_norm(trial, dim=-1).clamp_min(1e-30)
        tangent_impulse = (trial * (bound / norm).clamp_max(1)[:, None]).reshape(6)
    total = total + normal_jac.T @ normal_impulse + tangent_jac.T @ tangent_impulse
    impulse = torch.cat((tangent_impulse.reshape(3, 2), normal_impulse[:, None]), dim=1)
    return total, impulse


def resistance_step(
    state: State, action: Tensor, model: Robot, dt: float, sweeps=16, *, kernel=resistance
):
    total = torch.cat((state.p, state.spin), dim=-1)
    total, impulse = kernel(state.q, total, action, model, dt, sweeps)
    return State(state.q, total[..., :8], total[..., 8:], state.fallen), impulse


def finish_step(state: State, last: State, residual: Tensor, impulse: Tensor):
    fallen = state.fallen | (rotation(last.q)[2, 2] <= last.q.new_tensor(0.9).cos())
    result = State(
        torch.where(state.fallen, state.q, last.q),
        torch.where(state.fallen, state.p, last.p),
        torch.where(state.fallen, state.spin, last.spin),
        fallen,
    )
    return result, Diagnostics(
        torch.where(state.fallen, 0, residual),
        torch.where(state.fallen, 0, impulse),
    )


def step(
    state: State,
    action: Tensor,
    model: Robot,
    dt=0.001,
    *,
    kernels=(conservative_step, resistance_step, finish_step),
):
    """One physical step; compile_step accelerates these same kernels.

    The conservative core is symplectic up to its solve error; the complete
    dissipative map is not. Use float64 and check every step. Do not use
    torch.inference_mode: kinematics needs forward differentiation.
    """
    if dt <= 0:
        raise ValueError("The physical time step must be positive.")
    conservative, resist, finish = kernels
    first, impulse0 = resist(state, action, model, dt / 2)
    middle, residual = conservative(first, model, dt)
    last, impulse1 = resist(middle, action, model, dt / 2)
    return finish(state, last, residual, impulse0 + impulse1)


def compile_step(*, batched=False, model_batched=False, backend="inductor"):
    """Compile reusable resistance, conservative, and terminal kernels.

    Shapes, dtype, device and Python settings specialize kernels. Construct once
    and keep the batch size fixed during a rollout. Parameter values may change.
    """
    if model_batched and not batched:
        raise ValueError("Batched parameters require batched states.")
    md = 0 if model_batched else None

    def compile_kernel(fn, dims):
        fn = vmap(fn, in_dims=dims) if batched else fn
        return torch.compile(
            fn, backend=backend, fullgraph=True, dynamic=False, isolate_recompiles=True
        )

    conservative = compile_kernel(conservative_step, (0, md, None))
    resist = compile_kernel(resistance, (0, 0, 0, md, None, None))
    finish = compile_kernel(finish_step, (0, 0, 0, 0))
    return partial(
        step,
        kernels=(conservative, partial(resistance_step, kernel=resist), finish),
    )


def check(state: State, diagnostics: Diagnostics):
    """Numerical failure is a simulator error, never a robot fall or a score."""
    if not all(torch.isfinite(value).all() for value in (*state[:3], *diagnostics)):
        raise FloatingPointError("Non-finite robot dynamics.")
    if (diagnostics.residual > 1e-9).any():
        raise FloatingPointError("The variational step did not converge; reduce dt.")
