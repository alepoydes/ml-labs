import pytest
import torch
from torch.func import jacfwd, jacrev, vmap

from coursekit.labs import line_following as physics


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def moving_state(model, *, supported=False, airborne=False):
    state = physics.initial_state(model, supported=supported)
    q = state.q.clone()
    q[2] = 1 if airborne else q[2] - 0.0005
    q[3] = 0.003
    q[5] = 0.2
    v = q.new_tensor([0.1, 0.2, -0.3, 0.2, 0.1, -0.1, 1.0, -0.5])
    return state._replace(
        q=q, p=physics.momentum(q, v, model), spin=q.new_tensor([1e-5, -2e-5, 3e-5])
    )


def angular_momentum(state, model):
    v = physics.velocity(state, model)
    linear, omega, rotation = physics.motion(state.q, v, model)
    orbital = torch.linalg.cross(
        physics.centres(state.q, model), model.mass[:, None] * linear
    ).sum(0)
    local = rotation.T @ omega
    wheel = local[None] + torch.stack((v[6:] * 0, v[6:], v[6:] * 0), -1)
    return (
        orbital
        + rotation @ (model.inertia[0] * local + (model.inertia[1:3] * wheel).sum(0))
        + state.spin
    )


def test_kinematics_and_mass():
    model = physics.robot()
    state = moving_state(model, airborne=True)
    q, v = state.q, physics.velocity(state, model)
    matrix = physics.mass_matrix(q, v * 0, model)
    torch.testing.assert_close(matrix, matrix.T, atol=1e-14, rtol=0)
    torch.testing.assert_close(matrix, jacrev(physics.momentum, argnums=1)(q, v, model))
    assert torch.linalg.eigvalsh(matrix).min() > 0
    torch.testing.assert_close(matrix @ v, state.p)
    linear, omega, rotation = physics.motion(q, v, model)
    epsilon = 1e-6
    difference = (
        physics.centres(q + epsilon * v, model) - physics.centres(q - epsilon * v, model)
    ) / (2 * epsilon)
    torch.testing.assert_close(linear, difference, atol=1e-10, rtol=1e-8)
    arms = model.offset @ rotation.T
    torch.testing.assert_close(
        linear, v[:3] + torch.linalg.cross(omega.expand_as(arms), arms)
    )
    normal_jac = physics.contact_jacobian(q, q.new_zeros(11), model)[:, 2, :8]
    gap_jac = jacfwd(lambda position: physics.contact_geometry(position, model)[0])(q)
    torch.testing.assert_close(normal_jac, gap_jac, atol=1e-13, rtol=0)


def test_rolling_uses_material_contact_velocity():
    model = physics.robot()
    state = physics.initial_state(model)
    speeds = state.q.new_zeros(11)
    speeds[0] = 0.2
    speeds[6:8] = speeds[0] / model.radius[:2]
    speeds[9] = speeds[0] / model.radius[2]
    torch.testing.assert_close(
        physics.contact_velocity(state.q, speeds, model),
        state.q.new_zeros(3, 3),
        atol=1e-14,
        rtol=0,
    )
    speeds[6:] = 0
    slip = physics.contact_velocity(state.q, speeds, model)
    torch.testing.assert_close(slip[:, 0], speeds[0].expand(3))


def test_free_fall_and_cyclic_momenta():
    model = physics.robot()
    state = physics.initial_state(model, supported=False)
    state.q[2] = 1
    dt = 0.003
    result, residual = physics.conservative_step(state, model, dt)
    expected = state.q.clone()
    expected[2] -= model.gravity * dt**2 / 2
    torch.testing.assert_close(result.q, expected, atol=1e-13, rtol=0)
    torch.testing.assert_close(
        result.p[2], -model.mass.sum() * model.gravity * dt, atol=1e-13, rtol=0
    )
    assert residual < 1e-11
    state = moving_state(model, airborne=True)
    result, _ = physics.conservative_step(state, model, dt)
    indices = [0, 1, 6, 7]
    torch.testing.assert_close(result.p[indices], state.p[indices], atol=1e-12, rtol=0)

    def vertical_momentum(s):
        return s.p[5] + s.q[0] * s.p[1] - s.q[1] * s.p[0]

    torch.testing.assert_close(
        vertical_momentum(result), vertical_momentum(state), atol=1e-12, rtol=0
    )


def test_conservative_map_is_symplectic_and_reversible():
    model = physics.robot()
    state = moving_state(model, airborne=True)

    def evolve(z):
        result, _ = physics.conservative_step(
            state._replace(q=z[:8], p=z[8:]), model, 0.002
        )
        return torch.cat((result.q, result.p))

    z = torch.cat((state.q, state.p))
    derivative = jacfwd(evolve)(z)
    torch.testing.assert_close(jacrev(evolve)(z), derivative, atol=1e-10, rtol=1e-9)
    form = z.new_zeros(16, 16)
    form[:8, 8:] = torch.eye(8)
    form[8:, :8] = -torch.eye(8)
    torch.testing.assert_close(derivative.T @ form @ derivative, form, atol=1e-10, rtol=0)
    forward, _ = physics.conservative_step(state, model, 0.002)
    back, _ = physics.conservative_step(forward, model, -0.002)
    torch.testing.assert_close(back.q, state.q, atol=1e-12, rtol=0)
    torch.testing.assert_close(back.p, state.p, atol=1e-12, rtol=0)


@pytest.mark.parametrize("supported", [False, True])
def test_resistance_is_passive_and_friction_uses_one_disk(supported):
    model = physics.robot()
    state = moving_state(model, supported=supported)
    dt = 0.0005
    result, impulse = physics.resistance_step(state, state.q.new_zeros(2), model, dt)
    assert physics.energy(result, model) < physics.energy(state, model)
    gap, _ = physics.contact_geometry(state.q, model)
    normal = impulse[:, 2] - dt * model.stiffness * gap.clamp_max(0)
    assert (
        torch.linalg.vector_norm(impulse[:, :2], dim=-1) <= model.friction * normal + 1e-14
    ).all()
    assert (impulse[:, 2] >= 0).all()
    assert (impulse[gap > 0] == 0).all()


@pytest.mark.parametrize("action", [[0.0, 0.0], [0.7, -0.3]])
def test_internal_torques_conserve_world_angular_momentum(action):
    model = physics.robot()
    state = moving_state(model, airborne=True)
    result, impulse = physics.resistance_step(
        state, state.q.new_tensor(action), model, 0.001
    )
    assert (impulse == 0).all()
    torch.testing.assert_close(result.p[:3], state.p[:3], atol=1e-14, rtol=0)
    torch.testing.assert_close(
        angular_momentum(result, model), angular_momentum(state, model), atol=1e-13, rtol=0
    )
    assert not torch.equal(result.spin, state.spin)


def test_batch_matches_independent_robots():
    model = physics.robot()
    heavier = model._replace(mass=model.mass * 1.3, inertia=model.inertia * 1.3)
    models = [model, heavier]
    states = [physics.initial_state(model), moving_state(heavier, airborne=True)]
    batch = physics.State(*(torch.stack(items) for items in zip(*states)))
    parameters = physics.Robot(*(torch.stack(items) for items in zip(*models)))
    actions = batch.q.new_tensor([[0.1, 0.1], [-0.2, 0.2]])
    result, info = vmap(physics.step, in_dims=(0, 0, 0))(batch, actions, parameters)
    for i, state in enumerate(states):
        expected, diagnostics = physics.step(state, actions[i], models[i])
        for actual, value in zip(result, expected):
            torch.testing.assert_close(actual[i], value, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(
            info.residual[i], diagnostics.residual, atol=1e-12, rtol=0
        )


def test_fall_is_terminal():
    model = physics.robot()
    state = moving_state(model, airborne=True)
    state.q[4] = 0.91
    result, _ = physics.step(state, state.q.new_zeros(2), model)
    assert result.fallen
    frozen, _ = physics.step(result, state.q.new_ones(2), model)
    for actual, expected in zip(frozen, result):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_numerical_failure_is_distinct_from_a_fall():
    model = physics.robot()
    state = moving_state(model, airborne=True)
    result, residual = physics.conservative_step(state, model, 0.1, iterations=0)
    assert not result.fallen
    with pytest.raises(FloatingPointError, match="did not converge"):
        physics.check(result, physics.Diagnostics(residual, state.q.new_zeros(3, 3)))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        physics.check(
            state._replace(spin=state.spin * float("nan")),
            physics.Diagnostics(residual * 0, state.q.new_zeros(3, 3)),
        )
    with pytest.raises(ValueError, match="positive"):
        physics.step(state, state.q.new_zeros(2), model, dt=0)


def test_contact_iterations_converge():
    model = physics.robot()
    state = moving_state(model, supported=True)
    action = state.q.new_zeros(2)
    normal, impulse = physics.resistance_step(state, action, model, 0.0005)
    refined, reference = physics.resistance_step(state, action, model, 0.0005, sweeps=64)
    torch.testing.assert_close(impulse, reference, atol=1e-7, rtol=0)
    torch.testing.assert_close(normal.p, refined.p, atol=1e-7, rtol=0)


def test_rest_drive_and_turn_trajectories():
    model = physics.robot()
    initial = physics.initial_state(model)
    state = physics.State(*(value.expand(3, *value.shape).clone() for value in initial))
    action = state.q.new_tensor([[0, 0], [0.2, 0.2], [-0.2, 0.2]])
    advance = vmap(physics.step, in_dims=(0, 0, None))
    max_energy = float(physics.energy(initial, model))
    for _ in range(60):
        state, diagnostics = advance(state, action, model)
        physics.check(state, diagnostics)
        rest = physics.State(*(value[0] for value in state))
        assert physics.energy(rest, model) <= max_energy + 2e-6
        gap, _ = vmap(physics.contact_geometry, in_dims=(0, None))(state.q, model)
        assert gap.min() > -0.001
        assert not state.fallen.any()
    assert abs(state.q[0, 0]) < 0.0001
    assert abs(state.q[0, 4] - initial.q[4]) < 0.001
    assert state.q[1, 0] > 0.0008
    torch.testing.assert_close(
        state.q[:2, [1, 3, 5]], state.q.new_zeros(2, 3), atol=1e-10, rtol=0
    )
    assert state.q[2, 5] > 0.015


def test_contact_trajectory_converges_with_time_step():
    model = physics.robot()
    results = []
    for dt in (0.002, 0.001, 0.0005):
        state = physics.initial_state(model)
        for _ in range(round(0.02 / dt)):
            state, diagnostics = physics.step(
                state, state.q.new_tensor([0.2, 0.2]), model, dt
            )
            physics.check(state, diagnostics)
        results.append(torch.cat((state.q, physics.velocity(state, model))))
    coarse = torch.linalg.vector_norm(results[0] - results[1])
    fine = torch.linalg.vector_norm(results[1] - results[2])
    assert fine < 0.65 * coarse


def test_smooth_conservative_trajectory_has_second_order_accuracy():
    model = physics.robot()
    initial = moving_state(model, airborne=True)
    results = []
    for dt in (0.02, 0.01, 0.005):
        state = initial
        for _ in range(round(0.1 / dt)):
            state, residual = physics.conservative_step(state, model, dt)
            assert residual < 1e-10
            assert abs(physics.energy(state, model) - physics.energy(initial, model)) < 1e-6
        results.append(torch.cat((state.q, state.p)))
    coarse = torch.linalg.vector_norm(results[0] - results[1])
    fine = torch.linalg.vector_norm(results[1] - results[2])
    assert 0.2 < fine / coarse < 0.3


def test_compiler_reuses_kernels_and_matches_eager():
    batched, model_batched = True, True
    model = physics.robot()
    initial = moving_state(model, supported=True)
    # Graph capture is portable; the instructor pilot also checks native codegen.
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = physics.compile_step(
        batched=batched, model_batched=model_batched, backend=backend
    )
    eager = (
        vmap(physics.step, in_dims=(0, 0, 0 if model_batched else None))
        if batched
        else physics.step
    )
    counts = {}
    for size in (2, 3, 2) if batched else (1, 1):
        state, parameters = initial, model
        if batched:
            state = physics.State(
                *(value.expand(size, *value.shape).clone() for value in initial)
            )
        if model_batched:
            parameters = physics.Robot(
                *(value.expand(size, *value.shape).clone() for value in model)
            )
            parameters.mass[0] *= 1.2
            parameters.inertia[0] *= 1.2
        action = state.q.new_full((*state.q.shape[:-1], 2), 0.1)
        previous = len(graphs)
        for iteration in range(2):
            expected, reference_info = eager(state, action, parameters)
            actual, diagnostics = compiled(state, action, parameters)
            physics.check(actual, diagnostics)
            for observed, reference in zip(actual, expected):
                torch.testing.assert_close(observed, reference, atol=1e-12, rtol=1e-10)
            for observed, reference in zip(diagnostics, reference_info):
                torch.testing.assert_close(observed, reference, atol=1e-12, rtol=1e-10)
            if iteration == 0 and size not in counts:
                assert len(graphs) > previous
                counts[size] = len(graphs) - previous
                previous = len(graphs)
            assert len(graphs) == previous
            state = actual
            action = -action
            parameters = parameters._replace(
                mass=parameters.mass * 1.01, inertia=parameters.inertia * 1.01
            )
    assert len(set(counts.values())) == 1


def test_singular_solve_is_a_numerical_failure():
    model = physics.robot()
    broken = model._replace(mass=model.mass * 0, inertia=model.inertia * 0)
    state, diagnostics = physics.step(
        physics.initial_state(broken), torch.zeros(2, dtype=torch.float64), broken
    )
    with pytest.raises(FloatingPointError, match="Non-finite"):
        physics.check(state, diagnostics)
