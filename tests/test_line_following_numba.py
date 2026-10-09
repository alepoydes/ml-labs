"""Independent mechanical invariants and agreement with the autodiff derivation."""

import numpy as np
import pytest
import torch
from torch.func import grad

from coursekit.labs import line_following as reference
from coursekit.labs import line_following_geometry as geometry
from coursekit.labs import line_following_numba as native
from coursekit.labs.line_following_course import advance, load_courses, optical
from coursekit.labs.line_following_env import Environment, sample_robot


def arrays(model):
    return reference.Robot(*(v.numpy() for v in model))


def test_analytic_derivatives_and_material_contact_velocities():
    rng = np.random.default_rng(42)
    for height in (0.032, 0.2):
        for _ in range(4):
            model = sample_robot(rng)
            q = torch.from_numpy(rng.normal(0, 0.2, 8))
            q[2] = height
            v = torch.from_numpy(rng.normal(0, 0.8, 8))
            m = arrays(model)
            r, e, _, w, offsets, linear = native.kinematics(q.numpy(), m)
            g, contact, bearing = native.contacts(q.numpy(), r, w, offsets, linear, m)
            np.testing.assert_allclose(
                g, reference.contact_geometry(q, model)[0], atol=1e-15
            )
            speeds = torch.from_numpy(rng.normal(0, 1, 11))
            np.testing.assert_allclose(
                contact @ speeds.numpy(),
                reference.contact_velocity(q, speeds, model),
                atol=1e-15,
            )
            np.testing.assert_allclose(
                bearing @ speeds.numpy(),
                reference.relative_spin(q, speeds, model),
                atol=1e-15,
            )
            np.testing.assert_allclose(
                native.inertia(linear, e, m), reference.mass_matrix(q, v, model), atol=1e-15
            )
            lq, lv = native.lagrangian_derivatives(q.numpy(), v.numpy(), m)
            np.testing.assert_allclose(
                lq, grad(reference.lagrangian, 0)(q, v, model), atol=2e-12, rtol=2e-12
            )
            np.testing.assert_allclose(lv, reference.momentum(q, v, model), atol=1e-15)


def test_steps_match_autodiff_with_contacts_airborne_motion_and_changed_parameters():
    rng = np.random.default_rng(92)
    for airborne in (False, True):
        model = sample_robot(rng)
        state = reference.initial_state(model)
        if airborne:
            state.q[2] = 0.3
            state.q[3:6] = torch.tensor([0.1, 0.2, 0.3])
            state = state._replace(
                p=reference.momentum(
                    state.q, torch.from_numpy(rng.normal(0, 0.2, 8)), model
                )
            )
        q, p, spin = (v.numpy().copy() for v in state[:3])
        m = arrays(model)
        for _ in range(12):
            action = torch.from_numpy(rng.uniform(-0.3, 0.3, 2))
            q, p, spin, fell, residual, impulse = native.step(q, p, spin, action.numpy(), m)
            state, diagnostics = reference.step(state, action, model)
            reference.check(state, diagnostics)
            for a, b in zip((q, p, spin), state[:3]):
                np.testing.assert_allclose(a, b, atol=2e-11, rtol=1e-9)
            assert fell == bool(state.fallen)
            assert residual < 1e-9
            np.testing.assert_allclose(impulse, diagnostics.impulse, atol=2e-12, rtol=1e-8)


def test_free_motion_momentum_reversibility_and_passive_contact_losses():
    model = reference.robot()
    model = model._replace(gravity=model.gravity * 0)
    m = arrays(model)
    q = np.array([0.1, 0.2, 0.5, 0.1, 0.2, 0.3, 0.0, 0.0])
    velocity = np.array([0.3, -0.1, 0.2, 0.1, 0.3, 0.4, 2.0, -1.0])
    p = native.mass_matrix(q, m) @ velocity
    q1, p1, residual = native.conservative_step(q, p, m, 0.002)
    back_q, back_p, _ = native.conservative_step(q1, p1, m, -0.002)
    np.testing.assert_allclose(back_q, q, atol=1e-12)
    np.testing.assert_allclose(back_p, p, atol=1e-12)
    np.testing.assert_allclose(p1[:3], p[:3], atol=1e-13)  # free linear momentum
    assert residual < 1e-12
    total = np.r_[p, [1e-5, -2e-5, 3e-5]]

    def kinetic(momentum):
        return (
            momentum[:8] @ np.linalg.solve(native.mass_matrix(q, m), momentum[:8])
            + momentum[8:] @ momentum[8:] / m.inertia[3, 0]
        ) / 2

    before = kinetic(total)
    after, impulse = native.resistance(q, total, np.zeros(2), m, 0.001)
    assert kinetic(after) < before
    np.testing.assert_array_equal(impulse, 0)  # airborne bearing and motor losses only
    q[2] = 0.03
    after, impulse = native.resistance(q, total, np.zeros(2), m, 0.001)
    assert kinetic(after) <= kinetic(total) + 1e-12
    assert (impulse[:, 2] >= 0).all()
    r, _, _, w, offsets, linear = native.kinematics(q, m)
    gap, _, _ = native.contacts(q, r, w, offsets, linear, m)
    load = impulse[:, 2] + 0.001 * m.stiffness * np.maximum(-gap, 0)
    assert (np.linalg.norm(impulse[:, :2], axis=1) <= m.friction * load + 1e-14).all()
    assert (impulse[gap > 0] == 0).all()


def test_internal_drive_and_bearing_torques_conserve_angular_momentum():
    m = arrays(reference.robot())
    q = np.array([0.1, 0.2, 0.5, 0.1, 0.2, 0.3, 0.0, 0.0])
    p = native.mass_matrix(q, m) @ np.array([0.3, -0.1, 0.2, 0.1, 0.3, 0.4, 2, -1])
    total = np.r_[p, [1e-5, -2e-5, 3e-5]]
    r = native.rotation(q)
    e, _ = native.angular_map(q)

    def angular(momentum):
        return (
            np.cross(q[:3], momentum[:3])
            + np.linalg.solve((r @ e).T, momentum[3:6])
            + momentum[8:]
        )

    for action in (np.zeros(2), np.array([0.7, -0.3])):
        after, impulse = native.resistance(q, total, action, m, 0.001)
        np.testing.assert_array_equal(impulse, 0)
        np.testing.assert_allclose(after[:3], total[:3], atol=1e-14)
        np.testing.assert_allclose(angular(after), angular(total), atol=1e-13)


@pytest.mark.parametrize("contact", [False, True])
def test_trajectory_convergence_with_time_step(contact):
    model = reference.robot()
    initial = reference.initial_state(model)
    m = arrays(model)
    initial_q, initial_p, initial_spin = (v.numpy().copy() for v in initial[:3])
    if not contact:
        initial_q[2] = 0.5
        initial_q[3:6] = [0.1, 0.2, 0.3]
        initial_p = native.mass_matrix(initial_q, m) @ np.array(
            [0.3, -0.1, 0.2, 0.1, 0.3, 0.4, 2, -1]
        )
    results = []
    for dt in (0.002, 0.001, 0.0005) if contact else (0.01, 0.005, 0.0025):
        q, p, spin = initial_q.copy(), initial_p.copy(), initial_spin.copy()
        for _ in range(round((0.02 if contact else 0.05) / dt)):
            if contact:
                q, p, spin, fell, _, _ = native.step(
                    q, p, spin, np.array([0.2, 0.2]), m, dt
                )
                assert not fell
            else:
                q, p, residual = native.conservative_step(q, p, m, dt)
                assert residual < 1e-10
        results.append(np.r_[q, np.linalg.solve(native.mass_matrix(q, m), p)])
    ratio = np.linalg.norm(results[1] - results[2]) / np.linalg.norm(
        results[0] - results[1]
    )
    assert ratio < 0.65 if contact else 0.2 < ratio < 0.3


def test_native_step_rejects_invalid_time_and_nonconvergence():
    model = reference.robot()
    state = reference.initial_state(model)
    args = [v.numpy() for v in state[:3]] + [np.zeros(2), arrays(model)]
    for dt in (0, -0.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive"):
            native.step(*args, dt)
    with pytest.raises(FloatingPointError, match="converge"):
        native.step(*args, 0.1)


def test_spline_optics_and_ordered_coverage_match_certified_reference():
    from pathlib import Path

    courses = load_courses(
        Path(__file__).parents[1] / "labs/line_following/resources/courses.npz"
    )
    rng = np.random.default_rng(321)
    for course in courses[::5]:
        g = geometry.prepare(course)
        for _ in range(6):
            distance = rng.uniform(0, course.length)
            q = np.zeros(8)
            q[:2] = course.at(distance) + rng.normal(0, 0.004, 2)
            q[3:6] = rng.normal(0, 0.3, 3)
            direction = course.at(distance, 1)
            q[5] += np.arctan2(direction[1], direction[0])
            other = q.copy()
            other[:2] += rng.normal(0, 0.008, 2)
            other[3:6] += rng.normal(0, 0.01, 3)
            r, r1 = native.rotation(q), native.rotation(other)
            np.testing.assert_allclose(
                geometry.advance(g, distance, q, other, r, r1),
                advance(course, distance, q, other, r, r1),
                atol=1e-11,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                geometry.optical(g, q, r), optical(course, q, r), atol=1e-7
            )


def test_native_block_preserves_every_frame_and_terminal_rows_without_torch_compile(
    monkeypatch,
):
    from coursekit.labs.line_following_course import Course

    monkeypatch.setattr(
        torch, "compile", lambda *a, **k: pytest.fail("Optional compiler used")
    )
    course = Course.polyline("straight", np.array([[0.0, 0.0], [1.0, 0.0]]))
    plain = Environment([course, course], [13, 15], noise=False, time_limit=0.007)
    recorded = Environment([course, course], [13, 15], noise=False, time_limit=0.007)
    plain.status[1] = recorded.status[1] = "timeout"
    snapshots = []
    plain.step(torch.full((2, 2), 0.2))
    recorded.step(torch.full((2, 2), 0.2), on_step=lambda e: snapshots.append(e.info()))
    assert len(snapshots) == 7
    for a, b in zip(plain.state, recorded.state):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    np.testing.assert_array_equal(
        [v["time"][0] for v in snapshots], np.arange(1, 8) * 0.001
    )
    assert all(v["time"][1] == 0 for v in snapshots)
    assert snapshots[-1]["status"].tolist() == ["timeout", "timeout"]
    recorded.reset([True, False])
    recorded.state.p[0, 0] = float("nan")
    before = recorded.info()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        recorded.step(torch.zeros((2, 2)))
    np.testing.assert_array_equal(recorded.info()["time"], before["time"])
    np.testing.assert_array_equal(
        recorded.info()["travelled_distance"], before["travelled_distance"]
    )


def test_imported_colour_change_invalidates_native_cache(tmp_path):
    import json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import coursekit

    package = tmp_path / "coursekit"
    shutil.copytree(
        Path(coursekit.__file__).parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    env = os.environ | {
        "PYTHONPATH": str(tmp_path),
        "NUMBA_CACHE_DIR": str(tmp_path / "kernels"),
    }
    script = """
import json
import numpy as np
from coursekit.labs import line_following_geometry as g
from coursekit.labs.line_following_course import Course
from coursekit.labs.line_following_jit import CACHE_DIR
course = Course.polyline('line', [[0.,0.],[1.,0.]])
q = np.zeros(8); q[0] = .5
print(json.dumps(dict(rgb=g.optical(g.prepare(course), q, np.eye(3))[7].tolist(), cache=str(CACHE_DIR))))
"""

    def run():
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    before = run()
    helper = package / "labs/line_following_course.py"
    # Only an imported helper changes: the cached optical function is untouched.
    helper.write_text(helper.read_text() + "\nINK = np.array([0.25, 0.5, 0.75])\n")
    after = run()
    np.testing.assert_allclose(after["rgb"], [0.25, 0.5, 0.75], atol=1e-7)
    assert before["rgb"] != after["rgb"] and before["cache"] != after["cache"]
    assert all(any(Path(value["cache"]).rglob("*.nbc")) for value in (before, after))
