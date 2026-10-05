import numpy as np
import pytest
import torch
from torch.func import vmap

from coursekit.common import CourseError
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import (
    FOOTPRINT,
    GROUND,
    INK,
    Course,
    advance,
    load_courses,
    optical,
    reference_time,
    save_courses,
    swept_intervals,
)


def pose(x=0.0, y=0.0, yaw=0.0, roll=0.0, pitch=0.0):
    q = np.array([x, y, 0.035, roll, pitch, yaw, 0.0, 0.0])
    return q, physics.rotation(torch.from_numpy(q)).numpy()


def move(course, frontier, start, end):
    q0, r0 = start
    q1, r1 = end
    return advance(course, frontier, q0, q1, r0, r1)


def test_long_step_covers_whole_translation_without_progress_cap():
    course = Course("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    assert move(course, 0.0, pose(), pose(x=1.0)) == pytest.approx(1.07)
    assert move(course, 1.07, pose(x=1.0), pose(x=0.1)) == pytest.approx(1.07)
    assert move(course, 1.07, pose(x=0.1), pose(x=1.2)) == pytest.approx(1.27)


def test_route_gap_is_not_excused_by_correct_endpoint_positions():
    course = Course("detour", np.array([[0.0, 0.0], [0.2, 0.2], [1.0, 0.0]]))
    progress = move(course, 0.0, pose(), pose(x=1.0))
    assert progress == pytest.approx(FOOTPRINT[1] * np.sqrt(2))
    assert progress < course.length / 4


def test_rejoin_ahead_freezes_until_missing_interval_is_recovered():
    course = Course("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    frontier = 0.2
    assert move(course, frontier, pose(x=1.0), pose(x=1.2)) == frontier
    assert move(course, frontier, pose(x=1.2), pose(x=0.3, y=0.3)) == frontier
    assert move(course, frontier, pose(x=0.1), pose(x=0.2)) == pytest.approx(0.27)


def test_crossing_does_not_switch_route_branch():
    course = Course(
        "crossing", np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    )
    frontier = np.sqrt(2) - 0.03
    result = move(course, frontier, pose(x=-0.02, y=-0.02), pose(x=0.02, y=0.02))
    assert frontier < result < np.sqrt(2) + 0.15


def test_rotating_rectangle_does_not_accept_its_convex_hull():
    # The hull of the two poses contains (.078, 0), but the rotating chassis never does.
    course = Course("edge", np.array([[0.065, 0.0], [0.078, 0.0]]))
    start, end = pose(yaw=-0.2), pose(yaw=0.2)
    result = move(course, 0.0, start, end)
    assert result == pytest.approx(FOOTPRINT[0] / np.cos(0.2) - 0.065)
    assert result < course.length
    angles = np.linspace(-0.2, 0.2, 1001)
    assert (0.078 * np.cos(angles) > FOOTPRINT[0]).all()


def test_certified_sweep_is_inside_independently_sampled_motion():
    course = Course("curve", np.array([[-0.2, -0.04], [0.1, 0.01], [0.3, 0.08]]))
    start, end = (
        pose(roll=0.08, pitch=-0.1, yaw=-0.1),
        pose(x=0.12, y=0.02, yaw=0.15, pitch=0.05),
    )
    q0, r0 = start
    q1, r1 = end
    intervals = swept_intervals(course, q0, q1, r0, r1)
    queries = course.at(
        (intervals[:, :1] + np.array([0.2, 0.5, 0.8]) * np.diff(intervals)).ravel()
    )
    poses = q0 + np.linspace(0, 1, 4001)[:, None] * (q1 - q0)
    rotations = vmap(physics.rotation)(torch.from_numpy(poses)).numpy()
    local = np.einsum(
        "tij,tpj->tpi",
        np.linalg.inv(rotations[:, :2, :2]),
        queries[None] - poses[:, None, :2],
    )
    assert (np.abs(local) <= FOOTPRINT + 1e-10).all(axis=2).any(axis=0).all()


def test_optics_average_only_the_transverse_row():
    course = Course("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    rgb = optical(course, np.array([1.0, 0.0, 0.035]), np.eye(3))
    np.testing.assert_allclose(rgb[[7, 8]], np.broadcast_to(INK, (2, 3)), atol=1e-7)
    np.testing.assert_allclose(rgb[:7], np.broadcast_to(GROUND, (7, 3)), atol=1e-7)
    ahead = Course("ahead", np.array([[0.02, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(
        optical(ahead, np.zeros(3), np.eye(3)), np.broadcast_to(GROUND, (16, 3)), atol=1e-7
    )


def test_course_archive_and_reversed_coordinates(tmp_path):
    course = Course(
        "corner", np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), [False, True, False]
    )
    file = tmp_path / "courses.npz"
    save_courses(file, [course, Course("short", np.array([[0.0, 0.0], [0.2, 0.0]]))])
    restored = load_courses(file)
    np.testing.assert_array_equal(restored[0].points, course.points)
    np.testing.assert_array_equal(restored[0].corners, course.corners)
    np.testing.assert_allclose(
        course.reversed().at(np.array([0.2, 1.1])), course.at(np.array([1.8, 0.9]))
    )
    with pytest.raises(CourseError, match="distinct"):
        Course("invalid", np.zeros((2, 2)))


def test_straight_reference_matches_acceleration_limited_travel():
    course = Course("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    model = physics.robot()
    vmax = float((model.radius[:2] * model.speed).min())
    acceleration = min(
        float(model.friction[:2].min() * model.gravity),
        float((model.stall / model.radius[:2]).sum() / model.mass.sum()),
    )
    expected = 2 * course.length / vmax + 1.5 * vmax / acceleration
    assert reference_time(course, model) == pytest.approx(expected, rel=2e-3)
    slower = model._replace(speed=model.speed / 2)
    assert reference_time(course, slower) > reference_time(course, model)


def test_corner_reference_is_finite_and_slower_than_straight():
    corner = Course(
        "corner", np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), [False, True, False]
    )
    straight = Course("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    t = reference_time(corner, physics.robot())
    assert np.isfinite(t) and t > reference_time(straight, physics.robot())
