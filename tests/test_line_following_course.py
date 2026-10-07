import numpy as np
import pytest
import torch
from torch.func import vmap

from coursekit.common import CourseError
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import (
    GROUND,
    INK,
    SENSOR_DEPTH,
    SENSOR_HALF,
    SENSOR_WIDTH,
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
    course = Course.polyline("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    assert move(course, 0.0, pose(), pose(x=1.0)) == pytest.approx(1.004)
    assert move(course, 1.004, pose(x=1.0), pose(x=0.1)) == pytest.approx(1.004)
    assert move(course, 1.004, pose(x=0.1), pose(x=1.2)) == pytest.approx(1.204)


@pytest.mark.parametrize("offset", [0.0, 0.046, 0.048, 0.055, 0.060])
def test_route_must_pass_between_wheels_even_when_it_is_under_the_chassis(offset):
    course = Course.polyline("parallel", [[0, offset], [2, offset]])
    progress = move(course, 0.0, pose(), pose(x=1.0))
    assert progress == pytest.approx(1.004 if offset < 0.047 else 0.0)
    # A paint edge may still be visible when its centreline lies outside the row.
    if offset == 0.048:
        assert (optical(course, np.array([1, 0, 0.035]), np.eye(3)) < 0.9).any()
    ahead = Course.polyline("ahead", [[0.02, 0], [2, 0]])
    assert move(ahead, 0.0, pose(), pose()) == 0.0


def test_stationary_coverage_uses_the_projected_sensor_rectangle():
    for yaw, roll, pitch in [(0, 0, 0), (0.7, 0.1, -0.25), (-1.2, -0.2, 0.3)]:
        q, r = pose(x=1, y=0.4, yaw=yaw, roll=roll, pitch=pitch)
        for offset in (-0.048, -0.03, 0, 0.03, 0.048):
            points = np.array([[-0.2, offset], [0.2, offset]]) @ r[:2, :2].T + q[:2]
            course = Course.polyline("across-row", points)
            intervals = swept_intervals(course, q, q, r, r)
            if abs(offset) > SENSOR_WIDTH / 2:
                assert not len(intervals)
            else:
                assert len(intervals)
                expected = (0.2 + np.array([-1, 1]) * SENSOR_DEPTH / 2) * np.linalg.norm(
                    r[:2, 0]
                )
                np.testing.assert_allclose(
                    intervals, np.broadcast_to(expected, intervals.shape), atol=1e-12
                )


def test_route_gap_is_not_excused_by_correct_endpoint_positions():
    course = Course.polyline("detour", np.array([[0.0, 0.0], [0.2, 0.2], [1.0, 0.0]]))
    progress = move(course, 0.0, pose(), pose(x=1.0))
    assert progress == pytest.approx(SENSOR_HALF[1] * np.sqrt(2))
    assert progress < course.length / 4


def test_rejoin_ahead_freezes_until_missing_interval_is_recovered():
    course = Course.polyline("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    frontier = 0.2
    assert move(course, frontier, pose(x=1.0), pose(x=1.2)) == frontier
    assert move(course, frontier, pose(x=1.2), pose(x=0.3, y=0.3)) == frontier
    assert move(course, frontier, pose(x=0.1), pose(x=0.2)) == pytest.approx(0.204)


def test_crossing_does_not_switch_route_branch():
    course = Course.polyline(
        "crossing", np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    )
    frontier = np.sqrt(2) - 0.03
    result = move(course, frontier, pose(x=-0.02, y=-0.02), pose(x=0.02, y=0.02))
    assert frontier < result < np.sqrt(2) + 0.15


def test_rotating_rectangle_does_not_accept_its_convex_hull():
    # The hull of the two poses contains (.008, 0), but the rotating sensor never does.
    course = Course.polyline("edge", np.array([[0.003, 0.0], [0.008, 0.0]]))
    start, end = pose(yaw=-0.2), pose(yaw=0.2)
    result = move(course, 0.0, start, end)
    assert result == pytest.approx(SENSOR_HALF[0] / np.cos(0.2) - 0.003)
    assert result < course.length
    angles = np.linspace(-0.2, 0.2, 1001)
    assert (0.008 * np.cos(angles) > SENSOR_HALF[0]).all()


def test_certified_sweep_is_inside_independently_sampled_motion():
    course = Course.polyline("curve", np.array([[-0.2, -0.04], [0.1, 0.01], [0.3, 0.08]]))
    start, end = (
        pose(roll=0.08, pitch=-0.1, yaw=-0.1),
        pose(x=0.12, y=0.02, yaw=0.15, pitch=0.05),
    )
    q0, r0 = start
    q1, r1 = end
    intervals = swept_intervals(course, q0, q1, r0, r1)
    assert len(intervals)
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
    assert (np.abs(local) <= SENSOR_HALF + 1e-10).all(axis=2).any(axis=0).all()


def test_optics_average_only_the_transverse_row():
    course = Course.polyline("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    rgb = optical(course, np.array([1.0, 0.0, 0.035]), np.eye(3))
    np.testing.assert_allclose(rgb[[7, 8]], np.broadcast_to(INK, (2, 3)), atol=1e-7)
    np.testing.assert_allclose(rgb[:7], np.broadcast_to(GROUND, (7, 3)), atol=1e-7)
    ahead = Course.polyline("ahead", np.array([[0.02, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(
        optical(ahead, np.zeros(3), np.eye(3)), np.broadcast_to(GROUND, (16, 3)), atol=1e-7
    )


def test_course_archive_and_reversed_coordinates(tmp_path):
    course = Course.polyline(
        "corner", np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), [False, True, False]
    )
    file = tmp_path / "courses.npz"
    save_courses(
        file, [course, Course.polyline("short", np.array([[0.0, 0.0], [0.2, 0.0]]))]
    )
    restored = load_courses(file)
    np.testing.assert_array_equal(restored[0].controls, course.controls)
    np.testing.assert_array_equal(restored[0].corners, course.corners)
    np.testing.assert_allclose(
        course.reversed().at(np.array([0.2, 1.1])), course.at(np.array([1.8, 0.9]))
    )
    with pytest.raises(CourseError, match="distinct"):
        Course.polyline("invalid", np.zeros((2, 2)))


def test_straight_reference_matches_acceleration_limited_travel():
    course = Course.polyline("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
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
    corner = Course.polyline(
        "corner", np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), [False, True, False]
    )
    straight = Course.polyline("straight", np.array([[0.0, 0.0], [2.0, 0.0]]))
    t = reference_time(corner, physics.robot())
    assert np.isfinite(t) and t > reference_time(straight, physics.robot())


def test_sensor_stays_between_wheels_and_can_resolve_nearby_lines():
    from coursekit.labs.line_following_course import (
        LINE_WIDTH,
        SENSOR_DEPTH,
        SENSOR_WIDTH,
        sensor_areas,
    )

    centre = np.array([1.0, 0.0, 0.035])
    areas = sensor_areas(centre, np.eye(3))
    model = physics.robot()
    inner_edge = float(model.offset[1, 1]) - 0.016 / 2
    np.testing.assert_allclose(areas[..., 1].min(), -inner_edge)
    np.testing.assert_allclose(areas[..., 1].max(), inner_edge)
    np.testing.assert_allclose(np.ptp(areas[..., 0]), SENSOR_DEPTH)
    np.testing.assert_allclose(areas[:, 3, 1] - areas[:, 0, 1], SENSOR_WIDTH / 16)
    np.testing.assert_allclose(LINE_WIDTH, 2 * SENSOR_WIDTH / 16)
    under_wheel = Course.polyline("under-wheel", [[0, 0.055], [2, 0.055]])
    np.testing.assert_allclose(
        optical(under_wheel, centre, np.eye(3)), np.broadcast_to(GROUND, (16, 3)), atol=1e-7
    )
    paired = Course.polyline(
        "two-lines", [[0, 0], [2, 0], [2, SENSOR_WIDTH / 2], [0, SENSOR_WIDTH / 2]]
    )
    dark = optical(paired, centre, np.eye(3)).mean(axis=1) < 0.5
    assert np.count_nonzero(dark & ~np.r_[False, dark[:-1]]) == 2
    assert dark[7:9].all() and dark[-1]
    for yaw, roll, pitch in [(0, 0, 0), (0.7, 0.1, -0.25), (-1.2, -0.2, 0.3)]:
        q, r = pose(x=1, y=0.4, yaw=yaw, roll=roll, pitch=pitch)
        expected = (areas - centre[:2]) @ r[:2, :2].T + q[:2]
        np.testing.assert_allclose(sensor_areas(q[:3], r), expected, atol=1e-15)
