import numpy as np
import pytest
from scipy.integrate import quad

from coursekit.common import CourseError
from coursekit.labs.line_following import robot
from coursekit.labs.line_following_course import (
    GROUND,
    INK,
    advance,
    colour,
    reference_time,
)
from coursekit.labs.line_following_curve import Course, bezier, load_courses, split


def parabola():
    return Course("parabola", np.array([[[0, 0], [1 / 3, 0], [2 / 3, 1 / 3], [1, 1]]]))


def subdivide(course):
    left, right = split(course.controls)
    controls = np.stack((left, right), axis=1).reshape(-1, 4, 2)
    knots = np.sort(np.r_[course.knots, (course.knots[:-1] + course.knots[1:]) / 2])
    corners = np.zeros(len(knots), dtype=bool)
    corners[::2] = course.corners
    return Course(course.name, controls, knots, corners)


@pytest.mark.parametrize(
    "course",
    [
        parabola(),
        Course(
            "C2-turn",
            np.array(
                [
                    [[0, 0], [1 / 3, 0], [2 / 3, 0], [5 / 6, 1 / 6]],
                    [[5 / 6, 1 / 6], [1, 1 / 3], [1, 2 / 3], [1, 1]],
                ]
            ),
        ),
    ],
)
def test_exact_subdivision_preserves_geometry_length_and_reference(course):
    original = course
    distances = np.linspace(0, course.length, 73)
    expected_time = reference_time(course, robot())
    for _ in range(5):
        course = subdivide(course)
        assert course.length == pytest.approx(original.length, abs=2e-12)
        for order in range(4):
            np.testing.assert_allclose(
                course.at(distances, order), original.at(distances, order), atol=2e-9
            )
        np.testing.assert_allclose(
            course.geometry(distances), original.geometry(distances), atol=2e-9
        )
        assert reference_time(course, robot()) == pytest.approx(expected_time, rel=1e-9)


def test_arc_length_curvature_and_change_match_analytic_parabola():
    course = parabola()
    t = np.linspace(0, 1, 31)
    s = np.array(
        [quad(lambda u: np.sqrt(1 + 4 * u**2), 0, end, epsabs=1e-12)[0] for end in t]
    )
    np.testing.assert_allclose(course.at(s), np.c_[t, t**2], atol=1e-11)
    assert course.length == pytest.approx(s[-1], abs=1e-12)
    k, change = course.geometry(s)
    np.testing.assert_allclose(k, 2 / (1 + 4 * t**2) ** 1.5, atol=1e-10)
    np.testing.assert_allclose(change, -24 * t / (1 + 4 * t**2) ** 3, atol=1e-10)
    reverse = course.reversed()
    np.testing.assert_allclose(reverse.at(s), course.at(course.length - s), atol=1e-11)
    np.testing.assert_allclose(
        reverse.geometry(s)[0], -course.geometry(course.length - s)[0], atol=1e-10
    )
    values = [
        reference_time(course, robot(), spacing=h) for h in (0.01, 0.005, 0.0025, 0.00125)
    ]
    assert abs(values[-1] - values[-2]) / values[-1] < 1e-3
    assert abs(values[-1] - values[1]) / values[-1] < 1e-3


def test_sensor_uses_the_curve_and_certified_sweep_does_not_credit_its_chord():
    course = parabola()
    t = np.array([0.3, 0.5, 0.7])
    positions = np.c_[t, t**2]
    normals = np.c_[-2 * t, np.ones(3)] / np.sqrt(1 + 4 * t**2)[:, None]
    np.testing.assert_allclose(
        colour(course, positions + 0.007999 * normals), np.tile(INK, (3, 1))
    )
    np.testing.assert_allclose(
        colour(course, positions + 0.008001 * normals), np.tile(GROUND, (3, 1))
    )
    arch = Course("arch", np.array([[[0, 0], [0.3, 0.4], [0.7, 0.4], [1, 0]]]))
    start = np.array([0.0, 0, 0.035, 0, 0, 0, 0, 0])
    end = start.copy()
    end[0] = 1
    frontier = advance(arch, 0.0, start, end, np.eye(3), np.eye(3))
    assert 0 < frontier < arch.length / 4
    points = arch.at(np.linspace(0, frontier, 301))
    assert (np.abs(points[:, 1]) <= 0.064 + 1e-12).all()


def test_invalid_tangents_joins_and_old_archive_are_rejected(tmp_path):
    with pytest.raises(CourseError, match="tangents"):
        Course("cusp", np.array([[[-0.125, 0], [0.125, 0], [-0.125, 0], [0.125, 0]]]))
    bent = Course.polyline("corner", [[0, 0], [1, 0], [1, 1]])
    with pytest.raises(CourseError, match="C2"):
        Course("unmarked", bent.controls, bent.knots)
    archive = tmp_path / "old.npz"
    np.savez_compressed(archive, points=np.array([[0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(CourseError, match="format 2"):
        load_courses(archive)


def test_chord_certificates_bound_every_parameter_value():
    course = parabola()
    t = np.linspace(0, 1, 97)
    chord = course._leaves[:, None, 0] + t[None, :, None] * (
        course._leaves[:, None, -1] - course._leaves[:, None, 0]
    )
    actual = bezier(course._leaves[:, None], t)
    assert (np.linalg.norm(actual - chord, axis=-1) <= course._error[:, None] + 1e-14).all()


def test_arc_inversion_handles_nonuniform_parameter_speed():
    course = Course(
        "slow-parameter", np.array([[[0, 0], [1e-6 / 3, 0], [2e-6 / 3, 0], [1.000001, 0]]])
    )
    distance = np.r_[0.0, np.geomspace(1e-12, 1.0, 100), course.length]
    np.testing.assert_allclose(
        course.at(distance), np.c_[distance, np.zeros_like(distance)], atol=2e-12, rtol=0
    )
