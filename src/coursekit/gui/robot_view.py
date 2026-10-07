"""Shared scene, sensor and diagnostics for local and instructor recordings."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from coursekit.gui.common import Diagnostics, pixmap
from coursekit.inspection import copy_debug
from coursekit.labs.line_following_course import Course


def recorded_courses(record):
    """Validate renderable frames before replacing the currently open recording."""
    if (record["lab"], record["version"]) != ("line_following", 2):
        raise ValueError(
            "Expected Line Following spline recording version 2; old polyline recordings are incompatible."
        )
    courses = [Course(**c) for c in record["courses"]]
    n = len(courses)
    frames, controls, model = record["frames"], record["controls"], record["model"]
    count = len(frames["q"])
    if (
        not n
        or not count
        or not np.isfinite(record["physics_dt"])
        or record["physics_dt"] <= 0
    ):
        raise ValueError("A recording needs robots, frames and a positive time step.")
    for name, value in frames.items():
        value = np.asarray(value)
        if value.ndim == 0 or len(value) != count:
            raise ValueError(f"Invalid recorded frame field {name}.")
        frames[name] = value
    for name, shape in [
        ("q", (count, n, 8)),
        ("time", (count, n)),
        ("progress", (count, n)),
        ("score", (count, n)),
    ]:
        value = np.asarray(frames[name])
        if (
            value.shape != shape
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid recorded {name}.")
    if (
        np.asarray(frames["status"]).shape != (count, n)
        or "leg" in frames
        and np.asarray(frames["leg"]).shape != (count, n)
    ):
        raise ValueError("Invalid recorded robot status.")
    index = np.asarray(frames["control"])
    if (
        index.shape != (count,)
        or index.dtype.kind not in "iu"
        or (index < -1).any()
        or (index >= len(controls)).any()
        or (index[1:] < index[:-1]).any()
    ):
        raise ValueError("Invalid recorded controller indices.")
    for name, shape in [("radius", (n, 3)), ("offset", (n, 4, 3))]:
        if np.asarray(model[name]).shape != shape or not np.isfinite(model[name]).all():
            raise ValueError(f"Invalid recorded robot {name}.")
    for name, value in model.items():
        value = np.asarray(value)
        if value.ndim == 0 or len(value) != n:
            raise ValueError("Invalid recorded model batch.")
        model[name] = value
    for control in controls:
        control["action"] = np.asarray(control["action"])
        observation = control["observation"]
        if (
            np.asarray(control["action"]).shape != (n, 2)
            or not np.isfinite(control["action"]).all()
        ):
            raise ValueError("Invalid recorded motor commands.")
        if np.asarray(observation["rgb"]).shape != (n, 16, 3) or np.asarray(
            observation["time"]
        ).shape != (n,):
            raise ValueError("Invalid recorded sensors.")
        for name, value in observation.items():
            value = np.asarray(value)
            if (
                value.ndim == 0
                or len(value) != n
                or value.dtype.kind not in "biuf"
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid recorded sensor {name}.")
            observation[name] = value
        if (
            (observation["rgb"] < 0).any()
            or (observation["rgb"] > 1).any()
            or (observation["time"] < 0).any()
        ):
            raise ValueError("Invalid recorded sensor range.")
        debug = control.get("debug", [{}] * n)
        if len(debug) != n or any(not isinstance(row, dict) for row in debug):
            raise ValueError("Invalid recorded diagnostics.")
        control["debug"] = [copy_debug(row) for row in debug]
    return courses


class RobotView(QWidget):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.diagnostics = Diagnostics()
        self.sensor = QLabel("Sensor readings appear after the first controller call.")
        self.sensor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status = QLabel("Choose a course to preview the track.")
        self.status.setWordWrap(True)
        self.status.setObjectName("selection")
        self.follow, self.overhead = QCheckBox("Follow robot"), QCheckBox("Top view")
        self.follow.toggled.connect(lambda value: self.camera("follow", value))
        self.overhead.toggled.connect(lambda value: self.camera("top", value))
        self.fit = QPushButton("Fit track")
        self.fit.clicked.connect(self.fit_track)
        camera = QHBoxLayout()
        for widget in [self.follow, self.overhead, self.fit]:
            camera.addWidget(widget)
        camera.addStretch()
        hint = QLabel("Wheel: zoom · drag: pan")
        hint.setObjectName("muted")
        camera.addWidget(hint)
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(camera)
        layout.addWidget(scene, 1)
        layout.addWidget(QLabel("Optical sensor · left → right"))
        layout.addWidget(self.sensor)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.diagnostics)
        split.setSizes([780, 300])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.status)
        outer.addWidget(split, 1)

    def camera(self, name, value):
        setattr(self.scene, name, value)
        self.scene.update()

    def fit_track(self):
        self.follow.setChecked(False)
        self.scene.zoom, self.scene.pan = 1.0, np.zeros(2)
        self.scene.update()

    def show_frame(self, course, frame, model, controls, row=0, *, pitch_history=None):
        self.scene.show_pose(course, frame["q"][row], {k: v[row] for k, v in model.items()})
        leg = "return" if "leg" in frame and frame["leg"][row] else "outward"
        self.status.setText(
            f"{course.name} · {frame['time'][row]:.3f} s · {frame['status'][row]} · {leg} · "
            f"{frame['progress'][row]:.3f} / {2 * course.length:.3f} m · {frame['score'][row]:.2f} points"
        )
        index = int(frame["control"])
        if index < 0:
            self.diagnostics.show_snapshot({})
            self.sensor.clear()
            self.sensor.setText("No controller call at this frame.")
            return
        control = controls[index]
        observation = control["observation"]
        self.sensor.setPixmap(
            pixmap(np.rint(observation["rgb"][row][None] * 255).astype(np.uint8)).scaled(
                512, 32
            )
        )
        debug = control.get("debug", [{}] * len(control["action"]))[row]
        values = {
            **debug,
            **{f"sensor.{k}": v[row] for k, v in observation.items() if k != "rgb"},
            "left motor": float(control["action"][row, 0]),
            "right motor": float(control["action"][row, 1]),
            "pitch (rad)": float(frame["q"][row, 4]),
        }

        def history(key):
            # Build only the selected plot, including when selection changes on pause.
            if key == "pitch (rad)":
                return pitch_history() if callable(pitch_history) else pitch_history
            if key in ("left motor", "right motor"):
                column = int(key == "right motor")
                return (
                    [float(c["observation"]["time"][row]) for c in controls[: index + 1]],
                    [float(c["action"][row, column]) for c in controls[: index + 1]],
                )
            if isinstance(debug.get(key), (int, float)):
                samples = [
                    (
                        c["observation"]["time"][row],
                        c.get("debug", [{}] * len(c["action"]))[row].get(key),
                    )
                    for c in controls[: index + 1]
                ]
                samples = [(t, v) for t, v in samples if isinstance(v, (int, float))]
                if samples:
                    return tuple(zip(*samples))
            return None

        self.diagnostics.show_snapshot(values, history)
