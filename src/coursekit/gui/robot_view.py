"""Shared scene, sensor and diagnostics for local and instructor recordings."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from coursekit.gui.common import Diagnostics, pixmap
from coursekit.inspection import DebugHistory, copy_debug
from coursekit.labs.line_following_course import SENSOR_DEPTH, SENSOR_WIDTH, Course
from coursekit.labs.line_following_results import OUTCOMES
from coursekit.recording import robot_metadata


def recorded_courses(record):
    """Validate renderable frames before replacing the currently open recording."""
    expected = robot_metadata()
    controller = record.get("controller")
    if controller is not None:
        if not isinstance(controller, dict) or any(
            not isinstance(controller.get(key), str) or not controller[key]
            for key in (
                "name",
                "path",
                "preparation",
                "version",
                "source_hash",
                "config_hash",
                "artifacts_hash",
            )
        ):
            raise ValueError("Invalid recorded controller identity.")
    if any(record.get(key) != expected[key] for key in ("lab", "version")):
        raise ValueError(
            "Expected Line Following recording version 4. Earlier recordings do not "
            "identify compatible sensor geometry and coverage rules; create a new recording."
        )
    if any(
        record.get(key) != expected[key]
        for key in ("geometry", "coverage_version", "case_version")
    ):
        raise ValueError(
            "Recording geometry or coverage rules differ from this viewer. "
            "Use the matching coursekit version or create a new recording."
        )
    courses = [Course(**c) for c in record["courses"]]
    n = len(courses)
    for key in ("case_ids", "case_fingerprints"):
        value = np.asarray(record.get(key))
        if value.shape != (n,) or value.dtype.kind != "U" or any(not v for v in value):
            raise ValueError(f"Invalid recorded {key}.")
    if not isinstance(record.get("sensor_noise"), bool):
        raise ValueError("Recording must identify whether sensor noise was enabled.")
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
    if "travelled_distance" in frames:
        distance = frames["travelled_distance"]
        if (
            distance.shape != (count, n)
            or distance.dtype.kind not in "fiu"
            or not np.isfinite(distance).all()
            or (distance < 0).any()
            or (distance[1:] < distance[:-1]).any()
        ):
            raise ValueError("Invalid recorded travelled distance.")
    if "time_limit" in frames:
        limit = frames["time_limit"]
        if (
            limit.shape != (count, n)
            or limit.dtype.kind not in "fiu"
            or not np.isfinite(limit).all()
            or (limit <= 0).any()
            or (limit != limit[0]).any()
        ):
            raise ValueError("Invalid recorded time limit.")
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
    def __init__(self, scene, *, diagnostic_state=None):
        super().__init__()
        self.scene = scene
        self.diagnostics = Diagnostics(diagnostic_state)
        self.history = DebugHistory()
        self.sensor = QLabel("Sensor readings appear after the first controller call.")
        self.sensor.setWordWrap(True)
        self.sensor.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.sensor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status = QLabel("Choose a course to preview the track.")
        self.status.setWordWrap(True)
        self.status.setObjectName("selection")
        self.follow, self.overhead = QCheckBox("Follow"), QCheckBox("Top view")
        self.follow.setToolTip("Keep the camera centred on the robot.")
        self.follow.toggled.connect(lambda value: self.camera("follow", value))
        self.overhead.toggled.connect(lambda value: self.camera("top", value))
        self.sensing = QCheckBox("Sensor area")
        self.sensing.setChecked(True)
        self.sensing.setToolTip(
            "Show the 16 sensing and route-coverage areas through the chassis."
        )
        self.sensing.toggled.connect(lambda value: self.camera("sensing", value))
        self.fit = QPushButton("Fit track")
        self.fit.clicked.connect(self.fit_track)
        camera = QHBoxLayout()
        for widget in [self.follow, self.overhead, self.sensing, self.fit]:
            camera.addWidget(widget)
        camera.addStretch()
        scene.setToolTip("Wheel: zoom · drag: pan")
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(camera)
        layout.addWidget(scene, 1)
        sensor_label = QLabel(
            f"Optical sensor · {SENSOR_WIDTH * 1000:g}×{SENSOR_DEPTH * 1000:g} mm · body right (0) → left (15)"
        )
        sensor_label.setWordWrap(True)
        layout.addWidget(sensor_label)
        layout.addWidget(self.sensor)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.diagnostics)
        split.setSizes([400, 620])
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

    def show_frame(
        self,
        course,
        frame,
        model,
        controls,
        row=0,
        *,
        case_id=None,
        sensor_noise=True,
    ):
        self.scene.show_pose(course, frame["q"][row], {k: v[row] for k, v in model.items()})
        leg = "return" if "leg" in frame and frame["leg"][row] else "outward"
        outcome = OUTCOMES.get(str(frame["status"][row]), str(frame["status"][row]))
        distance = (
            f"{frame['travelled_distance'][row]:.3f} m"
            if "travelled_distance" in frame
            else "—"
        )
        limit = f"{frame['time_limit'][row]:.3f}" if "time_limit" in frame else "—"
        self.status.setText(
            f"{case_id or course.name}{' · noise OFF (diagnostic)' if not sensor_noise else ''} · {frame['time'][row]:.3f} / {limit} s · {outcome} · {leg} · {frame['score'][row]:.2f} points\n"
            f"Accepted: {frame['progress'][row]:.3f} / {2 * course.length:.3f} m "
            f"({100 * frame['progress'][row] / (2 * course.length):.1f}%) · Travelled: {distance}"
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
                max(16, self.sensor.width()), 32
            )
        )
        debug = control.get("debug", [{}] * len(control["action"]))[row]
        self.history.bind(controls, row)

        def history(key):
            value = debug.get(key)
            if isinstance(value, (int, float, np.ndarray)):
                return self.history.series(key, np.shape(value), index)
            return None

        self.diagnostics.show_snapshot(debug, history, time=float(frame["time"][row]))
