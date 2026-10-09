"""Shared scene, sensor and diagnostics for local and instructor recordings."""

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
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
    def __init__(self, scene, *, diagnostic_state=None, view_mode="large"):
        super().__init__()
        self.scene = scene
        self.diagnostics = Diagnostics(diagnostic_state)
        self.history = DebugHistory()
        self.sensor = QLabel("No optical reading yet")
        self.sensor.setWordWrap(True)
        self.sensor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.sensor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sensor.setFixedHeight(28)
        self.sensor.setScaledContents(True)
        self.sensor.setStyleSheet(
            "background: white; color: #1f2937; border: 1px solid #cbd5e1;"
        )
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
        camera_widget = QWidget()
        camera = QHBoxLayout(camera_widget)
        camera.setContentsMargins(0, 0, 0, 0)
        for widget in [self.follow, self.overhead, self.sensing, self.fit]:
            camera.addWidget(widget)
        camera_widget.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        scene.setMinimumSize(200, 120)
        scene.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scene.installEventFilter(self)
        self.sensor_panel = QWidget(scene)
        self.sensor_panel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.sensor_panel.setStyleSheet("background: rgba(255, 255, 255, 215);")
        self.sensor_panel.setToolTip(
            f"Optical sensor: {SENSOR_WIDTH * 1000:g}×{SENSOR_DEPTH * 1000:g} mm; "
            "sector 0 is on the robot's right, sector 15 on its left."
        )
        sensor_layout = QVBoxLayout(self.sensor_panel)
        sensor_layout.setContentsMargins(4, 2, 4, 4)
        sensor_layout.setSpacing(2)
        sensor_label = QLabel("Optical sensor · right 0 → left 15")
        sensor_layout.addWidget(sensor_label)
        sensor_layout.addWidget(self.sensor)
        self.header_left = QWidget()
        self.header_layout = QVBoxLayout(self.header_left)
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.header_layout.setSpacing(6)
        self.camera_grid = QGridLayout()
        self.camera_grid.setContentsMargins(0, 0, 0, 0)
        self.camera_grid.setSpacing(6)
        self.camera_grid.addWidget(camera_widget, 0, 0)
        self.camera_grid.addWidget(self.status, 1, 0, 1, 2)
        self.header_layout.addLayout(self.camera_grid)
        self.status_inline = False
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.grid.addWidget(self.header_left, 0, 0)
        self.grid.addWidget(scene, 0, 1)
        self.grid.addWidget(self.diagnostics, 1, 0, 1, 2)
        self.grid.setColumnStretch(0, 1)
        self.grid.setRowStretch(1, 1)
        self.large = None
        self.set_large(view_mode == "large")
        self.place_sensor()

    def attach_controls(self, controls):
        self.header_layout.insertWidget(0, controls)

    def set_large(self, large):
        if self.large == large:
            return
        self.large = large
        for widget in (self.header_left, self.scene, self.diagnostics):
            self.grid.removeWidget(widget)
        if large:
            self.grid.addWidget(self.header_left, 0, 0, 1, 2)
            self.grid.addWidget(self.scene, 1, 0, 1, 2)
            self.scene.setMinimumSize(200, 120)
            self.scene.setMaximumSize(self.maximumSize())
            self.diagnostics.hide()
            tip = "Double-click to show diagnostics and a compact 3D preview."
        else:
            self.grid.addWidget(self.header_left, 0, 0)
            self.grid.addWidget(self.scene, 0, 1)
            self.grid.addWidget(self.diagnostics, 1, 0, 1, 2)
            self.scene.setFixedWidth(270)
            self.diagnostics.show()
            tip = "Double-click to enlarge the 3D view."
        self.sensor_panel.setVisible(large)
        self.place_status()
        self.scene.setToolTip(tip + " Wheel: zoom · drag: pan.")
        self.scene.update()

    def place_status(self):
        available = self.width() - (276 if self.large is False else 0)
        inline = available >= 700
        if inline == self.status_inline:
            return
        self.camera_grid.removeWidget(self.status)
        self.camera_grid.addWidget(
            self.status, 0 if inline else 1, 1 if inline else 0, 1, 1 if inline else 2
        )
        self.camera_grid.setColumnStretch(1, int(inline))
        self.status_inline = inline

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.place_status()

    def place_sensor(self):
        self.sensor_panel.setGeometry(8, 8, max(1, min(360, self.scene.width() - 16)), 58)
        self.sensor_panel.raise_()

    def eventFilter(self, watched, event):
        if watched is self.scene and event.type() == QEvent.Type.Resize:
            self.place_sensor()
        if (
            watched is self.scene
            and event.type() == QEvent.Type.MouseButtonDblClick
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.set_large(not self.large)
            return True
        return super().eventFilter(watched, event)

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
            pixmap(np.rint(observation["rgb"][row][None] * 255).astype(np.uint8))
        )
        debug = control.get("debug", [{}] * len(control["action"]))[row]
        self.history.bind(controls, row)

        def history(key):
            value = debug.get(key)
            if isinstance(value, (int, float, np.ndarray)):
                return self.history.series(key, np.shape(value), index)
            return None

        self.diagnostics.show_snapshot(debug, history, time=float(frame["time"][row]))
