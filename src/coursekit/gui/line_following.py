from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from coursekit import recording, resources
from coursekit.gui.common import Diagnostics, SolutionPath, pixmap
from coursekit.gui.line_run import Run
from coursekit.gui.scene import Scene
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import Course, load_courses
from coursekit.labs.line_following_env import CONTROL_STEPS


class LineFollowingPage(QWidget):
    def __init__(self, project):
        super().__init__()
        self.project = Path(project)
        self.job = self.replay = None
        self.courses = []
        self.live, self.shown = True, None
        self.chooser = SolutionPath(self.project, "line_following")
        self.course = QComboBox()
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.batch = QSpinBox()
        self.batch.setRange(1, 32)
        self.episode = QSpinBox()
        self.episode.setRange(1, 1)
        self.start_button = QPushButton("New run")
        self.play_button = QPushButton("Resume live")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.physics_button = QPushButton("+1 ms")
        self.control_button = QPushButton("Next control")
        self.speed = QComboBox()
        self.speed.addItems(["0.1×", "0.25×", "0.5×", "1×", "2×", "5×", "20×"])
        self.speed.setCurrentIndex(3)
        self.scene = Scene()
        self.diagnostics = Diagnostics()
        self.sensors = QLabel("Sensor readings appear after the first controller call.")
        self.sensors.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message, self.status = QLabel(), QLabel()
        self.message.setWordWrap(True)
        self.scene.failed.connect(self.message.setText)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        top, controls, history = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        top.addWidget(self.chooser, 1)
        for widget in [
            self.course,
            QLabel("Seed"),
            self.seed,
            QLabel("Batch"),
            self.batch,
            self.start_button,
        ]:
            top.addWidget(widget)
        for widget in [
            self.play_button,
            self.pause_button,
            self.physics_button,
            self.control_button,
            self.stop_button,
            self.speed,
            QLabel("Robot"),
            self.episode,
        ]:
            controls.addWidget(widget)
        follow, overhead = QCheckBox("Follow robot"), QCheckBox("Top view")
        save, open_record = QPushButton("Save recording…"), QPushButton("Open recording…")
        for widget in [follow, overhead, self.timeline, save, open_record]:
            history.addWidget(widget)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.scene, 1)
        left_layout.addWidget(self.sensors)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.diagnostics)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(controls)
        layout.addWidget(split, 1)
        layout.addLayout(history)
        layout.addWidget(self.status)
        layout.addWidget(self.message)
        self.start_button.clicked.connect(self.start)
        self.play_button.clicked.connect(self.resume)
        self.pause_button.clicked.connect(self.pause)
        self.stop_button.clicked.connect(self.stop)
        self.physics_button.clicked.connect(lambda: self.step(False))
        self.control_button.clicked.connect(lambda: self.step(True))
        self.speed.currentIndexChanged.connect(self.change_speed)
        self.timeline.valueChanged.connect(self.seek)
        self.episode.valueChanged.connect(self.render)
        self.course.currentIndexChanged.connect(self.preview)
        save.clicked.connect(self.save)
        open_record.clicked.connect(self.open_recording)
        follow.toggled.connect(lambda value: self.camera("follow", value))
        overhead.toggled.connect(lambda value: self.camera("top", value))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)
        try:
            spec = resources.manifest(self.project / "labs/line_following/resources.toml")
            resources.sync(self.project / "data/line_following", spec)
            self.courses = load_courses(self.project / "data/line_following/courses.npz")
            self.course.addItems([c.name for c in self.courses])
            self.message.setText(
                "Choose a course and solution. New runs start paused; mouse wheel zooms the scene."
            )
        except Exception as exc:
            self.message.setText(str(exc))
        self.buttons()

    def preview(self):
        if self.job is not None or self.replay is not None or not self.courses:
            return
        course = self.courses[self.course.currentIndex()]
        model = physics.robot()
        q = physics.initial_state(model).q
        q[:2] = torch.from_numpy(course.at(0.0).copy())
        direction = course.at(0.0, derivative=1)
        q[5] = float(np.arctan2(direction[1], direction[0]))
        self.scene.show_pose(
            course, q.numpy(), {k: v.numpy() for k, v in model._asdict().items()}
        )

    def buttons(self):
        running = self.job is not None and self.job.isRunning()
        self.start_button.setEnabled(bool(self.courses) and not running)
        for widget in [self.play_button, self.pause_button, self.stop_button]:
            widget.setEnabled(running)
        for widget in [self.physics_button, self.control_button]:
            widget.setEnabled(running or self.replay is not None)

    def start(self):
        if self.job is not None and self.job.isRunning():
            return
        if self.job is not None:
            self.job.deleteLater()
        courses = [
            self.courses[(self.course.currentIndex() + i) % len(self.courses)]
            for i in range(self.batch.value())
        ]
        self.job = Run(
            Path(self.chooser.path.text()),
            courses,
            [self.seed.value() + i for i in range(len(courses))],
            self,
        )
        self.job.finished.connect(self.finished_run, Qt.ConnectionType.QueuedConnection)
        self.replay, self.live, self.shown = None, True, None
        self.episode.setMaximum(len(courses))
        self.message.setText(
            "Loading solution and preparing physics. The first native compilation can take time."
        )
        self.job.start()
        self.buttons()

    @Slot()
    def finished_run(self):
        if self.sender() is not self.job:
            return
        self.message.setText(
            self.job.error or "Run stopped. Recorded history remains available."
        )
        self.refresh()
        self.buttons()

    def resume(self):
        if self.job is not None:
            self.live = True
            self.job.command(paused=False, speed=float(self.speed.currentText()[:-1]))

    def pause(self):
        if self.job is not None:
            self.job.command()

    def stop(self):
        if self.job is not None:
            self.job.command(stop=True)

    def shutdown(self):
        self.timer.stop()
        self.scene.cleanup()

    def change_speed(self):
        if self.job is not None:
            self.job.command(
                paused=self.job.paused, speed=float(self.speed.currentText()[:-1])
            )

    def step(self, control):
        count = CONTROL_STEPS - self.timeline.value() % CONTROL_STEPS if control else 1
        if self.timeline.value() < self.timeline.maximum() or self.replay is not None:
            self.timeline.setValue(
                min(self.timeline.maximum(), self.timeline.value() + count)
            )
        elif self.job is not None and self.job.isRunning():
            self.live = True
            self.job.command(steps=count)

    def seek(self):
        self.live = False
        self.pause()
        self.render()

    def camera(self, name, value):
        setattr(self.scene, name, value)
        self.scene.update()

    def refresh(self):
        if self.job is not None:
            snapshot = self.job.snapshot()
            if snapshot is None:
                return
            count = snapshot[2]
            self.timeline.blockSignals(True)
            self.timeline.setMaximum(count - 1)
            if self.live:
                self.timeline.setValue(count - 1)
            self.timeline.blockSignals(False)
            if self.shown != (self.timeline.value(), self.episode.value()):
                self.render()

    def render(self):
        index, row = self.timeline.value(), self.episode.value() - 1
        if self.replay is not None:
            frames = self.replay["frames"]
            frame = {k: v[index] for k, v in frames.items()}
            control = (
                self.replay["controls"][frame["control"]] if frame["control"] >= 0 else None
            )
            course, model = self.replay_courses[row], self.replay["model"]
            controls = self.replay["controls"]
        elif self.job is not None:
            snapshot = self.job.snapshot(index)
            if snapshot is None:
                return
            frame, control, _ = snapshot
            course, model = self.job.courses[row], self.job.model
            controls = self.job.controls
        else:
            return
        self.shown = index, row + 1
        self.scene.show_pose(course, frame["q"][row], {k: v[row] for k, v in model.items()})
        self.status.setText(
            f"t = {frame['time'][row]:.3f} s · {frame['status'][row]} · {'return' if frame['leg'][row] else 'outward'} · accepted {frame['progress'][row]:.3f} / {2 * course.length:.3f} m · score {frame['score'][row]:.2f} · frame {index}"
        )
        if control is None:
            self.diagnostics.show_snapshot({})
            self.sensors.clear()
            self.sensors.setText("No controller call at this frame.")
            return
        observation = control["observation"]
        rgb = np.rint(observation["rgb"][row][None] * 255).astype(np.uint8)
        self.sensors.setPixmap(pixmap(rgb).scaled(512, 32))
        values = {
            **control["debug"][row],
            **{f"sensor.{k}": v[row] for k, v in observation.items() if k != "rgb"},
            "motor command": control["action"][row],
        }
        history = {}
        for key, value in control["debug"][row].items():
            if isinstance(value, (int, float)):
                samples = [
                    (c["observation"]["time"][row], c["debug"][row].get(key))
                    for c in controls[: frame["control"] + 1]
                ]
                samples = [(t, v) for t, v in samples if isinstance(v, (int, float))]
                if samples:
                    history[key] = tuple(zip(*samples))
        self.diagnostics.show_snapshot(values, history)

    def save(self):
        record = (
            self.replay
            if self.replay is not None
            else self.job.record()
            if self.job is not None
            else None
        )
        if record is None or not record["frames"]:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save recording",
            str(self.project / "runs/line-following.npz"),
            "Recording (*.npz)",
        )
        if filename:
            recording.save(filename, record)
            self.message.setText(f"Saved {filename}")

    def open_recording(self):
        if self.job is not None and self.job.isRunning():
            self.message.setText("Stop the current run before opening a recording.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open recording", str(self.project / "runs"), "Recording (*.npz)"
        )
        if not filename:
            return
        try:
            record = recording.load(filename)
            if (record["lab"], record["version"]) != ("line_following", 2):
                raise ValueError(
                    "Expected Line Following spline recording version 2; old polyline recordings are incompatible."
                )
            courses = [Course(**c) for c in record["courses"]]
            if record["frames"]["q"].shape[1:] != (len(courses), 8):
                raise ValueError("Invalid recorded robot poses.")
            if self.job is not None:
                self.job.deleteLater()
            self.job, self.replay, self.replay_courses = None, record, courses
            self.episode.blockSignals(True)
            self.timeline.blockSignals(True)
            self.episode.setMaximum(len(courses))
            self.timeline.setMaximum(len(record["frames"]["q"]) - 1)
            self.timeline.setValue(0)
            self.episode.blockSignals(False)
            self.timeline.blockSignals(False)
            self.render()
            self.buttons()
            self.message.setText(
                "Recorded playback. Opening history does not load or execute the solution."
            )
        except Exception as exc:
            self.message.setText(f"Cannot open recording: {exc}")
