from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from coursekit import recording, resources
from coursekit.gui.common import Job, Notice, SolutionPath, heading
from coursekit.gui.line_run import Run
from coursekit.gui.playback import Playback
from coursekit.gui.robot_view import RobotView, recorded_courses
from coursekit.gui.scene import Scene
from coursekit.gui.style import apply_style
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import load_courses
from coursekit.labs.line_following_env import CONTROL_STEPS, PHYSICS_DT


class LineFollowingPage(QWidget):
    def __init__(self, project):
        super().__init__()
        apply_style(self)
        self.project = Path(project)
        self.job = self.preparation = self.replay = None
        self.courses = []
        self.live, self.shown = True, None
        self.saved_frames = 0
        self.chooser = SolutionPath(self.project, "line_following")
        self.course = QComboBox()
        self.course.setMinimumContentsLength(14)
        self.seed, self.batch, self.episode = QSpinBox(), QSpinBox(), QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.batch.setRange(1, 32)
        self.batch.setToolTip(
            "Run consecutive courses with consecutive robot seeds in one batch."
        )
        self.episode.setRange(1, 1)
        self.start_button = QPushButton("New run")
        self.start_button.setProperty("primary", True)
        self.play_button = QPushButton("Resume live")
        self.pause_button, self.stop_button = (
            QPushButton("Pause live"),
            QPushButton("Stop run"),
        )
        self.physics_button, self.control_button = (
            QPushButton("Step 1 ms"),
            QPushButton("Step control"),
        )
        self.prepare_button = QPushButton("Prepare courses")
        self.save_button, self.open_button = (
            QPushButton("Save recording…"),
            QPushButton("Open recording…"),
        )
        self.setup_button = QPushButton("Run setup")
        self.setup_button.setCheckable(True)
        self.setup_button.setChecked(True)
        self.view = RobotView(Scene())
        self.scene, self.diagnostics, self.sensors, self.status = (
            self.view.scene,
            self.view.diagnostics,
            self.view.sensor,
            self.view.status,
        )
        self.message = Notice()
        self.scene.failed.connect(self.message.setText)
        self.playback = Playback()
        self.timeline, self.speed = self.playback.slider, self.playback.speed
        self.playback.changed.connect(self.seek)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(
            heading(
                "Line Following",
                "Run a controller, inspect its sensors, or replay a saved run.",
            ),
            1,
        )
        header.addWidget(self.open_button)
        header.addWidget(self.save_button)
        header.addWidget(self.setup_button)
        layout.addLayout(header)
        self.setup = QWidget()
        form = QVBoxLayout(self.setup)
        form.setContentsMargins(0, 0, 0, 0)
        form.addWidget(self.chooser)
        setup = QHBoxLayout()
        for widget in [
            QLabel("Course"),
            self.course,
            QLabel("Robot seed"),
            self.seed,
            QLabel("Batch"),
            self.batch,
            self.start_button,
            self.prepare_button,
        ]:
            setup.addWidget(widget)
        setup.addStretch()
        form.addLayout(setup)
        layout.addWidget(self.setup)
        self.control_bar = QWidget()
        live = QHBoxLayout(self.control_bar)
        live.setContentsMargins(0, 0, 0, 0)
        for widget in [
            self.play_button,
            self.pause_button,
            self.physics_button,
            self.control_button,
            self.stop_button,
        ]:
            live.addWidget(widget)
        live.addStretch()
        self.robot_label = QLabel("View robot")
        live.addWidget(self.robot_label)
        live.addWidget(self.episode)
        layout.addWidget(self.control_bar)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.playback)
        layout.addWidget(self.message)
        self.start_button.clicked.connect(self.start)
        self.play_button.clicked.connect(self.resume)
        self.pause_button.clicked.connect(self.pause)
        self.stop_button.clicked.connect(self.stop)
        self.physics_button.clicked.connect(lambda: self.step(False))
        self.control_button.clicked.connect(lambda: self.step(True))
        self.speed.currentIndexChanged.connect(self.change_speed)
        self.episode.valueChanged.connect(self.render)
        self.course.currentIndexChanged.connect(self.preview)
        self.save_button.clicked.connect(self.save)
        self.open_button.clicked.connect(self.open_recording)
        self.prepare_button.clicked.connect(self.prepare)
        self.setup_button.toggled.connect(self.setup.setVisible)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)
        self.prepare()

    def prepare(self):
        if self.preparation is not None:
            return

        def load():
            resources.sync(
                self.project / "data/line_following",
                resources.manifest(self.project / "labs/line_following/resources.toml"),
            )
            return load_courses(self.project / "data/line_following/courses.npz")

        self.message.setText("Preparing verified courses…")
        self.preparation = Job(load, self)
        self.preparation.finished.connect(self.prepared, Qt.ConnectionType.QueuedConnection)
        self.preparation.start()
        self.buttons()

    @Slot()
    def prepared(self):
        job, self.preparation = self.preparation, None
        if job.error:
            self.message.setText(job.error)
        else:
            self.courses = job.value
            self.course.clear()
            self.course.addItems([c.name for c in self.courses])
            self.message.setText(
                "Choose a solution and course, then New run. The run starts paused."
            )
        job.deleteLater()
        self.buttons()

    def preview(self):
        if (
            self.job is not None
            or self.replay is not None
            or not self.courses
            or self.course.currentIndex() < 0
        ):
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
        self.status.setText(
            f"{course.name} · {course.length:.2f} m each way · course preview"
        )

    def buttons(self):
        running = self.job is not None and self.job.isRunning()
        ready = self.job is not None and self.job.snapshot() is not None
        preparing = self.preparation is not None
        self.start_button.setEnabled(bool(self.courses) and not running and not preparing)
        for widget in [
            self.chooser,
            self.course,
            self.seed,
            self.batch,
            self.prepare_button,
        ]:
            widget.setEnabled(not running and not preparing)
        self.prepare_button.setVisible(not self.courses)
        self.play_button.setEnabled(
            running and ready and (self.job.paused or not self.live)
        )
        self.pause_button.setEnabled(running and ready and not self.job.paused)
        self.stop_button.setEnabled(running)
        self.physics_button.setEnabled(running and ready)
        self.control_button.setEnabled(running and ready)
        self.open_button.setEnabled(not running and not preparing)
        self.save_button.setEnabled(ready or self.replay is not None)
        self.episode.setEnabled(ready or self.replay is not None)
        batch = (ready or self.replay is not None) and self.episode.maximum() > 1
        self.control_bar.setVisible(running or batch)
        for widget in [
            self.play_button,
            self.pause_button,
            self.stop_button,
            self.physics_button,
            self.control_button,
        ]:
            widget.setVisible(running)
        self.robot_label.setVisible(batch)
        self.episode.setVisible(batch)
        self.playback.setVisible(ready or self.replay is not None)

    def start(self):
        if (
            not self.courses
            or self.preparation is not None
            or self.job is not None
            and self.job.isRunning()
        ):
            return
        if not self.confirm_reset():
            return
        self.playback.pause()
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
        self.saved_frames = 0
        self.episode.blockSignals(True)
        self.episode.setMaximum(len(courses))
        self.episode.setValue(1)
        self.episode.blockSignals(False)
        self.playback.set_frames(0, PHYSICS_DT)
        self.diagnostics.show_snapshot({})
        self.sensors.clear()
        self.sensors.setText("Waiting for the first controller call.")
        self.message.setText(
            "Loading solution and preparing physics. The first compilation can take time; Stop remains available."
        )
        self.job.start()
        self.setup_button.setChecked(False)
        self.buttons()

    @Slot()
    def finished_run(self):
        if self.sender() is not self.job:
            return
        self.message.setText(
            self.job.error
            or "Run ended. Play or inspect its recording below, or save it before starting another run."
        )
        self.refresh()
        self.buttons()

    def resume(self):
        if self.job is not None and self.job.isRunning():
            self.playback.pause()
            self.live = True
            self.job.command(paused=False, speed=self.speed.currentData())
            self.message.setText(
                "Live simulation. Pause or step to inspect; use the timeline to review recorded history."
            )
            self.buttons()

    def pause(self):
        if self.job is not None and self.job.isRunning():
            self.job.command()
        self.buttons()

    def stop(self):
        if self.job is not None:
            self.job.command(stop=True)
            self.message.setText("Stopping the run after the current computation finishes…")

    def shutdown(self):
        self.timer.stop()
        self.playback.pause()
        self.scene.cleanup()

    def confirm_reset(self):
        snapshot = self.job.snapshot() if self.job is not None else None
        return (
            snapshot is None
            or snapshot[2] <= max(1, self.saved_frames)
            or QMessageBox.question(
                self,
                "Replace current run?",
                "This run has unsaved frames. Discard them? Use Save recording to keep them.",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            == QMessageBox.StandardButton.Discard
        )

    def change_speed(self):
        if self.job is not None and self.job.isRunning():
            self.job.command(paused=self.job.paused, speed=self.speed.currentData())

    def step(self, control):
        self.playback.pause()
        index = self.timeline.value()
        count = (
            (1 if index == 0 else CONTROL_STEPS - (index - 1) % CONTROL_STEPS)
            if control
            else 1
        )
        if index < self.timeline.maximum() or self.replay is not None:
            if control:
                self.playback.next_control()
            else:
                self.playback.step(1)
        elif self.job is not None and self.job.isRunning():
            self.live = True
            self.job.command(steps=count)
            self.buttons()

    def seek(self):
        self.live = False
        self.pause()
        self.render()

    def refresh(self):
        if self.job is not None:
            snapshot = self.job.snapshot()
            if snapshot is None:
                return
            first = self.timeline.maximum() == 0 and self.shown is None
            self.playback.set_frames(
                snapshot[2], PHYSICS_DT, follow=self.live, control_steps=CONTROL_STEPS
            )
            if first and self.job.isRunning():
                self.message.setText(
                    "Ready · paused. Resume live to run, or Step 1 ms / Step control to advance precisely."
                )
            if self.shown != (self.timeline.value(), self.episode.value()):
                self.render()
            self.buttons()

    def render(self):
        index, row = self.timeline.value(), self.episode.value() - 1
        if self.replay is not None:
            frames = self.replay["frames"]
            frame = {k: v[index] for k, v in frames.items()}
            course, model, controls = (
                self.replay_courses[row],
                self.replay["model"],
                self.replay["controls"],
            )
            pitch = frames["time"][: index + 1, row], frames["q"][: index + 1, row, 4]
        elif self.job is not None:
            snapshot = self.job.snapshot(index)
            if snapshot is None:
                return
            frame = snapshot[0]
            course, model, controls = (
                self.job.courses[row],
                self.job.model,
                self.job.controls,
            )
            history = self.job.frames[: index + 1]
            pitch = [f["time"][row] for f in history], [f["q"][row, 4] for f in history]
        else:
            return
        self.shown = index, row + 1
        self.view.show_frame(course, frame, model, controls, row, pitch_history=pitch)

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
            try:
                recording.save(filename, record)
                self.saved_frames = len(record["frames"]["q"])
                self.message.setText(f"Saved {filename}")
            except Exception as exc:
                self.message.setText(f"Cannot save recording: {exc}")

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
            courses = recorded_courses(record)
            if not self.confirm_reset():
                return
            self.playback.pause()
            if self.job is not None:
                self.job.deleteLater()
            self.job, self.replay, self.replay_courses = None, record, courses
            self.setup_button.setChecked(False)
            self.episode.blockSignals(True)
            self.episode.setMaximum(len(courses))
            self.episode.setValue(1)
            self.episode.blockSignals(False)
            self.playback.set_frames(
                len(record["frames"]["q"]),
                record["physics_dt"],
                record["frames"]["control"],
            )
            self.timeline.blockSignals(True)
            self.timeline.setValue(0)
            self.timeline.blockSignals(False)
            self.playback.update_controls()
            self.render()
            self.buttons()
            self.message.setText(
                "Recorded playback. Play, change speed or drag the timeline; the solution is not executed."
            )
        except Exception as exc:
            self.message.setText(f"Cannot open recording: {exc}")
