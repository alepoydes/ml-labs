import time
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from coursekit import recording, resources, submissions, training
from coursekit.gui.common import Job, Notice, SolutionPath, heading
from coursekit.gui.line_run import Run, Worker
from coursekit.gui.playback import Playback
from coursekit.gui.robot_view import RobotView, recorded_courses
from coursekit.gui.scene import Scene
from coursekit.gui.style import apply_style
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import Course
from coursekit.labs.line_following_curve import iter_courses
from coursekit.labs.line_following_env import CONTROL_STEPS, PHYSICS_DT


class LineFollowingPage(QWidget):
    def __init__(self, project):
        super().__init__()
        apply_style(self)
        self.project = Path(project)
        self.job = self.preparation = self.replay = None
        self.warmup, self.warmed_batch = None, None
        self.worker = Worker()
        self.pending_start = False
        self.selected_course = -1
        self.courses = []
        self.live, self.shown = True, None
        self.chooser = SolutionPath(self.project, "line_following")
        self.solution_button = QPushButton("Create baseline")
        self.solution_button.setToolTip(
            "Create the baseline if missing, then run train.py to prepare its artifacts. Existing code is never replaced."
        )
        self.solution_hint = QLabel()
        self.solution_hint.setObjectName("muted")
        self.solution_hint.setTextFormat(Qt.TextFormat.PlainText)
        self.solution_hint.setWordWrap(True)
        self.course = QComboBox()
        self.course.setMinimumContentsLength(14)
        self.previous_course = QPushButton("Previous course")
        self.next_course = QPushButton("Next course")
        self.seed, self.batch, self.episode = QSpinBox(), QSpinBox(), QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.batch.setRange(1, 32)
        self.batch.setToolTip(
            "Run consecutive courses with consecutive robot seeds in one batch."
        )
        self.episode.setRange(1, 1)
        self.seed.setToolTip("Robot parameters for the next run; click Restart to apply.")
        self.start_button = QPushButton("Start")
        self.start_button.setProperty("primary", True)
        self.play_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.physics_button, self.control_button = (
            QPushButton("Step 1 ms"),
            QPushButton("Step control"),
        )
        self.prepare_button = QPushButton("Prepare courses")
        self.save_button, self.open_button = (
            QPushButton("Save recording…"),
            QPushButton("Open recording…"),
        )
        self.engine_status = QLabel("Preparing simulation…")
        self.engine_status.setObjectName("muted")
        self.engine_status.setWordWrap(True)
        self.engine_status.setMaximumWidth(180)
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
        layout.addLayout(header)
        self.setup = QWidget()
        form = QVBoxLayout(self.setup)
        form.setContentsMargins(0, 0, 0, 0)
        solution_row = QHBoxLayout()
        solution_row.addWidget(self.chooser, 1)
        solution_row.addWidget(self.solution_button)
        form.addLayout(solution_row)
        form.addWidget(self.solution_hint)
        setup = QHBoxLayout()
        for widget in [
            self.previous_course,
            QLabel("Course"),
            self.course,
            self.next_course,
            QLabel("Robot seed"),
            self.seed,
            QLabel("Batch"),
            self.batch,
            self.prepare_button,
        ]:
            setup.addWidget(widget)
        setup.setStretchFactor(self.course, 1)
        form.addLayout(setup)
        layout.addWidget(self.setup)
        self.control_bar = QWidget()
        live = QHBoxLayout(self.control_bar)
        live.setContentsMargins(0, 0, 0, 0)
        for widget in [
            self.start_button,
            self.play_button,
            self.physics_button,
            self.control_button,
            self.stop_button,
        ]:
            live.addWidget(widget)
        live.addStretch()
        live.addWidget(self.engine_status)
        self.robot_label = QLabel("View robot")
        live.addWidget(self.robot_label)
        live.addWidget(self.episode)
        layout.addWidget(self.control_bar)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.playback)
        layout.addWidget(self.message)
        self.start_button.clicked.connect(self.start)
        self.play_button.clicked.connect(self.toggle_run)
        self.stop_button.clicked.connect(self.stop)
        self.physics_button.clicked.connect(lambda: self.step(False))
        self.control_button.clicked.connect(lambda: self.step(True))
        self.speed.currentIndexChanged.connect(self.change_speed)
        self.episode.valueChanged.connect(self.render)
        self.course.currentIndexChanged.connect(self.change_course)
        self.previous_course.clicked.connect(
            lambda: self.course.setCurrentIndex(self.course.currentIndex() - 1)
        )
        self.next_course.clicked.connect(
            lambda: self.course.setCurrentIndex(self.course.currentIndex() + 1)
        )
        self.batch.valueChanged.connect(self.prepare_engine)
        self.save_button.clicked.connect(self.save)
        self.open_button.clicked.connect(self.open_recording)
        self.prepare_button.clicked.connect(self.prepare)
        self.solution_button.clicked.connect(self.prepare_solution)
        self.chooser.path.textChanged.connect(self.buttons)
        # Space controls the current run; the lower controls play recorded history.
        self.playback.play_action.setEnabled(False)
        action = QAction("Start / pause / continue", self)
        action.setShortcut("Space")
        action.triggered.connect(self.toggle_run)
        self.addAction(action)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)
        self.prepare()
        self.prepare_engine()

    def prepare_engine(self):
        if (
            not self.isEnabled()
            or self.warmup is not None
            or self.job is not None
            and self.job.isRunning()
            or self.warmed_batch == self.batch.value()
            and self.worker.process is not None
        ):
            return
        course = Course.polyline("warmup", np.array([[0.0, 0.0], [1.0, 0.0]]))
        size = self.batch.value()
        self.warmup = Run(
            None, [course] * size, list(range(size)), self, worker=self.worker
        )
        self.warm_started = time.monotonic()
        self.warmup.finished.connect(
            self.engine_prepared, Qt.ConnectionType.QueuedConnection
        )
        self.warmup.start()
        self.buttons()

    @Slot()
    def engine_prepared(self):
        task, self.warmup = self.warmup, None
        self.warmed_batch = (
            len(task.courses) if not task.error and not task.stopping else None
        )
        if task.error:
            self.message.setText(task.error)
        task.deleteLater()
        if not self.isEnabled():
            return
        if self.pending_start and not task.error and not task.stopping:
            self.pending_start = False
            self.start()
        elif self.warmed_batch is not None:
            self.prepare_engine()  # Batch selection may have changed during preparation.
        else:
            self.pending_start = False
        self.buttons()

    def prepare(self):
        if self.preparation is not None:
            return

        def load():
            resources.sync(
                self.project / "data/line_following",
                resources.manifest(self.project / "labs/line_following/resources.toml"),
            )
            courses = []
            for course in iter_courses(self.project / "data/line_following/courses.npz"):
                if QThread.currentThread().isInterruptionRequested():
                    return None
                courses.append(course)
            return courses

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
        elif job.value is None:
            self.message.setText("Course preparation cancelled. Prepare courses to retry.")
        else:
            self.courses = job.value
            self.course.blockSignals(True)
            self.course.clear()
            self.course.addItems([c.name for c in self.courses])
            self.course.blockSignals(False)
            self.selected_course = self.course.currentIndex()
            self.preview()
            self.message.setText(
                "Choose a course and click Start. Save recording keeps a run when you want to keep it."
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

    def change_course(self):
        index = self.course.currentIndex()
        if index < 0:
            return
        if self.job is not None:
            accepted = self.start()
        else:
            accepted = True
        if not accepted:
            self.course.blockSignals(True)
            self.course.setCurrentIndex(self.selected_course)
            self.course.blockSignals(False)
            return
        self.selected_course = index
        if self.job is not None:
            return
        if self.replay is not None:
            self.replay = None
            self.playback.pause()
            self.playback.set_frames(0, PHYSICS_DT)
            self.diagnostics.show_snapshot({})
            self.sensors.clear()
        self.preview()
        self.buttons()

    def solution_problem(self):
        path = Path(self.chooser.path.text())
        missing = [
            name
            for name in ("solution.py", "train.py", "config.toml", "submission.toml")
            if not (path / name).is_file()
        ]
        if missing:
            if path == self.project / "work/line_following/baseline" and not path.exists():
                return "No working solution yet. Create baseline to copy and prepare the example, or browse to your own solution."
            return "Choose a working solution. Missing files: " + ", ".join(missing) + "."
        try:
            if (
                submissions.metadata((path / "submission.toml").read_bytes())["lab"]
                != "line_following"
            ):
                return "Choose a Line Following solution."
        except Exception as exc:
            return str(exc)
        return ""

    def prepare_solution(self):
        if self.preparation is not None or self.job is not None and self.job.isRunning():
            return
        path = Path(self.chooser.path.text())
        default = self.project / "work/line_following/baseline"
        if self.solution_problem() and not (path == default and not path.exists()):
            self.message.setText(self.solution_problem())
            return

        def prepare():
            if path == default and not path.exists():
                training.new_solution(self.project, "line_following", "baseline")
            training.train_local(path, self.project)

        self.message.setText("Preparing solution: running train.py and saving artifacts…")
        self.preparation = Job(prepare, self)
        self.preparation.finished.connect(
            self.prepared_solution, Qt.ConnectionType.QueuedConnection
        )
        self.preparation.start()
        self.buttons()

    @Slot()
    def prepared_solution(self):
        job, self.preparation = self.preparation, None
        self.message.setText(
            job.error or "Solution prepared. Click Start to run it on the selected course."
        )
        job.deleteLater()
        self.buttons()

    def buttons(self):
        running = self.job is not None and self.job.isRunning()
        ready = self.job is not None and self.job.snapshot() is not None
        warming = self.warmup is not None
        preparing = self.preparation is not None
        path = Path(self.chooser.path.text())
        problem = self.solution_problem()
        created = (path / "artifacts/provenance.json").is_file()
        can_create = (
            path == self.project / "work/line_following/baseline" and not path.exists()
        )
        self.solution_button.setText(
            "Create baseline" if can_create else "Prepare solution"
        )
        self.solution_button.setEnabled(
            bool(self.courses)
            and not running
            and not preparing
            and not self.pending_start
            and (can_create or not problem)
        )
        self.solution_hint.setText(
            problem
            or (
                "Artifacts ready. Prepare solution runs train.py again."
                if created
                else "Prepare solution to create its artifacts before running."
            )
        )
        self.solution_hint.setVisible(bool(problem) or not created)
        self.start_button.setText(
            "Starting…"
            if self.pending_start
            else "Restart"
            if self.job is not None
            else "Start"
        )
        self.start_button.setEnabled(
            bool(self.courses)
            and not preparing
            and not problem
            and created
            and not self.pending_start
        )
        for widget in [
            self.chooser,
            self.seed,
            self.batch,
            self.prepare_button,
        ]:
            widget.setEnabled(not preparing)
        self.course.setEnabled(not preparing)
        self.previous_course.setEnabled(not preparing and self.course.currentIndex() > 0)
        self.next_course.setEnabled(
            not preparing and 0 <= self.course.currentIndex() < len(self.courses) - 1
        )
        self.prepare_button.setVisible(not self.courses)
        self.play_button.setText("Continue" if running and self.job.paused else "Pause")
        self.play_button.setEnabled(running and ready and not self.job.stopping)
        self.stop_button.setEnabled(running or warming or self.pending_start)
        self.stop_button.setText(
            "Cancel preparation" if warming and not running else "Stop"
        )
        stepping = running and ready and self.job.paused and not self.job.stopping
        self.physics_button.setEnabled(stepping)
        self.control_button.setEnabled(stepping)
        self.open_button.setEnabled(not running and not preparing)
        self.save_button.setEnabled(ready or self.replay is not None)
        self.episode.setEnabled(ready or self.replay is not None)
        batch = (ready or self.replay is not None) and self.episode.maximum() > 1
        for widget in [
            self.play_button,
            self.physics_button,
            self.control_button,
        ]:
            widget.setVisible(running)
        self.robot_label.setVisible(batch)
        self.episode.setVisible(batch)
        self.playback.setVisible(ready or self.replay is not None)
        if not warming:
            if ready and running:
                self.warmed_batch = len(self.job.courses)
            self.engine_status.setText(
                "Preparing simulation…"
                if running and not ready
                else "Simulation ready"
                if self.worker.process is not None
                else "Simulation not prepared"
            )
            self.engine_status.setToolTip("")

    def start(self):
        if not self.courses or self.preparation is not None:
            return False
        if (
            self.solution_problem()
            or not (Path(self.chooser.path.text()) / "artifacts/provenance.json").is_file()
        ):
            self.message.setText(
                self.solution_problem() or "Prepare solution before starting a run."
            )
            self.buttons()
            return False
        self.restart()
        return True

    def restart(self):
        if self.warmup is not None:
            self.pending_start = True
            self.message.setText(
                "The run will start automatically when preparation finishes."
            )
            self.buttons()
            return
        if self.job is not None and self.job.isRunning():
            self.pending_start = True
            self.job.command(stop=True)
            self.buttons()
            return
        self.pending_start = False
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
            worker=self.worker,
        )
        self.job.finished.connect(self.finished_run, Qt.ConnectionType.QueuedConnection)
        self.replay, self.live, self.shown = None, True, None
        self.episode.blockSignals(True)
        self.episode.setMaximum(len(courses))
        self.episode.setValue(1)
        self.episode.blockSignals(False)
        self.playback.set_frames(0, PHYSICS_DT)
        self.diagnostics.show_snapshot({})
        self.sensors.clear()
        self.sensors.setText("Waiting for the first controller call.")
        self.message.setText(
            "Starting simulation process. Stop or close the window to cancel."
        )
        self.job.command(paused=False, speed=self.speed.currentData())
        self.job.start()
        self.buttons()

    @Slot()
    def finished_run(self):
        if self.sender() is not self.job:
            return
        if self.job.snapshot() is not None and self.worker.process is not None:
            self.warmed_batch = len(self.job.courses)
        if self.pending_start and self.isEnabled():
            self.pending_start = False
            self.restart()
            return
        self.pending_start = False
        self.message.setText(
            self.job.error
            or "Run ended. Choose another course, Restart, or inspect the recording below. Save recording keeps it."
        )
        self.refresh()
        self.buttons()

    def resume(self):
        if self.job is not None and self.job.isRunning():
            self.playback.pause()
            self.live = True
            self.job.command(paused=False, speed=self.speed.currentData())
            self.buttons()

    def toggle_run(self):
        if self.job is not None and self.job.isRunning():
            if self.job.paused or not self.live:
                self.resume()
            else:
                self.pause()
        elif self.replay is not None or self.job is not None:
            self.playback.toggle()
        else:
            self.start()

    def pause(self):
        if self.job is not None and self.job.isRunning():
            self.job.command()
        self.buttons()

    def stop(self):
        self.pending_start = False
        if self.preparation is not None:
            self.preparation.requestInterruption()
        if self.warmup is not None:
            self.warmup.command(stop=True)
        if self.job is not None and self.job.isRunning() and not self.job.stopping:
            self.job.command(stop=True)
            self.message.setText("Stopping the simulation process…")

    def shutdown(self):
        self.worker.close()
        self.timer.stop()
        self.playback.pause()
        self.scene.cleanup()

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
        if self.warmup is not None:
            elapsed = time.monotonic() - self.warm_started
            self.engine_status.setText(f"Preparing simulation · {elapsed:.0f} s")
            self.engine_status.setToolTip(
                "You can choose a course and solution now. Start runs automatically when ready."
            )
        if self.job is not None:
            if self.job.isRunning() and self.isEnabled():
                phase = self.job.phase
                elapsed = time.monotonic() - self.job.phase_started
                if phase == "Paused":
                    text = "Paused. Continue resumes from the latest state; the timeline inspects recorded history."
                elif phase == "Simulating":
                    text = "Running. Pause or drag the timeline to inspect; changing course starts a new run."
                else:
                    text = f"{phase} · {elapsed:.1f} s. Stop or close to cancel."
                    if phase == "Warming physics":
                        text += " First preparation of this batch size can take longer."
                self.message.setText(text)
            snapshot = self.job.snapshot()
            if snapshot is None:
                return
            self.playback.set_frames(
                snapshot[2], PHYSICS_DT, follow=self.live, control_steps=CONTROL_STEPS
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

            def pitch():
                history = self.job.frames[: index + 1]
                return [f["time"][row] for f in history], [f["q"][row, 4] for f in history]
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
            self.playback.pause()
            if self.job is not None:
                self.job.deleteLater()
            self.job, self.replay, self.replay_courses = None, record, courses
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
