"""Wall-clock playback of saved frames; seeking never invokes a controller."""

import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class Playback(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.dt, self.position, self.last_tick = 0.001, 0.0, 0.0
        self.controls = None
        self.control_steps = None
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.tick)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Recorded frame")
        self.slider.setToolTip("Drag to seek. Home / End jump to the first / last frame.")
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.seek)
        self.play = QPushButton("Play recording")
        self.play.clicked.connect(self.toggle)
        self.previous = QPushButton("Previous frame")
        self.next = QPushButton("Next frame")
        self.control = QPushButton("Next control")
        self.previous.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(1))
        self.control.clicked.connect(self.next_control)
        self.speed = QComboBox()
        for speed in [0.1, 0.25, 0.5, 1, 2, 5, 20]:
            self.speed.addItem(f"{speed:g}×", speed)
        self.speed.setCurrentIndex(3)
        self.speed.currentIndexChanged.connect(self.speed_changed)
        self.label = QLabel("No recorded frames")
        self.label.setObjectName("muted")
        bar = QHBoxLayout()
        for widget in [
            self.play,
            self.previous,
            self.next,
            self.control,
            QLabel("Speed"),
            self.speed,
        ]:
            bar.addWidget(widget)
        bar.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        timeline = QHBoxLayout()
        timeline.addWidget(self.slider, 1)
        timeline.addWidget(self.label)
        layout.addLayout(timeline)
        action = self.play_action = QAction("Play / pause recording", self)
        action.setShortcut("Space")
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.triggered.connect(self.toggle)
        self.addAction(action)
        self.update_controls()

    def set_frames(self, count, dt, controls=None, *, follow=False, control_steps=None):
        self.dt, self.controls, self.control_steps = dt, controls, control_steps
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, count - 1))
        if follow:
            self.slider.setValue(max(0, count - 1))
        self.slider.blockSignals(False)
        self.update_controls()

    def update_controls(self):
        index, end = self.slider.value(), self.slider.maximum()
        self.play.setEnabled(end > 0)
        self.previous.setEnabled(index > 0)
        self.next.setEnabled(index < end)
        self.control.setEnabled(
            index < end and (self.controls is not None or self.control_steps is not None)
        )
        self.slider.setEnabled(end > 0)
        self.label.setText(f"{index * self.dt:.3f} / {end * self.dt:.3f} s · frame {index}")

    def pause(self):
        self.timer.stop()
        self.play.setText("Play recording")

    def toggle(self):
        if self.timer.isActive():
            self.pause()
        elif self.slider.maximum() > 0:
            if self.slider.value() == self.slider.maximum():
                self.slider.setValue(0)
            self.position = float(self.slider.value())
            self.last_tick = time.monotonic()
            self.play.setText("Pause recording")
            self.timer.start()
            self.changed.emit()

    def speed_changed(self):
        if self.timer.isActive():
            self.position = float(self.slider.value())
            self.last_tick = time.monotonic()

    def tick(self):
        now = time.monotonic()
        self.position += (now - self.last_tick) * self.speed.currentData() / self.dt
        self.last_tick = now
        index = min(self.slider.maximum(), int(self.position))
        if index != self.slider.value():
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
            self.update_controls()
            self.changed.emit()
        if index == self.slider.maximum():
            self.pause()

    def seek(self):
        self.pause()
        self.position = float(self.slider.value())
        self.update_controls()
        self.changed.emit()

    def step(self, amount):
        self.pause()
        self.slider.setValue(
            min(self.slider.maximum(), max(0, self.slider.value() + amount))
        )

    def next_control(self):
        index = self.slider.value()
        if self.controls is not None:
            boundary = int(
                np.searchsorted(self.controls, self.controls[index], side="right")
            )
            self.step(boundary - index)
        elif self.control_steps is not None:
            self.step(
                1 if index == 0 else self.control_steps - (index - 1) % self.control_steps
            )

    def inspect_time(self, seconds, controls, row=0):
        """Seek the first physical frame carrying the nearest controller snapshot."""
        if not controls:
            return
        times = np.array([float(c["observation"]["time"][row]) for c in controls])
        call = int(np.argmin(abs(times - seconds)))
        if self.controls is not None:
            frame = int(np.searchsorted(self.controls, call, side="left"))
        else:
            frame = 1 + call * self.control_steps
        self.pause()
        self.step(frame - self.slider.value())
