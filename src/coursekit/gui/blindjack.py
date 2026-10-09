import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from coursekit import recording, resources, training
from coursekit.gui.common import Card, Diagnostics, Job, Notice, SolutionPath, heading
from coursekit.gui.style import apply_style
from coursekit.inspection import BlindjackGame, copy_debug, inspect_hand
from coursekit.labs.blindjack import validate_mask


class BlindjackPage(QWidget):
    def __init__(self, project):
        super().__init__()
        apply_style(self)
        self.project = Path(project)
        self.solution = self.game = self.decision = self.job = None
        self.decisions = {}
        self.drafts = {}
        self.activity, self.dirty = 1, False
        self.solution_name = ""
        self.chooser = SolutionPath(self.project, "blindjack")
        self.mode = QComboBox()
        self.mode.addItems(["Inspect solution", "Play", "Human vs model"])
        self.mode.setCurrentIndex(1)
        self.number = QSpinBox()
        self.number.setMinimum(1)
        self.load_button = QPushButton("Load solution")
        self.run_button = QPushButton("Run model")
        self.commit_button = QPushButton("Commit selection")
        self.prepare_button = QPushButton("Prepare resources")
        self.save_button = QPushButton("Save session…")
        self.open_button = QPushButton("Open session…")
        self.reset_button = QPushButton("New session")
        self.previous_button, self.next_button = (
            QPushButton("Previous"),
            QPushButton("Next hand"),
        )
        self.unplayed_button = QPushButton("Next unplayed")
        self.message = Notice()
        self.totals = QLabel()
        self.totals.setObjectName("stat")
        self.totals.setWordWrap(True)
        self.diagnostics = Diagnostics(compact=True)
        self.tiles, self.cards, self.answers = [], [], []
        board = QWidget()
        grid = QGridLayout(board)
        grid.setContentsMargins(0, 0, 0, 0)
        for i in range(7):
            card = Card(i + 1)
            self.tiles.append(card)
            self.cards.append(card.choice)
            self.answers.append(card.answer)
            card.choice.toggled.connect(self.remember_selection)
            grid.addWidget(card, i // 4, i % 4, Qt.AlignmentFlag.AlignTop)
        grid.setRowStretch(2, 1)
        controls = QHBoxLayout()
        for widget in (
            self.previous_button,
            QLabel("Hand"),
            self.number,
            self.next_button,
            self.unplayed_button,
            self.run_button,
            self.commit_button,
        ):
            controls.addWidget(widget)
        controls.addStretch()
        self.solution_bar = QWidget()
        top = QHBoxLayout(self.solution_bar)
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.chooser, 1)
        top.addWidget(self.load_button)
        split = QSplitter()
        self.board = QScrollArea()
        self.board.setWidgetResizable(True)
        self.board.setFrameShape(QScrollArea.Shape.NoFrame)
        self.board.viewport().setAutoFillBackground(False)
        self.board.setWidget(board)
        board.setAutoFillBackground(False)
        split.addWidget(self.board)
        split.addWidget(self.diagnostics)
        split.setSizes([780, 280])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(
            heading("Blindjack", "Select cards to approach 21. A sum above 21 earns zero."),
            1,
        )
        header.addWidget(self.open_button)
        header.addWidget(self.save_button)
        layout.addLayout(header)
        activity = QHBoxLayout()
        activity.addWidget(QLabel("Activity"))
        activity.addWidget(self.mode)
        activity.addWidget(self.reset_button)
        activity.addStretch()
        activity.addWidget(self.prepare_button)
        layout.addLayout(activity)
        layout.addWidget(self.solution_bar)
        layout.addLayout(controls)
        layout.addWidget(self.message)
        layout.addWidget(split, 1)
        layout.addWidget(self.totals)
        self.commit_button.setProperty("primary", True)
        self.run_button.setProperty("primary", True)
        self.load_button.clicked.connect(self.load_solution)
        self.run_button.clicked.connect(self.predict)
        self.commit_button.clicked.connect(self.commit)
        self.prepare_button.clicked.connect(self.prepare)
        self.number.valueChanged.connect(self.show_hand)
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.save_button.clicked.connect(self.save)
        self.open_button.clicked.connect(self.open_recording)
        self.reset_button.clicked.connect(self.new_session)
        self.previous_button.clicked.connect(
            lambda: self.number.setValue(self.number.value() - 1)
        )
        self.next_button.clicked.connect(
            lambda: self.number.setValue(self.number.value() + 1)
        )
        self.unplayed_button.clicked.connect(self.next_unplayed)
        self.chooser.path.textChanged.connect(self.buttons)
        self.load_data()

    def load_data(self):
        try:
            spec = resources.manifest(self.project / "labs/blindjack/resources.toml")
            resource_hash = resources.ready(self.project / "data/blindjack", spec)
            if self.game is None or resource_hash != self.resource_hash:
                self.game = BlindjackGame(self.project / "data/blindjack")
            self.resource_hash = resource_hash
            self.number.setMaximum(len(self.game.hands))
            self.show_hand()
        except Exception as exc:
            self.message.setText(f"Cards are not ready. Click Prepare resources.\n{exc}")
            self.game = None
            self.buttons()

    def buttons(self):
        busy = self.job is not None
        ready = self.game is not None
        index = self.number.value() - 1
        committed = ready and index in self.game.records
        inspecting = self.mode.currentIndex() != 1
        self.solution_bar.setVisible(inspecting)
        self.run_button.setVisible(inspecting)
        self.commit_button.setVisible(self.mode.currentIndex() != 0)
        self.diagnostics.setVisible(
            inspecting and (self.mode.currentIndex() == 0 or committed)
        )
        loaded = self.solution is not None and (
            not self.solution_name or self.solution_name == self.chooser.path.text()
        )
        self.load_button.setText("Reload solution" if loaded else "Load solution")
        self.load_button.setEnabled(not busy)
        self.chooser.setEnabled(not busy)
        self.prepare_button.setEnabled(not busy)
        self.prepare_button.setText("Verify resources" if ready else "Prepare resources")
        self.save_button.setEnabled(
            ready and not busy and bool(self.game.records or self.decisions or self.drafts)
        )
        self.open_button.setEnabled(ready and not busy)
        self.reset_button.setEnabled(
            ready and not busy and bool(self.game.records or self.decisions or self.drafts)
        )
        self.previous_button.setEnabled(ready and not busy and index > 0)
        self.next_button.setEnabled(
            ready and not busy and index + 1 < self.number.maximum()
        )
        self.unplayed_button.setVisible(self.mode.currentIndex() != 0)
        self.unplayed_button.setEnabled(
            ready and not busy and len(self.game.records) < self.number.maximum()
        )
        self.number.setEnabled(ready and not busy)
        self.mode.setEnabled(not busy)
        self.run_button.setEnabled(
            ready
            and loaded
            and not busy
            and self.mode.currentIndex() != 1
            and (self.mode.currentIndex() == 0 or (not committed and self.decision is None))
        )
        self.commit_button.setEnabled(
            ready
            and not busy
            and not committed
            and self.mode.currentIndex() != 0
            and (self.mode.currentIndex() == 1 or self.decision is not None)
        )
        for card in self.cards:
            card.setEnabled(
                ready and not busy and not committed and self.mode.currentIndex() != 0
            )

    def work(self, function, callback):
        if self.job is not None:
            return
        job = self.job = Job(function, self)
        self.completion = callback
        job.finished.connect(self.finished_job, Qt.ConnectionType.QueuedConnection)
        self.buttons()
        job.start()

    @Slot()
    def finished_job(self):
        job, callback = self.job, self.completion
        self.job = self.completion = None
        if job.error:
            self.message.setText(job.error)
        else:
            callback(job.value)
        self.buttons()
        job.deleteLater()

    def prepare(self):
        self.message.setText("Preparing verified resources…")
        self.work(
            lambda: resources.sync(
                self.project / "data/blindjack",
                resources.manifest(self.project / "labs/blindjack/resources.toml"),
            ),
            lambda _: self.load_data(),
        )

    def load_solution(self):
        if not self.confirm_reset():
            return
        path = Path(self.chooser.path.text())
        self.message.setText("Loading solution…")

        def loaded(solution):
            self.solution = solution
            self.solution_name = str(path)
            self.dirty = False
            self.change_mode()
            self.message.setText("Solution loaded. Run it on the selected hand.")

        self.work(lambda: training.load_solution(path), loaded)

    def change_mode(self):
        if self.confirm_reset():
            self.activity = self.mode.currentIndex()
            self.clear_session()
        else:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(self.activity)
            self.mode.blockSignals(False)

    def new_session(self):
        if self.confirm_reset():
            self.clear_session()

    def confirm_reset(self):
        return (
            not self.dirty
            or QMessageBox.question(
                self,
                "Replace current session?",
                "This session has unsaved choices. Continue and discard them? Use Save session to keep them.",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            == QMessageBox.StandardButton.Discard
        )

    def clear_session(self):
        self.dirty = False
        self.decisions.clear()
        self.drafts.clear()
        if self.game is not None:
            self.game.records.clear()
        self.show_hand()

    def remember_selection(self):
        if (
            self.game is not None
            and self.mode.currentIndex() != 0
            and self.number.value() - 1 not in self.game.records
        ):
            self.drafts[self.number.value() - 1] = [c.isChecked() for c in self.cards]
            self.dirty = True
            self.buttons()

    def next_unplayed(self):
        if self.game is None:
            return
        count = len(self.game.hands)
        for offset in range(1, count + 1):
            index = (self.number.value() - 1 + offset) % count
            if index not in self.game.records:
                self.number.setValue(index + 1)
                return

    def show_hand(self):
        if self.game is None:
            return
        self.decision = self.decisions.get(self.number.value() - 1)
        self.diagnostics.show_snapshot({})
        index = self.number.value() - 1
        draft = self.drafts.get(index, [False] * 7)
        for i, (image, tile, card, answer) in enumerate(
            zip(self.game.hand(index), self.tiles, self.cards, self.answers)
        ):
            tile.set_image(image)
            card.blockSignals(True)
            card.setChecked(draft[i])
            card.blockSignals(False)
            tile.highlight(card.isChecked())
            answer.setText("")
        self.started = time.perf_counter()
        if index in self.game.records:
            self.show_result(self.game.records[index])
            if self.decision is not None:
                self.diagnostics.show_snapshot(self.decision["debug"])
        elif self.mode.currentIndex() == 0 and self.decision is not None:
            self.show_prediction(index, self.decision)
        else:
            self.message.setText(
                "Choose cards, then commit your answer."
                if self.mode.currentIndex() == 1
                else "Choose a trained solution above, then click Load solution."
                if self.solution is None
                else "Run the model. Its answer stays hidden until you commit."
                if self.mode.currentIndex() == 2
                else "Run the model to inspect its selection and diagnostics."
            )
        self.update_totals()
        self.buttons()

    def predict(self):
        index = self.number.value() - 1
        images = self.game.hand(index)

        def finished(decision):
            self.dirty = True
            self.decision = decision
            self.decisions[index] = decision
            if self.mode.currentIndex() == 0:
                self.show_prediction(index, decision)
            else:
                self.message.setText(
                    "Model answer is ready. Commit your selection to compare."
                )

        self.message.setText("Running model…")
        self.work(lambda: inspect_hand(self.solution, images), finished)

    def show_prediction(self, index, decision):
        from coursekit.labs.blindjack import reward

        values = self.game.labels[self.game.hands[index]]
        for card, answer, value, selected in zip(
            self.cards, self.answers, values, decision["mask"]
        ):
            card.setChecked(bool(selected))
            answer.setText(f"Value: {value}")
        self.message.setText(
            f"Model reward: {reward(values, decision['mask'])} / 21. Inference: {decision['seconds']:.4f} s."
        )
        self.diagnostics.show_snapshot(decision["debug"])

    def commit(self):
        self.dirty = True
        index = self.number.value() - 1
        human = np.array([c.isChecked() for c in self.cards], dtype=bool)
        record = self.game.commit(
            index,
            human,
            self.decision if self.mode.currentIndex() == 2 else None,
            seconds=time.perf_counter() - self.started,
        )
        self.show_result(record)
        if self.decision is not None:
            self.diagnostics.show_snapshot(self.decision["debug"])
        self.update_totals()
        self.buttons()

    def show_result(self, record):
        for i, (card, label, value) in enumerate(
            zip(self.cards, self.answers, self.game.reveal(record["hand"]))
        ):
            card.setChecked(record["human"][i])
            model = (
                ""
                if record["model"] is None
                else (" · model: yes" if record["model"][i] else " · model: no")
            )
            label.setText(f"Value: {value}{model}")
        text = f"Your reward: {record['human_reward']} / 21."
        if record["model_reward"] is not None:
            text += f" Model reward: {record['model_reward']} / 21."
        text += " Use Next unplayed to continue; this answer is saved in the session."
        self.message.setText(text)

    def update_totals(self):
        summary = self.game.summary()
        text = f"Committed hands: {summary['hands']}"
        if summary["human_score"] is not None:
            text += f" · Your score: {summary['human_score']:.2f} / 100"
        if summary["paired_hands"]:
            text += f" · Model: {summary['model_score']:.2f} / 100 on {summary['paired_hands']} paired hands"
        self.totals.setText(text)

    def save(self):
        if self.game is None or self.job is not None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save comparison",
            str(self.project / "runs/blindjack-comparison.npz"),
            "Recording (*.npz)",
        )
        if filename:
            path = Path(filename)
            try:
                recording.save(
                    path,
                    {
                        "lab": "blindjack",
                        "version": 1,
                        "resource_hash": self.resource_hash,
                        "solution": self.solution_name,
                        "mode": self.mode.currentIndex(),
                        "summary": self.game.summary(),
                        "hands": list(self.game.records.values()),
                        "hand": self.number.value() - 1,
                        "drafts": {str(i): d for i, d in self.drafts.items()},
                        "decisions": {str(i): d for i, d in self.decisions.items()},
                    },
                )
                self.message.setText(f"Saved {path}")
                self.dirty = False
            except Exception as exc:
                self.message.setText(f"Cannot save session: {exc}")

    def open_recording(self):
        if self.game is None or self.job is not None:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open recording", str(self.project / "runs"), "Recording (*.npz)"
        )
        if not filename:
            return
        try:
            record = recording.load(filename)
            if (record["lab"], record["version"], record["resource_hash"]) != (
                "blindjack",
                1,
                self.resource_hash,
            ):
                raise ValueError(
                    "This recording belongs to a different lab or resource set."
                )
            game = BlindjackGame(self.project / "data/blindjack")
            decisions = {int(i): d for i, d in record["decisions"].items()}
            if (
                type(record["mode"]) is not int
                or record["mode"] not in range(3)
                or not isinstance(record["solution"], str)
            ):
                raise ValueError("Invalid recorded activity or solution name.")
            for index, decision in decisions.items():
                game.hand(index)
                validate_mask(decision["mask"])
                if not np.isfinite(decision["seconds"]) or decision["seconds"] < 0:
                    raise ValueError("Invalid recorded inference time.")
                decision["debug"] = copy_debug(decision["debug"])
            for row in record["hands"]:
                model = None if row["model"] is None else decisions[row["hand"]]
                game.commit(
                    row["hand"],
                    np.asarray(row["human"]),
                    model,
                    seconds=row["human_seconds"],
                )
            drafts = {int(i): d for i, d in record.get("drafts", {}).items()}
            for index, draft in drafts.items():
                game.hand(index)
                validate_mask(np.asarray(draft))
            current = record.get(
                "hand",
                min([r["hand"] for r in record["hands"]] + list(decisions), default=0),
            )
            game.hand(current)
            if not self.confirm_reset():
                return
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(record["mode"])
            self.mode.blockSignals(False)
            self.activity, self.dirty = record["mode"], False
            self.game, self.decisions = game, decisions
            self.drafts = drafts
            self.solution_name = record["solution"]
            self.solution = None
            if self.solution_name:
                self.chooser.path.setText(self.solution_name)
            self.number.blockSignals(True)
            self.number.setValue(current + 1)
            self.number.blockSignals(False)
            self.show_hand()
        except Exception as exc:
            self.message.setText(f"Cannot open recording: {exc}")
