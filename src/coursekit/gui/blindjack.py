import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from coursekit import recording, resources, training
from coursekit.gui.common import Diagnostics, Job, SolutionPath, pixmap
from coursekit.inspection import BlindjackGame, inspect_hand


class BlindjackPage(QWidget):
    def __init__(self, project):
        super().__init__()
        self.project = Path(project)
        self.solution = self.game = self.decision = self.job = None
        self.decisions = {}
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
        save = QPushButton("Save results…")
        open_record = QPushButton("Open recording…")
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.totals = QLabel()
        self.diagnostics = Diagnostics()
        self.pictures, self.cards, self.answers = [], [], []
        board = QWidget()
        grid = QGridLayout(board)
        for i in range(7):
            picture, select, answer = QLabel(), QCheckBox(f"Card {i + 1}"), QLabel()
            picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pictures.append(picture)
            self.cards.append(select)
            self.answers.append(answer)
            card = QWidget()
            card_layout = QVBoxLayout(card)
            card_layout.addWidget(picture)
            card_layout.addWidget(select)
            card_layout.addWidget(answer)
            card.setMaximumHeight(220)
            grid.addWidget(card, i // 4, i % 4, Qt.AlignmentFlag.AlignTop)
        grid.setRowStretch(2, 1)
        controls = QHBoxLayout()
        for widget in (
            self.mode,
            QLabel("Hand"),
            self.number,
            self.run_button,
            self.commit_button,
            save,
            open_record,
        ):
            controls.addWidget(widget)
        top = QHBoxLayout()
        top.addWidget(self.chooser, 1)
        top.addWidget(self.load_button)
        top.addWidget(self.prepare_button)
        split = QSplitter()
        split.addWidget(board)
        split.addWidget(self.diagnostics)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(controls)
        layout.addWidget(split, 1)
        layout.addWidget(self.message)
        layout.addWidget(self.totals)
        self.load_button.clicked.connect(self.load_solution)
        self.run_button.clicked.connect(self.predict)
        self.commit_button.clicked.connect(self.commit)
        self.prepare_button.clicked.connect(self.prepare)
        self.number.valueChanged.connect(self.show_hand)
        self.mode.currentIndexChanged.connect(self.change_mode)
        save.clicked.connect(self.save)
        open_record.clicked.connect(self.open_recording)
        self.load_data()

    def load_data(self):
        try:
            spec = resources.manifest(self.project / "labs/blindjack/resources.toml")
            self.resource_hash = resources.ready(self.project / "data/blindjack", spec)
            self.game = BlindjackGame(self.project / "data/blindjack")
            self.number.setMaximum(len(self.game.hands))
            self.show_hand()
        except Exception as exc:
            self.message.setText(str(exc))
            self.game = None
            self.buttons()

    def buttons(self):
        busy = self.job is not None
        ready = self.game is not None
        index = self.number.value() - 1
        committed = ready and index in self.game.records
        self.load_button.setEnabled(not busy)
        self.prepare_button.setEnabled(not busy)
        self.number.setEnabled(ready and not busy)
        self.mode.setEnabled(not busy)
        self.run_button.setEnabled(
            ready
            and self.solution is not None
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
            card.setEnabled(ready and not committed and self.mode.currentIndex() != 0)

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
        path = Path(self.chooser.path.text())
        self.message.setText("Loading solution…")

        def loaded(solution):
            self.solution = solution
            self.solution_name = str(path)
            self.change_mode()
            self.message.setText("Solution loaded. Run it on the selected hand.")

        self.work(lambda: training.load_solution(path), loaded)

    def change_mode(self):
        self.decisions.clear()
        if self.game is not None:
            self.game.records.clear()
        self.show_hand()

    def show_hand(self):
        if self.game is None:
            return
        self.decision = self.decisions.get(self.number.value() - 1)
        self.diagnostics.show_snapshot({})
        index = self.number.value() - 1
        for image, picture, card, answer in zip(
            self.game.hand(index), self.pictures, self.cards, self.answers
        ):
            picture.setPixmap(pixmap(image))
            card.setChecked(False)
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
                else (" · model selected" if record["model"][i] else " · model skipped")
            )
            label.setText(f"Value: {value}{model}")
        text = f"Your reward: {record['human_reward']} / 21."
        if record["model_reward"] is not None:
            text += f" Model reward: {record['model_reward']} / 21."
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
        if self.game is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save comparison",
            str(self.project / "runs/blindjack-comparison.npz"),
            "Recording (*.npz)",
        )
        if filename:
            path = Path(filename)
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
                    "decisions": {str(i): d for i, d in self.decisions.items()},
                },
            )
            self.message.setText(f"Saved {path}")

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
            for row in record["hands"]:
                model = None if row["model"] is None else decisions[row["hand"]]
                game.commit(
                    row["hand"],
                    np.array(row["human"], dtype=bool),
                    model,
                    seconds=row["human_seconds"],
                )
            self.mode.setCurrentIndex(record["mode"])
            self.game, self.decisions = game, decisions
            self.solution_name = record["solution"]
            self.solution = None
            self.show_hand()
        except Exception as exc:
            self.message.setText(f"Cannot open recording: {exc}")
