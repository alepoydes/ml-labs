"""Image/question inspection, recorded answers, and whole-batch debugging."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSettings, Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from coursekit import recording, resources
from coursekit.gui.common import Diagnostics, Job, Notice, SolutionPath, pixmap
from coursekit.gui.query_run import Run
from coursekit.labs.visual_queries_data import metadata, split_label
from coursekit.labs.visual_queries_inspection import (
    Dataset,
    new_record,
    probabilities,
    read_record,
)


class Answers(QAbstractTableModel):
    headers = ["Row", "Batch", "Question", "Model", "Expected", "P(answer)"]

    def __init__(self, page):
        super().__init__(page)
        self.page, self.numbers = page, []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.numbers)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        page, number = self.page, self.numbers[index.row()]
        row = page.dataset.examples[number - 1]
        found = page.predictions.get(number)
        answer, probability, batch_id = "—", "—", "—"
        if found:
            batch_id, i = found
            item = page.active_run["batches"][batch_id]
            answer = page.dataset.words[item["answers"][i]]
            debug = item.get("debug", [{}] * len(item["rows"]))[i]
            p = probabilities(debug, len(page.dataset.words))
            if p is not None:
                probability = f"{p[item['answers'][i]]:.1%}"
            batch_id += 1
        return str(
            [
                number,
                batch_id,
                row["question"],
                answer,
                row["answer"] if page.reveal.isChecked() else "Hidden",
                probability,
            ][index.column()]
        )


class VisualQueriesPage(QWidget):
    def __init__(self, project, *, data=None, record_path=None, readonly=False):
        super().__init__()
        self.project, self.readonly = Path(project), readonly
        self.job = self.preparation = self.image_job = None
        self.dataset = self.active_run = None
        self.record, self.datasets, self.predictions = new_record(), [], {}
        self.record_path = None
        self.loading, self.number = False, 1
        self.data_path = QLineEdit(str(data or self.project / "data/visual_queries"))
        self.split = QComboBox()
        self.split.setMinimumContentsLength(22)
        self.split.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.open_data = QPushButton("Open data…")
        self.open_record = QPushButton("Open recording…")
        self.save_record = QPushButton("Save recording…")
        self.chooser = SolutionPath(self.project, "visual_queries")
        self.new_solution = QPushButton("New solution…")
        self.artifacts = QLineEdit()
        self.artifacts.setPlaceholderText("Default: solution/artifacts")
        self.artifact_browse = QPushButton("Weights…")
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["cpu", "cuda:0"])
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(16)
        self.start = QPushButton("New run")
        self.next_batch = QPushButton("Next batch")
        self.play = QPushButton("Run remaining")
        self.play.setCheckable(True)
        self.stop_button = QPushButton("Stop")
        self.previous = QPushButton("Previous row")
        self.next_row = QPushButton("Next row")
        self.errors = QCheckBox("Errors only")
        self.same_image = QCheckBox("Same image")
        self.reveal = QCheckBox("Show expected answers")
        self.reveal.setChecked(True)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find question…")
        self.identity = QLabel("No model loaded. Browse data or open a recording.")
        self.identity.setWordWrap(True)
        self.identity.setTextFormat(Qt.TextFormat.PlainText)
        self.message = Notice()
        self.picture = QLabel("Select a question")
        self.picture.setMinimumSize(250, 250)
        self.picture.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_reference = QLabel()
        self.size_reference.setWordWrap(True)
        self.question, self.answer = QLabel(), QLabel()
        for widget in (self.question, self.answer):
            widget.setWordWrap(True)
            widget.setTextFormat(Qt.TextFormat.PlainText)
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.table = QTableView()
        self.model = Answers(self)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        for i, width in ((0, 45), (1, 48), (3, 70), (4, 78), (5, 80)):
            self.table.setColumnWidth(i, width)
        self.figure = Figure(figsize=(5, 2.5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.full_distribution = QCheckBox("All tokens")
        self.probability_table = QTableWidget(0, 3)
        self.probability_table.setHorizontalHeaderLabels(
            ["Token ID", "Token", "Probability"]
        )
        self.probability_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.probability_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.probability_table.hide()
        self.diagnostics = Diagnostics(dict(plots=[dict(mode=3, fields=[])]), compact=True)
        self.tabs = QTabWidget()
        probability_panel = QWidget()
        probability_layout = QVBoxLayout(probability_panel)
        probability_layout.addWidget(self.full_distribution)
        probability_layout.addWidget(self.canvas, 1)
        probability_layout.addWidget(self.probability_table, 1)
        self.tabs.addTab(probability_panel, "Answer probabilities")
        self.tabs.addTab(self.diagnostics, "Model debug")
        left, right = QWidget(), QWidget()
        ll, rr = QVBoxLayout(left), QVBoxLayout(right)
        ll.addWidget(self.picture, 1)
        ll.addWidget(self.size_reference)
        ll.addWidget(self.question)
        ll.addWidget(self.answer)
        rr.addWidget(self.table, 1)
        rr.addWidget(self.tabs, 1)
        self.panes = QSplitter()
        self.panes.addWidget(left)
        self.panes.addWidget(right)
        self.panes.setSizes([410, 820])
        layout = QVBoxLayout(self)

        def row(*widgets):
            line = QHBoxLayout()
            for widget in widgets:
                line.addWidget(widget)
            layout.addLayout(line)

        row(
            QLabel("Data"),
            self.data_path,
            self.open_data,
            self.split,
            self.open_record,
            self.save_record,
        )
        self.model_controls = QWidget()
        controls = QHBoxLayout(self.model_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        for widget in (
            self.chooser,
            self.new_solution,
            self.artifacts,
            self.artifact_browse,
            self.device,
        ):
            controls.addWidget(widget)
        layout.addWidget(self.model_controls)
        self.run_controls = QWidget()
        controls = QHBoxLayout(self.run_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        for widget in (
            QLabel("Batch"),
            self.batch_size,
            self.start,
            self.next_batch,
            self.play,
            self.stop_button,
        ):
            controls.addWidget(widget)
        controls.addStretch()
        layout.addWidget(self.run_controls)
        layout.addWidget(self.identity)
        row(
            self.previous,
            self.next_row,
            self.errors,
            self.same_image,
            self.reveal,
            self.search,
        )
        layout.addWidget(self.panes, 1)
        layout.addWidget(self.message)
        self.model_controls.setVisible(not readonly)
        self.run_controls.setVisible(not readonly)
        self.open_data.setVisible(not readonly)
        self.data_path.setReadOnly(readonly)
        self.start.clicked.connect(self.start_run)
        self.next_batch.clicked.connect(lambda: self.job.next() if self.job else None)
        self.play.toggled.connect(lambda value: self.job.play(value) if self.job else None)
        self.stop_button.clicked.connect(self.stop)
        self.previous.clicked.connect(lambda: self.move_row(-1))
        self.next_row.clicked.connect(lambda: self.move_row(1))
        self.errors.toggled.connect(self.refresh_rows)
        self.same_image.toggled.connect(self.refresh_rows)
        self.search.textChanged.connect(self.refresh_rows)
        self.reveal.toggled.connect(self.refresh_rows)
        self.full_distribution.toggled.connect(self.show_selected)
        self.table.selectionModel().currentRowChanged.connect(self.show_selected)
        self.open_data.clicked.connect(self.browse_data)
        self.open_record.clicked.connect(self.browse_record)
        self.save_record.clicked.connect(self.save)
        self.artifact_browse.clicked.connect(self.browse_artifacts)
        self.new_solution.clicked.connect(self.create_solution)
        self.split.currentIndexChanged.connect(self.select_split)
        self.data_path.returnPressed.connect(
            lambda: self.load_data(Path(self.data_path.text()))
        )
        self.settings = QSettings(
            str(self.project / "runs/visual_queries/inspection.ini"),
            QSettings.Format.IniFormat,
        )
        if self.settings.contains("panes"):
            self.panes.restoreState(self.settings.value("panes"))
        self.update_buttons()
        if record_path:
            self.load_record(record_path, data_override=data)
        else:
            self.load_data(Path(self.data_path.text()), sync=data is None)

    def prepare(self, function, callback):
        if self.preparation and self.preparation.isRunning():
            return
        self.message.setText("Loading and verifying data…")
        self.preparation = Job(function, self)
        self.preparation.callback = callback
        self.preparation.finished.connect(
            self.preparation_done, Qt.ConnectionType.QueuedConnection
        )
        self.preparation.start()
        self.update_buttons()

    @Slot()
    def preparation_done(self):
        if self.sender() is not self.preparation:
            return
        if self.preparation.error:
            self.message.setText(self.preparation.error)
        else:
            try:
                self.preparation.callback(self.preparation.value)
                self.message.setText(
                    (self.active_run or {}).get("error")
                    or "Ready. Select any question; model calls always use complete batches."
                )
            except Exception as exc:
                self.message.setText(str(exc))
        self.update_buttons()

    def set_data_splits(self, data):
        self.loading = True
        self.split.clear()
        for name in data.meta["splits"]:
            self.split.addItem(split_label(data.meta, name), name)
            self.split.setItemData(
                self.split.count() - 1, name, Qt.ItemDataRole.ToolTipRole
            )
        self.split.setCurrentIndex(self.split.findData(data.split))
        self.loading = False

    def load_data(self, root, *, sync=False):
        def load():
            if sync:
                resources.sync(
                    root,
                    resources.manifest(self.project / "labs/visual_queries/resources.toml"),
                )
            meta = metadata(root)
            name = (
                "base/validation"
                if "base/validation" in meta["splits"]
                else meta["splits"][0]
            )
            return Dataset(root, name)

        def install(data):
            self.record, self.datasets = new_record(), []
            self.record_path = None
            self.data_path.setText(str(data.root))
            self.set_data_splits(data)
            self.install_dataset(data, None)

        self.prepare(load, install)

    def load_record(self, path, *, data_override=None):
        def install(value):
            self.record, self.datasets = value
            self.record_path = Path(path).resolve()
            self.loading = True
            self.split.clear()
            for i, run in enumerate(self.record["runs"]):
                label = split_label(self.datasets[i].meta, run["split"])
                self.split.addItem(
                    f"{run.get('dataset', '')} · {label}".removeprefix(" · "), i
                )
                self.split.setItemData(i, run["split"], Qt.ItemDataRole.ToolTipRole)
            self.loading = False
            self.select_split(0)

        self.prepare(lambda: read_record(path, data_override=data_override), install)

    def install_dataset(self, data, run):
        self.dataset, self.active_run = data, run
        self.number = 1
        self.data_path.setText(str(data.root))
        self.batch_size.setMaximum(data.settings["max_batch"])
        sizes = sorted(
            data.meta.get("observable_rules", {}).get("size_pixels", {}).items(),
            key=lambda item: item[1][0],
        )
        self.size_reference.setText(
            "Size (maximum ink dimension): "
            + " · ".join(f"{name} {lo}–{hi} px" for name, (lo, hi) in sizes)
            if sizes
            else ""
        )
        self.predictions = {}
        if run:
            for b, item in enumerate(run["batches"]):
                for i, number in enumerate(item["rows"]):
                    self.predictions[int(number)] = b, i
            model = run["model"]
            version = (
                model.get("version")
                or model.get("artifacts_hash")
                or model.get("archive_sha256")
                or model.get("source", "unknown")
            )
            kind = (
                "Diagnostic run" if self.record.get("diagnostic") else "Recorded evaluation"
            )
            state = (
                "Complete"
                if run["complete"] and self.record["complete"]
                else "Partial recording"
            )
            self.identity.setText(
                f"{kind}: {model.get('name', 'solution')} · {version[:12]} · {state}"
            )
        else:
            self.identity.setText("Dataset only. No model has been run.")
        self.refresh_rows()

    @Slot(int)
    def select_split(self, index):
        if self.loading or index < 0:
            return
        if self.datasets:
            self.install_dataset(self.datasets[index], self.record["runs"][index])
        else:
            root, split = Path(self.data_path.text()), self.split.currentData()
            self.prepare(
                lambda: Dataset(root, split), lambda data: self.install_dataset(data, None)
            )

    def refresh_rows(self, *_):
        if self.dataset is None:
            return
        selected = self.number
        reference = self.dataset.examples[selected - 1]
        numbers = []
        for number, row in enumerate(self.dataset.examples, 1):
            if self.search.text().lower() not in row["question"].lower():
                continue
            if self.same_image.isChecked() and any(
                row[k] != reference[k] for k in ("sheet", "row", "column")
            ):
                continue
            if self.errors.isChecked():
                found = self.predictions.get(number)
                if not found or not row["answer"]:
                    continue
                b, i = found
                if (
                    self.dataset.words[self.active_run["batches"][b]["answers"][i]]
                    == row["answer"]
                ):
                    continue
            numbers.append(number)
        self.model.beginResetModel()
        self.model.numbers = numbers
        self.model.endResetModel()
        if numbers:
            self.table.selectRow(numbers.index(selected) if selected in numbers else 0)
        else:
            self.picture.clear()
            self.current_picture = None
            self.question.setText("No examples match this filter.")
            self.answer.clear()
            self.diagnostics.show_snapshot({})
            self.probability_table.setRowCount(0)
            self.probability_table.hide()
            self.canvas.show()
            self.figure.clear()
            self.canvas.draw_idle()

    @Slot()
    def show_selected(self, *_):
        index = self.table.currentIndex().row()
        if self.dataset is None or not 0 <= index < len(self.model.numbers):
            return
        self.number = self.model.numbers[index]
        row = self.dataset.examples[self.number - 1]
        self.question.setText(row["question"])
        self.request_image()
        found = self.predictions.get(self.number)
        debug, answer, position = {}, "Not predicted", ""
        if found:
            b, i = found
            item = self.active_run["batches"][b]
            answer = self.dataset.words[item["answers"][i]]
            debug = item.get("debug", [{}] * len(item["rows"]))[i]
            position = f" · batch {b + 1}, row {i + 1}/{len(item['rows'])}"
        expected = (
            f"\nExpected: {row['answer'] or 'unlabeled'}" if self.reveal.isChecked() else ""
        )
        self.answer.setText(f"Row {self.number}{position}\nModel: {answer}{expected}")
        self.diagnostics.show_snapshot(debug)
        self.figure.clear()
        axis = self.figure.add_subplot()
        p = probabilities(debug, len(self.dataset.words))
        show_all = p is not None and self.full_distribution.isChecked()
        self.probability_table.setVisible(show_all)
        self.canvas.setVisible(not show_all)
        if p is None:
            axis.text(0.5, 0.5, "Probabilities not provided", ha="center", va="center")
            axis.set_axis_off()
        else:
            order = np.argsort(-p[1:]) + 1
            if show_all:
                self.probability_table.setRowCount(len(order))
                for row, token in enumerate(order):
                    for column, text in enumerate(
                        (str(token), self.dataset.words[token], f"{p[token]:.8f}")
                    ):
                        self.probability_table.setItem(row, column, QTableWidgetItem(text))
            order = order[:10]
            axis.bar(np.arange(len(order)), p[order])
            axis.set_xticks(
                np.arange(len(order)),
                [self.dataset.words[i] for i in order],
                rotation=60 if len(order) > 12 else 0,
            )
            axis.set_ylim(0, 1)
            axis.set_ylabel("Probability")
            axis.set_title(
                "All answer probabilities"
                if self.full_distribution.isChecked()
                else "Most likely answers"
            )
        self.canvas.draw_idle()

    def request_image(self):
        if getattr(self, "displayed_image", None) != (self.dataset, self.number):
            self.current_picture = None
            self.picture.setText("Loading image…")
        if self.image_job and self.image_job.isRunning():
            return
        data, number = self.dataset, self.number
        self.image_request = data, number
        self.image_job = Job(lambda: data.image(number), self)
        self.image_job.finished.connect(
            self.image_ready, Qt.ConnectionType.QueuedConnection
        )
        self.image_job.start()

    @Slot()
    def image_ready(self):
        if self.sender() is not self.image_job:
            return
        if not self.table.currentIndex().isValid():
            return
        if self.image_request != (self.dataset, self.number):
            self.request_image()
        elif self.image_job.error:
            self.message.setText(self.image_job.error)
        else:
            try:
                self.current_picture = pixmap(self.image_job.value)
                self.displayed_image = self.image_request
                self.scale_picture()
            except Exception as exc:
                self.message.setText(str(exc))

    def scale_picture(self):
        if getattr(self, "current_picture", None) is not None:
            self.picture.setPixmap(
                self.current_picture.scaled(
                    self.picture.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scale_picture()

    def move_row(self, delta):
        row = self.table.currentIndex().row() + delta
        if 0 <= row < len(self.model.numbers):
            self.table.selectRow(row)

    @Slot()
    def start_run(self):
        if self.dataset is None or self.readonly or self.job and self.job.isRunning():
            return
        path = Path(self.chooser.path.text()).resolve()
        if not (path / "solution.py").is_file():
            self.message.setText(
                "Choose a prepared solution directory containing solution.py and artifacts. Create a working copy with step_1_new_solution first."
            )
            return
        artifacts = (
            Path(self.artifacts.text()).resolve()
            if self.artifacts.text()
            else path / "artifacts"
        )
        self.record = new_record(diagnostic=True)
        self.record_path = None
        self.datasets = []
        self.set_data_splits(self.dataset)
        self.active_run = self.dataset.new_run()
        self.record["runs"].append(self.active_run)
        self.predictions = {}
        self.play.setChecked(False)
        self.job = Run(
            self.dataset,
            path,
            artifacts,
            self.device.currentText(),
            self.batch_size.value(),
            self,
        )
        self.job.loaded.connect(self.model_loaded, Qt.ConnectionType.QueuedConnection)
        self.job.batch_ready.connect(self.batch_ready, Qt.ConnectionType.QueuedConnection)
        self.job.phase.connect(self.message.setText, Qt.ConnectionType.QueuedConnection)
        self.job.finished.connect(self.run_done, Qt.ConnectionType.QueuedConnection)
        self.job.start()
        self.refresh_rows()
        self.update_buttons()

    @Slot(object)
    def model_loaded(self, identity):
        self.active_run["model"] = identity
        self.identity.setText(
            f"Loaded model: {identity['name']} · code {identity['source_hash'][:12]} · weights {identity['artifacts_hash'][:12]} · {identity['device']}"
        )

    @Slot(object)
    def batch_ready(self, item):
        b = len(self.active_run["batches"])
        self.active_run["batches"].append(item)
        for i, number in enumerate(item["rows"]):
            self.predictions[int(number)] = b, i
        self.number = int(item["rows"][0])
        self.refresh_rows()

    @Slot()
    def run_done(self):
        self.record["complete"] = self.active_run["complete"] = self.job.complete
        self.active_run["error"] = self.job.error
        self.message.setText(
            self.job.error
            or (
                "Completed."
                if self.job.complete
                else "Stopped; completed batches are available for inspection and saving."
            )
        )
        self.play.setChecked(False)
        self.update_buttons()

    def update_buttons(self):
        busy = bool(self.job and self.job.isRunning())
        preparing = bool(self.preparation and self.preparation.isRunning())
        self.start.setEnabled(not busy and not preparing and self.dataset is not None)
        for widget in (self.next_batch, self.play, self.stop_button):
            widget.setEnabled(busy)
        for widget in (
            self.open_data,
            self.open_record,
            self.split,
            self.data_path,
            self.model_controls,
            self.batch_size,
        ):
            widget.setEnabled(not busy and not preparing)
        self.save_record.setEnabled(bool(self.record["runs"]))

    def stop(self):
        if self.job:
            self.job.stop()

    def browse_data(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose Visual Queries corpus", self.data_path.text()
        )
        if path:
            self.load_data(Path(path))

    def browse_artifacts(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose saved model artifacts", self.chooser.path.text()
        )
        if path:
            self.artifacts.setText(path)

    def create_solution(self):
        from coursekit.training import new_solution

        template, ok = QInputDialog.getItem(
            self, "Create working copy", "Template", ["baseline", "starter"], editable=False
        )
        if not ok:
            return
        name, ok = QInputDialog.getText(
            self, "Create working copy", "Variant name", text=template
        )
        if ok:
            try:
                path = new_solution(self.project, "visual_queries", template, name)
                self.chooser.path.setText(str(path))
                self.message.setText(
                    f"Created {path}. Train the working copy with step_2_train before inference."
                )
            except Exception as exc:
                self.message.setText(str(exc))

    def browse_record(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open predictions", str(self.project), "Recording (*.npz)"
        )
        if path:
            self.load_record(path)

    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save predictions",
            str(self.project / "runs/visual_queries/predictions.npz"),
            "Recording (*.npz)",
        )
        if path:
            try:
                recording.save(path, self.record)
                self.message.setText(f"Saved: {path}")
            except Exception as exc:
                self.message.setText(str(exc))

    def shutdown(self):
        self.stop()
        for job in (self.job, self.preparation, self.image_job):
            if job and job.isRunning():
                job.wait()
        self.settings.setValue("panes", self.panes.saveState())
        self.settings.sync()
