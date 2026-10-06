import traceback

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def heading(title, subtitle):
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    name, description = QLabel(title), QLabel(subtitle)
    name.setTextFormat(Qt.TextFormat.PlainText)
    description.setTextFormat(Qt.TextFormat.PlainText)
    name.setObjectName("page-title")
    description.setObjectName("muted")
    description.setWordWrap(True)
    layout.addWidget(name)
    layout.addWidget(description)
    return panel


class Notice(QWidget):
    """A short status with an expandable traceback, without resizing the whole page."""

    def __init__(self):
        super().__init__()
        self.setObjectName("notice")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(130)
        self.more = QPushButton("Show details")
        self.more.setCheckable(True)
        self.more.toggled.connect(self.details.setVisible)
        row = QHBoxLayout()
        row.addWidget(self.label, 1)
        row.addWidget(self.more)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.details)
        self.setText("")

    def setText(self, text):
        self.value = str(text)
        traceback = "Traceback (most recent call last)" in self.value
        self.label.setText(self.value.strip().splitlines()[-1] if traceback else self.value)
        self.details.setPlainText(self.value if traceback else "")
        self.more.setChecked(False)
        self.more.setVisible(traceback)
        self.details.hide()
        self.setVisible(bool(self.value))

    def text(self):
        return self.value


class Card(QWidget):
    """One canonical image, with matching mouse and keyboard selection controls."""

    def __init__(self, number):
        super().__init__()
        self.setObjectName("card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.picture = QToolButton()
        self.picture.setObjectName("card-image")
        self.picture.setIconSize(QSize(112, 112))
        self.picture.setAccessibleName(f"Select card {number}")
        self.choice = QCheckBox(f"Card {number}")
        self.choice.setShortcut(f"Alt+{number}")
        self.choice.setToolTip(f"Select card {number} (Alt+{number})")
        self.answer = QLabel()
        self.answer.setWordWrap(True)
        self.answer.setMinimumHeight(18)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self.picture, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.choice)
        layout.addWidget(self.answer)
        self.picture.clicked.connect(self.choice.click)
        self.choice.toggled.connect(self.highlight)
        self.setMinimumWidth(136)

    def set_image(self, image):
        self.picture.setIcon(QIcon(pixmap(image)))

    def highlight(self, selected):
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)


class Job(QThread):
    """Keep ordinary Python calls and tracebacks while Qt processes window events."""

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function, self.value, self.error = function, None, None

    def run(self):
        try:
            self.value = self.function()
        except BaseException:
            self.error = traceback.format_exc()
            print(self.error, end="")


def pixmap(array):
    array = np.ascontiguousarray(array, dtype=np.uint8)
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError("Expected an RGB image.")
    return QPixmap.fromImage(
        QImage(
            array.data, width, height, array.strides[0], QImage.Format.Format_RGB888
        ).copy()
    )


class SolutionPath(QWidget):
    def __init__(self, project, lab):
        super().__init__()
        self.path = QLineEdit(str(project / "work" / lab / "baseline"))
        choose = QPushButton("Browse…")
        choose.clicked.connect(self.browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Solution"))
        layout.addWidget(self.path, 1)
        layout.addWidget(choose)

    def browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose a working solution", self.path.text()
        )
        if path:
            self.path.setText(path)


class Diagnostics(QWidget):
    def __init__(self):
        super().__init__()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.snapshot, self.history = {}, None
        self.setMinimumWidth(240)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.draw)
        self.figure = Figure(figsize=(4, 2.3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(220, 160)
        self.empty = QLabel(
            "No diagnostics for this frame.\nSelect a field to inspect its value or plot."
        )
        self.empty.setWordWrap(True)
        self.empty.setObjectName("muted")
        self.value = QPlainTextEdit()
        self.value.setReadOnly(True)
        self.value.setMaximumHeight(140)
        self.save_plot = QPushButton("Save plot…")
        self.save_plot.clicked.connect(self.export_plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        title = QLabel("Diagnostics")
        title.setObjectName("section-title")
        layout.addWidget(title)
        layout.addWidget(self.empty)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.value)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.save_plot, 0, Qt.AlignmentFlag.AlignRight)
        layout.addStretch()
        self.show_snapshot({})

    def show_snapshot(self, snapshot, history=None):
        item = self.table.item(self.table.currentRow(), 0)
        name = item.text() if item else None
        self.snapshot, self.history = snapshot, history
        row = list(snapshot).index(name) if name in snapshot else 0
        self.table.blockSignals(True)
        self.table.setRowCount(len(snapshot))
        for i, (name, value) in enumerate(snapshot.items()):
            text = (
                f"array {value.shape}"
                if isinstance(value, np.ndarray)
                else f"{value:.5g}"
                if isinstance(value, float)
                else str(value)
            )
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.item(i, 0).setToolTip(name)
            self.table.setItem(i, 1, QTableWidgetItem(text))
        if snapshot:
            self.table.selectRow(row)
        self.table.setColumnWidth(0, 140)
        self.table.blockSignals(False)
        self.draw()

    def draw(self):
        self.figure.clear()
        self.empty.setVisible(not self.snapshot)
        self.table.setVisible(bool(self.snapshot))
        self.canvas.hide()
        self.value.hide()
        self.save_plot.hide()
        row = self.table.currentRow()
        if 0 <= row < len(self.snapshot):
            name = list(self.snapshot)[row]
            value = self.snapshot[name]
            plotted = (
                isinstance(value, np.ndarray)
                and value.size > 0
                or self.history is not None
                and name in self.history
            )
            self.canvas.setVisible(plotted)
            self.save_plot.setVisible(plotted)
            self.value.setVisible(not plotted)
            if not plotted:
                self.value.setPlainText(
                    f"Empty array {value.shape}"
                    if isinstance(value, np.ndarray) and not value.size
                    else str(value)
                )
                return
            axes = self.figure.add_subplot()
            axes.set_title(name, fontsize=10, wrap=True)
            if isinstance(value, np.ndarray) and value.ndim == 2:
                axes.imshow(value, aspect="auto", interpolation="nearest")
            elif isinstance(value, np.ndarray):
                axes.plot(np.arange(value.size), value.ravel(), marker="o")
            elif self.history is not None and name in self.history:
                times, values = self.history[name]
                axes.plot(
                    times, values, marker="o" if len(times) <= 20 else None, markersize=3
                )
                axes.set_xlabel("Simulated seconds")
            axes.tick_params(labelsize=8)
        self.canvas.draw_idle()

    def export_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save diagnostic plot", "diagnostic.png", "PNG (*.png);;SVG (*.svg)"
        )
        if path:
            try:
                self.figure.savefig(path)
            except Exception as exc:
                QMessageBox.warning(self, "Cannot save plot", str(exc))
