import traceback

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


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
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self.draw)
        self.figure = Figure(figsize=(4, 2.3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Controller diagnostics"))
        layout.addWidget(self.table)
        layout.addWidget(self.canvas)

    def show_snapshot(self, snapshot, history=None):
        self.snapshot, self.history = snapshot, history
        row = max(0, self.table.currentRow())
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
            self.table.setItem(i, 1, QTableWidgetItem(text))
        self.table.selectRow(min(row, len(snapshot) - 1))
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self.draw()

    def draw(self):
        self.figure.clear()
        row = self.table.currentRow()
        if 0 <= row < len(self.snapshot):
            name = list(self.snapshot)[row]
            value = self.snapshot[name]
            axes = self.figure.add_subplot()
            axes.set_title(name)
            if isinstance(value, np.ndarray) and value.ndim == 2:
                axes.imshow(value, aspect="auto", interpolation="nearest")
            elif isinstance(value, np.ndarray):
                axes.plot(np.arange(value.size), value.ravel(), marker="o")
            elif self.history is not None and name in self.history:
                times, values = self.history[name]
                axes.plot(times, values)
                axes.set_xlabel("Simulated seconds")
            else:
                axes.text(
                    0.5, 0.5, str(value), transform=axes.transAxes, ha="center", va="center"
                )
                axes.set_axis_off()
        self.canvas.draw_idle()
