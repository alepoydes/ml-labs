import traceback

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from coursekit.gui.diagnostics import Diagnostics as Diagnostics


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
