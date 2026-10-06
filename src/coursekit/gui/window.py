"""Native window actions and visible standard controls for every desktop tool."""

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QStyle, QToolButton, QWidget


class WindowControls(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window, self.was_maximized = window, False
        window.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.fullscreen = QAction("Full screen", window, checkable=True)
        self.fullscreen.setShortcut("F11")
        self.fullscreen.toggled.connect(self.toggle_fullscreen)
        leave = QAction("Leave full screen", window)
        leave.setShortcut("Escape")
        leave.triggered.connect(lambda: self.fullscreen.setChecked(False))
        maximize = QAction("Maximize / restore", window)
        maximize.triggered.connect(self.toggle_maximized)
        minimize = QAction("Minimize", window)
        minimize.triggered.connect(window.showMinimized)
        close = QAction("Close window", window)
        close.setShortcut("Ctrl+W")
        close.triggered.connect(window.close)
        menu = window.menuBar().addMenu("&Window")
        menu.addActions([minimize, maximize, self.fullscreen, leave, close])
        self.actions = dict(
            fullscreen=self.fullscreen, maximize=maximize, minimize=minimize, close=close
        )
        # GNOME commonly requests only a close title-bar button. These standard Qt
        # controls keep minimize/maximize discoverable without changing desktop settings.
        bar = QWidget(window)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for name, icon in [
            ("minimize", QStyle.StandardPixmap.SP_TitleBarMinButton),
            ("maximize", QStyle.StandardPixmap.SP_TitleBarMaxButton),
            ("close", QStyle.StandardPixmap.SP_TitleBarCloseButton),
        ]:
            action = self.actions[name]
            action.setIcon(window.style().standardIcon(icon))
            button = QToolButton(bar)
            button.setDefaultAction(action)
            button.setAccessibleName(name.title() + " window")
            button.setAutoRaise(True)
            layout.addWidget(button)
        window.menuBar().setCornerWidget(bar, Qt.Corner.TopRightCorner)
        window.installEventFilter(self)

    def toggle_fullscreen(self, checked):
        if checked:
            self.was_maximized = self.window.isMaximized()
            self.window.showFullScreen()
        else:
            self.window.showMaximized() if self.was_maximized else self.window.showNormal()

    def toggle_maximized(self):
        if self.window.isFullScreen():
            self.fullscreen.setChecked(False)
        self.window.showNormal() if self.window.isMaximized() else self.window.showMaximized()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.WindowStateChange:
            with QSignalBlocker(self.fullscreen):
                self.fullscreen.setChecked(self.window.isFullScreen())
            icon = (
                QStyle.StandardPixmap.SP_TitleBarNormalButton
                if self.window.isMaximized()
                else QStyle.StandardPixmap.SP_TitleBarMaxButton
            )
            self.actions["maximize"].setIcon(self.window.style().standardIcon(icon))
        return False


def window_actions(window):
    window.controls = WindowControls(window)
    return window.controls.actions
