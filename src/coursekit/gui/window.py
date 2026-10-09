"""Window shortcuts; the desktop owns the title bar and its system menu."""

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt
from PySide6.QtGui import QAction


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
        window.addActions([minimize, maximize, self.fullscreen, leave, close])
        menu = window.menuBar().addMenu("&Window")
        for action in (minimize, maximize, self.fullscreen, leave, close):
            menu.addAction(action)
        self.actions = dict(
            fullscreen=self.fullscreen, maximize=maximize, minimize=minimize, close=close
        )
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
            if self.window.isMaximized():
                return
        self.window.showNormal() if self.window.isMaximized() else self.window.showMaximized()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.WindowStateChange:
            with QSignalBlocker(self.fullscreen):
                self.fullscreen.setChecked(self.window.isFullScreen())
        return False


def window_actions(window):
    window.controls = WindowControls(window)
    return window.controls.actions
