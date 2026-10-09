import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QSurfaceFormat
from PySide6.QtWidgets import QApplication, QMainWindow


class Window(QMainWindow):
    def __init__(self, page, lab):
        super().__init__()
        from coursekit.gui.style import apply_style
        from coursekit.gui.window import window_actions

        apply_style(self)
        self.window_actions = window_actions(self)
        self.page = page
        self.closing = False
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCentralWidget(page)
        if hasattr(page, "diagnostics"):
            self.menuBar().addMenu(page.diagnostics.panels_menu)
        self.setWindowTitle(f"Coursekit · {lab.replace('_', ' ').title()}")
        self.resize(1280, 820)
        settings_file = Path(page.project).resolve() / "runs" / lab / "window.ini"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings = QSettings(str(settings_file), QSettings.Format.IniFormat)
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
            screen = QGuiApplication.primaryScreen()
            if screen is not None and not self.isMaximized() and not self.isFullScreen():
                available = screen.availableGeometry()
                self.resize(
                    min(self.width(), available.width()),
                    min(self.height(), available.height()),
                )
        mode = self.settings.value("mode", "normal")
        if mode == "maximized":
            self.setWindowState(Qt.WindowState.WindowMaximized)
        elif mode == "fullscreen":
            self.controls.was_maximized = self.settings.value(
                "maximized_before_fullscreen", False, type=bool
            )
            self.setWindowState(Qt.WindowState.WindowFullScreen)

    def closeEvent(self, event):
        if not self.closing:
            if hasattr(self.page, "confirm_reset") and not self.page.confirm_reset():
                event.ignore()
                return
            self.closing = True
            self.page.setEnabled(False)
        jobs = [
            self.page.job,
            getattr(self.page, "preparation", None),
            getattr(self.page, "warmup", None),
        ]
        if any(job is not None and job.isRunning() for job in jobs):
            if hasattr(self.page, "stop"):
                self.page.stop()
            self.page.message.setText(
                "Stopping the simulation process…"
                if getattr(self.page.job, "stopping", False)
                else "Finishing preparation before closing…"
            )
            event.ignore()
            QTimer.singleShot(50, self.close)
        else:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue(
                "mode",
                "fullscreen"
                if self.isFullScreen()
                else "maximized"
                if self.isMaximized()
                else "normal",
            )
            self.settings.setValue(
                "maximized_before_fullscreen", self.controls.was_maximized
            )
            self.settings.sync()
            if hasattr(self.page, "shutdown"):
                self.page.shutdown()
            event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a solution or play a public lab.")
    parser.add_argument(
        "lab", choices=["blindjack", "line_following"], nargs="?", default="blindjack"
    )
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--solution", type=Path)
    args = parser.parse_args(argv)
    surface = QSurfaceFormat()
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(surface)
    app = QApplication(sys.argv)
    if args.lab == "blindjack":
        from coursekit.gui.blindjack import BlindjackPage

        page = BlindjackPage(args.project)
    else:
        from coursekit.gui.line_following import LineFollowingPage

        page = LineFollowingPage(args.project)
    if args.solution is not None:
        page.chooser.path.setText(str(args.solution))
        if args.lab == "line_following":
            page.controller_mode.setCurrentIndex(1)
    window = Window(page, args.lab)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
