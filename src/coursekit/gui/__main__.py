import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication, QMainWindow


class Window(QMainWindow):
    def __init__(self, page, lab):
        super().__init__()
        self.page = page
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCentralWidget(page)
        self.setWindowTitle(f"Coursekit · {lab.replace('_', ' ').title()}")
        self.resize(1280, 820)

    def closeEvent(self, event):
        job = self.page.job
        if job is not None and job.isRunning():
            if hasattr(self.page, "stop"):
                self.page.stop()
            self.page.message.setText(
                "Waiting for the current Python call to finish before closing…"
            )
            event.ignore()
            QTimer.singleShot(50, self.close)
        else:
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
    window = Window(page, args.lab)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
