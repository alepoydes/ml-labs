"""Shared visual language for the local course tools."""

STYLE = """
QWidget#workbench { background: #f4f6fa; color: #24334b; }
QWidget { font-size: 10pt; }
QWidget#sidebar { background: #edf1f7; border-right: 1px solid #dbe2ed; }
QLabel#brand { color: #172f53; font-size: 18pt; font-weight: 700; }
QLabel#page-title { color: #172f53; font-size: 21pt; font-weight: 600; }
QLabel#section-title { color: #263c59; font-size: 12pt; font-weight: 600; }
QLabel#muted { color: #607089; }
QLabel#selection { color: #294b77; padding: 5px 0; }
QListWidget#navigation { background: transparent; border: none; outline: none; }
QListWidget#navigation::item { padding: 10px 12px; margin: 2px 0; border-radius: 7px; }
QListWidget#navigation::item:selected { background: #d8e6fa; color: #15477e; font-weight: 600; }
QTreeWidget { background: transparent; border: none; outline: none; }
QTreeWidget::item { padding: 5px 0; }
QTreeWidget::item:selected { background: #d8e6fa; color: #15477e; }
QToolButton, QPushButton { background: white; color: #263c59; border: 1px solid #ccd6e5;
    border-radius: 6px; padding: 6px 10px; }
QToolButton:hover, QPushButton:hover { background: #e9f1fc; border-color: #85a8d6; }
QToolButton:pressed, QPushButton:pressed { background: #dae8fa; }
QToolButton:disabled, QPushButton:disabled { color: #929dad; background: #f3f5f8; border-color: #e0e5ed; }
QToolButton[primary="true"], QPushButton[primary="true"] { background: #2463a9; color: white; border-color: #2463a9; }
QToolButton[primary="true"]:hover, QPushButton[primary="true"]:hover { background: #1c548f; }
QToolButton[primary="true"]:disabled, QPushButton[primary="true"]:disabled { background: #dce5f1; color: #8395ae; border-color: #dce5f1; }
QLineEdit, QComboBox, QSpinBox { background: white; border: 1px solid #ccd6e5; border-radius: 5px; padding: 5px 8px; }
QLineEdit:focus, QComboBox:focus { border-color: #2463a9; }
QTableView { background: white; alternate-background-color: #f8fafe; border: 1px solid #dde4ee;
    border-radius: 6px; gridline-color: #edf1f6; selection-background-color: #dceafd; selection-color: #173c69; }
QHeaderView::section { background: #edf2f8; color: #485d79; border: none;
    border-bottom: 1px solid #dbe3ef; padding: 8px; font-weight: 600; }
QTabBar::tab { background: transparent; color: #66768b; padding: 9px 15px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #205e9f; border-bottom-color: #2463a9; font-weight: 600; }
QTabWidget::pane { border: 1px solid #dce3ee; background: white; }
QTextBrowser, QPlainTextEdit { background: white; border: none; padding: 8px; }
QWidget#empty { background: white; border-radius: 8px; }
QWidget#card { background: white; border: 2px solid #dbe2ed; border-radius: 8px; }
QWidget#card[selected="true"] { background: #eaf3ff; border-color: #2463a9; }
QToolButton#card-image { background: transparent; border: none; padding: 0; }
QWidget#notice { background: #e9eff8; border-radius: 6px; }
QLabel#stat { color: #172f53; font-size: 14pt; font-weight: 600; }
QStatusBar { background: #edf1f7; color: #566781; }
QProgressBar { border: none; background: #dce4ef; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #5a92d0; border-radius: 4px; }
QSplitter::handle { background: #e3e9f2; }
"""


def apply_style(widget):
    widget.setObjectName("workbench")
    widget.setStyleSheet(STYLE)
