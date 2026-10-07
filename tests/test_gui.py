import os
import time

import numpy as np
import pytest


@pytest.fixture(scope="session")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    instance = widgets.QApplication.instance() or widgets.QApplication([])
    yield instance
    instance.processEvents()


def wait_for(app, predicate):
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert predicate()


def test_blindjack_visible_cards_drafts_sessions_and_small_window(
    app, student_project, monkeypatch
):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox

    from coursekit.gui.__main__ import Window
    from coursekit.gui.blindjack import BlindjackPage

    page = BlindjackPage(student_project)
    window = Window(page, "blindjack")
    window.resize(1024, 700)
    window.show()
    app.processEvents()
    QTest.mouseClick(page.tiles[1].picture, Qt.MouseButton.LeftButton)
    assert page.cards[1].isChecked() and page.tiles[1].property("selected")
    page.next_button.click()
    assert not any(c.isChecked() for c in page.cards)
    page.previous_button.click()
    assert page.cards[1].isChecked()
    page.commit_button.click()
    assert page.game.summary()["hands"] == 1 and not page.commit_button.isEnabled()
    record = page.game.records[0].copy()
    page.prepare_button.click()
    wait_for(app, lambda: page.job is None)
    assert (
        page.game.records[0] == record
    )  # Verification must not discard a practice session.
    page.unplayed_button.click()
    assert page.number.value() == 2 and all(a.text() == "" for a in page.answers)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Cancel
    )
    page.mode.setCurrentIndex(0)
    assert page.mode.currentIndex() == 1 and page.game.records[0] == record
    window.close()
    assert window.isVisible()  # Unsaved choices remain available after cancelling closure.
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Discard
    )
    page.mode.setCurrentIndex(0)
    assert page.game.records == {}
    app.processEvents()
    assert window.width() <= 1024 and window.height() <= 700
    assert all(tile.isVisible() for tile in page.tiles)
    assert all(
        page.board.viewport()
        .rect()
        .contains(
            tile.picture.mapTo(page.board.viewport(), tile.picture.rect().bottomRight())
        )
        for tile in page.tiles
    )
    assert page.diagnostics.empty.isVisible()
    window.close()
    app.processEvents()


def test_playback_clock_controls_and_diagnostic_field_identity(app, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from coursekit.gui import playback
    from coursekit.gui.common import Diagnostics

    clock = [0.0]
    monkeypatch.setattr(playback, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    player = playback.Playback()
    player.set_frames(22, 0.02, np.array([-1] + [0] * 20 + [1]))
    player.control.click()
    assert player.slider.value() == 1
    player.control.click()
    assert player.slider.value() == 21
    player.previous.click()
    assert player.slider.value() == 20
    player.slider.setValue(0)
    player.speed.setCurrentIndex(0)
    player.play.click()
    clock[0] = 0.1
    player.tick()
    assert player.slider.value() == 0
    clock[0] = 0.41
    player.tick()
    assert player.slider.value() == 2
    player.slider.setValue(5)
    assert not player.timer.isActive()
    player.next.click()
    assert player.slider.value() == 6
    player.step(-100)
    assert player.slider.value() == 0 and not player.previous.isEnabled()
    diagnostics = Diagnostics()
    diagnostics.show()
    diagnostics.show_snapshot({"state": "ready", "error": 2.0})
    diagnostics.table.selectRow(1)
    diagnostics.show_snapshot(
        {"error": 1.0, "state": "tracking"}, {"error": ([0, 1], [2, 1])}
    )
    assert diagnostics.table.currentRow() == 0
    np.testing.assert_array_equal(diagnostics.figure.axes[0].lines[0].get_ydata(), [2, 1])
    line = diagnostics.figure.axes[0].lines[0]
    requested = []

    def history(name):
        requested.append(name)
        return ([0, 1, 2], [2, 1, 0.5]) if name == "error" else None

    diagnostics.show_snapshot({"error": 0.5, "state": "tracking"}, history)
    assert requested == ["error"]
    assert diagnostics.figure.axes[0].lines[0] is line
    np.testing.assert_array_equal(line.get_ydata(), [2, 1, 0.5])
    destination = tmp_path / "plot.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a: (str(destination), "PNG")
    )
    diagnostics.save_plot.click()
    assert destination.read_bytes().startswith(b"\x89PNG")
    destination = tmp_path / "missing" / "plot.png"
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: errors.append(a[1:]))
    diagnostics.save_plot.click()
    assert errors and errors[0][0] == "Cannot save plot"
    assert diagnostics.table.currentRow() == 0
    diagnostics.table.selectRow(1)
    assert (
        diagnostics.value.toPlainText() == "tracking" and not diagnostics.canvas.isVisible()
    )
    diagnostics.show_snapshot({})
    assert diagnostics.empty.isVisible() and not diagnostics.table.isVisible()
    diagnostics.show_snapshot({"not yet initialized": np.empty((0, 0))})
    assert (
        diagnostics.value.toPlainText() == "Empty array (0, 0)"
        and not diagnostics.canvas.isVisible()
    )
    diagnostics.close()
    diagnostics.deleteLater()
    player.close()
    player.deleteLater()
    app.processEvents()


def test_worker_completion_runs_on_gui_thread(app, student_project):
    import threading

    from coursekit.gui.blindjack import BlindjackPage

    page = BlindjackPage(student_project)
    main = threading.get_ident()
    calls = []
    page.work(
        threading.get_ident, lambda worker: calls.append((worker, threading.get_ident()))
    )
    deadline = time.monotonic() + 10
    while page.job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert len(calls) == 1
    assert calls[0][0] != main and calls[0][1] == main
    page.close()
    page.deleteLater()
    app.processEvents()


def test_blindjack_human_model_reveal_history_and_save(
    app, student_project, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QFileDialog

    from coursekit.gui.blindjack import BlindjackPage

    class Policy:
        def select(self, images):
            return np.array([True, False, False, False, False, False, False])

        def debug(self, index=0):
            return {"values": np.arange(7), "state": "selected"}

    page = BlindjackPage(student_project)
    page.show()
    page.mode.setCurrentIndex(2)
    page.solution = Policy()
    page.buttons()
    assert page.run_button.isEnabled()
    page.run_button.click()
    deadline = time.monotonic() + 10
    while page.job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert page.job is None and page.decision is not None
    assert all(label.text() == "" for label in page.answers)
    assert page.diagnostics.snapshot == {}
    assert not page.run_button.isEnabled()
    page.cards[1].setChecked(True)
    page.commit_button.click()
    assert page.game.summary()["paired_hands"] == 1
    assert all("Value:" in label.text() for label in page.answers)
    assert "values" in page.diagnostics.snapshot
    assert not page.commit_button.isEnabled()
    page.number.setValue(2)
    assert all(label.text() == "" for label in page.answers)
    page.number.setValue(1)
    assert page.cards[1].isChecked()
    np.testing.assert_array_equal(page.diagnostics.snapshot["values"], np.arange(7))
    page.number.setValue(2)
    page.cards[3].click()
    destination = tmp_path / "comparison.npz"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args: (str(destination), "JSON")
    )
    page.save()
    from coursekit import recording

    saved = recording.load(destination)
    assert saved["summary"]["paired_hands"] == 1
    assert saved["hands"][0]["human"][1]
    assert saved["resource_hash"] == page.resource_hash
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args: (str(destination), "NPZ")
    )
    page.change_mode()
    page.open_recording()
    assert page.game.summary()["paired_hands"] == 1
    assert page.number.value() == 2 and page.cards[3].isChecked()
    assert all(answer.text() == "" for answer in page.answers)
    page.number.setValue(1)
    np.testing.assert_array_equal(page.diagnostics.snapshot["values"], np.arange(7))
    page.close()
    page.deleteLater()
    app.processEvents()


@pytest.mark.skipif(
    os.environ.get("COURSE_TEST_GUI") != "1",
    reason="Set COURSE_TEST_GUI=1 for native lab windows",
)
def test_native_student_windows_play_compare_and_replay(student_project, tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    shutil.copytree(
        Path(__file__).resolve().parents[1] / "labs/line_following",
        student_project / "labs/line_following",
    )
    script = r"""
import sys,time,traceback,json
from pathlib import Path
import numpy as np
from PySide6.QtCore import QTimer,Qt,QPoint
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox
from coursekit import recording
from coursekit.gui.__main__ import Window
from coursekit.gui.blindjack import BlindjackPage
from coursekit.gui.line_following import LineFollowingPage
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import Course
format=QSurfaceFormat(); format.setVersion(3,3); format.setProfile(QSurfaceFormat.CoreProfile)
format.setDepthBufferSize(24); QSurfaceFormat.setDefaultFormat(format)
app=QApplication([]); project=Path(sys.argv[1]); output=Path(sys.argv[2]); errors=[]; windows=[]
app.setQuitOnLastWindowClosed(False)
def failed(kind,value,tb):
    errors.append(''.join(traceback.format_exception(kind,value,tb)))
    print(errors[-1],file=sys.stderr)
    for w in windows:
        try:
            w.closing=True
            for job in [w.page.job,getattr(w.page,'preparation',None)]:
                if job is not None:
                    if hasattr(job,'command'): job.command(stop=True)
                    job.wait(3000)
            w.close()
        except RuntimeError: pass
    app.quit()
sys.excepthook=failed
def settled(predicate,callback,timeout=8):
    deadline=time.monotonic()+timeout
    def poll():
        if predicate(): QTimer.singleShot(0,callback)
        elif time.monotonic()>deadline: raise AssertionError('Did not settle: '+callback.__name__)
        else: QTimer.singleShot(20,poll)
    poll()
class Policy:
    def select(self,images): return np.array([1,0,1,0,0,0,0],dtype=bool)
    def debug(self,index=0): return {'estimated values':np.arange(7),'confidence':.8}
blind=BlindjackPage(project); cards=Window(blind,'blindjack'); windows.append(cards)
cards.resize(1024,700); cards.show(); blind.mode.setCurrentIndex(2); blind.solution=Policy(); blind.buttons()
def run_model():
    QTest.mouseClick(blind.run_button,Qt.LeftButton)
    settled(lambda: blind.job is None,commit)
def commit():
    assert all(a.text()=='' for a in blind.answers) and blind.diagnostics.snapshot=={}
    QTest.mouseClick(blind.tiles[1].picture,Qt.LeftButton)
    QTest.mouseClick(blind.commit_button,Qt.LeftButton)
    assert blind.game.summary()['paired_hands']==1
    QFileDialog.getSaveFileName=lambda *a:(str(output/'session.npz'),'NPZ')
    QTest.mouseClick(blind.save_button,Qt.LeftButton)
    assert (output/'session.npz').is_file()
    settled(lambda: cards.width()<=1024 and cards.height()<=700,capture_cards)
def capture_cards():
    assert all(blind.board.viewport().rect().contains(tile.picture.mapTo(blind.board.viewport(),tile.picture.rect().bottomRight())) for tile in blind.tiles)
    cards.grab().save(str(output/'blindjack.png')); cards.close()
    global robot,window
    robot=LineFollowingPage(project); window=Window(robot,'line_following'); windows.append(window)
    robot.scene.failed.connect(errors.append); window.resize(1024,700); window.show()
    settled(lambda: robot.preparation is None and robot.scene.renderer is not None,prepare_baseline)
def prepare_baseline():
    assert not (project/'work/line_following/baseline').exists()
    assert not robot.start_button.isEnabled() and robot.solution_button.isEnabled()
    assert window.width()<=1024 and window.height()<=700
    window.grab().save(str(output/'robot-first-launch.png'))
    QTest.mouseClick(robot.solution_button,Qt.LeftButton)
    settled(lambda: robot.preparation is None,open_recording)
def open_recording():
    assert robot.courses
    assert robot.start_button.isEnabled()
    assert (project/'work/line_following/baseline/artifacts/controller.json').is_file()
    course=Course.polyline('recorded straight',np.array([[0.,0.],[1.,0.]]))
    model=physics.robot(); q=physics.initial_state(model).q.numpy(); end=q.copy(); end[0]+=.02
    record=dict(lab='line_following',version=2,physics_dt=.02,
      courses=[dict(name=course.name,controls=course.controls,knots=course.knots,corners=course.corners)],
      model={k:v.numpy()[None] for k,v in model._asdict().items()},
      frames=dict(q=np.array([q,end])[:,None,:],time=np.array([[0.],[.02]]),progress=np.array([[0.],[.02]]),score=np.zeros((2,1)),status=np.array([['running'],['running']]),control=np.array([-1,0])),
      controls=[dict(observation=dict(time=np.array([0.]),rgb=np.full((1,16,3),.5)),action=np.array([[.2,-.1]]),debug=[{'state':'tracking'}])])
    recording.save(output/'robot.npz',record)
    QFileDialog.getOpenFileName=lambda *a:(str(output/'robot.npz'),'NPZ')
    QTest.mouseClick(robot.open_button,Qt.LeftButton)
    QTest.mouseClick(robot.playback.control,Qt.LeftButton)
    assert robot.timeline.value()==1 and robot.diagnostics.snapshot['left motor']==.2
    robot.diagnostics.table.selectRow(list(robot.diagnostics.snapshot).index('left motor'))
    QTest.mousePress(robot.scene,Qt.LeftButton,pos=QPoint(100,100))
    QTest.mouseMove(robot.scene,QPoint(120,110)); QTest.mouseRelease(robot.scene,Qt.LeftButton,pos=QPoint(120,110))
    assert np.linalg.norm(robot.scene.pan)>0
    QTest.mouseClick(robot.view.fit,Qt.LeftButton); assert np.linalg.norm(robot.scene.pan)==0
    settled(lambda: window.width()<=1024 and window.height()<=700,capture_robot)
def capture_robot():
    global run_started
    assert robot.scene.isValid() and not robot.scene.grabFramebuffer().isNull()
    assert robot.scene.renderer.ctx.error=='GL_NO_ERROR'
    window.grab().save(str(output/'robot.png'))
    assert not window.windowFlags() & Qt.FramelessWindowHint
    if app.platformName()=='wayland': assert window.windowHandle().frameMargins().top()>0
    run_started=time.monotonic()
    QTest.mouseClick(robot.start_button,Qt.LeftButton)
    assert not robot.play_button.isEnabled()
    settled(lambda: robot.job.snapshot() is not None,start_motion,timeout=180)
def start_motion():
    global first_ready
    first_ready=time.monotonic()-run_started
    assert robot.job.error is None,robot.job.error
    np.testing.assert_array_equal(robot.job.snapshot()[0]['time'],[0.])
    assert robot.job.controls==[]
    robot.refresh()
    QTest.mouseClick(robot.play_button,Qt.LeftButton)
    settled(lambda: robot.job.snapshot()[0]['time'][0]>=.04,measure_motion,timeout=15)
def measure_motion():
    global motion_started,sim_started
    robot.refresh()
    robot.diagnostics.table.selectRow(list(robot.diagnostics.snapshot).index('pitch (rad)'))
    motion_started=time.monotonic()
    sim_started=float(robot.job.snapshot()[0]['time'][0])
    settled(lambda: robot.job.snapshot()[0]['time'][0]>=sim_started+5.,capture_motion,timeout=15)
def capture_motion():
    global process,restart_started,measurements
    assert robot.job.error is None,robot.job.error
    assert np.linalg.norm(robot.job.snapshot()[0]['q'][0,:2]-robot.job.frames[0]['q'][0,:2])>1e-4
    assert robot.timeline.value()>0
    measurements={'first_ready_seconds':first_ready,'live_wall_seconds':time.monotonic()-motion_started,'live_simulated_seconds':float(robot.job.snapshot()[0]['time'][0])-sim_started}
    window.grab().save(str(output/'robot-live.png'))
    process=robot.job.process
    QMessageBox.question=lambda *a:QMessageBox.Discard
    restart_started=time.monotonic()
    robot.setup_button.setChecked(True)
    robot.course.setCurrentIndex(1)
    settled(lambda: robot.job.courses[0] is robot.courses[1] and robot.job.snapshot() is not None,course_restarted,timeout=4)
def course_restarted():
    global restart_started
    measurements['course_restart_seconds']=time.monotonic()-restart_started
    assert robot.job.process is process
    np.testing.assert_array_equal(robot.job.snapshot()[0]['time'],[0.])
    assert robot.job.controls==[]
    robot.refresh()
    assert robot.courses[1].name in robot.status.text()
    window.grab().save(str(output/'robot-new-course.png'))
    previous=robot.job
    restart_started=time.monotonic()
    QTest.mouseClick(robot.start_button,Qt.LeftButton)
    settled(lambda: robot.job is not previous and robot.job.snapshot() is not None,new_run_ready,timeout=4)
def new_run_ready():
    global close_started
    measurements['new_run_seconds']=time.monotonic()-restart_started
    assert robot.job.process is process
    np.testing.assert_array_equal(robot.job.snapshot()[0]['time'],[0.])
    (output/'robot-performance.json').write_text(json.dumps(measurements,indent=2))
    close_started=time.monotonic(); window.close()
    settled(lambda: not process.is_alive(),finish,timeout=2)
def finish():
    assert time.monotonic()-close_started<2
    QTimer.singleShot(100,app.quit)
settled(lambda: cards.isVisible(),run_model)
app.exec()
sys.excepthook=sys.__excepthook__
assert not errors,errors
"""
    env = os.environ | {
        "QT_QPA_PLATFORM": os.environ.get("COURSE_TEST_GUI_PLATFORM", "wayland")
    }
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-c",
            script,
            str(student_project),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_blindjack_model_error_is_visible_without_committing(app, student_project):
    from coursekit.gui.blindjack import BlindjackPage

    class Broken:
        def select(self, images):
            raise RuntimeError("example controller error")

    page = BlindjackPage(student_project)
    page.mode.setCurrentIndex(2)
    page.solution = Broken()
    page.buttons()
    page.run_button.click()
    deadline = time.monotonic() + 10
    while page.job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert page.job is None
    assert "example controller error" in page.message.text()
    assert page.game.records == {}
    assert not page.commit_button.isEnabled()
    page.close()
    page.deleteLater()
    app.processEvents()


def test_robot_pause_steps_history_recording_and_shutdown(app, tmp_path):
    from coursekit import recording
    from coursekit.gui import line_run
    from coursekit.labs.line_following_course import Course

    (tmp_path / "config.toml").write_text("")
    (tmp_path / "solution.py").write_text("""
import torch
class Solution:
    def __init__(self, artifacts, config, seed): self.calls = 0
    def act(self, observation, reset):
        self.calls += 1
        return torch.zeros((len(reset), 2))
    def debug(self, index=0): return {"calls": self.calls, "row": index}
""")
    course = Course.polyline("straight", np.array([[0.0, 0.0], [1.0, 0.0]]))
    worker = line_run.Worker()
    run = line_run.Run(
        tmp_path, [course, course], [1, 2], worker=worker, compiled=False, time_limit=0.05
    )
    run.start()

    def wait_for(predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert predicate()

    try:
        wait_for(lambda: run.snapshot() is not None)
        assert run.controls == []
        run.command(steps=1)
        wait_for(lambda: run.snapshot()[2] == 2)
        assert len(run.controls) == 1
        run.command(steps=19)
        wait_for(lambda: run.snapshot()[2] == 21)
        assert run.snapshot()[0]["time"].tolist() == [0.02, 0.02]
        for _ in range(4):
            assert run.snapshot(0)[0]["control"] == -1
            assert run.snapshot(1)[1]["debug"][1]["row"] == 1
        assert len(run.controls) == 1
        run.command(paused=False, speed=20)
        wait_for(lambda: not run.isRunning())
        assert run.error is None
        assert run.snapshot()[0]["status"].tolist() == ["timeout", "timeout"]
        assert len(run.controls) == 3
        assert run.controls[0]["reset"].all() and not run.controls[1]["reset"].any()
        path = tmp_path / "recording.npz"
        recording.save(path, run.record())
        saved = recording.load(path)
        assert saved["frames"]["q"].shape == (51, 2, 8)
        np.testing.assert_array_equal(
            saved["controls"][1]["observation"]["time"],
            run.controls[1]["observation"]["time"],
        )
        assert saved["controls"][1]["debug"][1]["row"] == 1
        assert len(run.controls) == 3
        process = worker.process
        run = line_run.Run(tmp_path, [course], [31], worker=worker, compiled=False)
        run.start()
        wait_for(lambda: run.snapshot() is not None)
        assert run.process is process and run.snapshot()[2] == 1
        run.command(steps=1)
        wait_for(lambda: run.snapshot()[2] == 2)
        assert run.controls[0]["debug"][0]["calls"] == 1
        assert run.controls[0]["reset"].tolist() == [True]
        np.testing.assert_array_equal(run.controls[0]["observation"]["time"], [0.0])
        run.command(stop=True)
        assert run.wait(2000) and worker.process is process and process.is_alive()
    finally:
        run.command(stop=True)
        assert run.wait(10000)
        worker.close()


def test_robot_window_live_rewind_saved_playback_and_close(app, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

    from coursekit import recording
    from coursekit.gui import line_following, line_run
    from coursekit.gui.__main__ import Window

    # Native Qt/OpenGL rendering is exercised separately; this checks the UI lifecycle.
    class Scene(QWidget):
        failed = Signal(str)

        def show_pose(self, *args):
            self.pose = args

        def cleanup(self):
            self.cleaned = True

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "labs/line_following", tmp_path / "labs/line_following")
    work = tmp_path / "work/line_following/baseline"
    monkeypatch.setattr(line_following, "Scene", Scene)

    monkeypatch.setattr(
        line_following,
        "Run",
        lambda *args, **kwargs: line_run.Run(
            *args, **kwargs, compiled=False, time_limit=0.003
        ),
    )
    page = line_following.LineFollowingPage(tmp_path)
    window = Window(page, "line_following")
    window.show()

    def wait_for(predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert predicate()

    wait_for(lambda: page.preparation is None)
    assert not work.exists() and page.scene.pose is not None
    assert not page.start_button.isEnabled() and page.solution_button.isEnabled()
    page.start()
    assert page.job is None and not work.exists()
    page.solution_button.click()
    wait_for(lambda: page.preparation is None)
    assert (work / "solution.py").is_file()
    assert (work / "artifacts/controller.json").is_file()
    assert page.start_button.isEnabled()
    edited = (work / "solution.py").read_text() + "\n# Student customization\n"
    (work / "solution.py").write_text(edited)
    page.solution_button.click()
    wait_for(lambda: page.preparation is None)
    assert (work / "solution.py").read_text() == edited
    page.chooser.path.setText(str(tmp_path / "does-not-exist"))
    assert not page.start_button.isEnabled() and not page.solution_button.isEnabled()
    page.chooser.path.setText(str(work))
    page.batch.setValue(2)
    page.start()
    job = page.job
    try:
        wait_for(lambda: job.snapshot() is not None)
        page.step(False)
        wait_for(lambda: job.snapshot()[2] == 2)
        page.refresh()
        assert page.timeline.value() == 1
        assert "command" in page.diagnostics.snapshot
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Cancel
        )
        window.close()
        assert window.isVisible() and job.isRunning() and not job.stopping
        page.course.setCurrentIndex(1)
        assert page.course.currentIndex() == 0
        assert page.job is job and not job.stopping
        page.timeline.setValue(0)
        assert page.diagnostics.snapshot == {}
        assert "No controller call" in page.sensors.text()
        assert len(job.controls) == 1
        page.resume()
        wait_for(lambda: not job.isRunning())
        app.processEvents()
        page.refresh()
        assert page.timeline.value() == 3
        assert len(job.controls) == 1
        page.episode.setValue(2)
        target = tmp_path / "run.npz"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *args: (str(target), "NPZ")
        )
        page.save()
        saved = recording.load(target)
        assert saved["version"] == 2
        assert "controls" in saved["courses"][0]
        assert saved["frames"]["q"].shape == (4, 2, 8)
        process = page.worker.process
        page.chooser.path.setText(str(tmp_path / "does-not-exist"))
        page.course.setCurrentIndex(1)
        assert page.job is job and page.course.currentIndex() == 0
        assert "Missing files" in page.message.text()
        assert page.worker.process is process
        page.chooser.path.setText(str(work))
        # A new course replaces the completed run and its visible scene.
        page.course.setCurrentIndex(1)
        wait_for(lambda: page.job.snapshot() is not None)
        page.refresh()
        assert page.worker.process is process and process.is_alive()
        assert page.job.courses[0] is page.courses[1]
        assert page.scene.pose[0] is page.courses[1]
        assert page.job.controls == []
        np.testing.assert_array_equal(page.job.snapshot()[0]["time"], [0.0, 0.0])
        # Changing a course while paused, and New run while live, reuse the worker.
        previous_job = page.job
        page.course.setCurrentIndex(2)
        wait_for(lambda: page.job is not previous_job and page.job.snapshot() is not None)
        assert page.worker.process is process
        page.refresh()
        assert page.scene.pose[0] is page.courses[2]
        page.start_button.click()
        previous_job = page.job
        wait_for(lambda: page.job is not previous_job and page.job.snapshot() is not None)
        assert page.worker.process is process
        page.stop()
        wait_for(lambda: not page.job.isRunning())
        # Switching to a shorter, smaller recorded batch must clamp both controls first.
        saved["courses"] = saved["courses"][:1]
        saved["seeds"] = saved["seeds"][:1]
        saved["controls"] = []
        saved["model"] = {k: v[:1] for k, v in saved["model"].items()}
        saved["frames"] = {k: v[:1] for k, v in saved["frames"].items()}
        for key in saved["frames"]:
            if key != "control":
                saved["frames"][key] = saved["frames"][key][:, :1]
        recording.save(target, saved)
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName", lambda *args: (str(target), "NPZ")
        )
        page.open_recording()
        assert page.job is None and page.timeline.maximum() == 0
        assert page.episode.maximum() == 1
        assert page.replay["frames"]["q"].shape == (1, 1, 8)
        saved["version"] = 1
        recording.save(target, saved)
        page.open_recording()
        assert "old polyline recordings are incompatible" in page.message.text()
        previous = page.replay
        saved["version"] = 2
        saved["frames"]["control"][0] = 999
        recording.save(target, saved)
        page.open_recording()
        assert page.replay is previous
        assert "Invalid recorded controller indices" in page.message.text()
        saved["frames"]["control"][0] = -1
        saved["frames"]["extra"] = []
        recording.save(target, saved)
        page.open_recording()
        assert (
            page.replay is previous
            and "Invalid recorded frame field" in page.message.text()
        )
    finally:
        if page.job is not None:
            page.job.command(stop=True)
            page.job.wait(10000)
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Discard
        )
        window.close()
        app.processEvents()


@pytest.mark.parametrize("stage", ["load", "act", "physics"])
def test_robot_stop_interrupts_blocked_calls_and_preserves_history(app, tmp_path, stage):
    from coursekit import recording
    from coursekit.gui.line_run import Run
    from coursekit.labs.line_following_course import Course

    # Deliberately non-cooperative student code, including a blocked engine warmup.
    # The stop operation must not depend on a callback inside any of these calls.
    (tmp_path / "config.toml").write_text(f'stage = "{stage}"\n')
    (tmp_path / "solution.py").write_text("""
import os, subprocess, sys, time
from pathlib import Path
class Solution:
    def __init__(self, artifacts, config, seed):
        self.stage = config["stage"]
        self.marker = Path(artifacts).parent / "entered"
        if self.stage == "load": self.block()
        if self.stage == "physics":
            from coursekit.labs.line_following_env import Environment
            # This test injects a blocked warmup; constructor sensing must not compile.
            Environment._sense = lambda *a, **kw: None
            Environment.step = lambda *a, **kw: self.block()
    def block(self):
        if os.name == "posix":
            code = 'import sys,time; from pathlib import Path; p=Path(sys.argv[1]); i=0\\nwhile True: p.write_text(str(i)); i+=1; time.sleep(.01)'
            subprocess.Popen([sys.executable, '-c', code, str(self.marker.parent / 'heartbeat')])
        self.marker.write_text("entered")
        time.sleep(3600)
    def act(self, observation, reset): self.block()
""")
    course = Course.polyline("straight", np.array([[0.0, 0.0], [1.0, 0.0]]))
    run = Run(tmp_path, [course], [0], compiled=stage == "physics")
    run.start()
    try:
        if stage == "act":
            wait_for(app, lambda: run.snapshot() is not None)
            run.command(paused=False)
        wait_for(app, lambda: (tmp_path / "entered").exists())
        if os.name == "posix":
            wait_for(app, lambda: (tmp_path / "heartbeat").exists())
        process = run.process
        assert process.is_alive()
        assert (
            run.phase
            == {
                "load": "Loading solution",
                "act": "Calling controller",
                "physics": "Warming physics",
            }[stage]
        )
        if stage == "physics":
            assert run.snapshot() is None  # Never claim readiness before warmup finishes.
        started = time.monotonic()
        run.command(stop=True)
        assert time.monotonic() - started < 0.2  # The GUI thread does not join a worker.
        assert run.wait(2000)
        assert not process.is_alive() and run.error is None
        if os.name == "posix":
            beat = (tmp_path / "heartbeat").read_text()
            time.sleep(0.1)
            assert (tmp_path / "heartbeat").read_text() == beat
        if stage == "act":
            recording.save(tmp_path / "stopped.npz", run.record())
            assert len(recording.load(tmp_path / "stopped.npz")["frames"]["q"]) == 1
    finally:
        run.command(stop=True)
        assert run.wait(3000)


def test_robot_window_closes_during_blocked_controller(app, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget

    from coursekit import training
    from coursekit.gui import line_following, line_run
    from coursekit.gui.__main__ import Window

    class Scene(QWidget):
        failed = Signal(str)

        def show_pose(self, *args):
            pass

        def cleanup(self):
            pass

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "labs/line_following", tmp_path / "labs/line_following")
    path = training.new_solution(tmp_path, "line_following", "baseline")
    (path / "artifacts").mkdir()
    (path / "artifacts/provenance.json").write_text("{}")
    (path / "solution.py").write_text("""
import time
class Solution:
    def __init__(self, *args): pass
    def act(self, observation, reset): time.sleep(3600)
""")
    monkeypatch.setattr(line_following, "Scene", Scene)
    monkeypatch.setattr(
        line_following,
        "Run",
        lambda *args, **kwargs: line_run.Run(*args, **kwargs, compiled=False),
    )
    page = line_following.LineFollowingPage(tmp_path)
    window = Window(page, "line_following")
    closed = []
    window.destroyed.connect(lambda: closed.append(True))
    window.show()
    try:
        wait_for(app, lambda: page.preparation is None)
        page.start()
        wait_for(app, lambda: page.job.snapshot() is not None)
        page.resume()
        wait_for(app, lambda: page.job.phase == "Calling controller")
        process = page.job.process
        started = time.monotonic()
        window.close()
        wait_for(app, lambda: bool(closed))
        assert time.monotonic() - started < 2
        assert not process.is_alive()
    finally:
        if not closed:
            page.stop()
            if page.job is not None:
                page.job.wait(3000)
            window.closing = True
            window.close()
            app.processEvents()
