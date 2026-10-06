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
import sys,time,traceback
from pathlib import Path
import numpy as np
from PySide6.QtCore import QTimer,Qt,QPoint
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QFileDialog
from coursekit import recording
from coursekit.gui.__main__ import Window
from coursekit.gui.blindjack import BlindjackPage
from coursekit.gui.line_following import LineFollowingPage
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import Course
format=QSurfaceFormat(); format.setVersion(3,3); format.setProfile(QSurfaceFormat.CoreProfile)
format.setDepthBufferSize(24); QSurfaceFormat.setDefaultFormat(format)
app=QApplication([]); project=Path(sys.argv[1]); output=Path(sys.argv[2]); errors=[]; windows=[]
def failed(kind,value,tb):
    errors.append(''.join(traceback.format_exception(kind,value,tb)))
    print(errors[-1],file=sys.stderr)
    for w in windows:
        try:
            w.closing=True
            for job in [w.page.job,getattr(w.page,'preparation',None)]:
                if job is not None: job.wait(10000)
            w.close()
        except RuntimeError: pass
    app.quit()
sys.excepthook=failed
def settled(predicate,callback):
    deadline=time.monotonic()+8
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
    settled(lambda: robot.preparation is None and robot.scene.renderer is not None,open_recording)
def open_recording():
    assert robot.courses
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
    assert robot.scene.isValid() and not robot.scene.grabFramebuffer().isNull()
    assert robot.scene.renderer.ctx.error=='GL_NO_ERROR'
    window.grab().save(str(output/'robot.png'))
    window.close(); QTimer.singleShot(20,app.quit)
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
        timeout=40,
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


def test_robot_pause_steps_history_recording_and_shutdown(app, tmp_path, monkeypatch):
    import torch

    from coursekit import recording
    from coursekit.gui import line_run
    from coursekit.labs import line_following as physics
    from coursekit.labs.line_following_course import Course
    from coursekit.labs.line_following_env import Environment

    calls = []

    class Policy:
        def act(self, obs, reset):
            calls.append((obs["time"].clone(), reset.clone()))
            return torch.zeros((len(reset), 2))

        def debug(self, index=0):
            return {"calls": len(calls), "row": index}

    def factory(courses, seeds):
        env = Environment(courses, seeds, compiled=False, time_limit=0.05)
        env._advance = lambda state, action, model: (
            state,
            physics.Diagnostics(torch.zeros(env.size), torch.zeros((env.size, 3, 3))),
        )
        return env

    monkeypatch.setattr(line_run, "load_solution", lambda path: Policy())
    course = Course.polyline("straight", np.array([[0.0, 0.0], [1.0, 0.0]]))
    run = line_run.Run(tmp_path, [course, course], [1, 2], factory=factory)
    run.start()

    def wait_for(predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert predicate()

    try:
        wait_for(lambda: run.snapshot() is not None)
        assert calls == []
        run.command(steps=1)
        wait_for(lambda: run.snapshot()[2] == 2)
        assert len(calls) == 1
        run.command(steps=19)
        wait_for(lambda: run.snapshot()[2] == 21)
        assert run.snapshot()[0]["time"].tolist() == [0.02, 0.02]
        for _ in range(4):
            assert run.snapshot(0)[0]["control"] == -1
            assert run.snapshot(1)[1]["debug"][1]["row"] == 1
        assert len(calls) == 1
        run.command(paused=False, speed=20)
        wait_for(lambda: not run.isRunning())
        assert run.error is None
        assert run.snapshot()[0]["status"].tolist() == ["timeout", "timeout"]
        assert len(calls) == 3
        assert calls[0][1].all() and not calls[1][1].any()
        path = tmp_path / "recording.npz"
        recording.save(path, run.record())
        saved = recording.load(path)
        assert saved["frames"]["q"].shape == (51, 2, 8)
        np.testing.assert_array_equal(
            saved["controls"][1]["observation"]["time"], calls[1][0].numpy()
        )
        assert saved["controls"][1]["debug"][1]["row"] == 1
        assert len(calls) == 3
    finally:
        run.command(stop=True)
        assert run.wait(10000)


def test_robot_window_live_rewind_saved_playback_and_close(app, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

    from coursekit import recording, training
    from coursekit.gui import line_following, line_run
    from coursekit.gui.__main__ import Window
    from coursekit.labs.line_following_env import Environment

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
    shutil.copytree(root / "labs/line_following/baseline", work)
    monkeypatch.setattr(line_following, "Scene", Scene)

    def factory(courses, seeds):
        return Environment(courses, seeds, compiled=False, time_limit=0.003)

    monkeypatch.setattr(
        line_following, "Run", lambda *args: line_run.Run(*args, factory=factory)
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
    training.train_local(work, tmp_path)
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
        job.command(stop=True)
        job.wait(10000)
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Discard
        )
        window.close()
        app.processEvents()
