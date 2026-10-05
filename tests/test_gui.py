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
    np.testing.assert_array_equal(page.diagnostics.snapshot["values"], np.arange(7))
    page.close()
    page.deleteLater()
    app.processEvents()


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
    from PySide6.QtWidgets import QFileDialog, QWidget

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
    training.train_local(work, tmp_path)
    window = Window(page, "line_following")
    window.show()
    page.batch.setValue(2)
    page.start()

    def wait_for(predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert predicate()

    job = page.job
    try:
        wait_for(lambda: job.snapshot() is not None)
        page.step(False)
        wait_for(lambda: job.snapshot()[2] == 2)
        page.refresh()
        assert page.timeline.value() == 1
        assert "command" in page.diagnostics.snapshot
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
    finally:
        job.command(stop=True)
        job.wait(10000)
        window.close()
        app.processEvents()
