import os
import subprocess
import sys

import numpy as np
import pytest


def test_headless_scene_renders_course_markers_and_robot():
    moderngl = pytest.importorskip("moderngl")
    pytest.importorskip("PySide6.QtOpenGLWidgets")
    from coursekit.gui.scene import Renderer
    from coursekit.labs import line_following as physics
    from coursekit.labs.line_following_course import Course

    try:
        context = moderngl.create_context(standalone=True, backend="egl")
    except Exception as exc:
        pytest.skip(f"Headless EGL unavailable: {exc}")
    renderer = Renderer(context)
    target = context.simple_framebuffer((600, 300))
    model = physics.robot()
    q = physics.initial_state(model).q.numpy()
    course = Course.polyline("straight", np.array([[0.0, 0.0], [1.0, 0.0]]))
    renderer.draw(
        target,
        600,
        300,
        course,
        q,
        {k: v.numpy() for k, v in model._asdict().items()},
        top=True,
    )
    pixels = np.frombuffer(target.read(), dtype=np.uint8).reshape(300, 600, 3).astype(int)
    assert (pixels.max(axis=-1) - pixels.min(axis=-1) > 80).sum() > 300
    assert ((pixels[..., 0] - pixels[..., 2]) > 100).sum() > 20
    assert ((pixels[..., 2] - pixels[..., 0]) > 100).sum() > 20
    assert (pixels.max(axis=-1) < 40).sum() > 200
    renderer.release()
    target.release()
    context.release()


@pytest.mark.skipif(
    os.environ.get("COURSE_TEST_GUI") != "1",
    reason="Set COURSE_TEST_GUI=1 on a desktop for actual Qt/OpenGL integration",
)
@pytest.mark.parametrize("broken", [False, True])
def test_qt_context_rendering_errors_and_shutdown(broken):
    pytest.importorskip("PySide6.QtOpenGLWidgets")
    pytest.importorskip("moderngl")
    script = r"""
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication
from coursekit.gui import scene as graphics
from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import Course
format = QSurfaceFormat()
format.setVersion(3, 3)
format.setProfile(QSurfaceFormat.CoreProfile)
format.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(format)
app = QApplication([])
broken = BROKEN
if broken:
    def fail(*args): raise RuntimeError('expected graphics initialization failure')
    graphics.moderngl.init_context = fail
scene = graphics.Scene()
failures, results = [], []
scene.failed.connect(failures.append)
model = physics.robot()
scene.show_pose(Course.polyline('straight', np.array([[0., 0.], [1., 0.]])), physics.initial_state(model).q.numpy(), {k: v.numpy() for k, v in model._asdict().items()})
scene.resize(700, 400)
scene.show()
def finish():
    if scene.isValid():
        frame = scene.grabFramebuffer()
        results.append(not frame.isNull())
        if scene.renderer is not None:
            results.append(scene.renderer.ctx.error == 'GL_NO_ERROR')
    scene.cleanup()
    scene.close()
    scene.deleteLater()
    QTimer.singleShot(20, app.quit)
QTimer.singleShot(200, lambda: scene.resize(800, 450))
QTimer.singleShot(500, finish)
app.exec()
if broken:
    assert any('expected graphics initialization failure' in error for error in failures)
else:
    assert not failures, failures
    assert results == [True, True], results
""".replace("BROKEN", repr(broken))
    env = os.environ.copy()
    env.pop("QT_QPA_PLATFORM", None)
    if "COURSE_TEST_GUI_PLATFORM" in env:
        env["QT_QPA_PLATFORM"] = env["COURSE_TEST_GUI_PLATFORM"]
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
