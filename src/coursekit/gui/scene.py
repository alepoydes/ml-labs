"""Small ModernGL scene built from the same course points and robot geometry."""

import traceback

import moderngl
import numpy as np
import torch
from PySide6.QtCore import Signal, Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from coursekit.labs import line_following as physics
from coursekit.labs.line_following_course import (
    END,
    INK,
    LINE_WIDTH,
    MARKER_LENGTH,
    START,
    sensor_areas,
)


def triangles(points, faces, colour):
    vertices = np.asarray(points)[np.asarray(faces)].reshape(-1, 3)
    return np.c_[vertices, np.broadcast_to(colour, vertices.shape)].astype("f4")


def box(half, colour):
    points = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    faces = [
        [0, 1, 3],
        [0, 3, 2],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [2, 3, 7],
        [2, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [1, 5, 7],
        [1, 7, 3],
    ]
    mesh = triangles(points, faces, colour)
    mesh[:, 3:] *= np.repeat(
        [0.72, 0.72, 0.82, 0.82, 0.65, 0.65, 0.9, 0.9, 0.6, 0.6, 1, 1], 3
    )[:, None]
    return mesh


def round_body(radius, *, sphere=False):
    a = np.linspace(0, 2 * np.pi, 25)
    b = np.linspace(-np.pi / 2, np.pi / 2, 13) if sphere else np.array([-0.008, 0.008])
    if sphere:
        points = np.stack(
            np.broadcast_arrays(
                radius * np.cos(b[:, None]) * np.cos(a),
                radius * np.sin(b[:, None]),
                radius * np.cos(b[:, None]) * np.sin(a),
            ),
            axis=-1,
        )
    else:
        points = np.stack(
            np.broadcast_arrays(radius * np.cos(a), b[:, None], radius * np.sin(a)), axis=-1
        )
    rows = []
    for i in range(len(b) - 1):
        for j in range(len(a) - 1):
            quad = points[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j]]
            rows.append(
                triangles(quad, [[0, 1, 2], [0, 2, 3]], [0.18 + 0.12 * (j % 2)] * 3)
            )
    if not sphere:
        for i in range(2):
            points_cap = np.vstack(([0, b[i], 0], points[i]))
            rows.append(
                triangles(
                    points_cap,
                    [[0, j + 1, j + 2] for j in range(len(a) - 1)],
                    [0.28, 0.3, 0.34],
                )
            )
    return np.concatenate(rows)


def painted_course(course):
    # Segment rectangles with round joins reproduce the finite-width painted line.
    rows = []
    angle = np.linspace(0, 2 * np.pi, 17)
    circle = np.c_[np.cos(angle), np.sin(angle)] * LINE_WIDTH / 2
    points, arc = course.mesh()
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        cuts = sorted(
            {
                arc[i],
                arc[i + 1],
                *[
                    s
                    for s in (MARKER_LENGTH, course.length - MARKER_LENGTH)
                    if arc[i] < s < arc[i + 1]
                ],
            }
        )
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            start, end = course.at(np.array([s0, s1]))
            normal = (
                np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a) * LINE_WIDTH / 2
            )
            quad = np.array([start + normal, start - normal, end - normal, end + normal])
            colour = (
                START
                if s1 <= MARKER_LENGTH
                else END
                if s0 >= course.length - MARKER_LENGTH
                else INK
            )
            rows.append(triangles(np.c_[quad, np.zeros(4)], [[0, 1, 2], [0, 2, 3]], colour))
    for point, s in zip(points, arc):
        colour = (
            START
            if s < MARKER_LENGTH
            else END
            if s > course.length - MARKER_LENGTH
            else INK
        )
        points = np.vstack((point, point + circle))
        rows.append(
            triangles(
                np.c_[points, np.zeros(len(points))],
                [[0, i + 1, i + 2] for i in range(16)],
                colour,
            )
        )
    return np.concatenate(rows)


class Renderer:
    def __init__(self, context):
        self.ctx = context
        self.program = context.program(
            vertex_shader="""
            #version 330
            uniform mat4 camera;
            in vec3 position;
            in vec3 colour;
            out vec3 tint;
            void main() { gl_Position = camera * vec4(position, 1.0); tint = colour; }
        """,
            fragment_shader="""
            #version 330
            in vec3 tint;
            uniform float opacity;
            out vec4 fragment;
            void main() { fragment = vec4(tint, opacity); }
        """,
        )
        self.buffer = context.buffer(reserve=1024, dynamic=True)
        self.vao = context.vertex_array(
            self.program, [(self.buffer, "3f 3f", "position", "colour")]
        )
        self.course, self.ground = None, None

    def draw(
        self,
        framebuffer,
        width,
        height,
        course,
        q,
        model,
        *,
        top=False,
        follow=False,
        sensing=True,
        zoom=1.0,
        pan=(0.0, 0.0),
    ):
        if self.course is not course:
            self.course, self.ground = course, painted_course(course)
        rotation = physics.rotation(torch.from_numpy(q)).numpy()
        pieces = []
        for mesh, offset in [
            (
                box(np.array([0.070, 0.064, 0.005]), [0.16, 0.5, 0.67]),
                np.array([0, 0, 0.008]),
            ),
            (box(np.array([0.03, 0.035, 0.065]), [0.94, 0.58, 0.17]), model["offset"][0]),
            *[
                (round_body(r, sphere=i == 2), model["offset"][i + 1])
                for i, r in enumerate(model["radius"])
            ],
        ]:
            mesh[:, :3] = (mesh[:, :3] + offset) @ rotation.T + q[:3]
            pieces.append(mesh)
        # A bright nose distinguishes heading from the permitted direction of travel.
        nose = triangles(
            [[0.07, -0.016, 0.018], [0.09, 0, 0.018], [0.07, 0.016, 0.018]],
            [[0, 1, 2]],
            [0.9, 0.1, 0.08],
        )
        nose[:, :3] = nose[:, :3] @ rotation.T + q[:3]
        pieces.append(nose)
        vertices = np.concatenate([self.ground, *pieces])
        centre = (
            q[:2] if follow else (course.bounds[0] + course.bounds[1]) / 2
        ) + np.asarray(pan)
        span = (
            0.6 if follow else max((course.bounds[1] - course.bounds[0]).max() + 0.4, 0.8)
        )
        scale = span / zoom
        view = np.eye(4)
        if not top:
            angle = np.pi / 5
            view[:3, :3] = [
                [1, 0, 0],
                [0, np.cos(angle), np.sin(angle)],
                [0, -np.sin(angle), np.cos(angle)],
            ]
        view[:3, 3] = -view[:3, :3] @ np.r_[centre, 0]
        aspect = width / max(1, height)
        projection = np.diag(
            [
                2 / (scale * max(1, aspect)),
                2 / (scale * max(1, 1 / aspect)),
                -1 / max(1, span),
                1,
            ]
        )
        self.program["camera"].write(
            np.asarray((projection @ view).T, dtype="f4").tobytes()
        )
        self.buffer.orphan(vertices.nbytes)
        self.buffer.write(vertices.tobytes())
        framebuffer.use()
        self.ctx.viewport = (0, 0, width, height)
        framebuffer.clear(0.94, 0.94, 0.94, 1, depth=1)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program["opacity"].value = 1.0
        self.vao.render(moderngl.TRIANGLES, vertices=len(vertices))
        if sensing:
            # Inspection overlay: show the actual ground footprint through the chassis.
            overlay = np.concatenate(
                [
                    triangles(
                        np.c_[area, np.zeros(4)],
                        [[0, 1, 2], [0, 2, 3]],
                        [0.05, 0.85, 0.95] if i % 2 else [0.95, 0.8, 0.1],
                    )
                    for i, area in enumerate(sensor_areas(q[:3], rotation))
                ]
            )
            self.buffer.orphan(overlay.nbytes)
            self.buffer.write(overlay.tobytes())
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.program["opacity"].value = 0.7
            self.vao.render(moderngl.TRIANGLES, vertices=len(overlay))
            self.ctx.disable(moderngl.BLEND)

    def release(self):
        self.vao.release()
        self.buffer.release()
        self.program.release()


class QtFunctions:
    """ModernGL's public loader interface, using the OpenGL context owned by Qt."""

    def __init__(self, context):
        self.context = context

    def load_opengl_function(self, name):
        address = self.context.getProcAddress(name.encode())
        return int(address) if address is not None else 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Scene(QOpenGLWidget):
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 200)
        self.renderer = self.framebuffer = None
        self.pose = None
        self.top, self.follow, self.zoom = False, False, 1.0
        self.sensing = True
        self.pan, self.drag = np.zeros(2), None

    def initializeGL(self):
        try:
            moderngl.init_context(QtFunctions(self.context()))
            context = moderngl.get_context()
            if context.version_code < 330:
                raise RuntimeError("The robot scene requires OpenGL 3.3 or newer.")
            self.renderer = Renderer(context)
            self.context().aboutToBeDestroyed.connect(self.cleanup)
        except Exception:
            self.failure()

    def failure(self):
        message = traceback.format_exc()
        print(message, end="")
        self.pose = None
        self.failed.emit(message)

    def paintGL(self):
        if self.renderer is None:
            return
        try:
            if (
                self.framebuffer is None
                or self.framebuffer.glo != self.defaultFramebufferObject()
            ):
                self.framebuffer = self.renderer.ctx.detect_framebuffer(
                    self.defaultFramebufferObject()
                )
            if self.pose is None:
                self.framebuffer.clear(0.94, 0.94, 0.94, 1)
                return
            ratio = self.devicePixelRatioF()
            self.renderer.draw(
                self.framebuffer,
                int(self.width() * ratio),
                int(self.height() * ratio),
                *self.pose,
                top=self.top,
                follow=self.follow,
                sensing=self.sensing,
                zoom=self.zoom,
                pan=self.pan,
            )
        except Exception:
            self.failure()

    def show_pose(self, course, q, model):
        self.pose = course, q, model
        self.update()

    def wheelEvent(self, event):
        self.zoom = np.clip(self.zoom * np.exp(event.angleDelta().y() / 1000), 0.2, 20)
        self.update()

    def mousePressEvent(self, event):
        from PySide6.QtCore import Qt

        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = event.position()

    def mouseMoveEvent(self, event):
        if self.drag is None or self.pose is None:
            return
        delta = event.position() - self.drag
        self.drag = event.position()
        course = self.pose[0]
        span = (
            0.6
            if self.follow
            else max((course.bounds[1] - course.bounds[0]).max() + 0.4, 0.8)
        )
        per_pixel = span / self.zoom / min(self.width(), self.height())
        self.pan += (
            np.array([-delta.x(), delta.y() / (1 if self.top else np.cos(np.pi / 5))])
            * per_pixel
        )
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None

    @Slot()
    def cleanup(self):
        if self.renderer is not None:
            self.makeCurrent()
            self.renderer.release()
            self.renderer = self.framebuffer = None
            self.doneCurrent()
