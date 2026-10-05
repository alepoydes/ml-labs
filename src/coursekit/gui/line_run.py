"""Local computation thread. Playback only reads records; it never calls the policy."""

import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread

from coursekit.inspection import debug_snapshot
from coursekit.labs.line_following_env import (
    CONTROL_STEPS,
    PHYSICS_DT,
    Environment,
    validate_action,
)
from coursekit.training import load_solution


class Stopped(Exception):
    pass


class Run(QThread):
    def __init__(self, path, courses, seeds, parent=None, *, factory=Environment):
        super().__init__(parent)
        self.path, self.courses, self.seeds = path, courses, seeds
        self.factory = factory
        self.condition = threading.Condition()
        self.paused, self.stopping, self.budget = True, False, 0
        self.speed, self.deadline = 1.0, time.monotonic()
        self.frames, self.controls = [], []
        self.model = None
        self.error = None

    def command(self, *, paused=True, steps=0, speed=None, stop=False):
        with self.condition:
            self.paused, self.budget = paused, steps
            self.stopping |= stop
            if speed is not None:
                self.speed = speed
            self.deadline = time.monotonic()
            self.condition.notify_all()

    def gate(self):
        with self.condition:
            while True:
                if self.stopping:
                    raise Stopped
                if self.paused:
                    if self.budget:
                        self.budget -= 1
                        return
                    self.condition.wait()
                else:
                    delay = self.deadline - time.monotonic()
                    if delay <= 0:
                        self.deadline = (
                            max(self.deadline, time.monotonic()) + PHYSICS_DT / self.speed
                        )
                        return
                    self.condition.wait(delay)

    def append(self, env):
        info = env.info()
        frame = {
            key: info[key] for key in ("q", "time", "progress", "leg", "status", "score")
        }
        frame["control"] = len(self.controls) - 1
        with self.condition:
            self.frames.append(frame)

    def run(self):
        try:
            solution = load_solution(self.path)
            env = self.factory(self.courses, self.seeds)
            with self.condition:
                self.model = {
                    key: value.numpy().copy() for key, value in env.model._asdict().items()
                }
            self.append(env)
            self.gate()

            def after_step(env):
                self.append(env)
                if not env.done.all():
                    self.gate()

            while not env.done.all():
                observation = env.observe()
                reset = env.first
                saved_observation = {k: v.numpy().copy() for k, v in observation.items()}
                saved_reset = reset.numpy().copy()
                started = time.perf_counter()
                action = solution.act(observation, reset)
                elapsed = time.perf_counter() - started
                validate_action(action, env.size)
                control = {
                    "observation": saved_observation,
                    "reset": saved_reset,
                    "action": action.detach().cpu().numpy().copy(),
                    "seconds": elapsed,
                    "debug": [debug_snapshot(solution, i) for i in range(env.size)],
                }
                with self.condition:
                    self.controls.append(control)
                env.step(action, on_step=after_step)
        except Stopped:
            pass
        except BaseException:
            self.error = traceback.format_exc()
            print(self.error, end="")

    def snapshot(self, index=-1):
        with self.condition:
            if not self.frames:
                return None
            frame = self.frames[index]
            control = self.controls[frame["control"]] if frame["control"] >= 0 else None
            return frame, control, len(self.frames)

    def record(self):
        with self.condition:
            return {
                "lab": "line_following",
                "version": 1,
                "solution": str(self.path),
                "seeds": np.array(self.seeds),
                "physics_dt": PHYSICS_DT,
                "control_steps": CONTROL_STEPS,
                "courses": [
                    {"name": c.name, "points": c.points, "corners": c.corners}
                    for c in self.courses
                ],
                "model": self.model,
                "frames": {k: np.array([f[k] for f in self.frames]) for k in self.frames[0]}
                if self.frames
                else {},
                "controls": list(self.controls),
                "error": self.error,
            }
