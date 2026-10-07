"""A reusable, cancellable physics process; Qt retains each run's history."""

import multiprocessing
import os
import queue
import signal
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread

from coursekit.labs.line_following_env import CONTROL_STEPS, PHYSICS_DT


class _StopRun(Exception):
    pass


def _simulate(connection, path, courses, seeds, options):
    import torch

    from coursekit.inspection import debug_snapshot
    from coursekit.labs.line_following_env import Environment, validate_action
    from coursekit.training import load_solution

    current_phase = None
    pending = []

    def flush():
        if pending:
            # One array per field avoids serializing dozens of tiny arrays per ms.
            connection.send(
                ("frames", {k: np.array([f[k] for f in pending]) for k in pending[0]})
            )
            pending.clear()

    def phase(value):
        nonlocal current_phase
        if value != current_phase:
            connection.send(("phase", value))
            current_phase = value

    try:
        if path is not None:
            phase("Loading solution")
            solution = load_solution(path)
        phase("Preparing physics")
        env = Environment(courses, seeds, **options)
        if options.get("compiled", True):
            phase("Warming physics")
            # The process retains compiled kernels; only a new specialization compiles.
            # Restore physical state and sensor RNG before the first controller call.
            env.step(torch.zeros((env.size, 2)))
            env.reset(seeds=seeds)
        if path is None:
            return  # Startup preparation warms this process without student code.
        connection.send(
            ("model", {k: v.numpy().copy() for k, v in env.model._asdict().items()})
        )
        control_index = -1
        paused, budget, speed, deadline = True, 0, 1.0, time.monotonic()

        def append():
            info = env.info()
            frame = {
                k: info[k] for k in ("q", "time", "progress", "leg", "status", "score")
            }
            pending.append(frame | {"control": control_index})
            if len(pending) >= CONTROL_STEPS:
                flush()

        def gate():
            nonlocal paused, budget, speed, deadline
            while True:
                while connection.poll():
                    kind, command = connection.recv()
                    if kind == "stop":
                        raise _StopRun
                    paused, budget, speed = command
                    deadline = time.monotonic()
                if paused and budget:
                    budget -= 1
                    phase("Simulating")
                    return
                if not paused:
                    delay = deadline - time.monotonic()
                    if delay <= 0:
                        # Absolute deadlines include controller and physics time.
                        deadline += PHYSICS_DT / speed
                        phase("Simulating")
                        return
                else:
                    phase("Paused")
                    flush()
                    delay = None
                connection.poll(delay)

        def after_step(env):
            append()
            if (env.status == "running").any():
                gate()

        append()
        flush()
        gate()
        while not env.done.all():
            phase("Calling controller")
            observation, reset = env.observe(), env.first
            saved_observation = {k: v.numpy().copy() for k, v in observation.items()}
            saved_reset = reset.numpy().copy()
            started = time.perf_counter()
            action = solution.act(observation, reset)
            elapsed = time.perf_counter() - started
            validate_action(action, env.size)
            flush()
            connection.send(
                (
                    "control",
                    {
                        "observation": saved_observation,
                        "reset": saved_reset,
                        "action": action.detach().cpu().numpy().copy(),
                        "seconds": elapsed,
                        "debug": [debug_snapshot(solution, i) for i in range(env.size)],
                    },
                )
            )
            control_index += 1
            phase("Simulating")
            env.step(action, on_step=after_step)
    finally:
        flush()


def _worker(connection):
    # Include compiler subprocesses in cancellation, without touching other runs.
    if os.name == "posix":
        os.setsid()
    import torch

    torch.set_num_threads(1)
    try:
        while True:
            kind, request = connection.recv()
            if kind != "run":
                continue
            try:
                _simulate(connection, *request)
            except _StopRun:
                pass
            except BaseException:
                connection.send(("error", traceback.format_exc()))
            connection.send(("done", None))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def _kill(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            pass  # Spawn may still be importing, before setsid().
    if process.is_alive():
        process.kill()


class Worker:
    """Own the warm process across sequential runs; a hung call invalidates it."""

    def __init__(self):
        self.process = self.connection = None

    def acquire(self):
        if self.process is None or not self.process.is_alive():
            self.close()
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(target=_worker, args=(child,))
            try:
                self.process.start()
            finally:
                child.close()
        return self.connection, self.process

    def close(self):
        if self.process is not None:
            if self.process.pid is not None:
                _kill(self.process)
                self.process.join()
            self.connection.close()
            self.process = self.connection = None


class Run(QThread):
    """Run a solution, or prepare the engine when path is None."""

    def __init__(self, path, courses, seeds, parent=None, *, worker=None, **options):
        super().__init__(parent)
        self.path, self.courses, self.seeds = path, courses, seeds
        self.options = options
        self.worker = worker if worker is not None else Worker()
        self.shared_worker = worker is not None
        self.lock = threading.Lock()
        self.commands = queue.SimpleQueue()
        self.process = None
        self.paused, self.stopping, self.speed = True, False, 1.0
        self.phase, self.phase_started = "Starting process", time.monotonic()
        self.frames, self.controls = [], []
        self.model = self.error = None
        self.stop_deadline = None

    def command(self, *, paused=True, steps=0, speed=None, stop=False):
        with self.lock:
            self.paused = paused
            if speed is not None:
                self.speed = speed
            if stop and not self.stopping:
                self.stopping = True
                self.phase = "Stopping"
                self.stop_deadline = time.monotonic() + 0.25
                self.commands.put(("stop", None))
            elif not self.stopping:
                self.commands.put(("control", (paused, steps, self.speed)))

    def run(self):
        completed = False
        try:
            connection, process = self.worker.acquire()
            self.process = process
            connection.send(("run", (self.path, self.courses, self.seeds, self.options)))
            while True:
                while not self.commands.empty():
                    connection.send(self.commands.get_nowait())
                if self.stopping and time.monotonic() >= self.stop_deadline:
                    break
                if connection.poll(0.01):
                    kind, value = connection.recv()
                    if kind == "done":
                        completed = True
                        break
                    with self.lock:
                        if kind == "frames":
                            self.frames.extend(
                                dict(zip(value, row)) for row in zip(*value.values())
                            )
                        elif kind == "control":
                            self.controls.append(value)
                        elif kind == "phase":
                            if not self.stopping and self.phase != value:
                                self.phase, self.phase_started = value, time.monotonic()
                        else:
                            setattr(self, kind, value)
                elif not process.is_alive():
                    break
        except (EOFError, BrokenPipeError):
            pass
        except BaseException:
            if not self.stopping:
                self.error = traceback.format_exc()
        finally:
            if not completed and not self.stopping and self.error is None:
                code = self.process.exitcode if self.process is not None else None
                self.error = f"Simulation process exited with code {code}."
            if not completed or not self.shared_worker or self.error:
                self.worker.close()
            with self.lock:
                self.process = None
                self.phase = "Stopped" if self.stopping else "Finished"
            if self.error:
                print(self.error, end="")

    def snapshot(self, index=-1):
        with self.lock:
            if not self.frames:
                return None
            frame = self.frames[index]
            control = self.controls[frame["control"]] if frame["control"] >= 0 else None
            return frame, control, len(self.frames)

    def record(self):
        with self.lock:
            return {
                "lab": "line_following",
                "version": 2,
                "solution": str(self.path),
                "seeds": np.array(self.seeds),
                "physics_dt": PHYSICS_DT,
                "control_steps": CONTROL_STEPS,
                "courses": [
                    {
                        "name": c.name,
                        "controls": c.controls,
                        "knots": c.knots,
                        "corners": c.corners,
                    }
                    for c in self.courses
                ],
                "model": self.model,
                "frames": {k: np.array([f[k] for f in self.frames]) for k in self.frames[0]}
                if self.frames
                else {},
                "controls": list(self.controls),
                "error": self.error,
            }
