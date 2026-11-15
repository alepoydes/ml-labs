"""Cancellable model process; the host sends complete numeric batches only."""

import multiprocessing
import os
import signal
import threading
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal


def _worker(connection, path, artifacts, device, seed, settings):
    if os.name == "posix":
        os.setsid()
    try:
        from coursekit.labs.visual_queries_inspection import model_identity, predict_batch
        from coursekit.training import load_solution

        def identity():
            return model_identity(path, artifacts, device, seed)

        before = identity()
        solution = load_solution(path, seed, device=device, artifact_dir=artifacts)
        if identity() != before:
            raise RuntimeError("Solution changed while loading; restart the run.")
        connection.send(("ready", before))
        while True:
            arguments = connection.recv()
            connection.send(("batch", predict_batch(solution, arguments, settings)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class Run(QThread):
    loaded = Signal(object)
    batch_ready = Signal(object)
    phase = Signal(str)

    def __init__(self, dataset, path, artifacts, device, batch_size, parent=None):
        super().__init__(parent)
        self.dataset, self.path, self.artifacts = dataset, path, artifacts
        self.device, self.batch_size = device, batch_size
        self.cancelled, self.advance = threading.Event(), threading.Event()
        self.advance.set()
        self.automatic = False
        self.stopping = False
        self.error = None
        self.complete = False

    def next(self):
        self.advance.set()

    def play(self, automatic):
        self.automatic = automatic
        if automatic:
            self.advance.set()

    def stop(self):
        self.stopping = True
        self.cancelled.set()
        self.advance.set()

    def run(self):
        context = multiprocessing.get_context("spawn")
        connection, child = context.Pipe()
        process = context.Process(
            target=_worker,
            args=(
                child,
                self.path,
                self.artifacts,
                self.device,
                0,
                self.dataset.settings,
            ),
        )

        def receive():
            while not self.cancelled.is_set():
                if connection.poll(0.02):
                    kind, value = connection.recv()
                    if kind == "error":
                        raise RuntimeError(value)
                    return value
                if not process.is_alive():
                    raise RuntimeError(
                        f"Model process exited with code {process.exitcode}."
                    )
            return None

        def send(arguments):
            failure = []

            def write():
                try:
                    connection.send(arguments)
                except (OSError, EOFError) as exc:
                    failure.append(exc)

            writer = threading.Thread(target=write, daemon=True)
            writer.start()
            while writer.is_alive() and not self.cancelled.is_set():
                writer.join(0.02)
            if failure:
                raise failure[0]

        try:
            self.phase.emit("Loading model…")
            process.start()
            child.close()
            identity = receive()
            if identity is None:
                return
            self.loaded.emit(identity)
            for start in range(0, len(self.dataset.examples), self.batch_size):
                while not self.advance.wait(0.02):
                    if self.cancelled.is_set():
                        return
                if self.cancelled.is_set():
                    return
                self.advance.clear()
                numbers = np.arange(
                    start + 1,
                    min(start + self.batch_size, len(self.dataset.examples)) + 1,
                    dtype=np.int64,
                )
                self.phase.emit(f"Predicting rows {numbers[0]}–{numbers[-1]}…")
                send(self.dataset.arguments(numbers))
                value = receive()
                if value is None:
                    return
                self.batch_ready.emit(value | {"rows": numbers})
                if value.get("debug_error"):
                    raise RuntimeError(value["debug_error"])
                if self.automatic:
                    self.advance.set()
                else:
                    self.phase.emit("Paused. Select any row or predict the next batch.")
            self.complete = True
        except BaseException:
            if not self.cancelled.is_set():
                self.error = traceback.format_exc()
        finally:
            child.close()
            if process.pid is not None:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.is_alive():
                    process.kill()
                process.join()
            connection.close()
