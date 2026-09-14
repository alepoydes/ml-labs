import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from coursekit.common import CourseError
from coursekit.labs.blindjack import validate_images

DEFAULT_LIMITS = {
    "cpus": 2,
    "memory_mb": 2048,
    "pids": 128,
    "init_seconds": 60,
    "hand_seconds": 5,
    "total_seconds": 300,
    "log_bytes": 65536,
}


class EvaluationError(CourseError):
    def __init__(self, kind: str, message: str = ""):
        self.kind = kind
        super().__init__(message or kind.replace("_", " "))


def engine() -> str:
    executable = os.environ.get("COURSE_CONTAINER_ENGINE")
    if executable and shutil.which(executable):
        return executable
    for executable in ("docker", "podman"):
        if shutil.which(executable):
            return executable
    raise EvaluationError(
        "infrastructure_failure",
        "Install Docker or Podman and build the reference image; see docs/environment.md.",
    )


def image_id(executable: str, image: str) -> str:
    result = subprocess.run(
        [executable, "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise EvaluationError(
            "infrastructure_failure",
            f"Reference image unavailable: {image}. Build it from public/Dockerfile.",
        )
    return result.stdout.strip()


class Worker:
    """One request at a time, bounded output and time, no hidden-data mounts."""

    def __init__(
        self,
        solution: Path,
        seed=0,
        *,
        container=True,
        image="mlcourse-reference:local",
        limits=None,
    ):
        self.limits = DEFAULT_LIMITS | (limits or {})
        self.container, self.name = container, "course-" + uuid.uuid4().hex
        self.engine = engine() if container else None
        self.log = bytearray()
        self.responses = queue.Queue(maxsize=2)
        self.started, self.request = time.monotonic(), 0
        self.output_exceeded = False
        self.process = None
        if container:
            identity = image_id(self.engine, image)
            command = [
                self.engine,
                "run",
                "--name",
                self.name,
                "-i",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--security-opt=label=disable",
                "--user=65534:65534",
                f"--cpus={self.limits['cpus']}",
                f"--memory={self.limits['memory_mb']}m",
                f"--memory-swap={self.limits['memory_mb']}m",
                f"--pids-limit={self.limits['pids']}",
                "--ulimit=nofile=256:256",
                "--ulimit=fsize=67108864:67108864",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=256m,mode=1777",
                "--mount",
                f"type=bind,src={solution.resolve()},dst=/submission,readonly",
                "--env=HOME=/tmp",
                "--env=PYTHONDONTWRITEBYTECODE=1",
                "--env=OMP_NUM_THREADS=1",
                "--env=OPENBLAS_NUM_THREADS=1",
                "--env=MKL_NUM_THREADS=1",
                "--env=MPLCONFIGDIR=/tmp/matplotlib",
                identity,
                "/app/.venv/bin/python",
                "-m",
                "coursekit.worker",
                "/submission",
                str(seed),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "coursekit.worker",
                str(solution.resolve()),
                str(seed),
            ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=os.environ
            | {
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )

        def read_output():
            while True:
                line = self.process.stdout.readline(4097)
                try:
                    self.responses.put_nowait(line)
                except queue.Full:
                    self.output_exceeded = True
                    self.process.kill()
                    return
                if not line or len(line) > 4096:
                    return

        def read_logs():
            while block := self.process.stderr.read(4096):
                remaining = self.limits["log_bytes"] - len(self.log)
                self.log.extend(block[: max(0, remaining)])
                if len(block) > remaining:
                    self.output_exceeded = True
                    self.process.kill()
                    return

        self.threads = [
            threading.Thread(target=f, daemon=True) for f in (read_output, read_logs)
        ]
        for thread in self.threads:
            thread.start()
        try:
            response = self._response(-1, self.limits["init_seconds"])
            if response != {"id": -1, "status": "ready"}:
                raise EvaluationError("protocol_failure")
        except BaseException as exc:
            if isinstance(exc, EvaluationError):
                exc.log = bytes(self.log)
            self.close()
            raise

    def _response(self, expected, timeout):
        remaining = self.limits["total_seconds"] - (time.monotonic() - self.started)
        try:
            line = self.responses.get(timeout=max(0, min(timeout, remaining)))
        except queue.Empty as exc:
            raise EvaluationError("timeout") from exc
        if self.output_exceeded:
            raise EvaluationError("output_limit")
        if not line:
            kind = "runtime_exception"
            if self.container:
                state = subprocess.run(
                    [self.engine, "inspect", self.name, "--format", "{{json .State}}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                try:
                    info = json.loads(state.stdout)
                    if info.get("OOMKilled"):
                        kind = "memory_or_process_limit"
                    elif info.get("ExitCode") in (125, 126, 127) or info.get("Error"):
                        kind = "infrastructure_failure"
                except ValueError:
                    kind = "infrastructure_failure"
            raise EvaluationError(kind, self.log.decode(errors="replace")[-2000:] or kind)
        try:
            if len(line) > 4096:
                raise ValueError("oversized response")
            response = json.loads(line)
            if type(response.get("id")) is not int or response["id"] != expected:
                raise ValueError("request id mismatch")
            if "error" in response:
                kind = response["error"]
                if kind not in {
                    "model_loading_failure",
                    "runtime_exception",
                    "invalid_output",
                    "protocol_failure",
                    "memory_or_process_limit",
                }:
                    raise ValueError("unknown error")
                raise EvaluationError(kind)
            return response
        except (ValueError, AttributeError, TypeError, RecursionError) as exc:
            raise EvaluationError("protocol_failure", str(exc)) from exc

    def select(self, images):
        validate_images(images)
        output = io.BytesIO()
        Image.fromarray(images.transpose(1, 0, 2, 3).reshape(128, 7 * 128, 3)).save(
            output, format="PNG"
        )
        raw = output.getvalue()
        try:
            # A separate writer keeps a student that stops reading stdin bounded.
            failure = []

            def send():
                try:
                    data = memoryview(
                        json.dumps({"id": self.request, "bytes": len(raw)}).encode()
                        + b"\n"
                        + raw
                    )
                    while data:
                        written = self.process.stdin.write(data)
                        if not written:
                            raise BrokenPipeError("Worker stopped reading observations")
                        data = data[written:]
                    self.process.stdin.flush()
                except OSError as exc:
                    failure.append(exc)

            writer = threading.Thread(target=send, daemon=True)
            writer.start()
            response = self._response(self.request, self.limits["hand_seconds"])
            writer.join(timeout=0.1)
            if failure or writer.is_alive():
                raise EvaluationError("protocol_failure")
            mask = response.get("mask")
            if (
                set(response) != {"id", "mask"}
                or not isinstance(mask, list)
                or len(mask) != 7
                or any(type(x) is not bool for x in mask)
            ):
                raise EvaluationError("invalid_output")
            self.request += 1
            return np.array(mask, dtype=bool)
        except OSError as exc:
            raise EvaluationError("runtime_exception", str(exc)) from exc

    def close(self, graceful=False):
        if graceful and self.process and self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if self.container and self.engine:
            subprocess.run(
                [self.engine, "rm", "-f", self.name], capture_output=True, timeout=20
            )
        if self.process:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=10)
            for thread in self.threads:
                thread.join(timeout=1)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(graceful=exc_type is None)
