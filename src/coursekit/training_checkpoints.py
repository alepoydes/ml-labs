"""Complete adaptation snapshots, shared by local and isolated execution."""

import base64
import inspect
import shutil
import time
import uuid
from pathlib import Path

from coursekit.common import CourseError, digest, safe_path, write_json

CHUNK_BYTES = 262144


class AdaptationTimeout(TimeoutError):
    """The adaptation budget expired; only an acknowledged snapshot may survive."""


def output_manifest(files, maximum, *, artifact_bytes=None):
    """Validate descriptors before creating files or requesting any model bytes."""
    if not isinstance(files, dict) or len(files) > 4096:
        raise CourseError("Invalid training file manifest.")
    total = models = 0
    names = set(files)
    for name, item in files.items():
        safe_path(name)
        if any(parent.as_posix() in names for parent in Path(name).parents):
            raise CourseError("Conflicting training file paths.")
        if (
            Path(name).suffix.lower() == ".npy"
            or name == "artifacts/provenance.json"
            or name != "history.json"
            and not name.startswith(("artifacts/", "history/"))
            or not isinstance(item, dict)
            or set(item) != {"size", "sha256"}
            or type(item["size"]) is not int
            or item["size"] < 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in item["sha256"])
        ):
            raise CourseError("Invalid training file descriptor.")
        total += item["size"]
        if name.startswith("artifacts/"):
            models += item["size"]
    if total > maximum or artifact_bytes is not None and models > artifact_bytes:
        raise CourseError("Training output exceeds storage limit.")
    return files


class Checkpoints:
    """Receive bounded chunks; promote only a complete, hash-verified snapshot."""

    def __init__(self, directory, limits, started=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.limits = limits
        self.started = time.monotonic() if started is None else started
        self.deadline = self.started + limits["wall_seconds"]
        self.pending = self.last = self.receipt = None
        self.files, self.offsets = {}, {}
        self.index = 0

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise AdaptationTimeout("Adaptation time limit exceeded.")

    def __call__(self, message):
        self.check_time()
        if not isinstance(message, dict):
            raise CourseError("Invalid checkpoint message.")
        kind = message.get("kind")
        if kind == "begin" and set(message) == {"kind", "files"}:
            if self.pending is not None:
                raise CourseError("A checkpoint transfer is already active.")
            self.files = output_manifest(
                message["files"],
                self.limits.get("output_mb", 272) * 1024**2,
                artifact_bytes=self.limits["artifact_bytes"],
            )
            self.pending = self.directory / uuid.uuid4().hex
            for folder in ("artifacts", "history"):
                (self.pending / folder).mkdir(parents=True)
            self.offsets = dict.fromkeys(self.files, 0)
            for name in self.files:
                path = self.pending / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=False)
            return {}
        if self.pending is None:
            raise CourseError("No checkpoint transfer is active.")
        if kind == "chunk" and set(message) == {"kind", "path", "offset", "data"}:
            name, offset, data = message["path"], message["offset"], message["data"]
            if (
                not isinstance(name, str)
                or name not in self.files
                or type(offset) is not int
                or offset != self.offsets[name]
                or not isinstance(data, str)
                or len(data) > 4 * ((CHUNK_BYTES + 2) // 3)
            ):
                raise CourseError("Invalid checkpoint chunk.")
            block = base64.b64decode(data, validate=True)
            if not 0 < len(block) <= min(CHUNK_BYTES, self.files[name]["size"] - offset):
                raise CourseError("Invalid checkpoint chunk size.")
            with (self.pending / name).open("ab") as stream:
                stream.write(block)
            self.offsets[name] += len(block)
            return {}
        if kind == "commit" and set(message) == {"kind"}:
            for name, item in self.files.items():
                if (
                    self.offsets[name] != item["size"]
                    or digest(self.pending / name) != item["sha256"]
                ):
                    raise CourseError("Incomplete checkpoint or hash mismatch.")
            self.check_time()
            previous, self.last = self.last, self.pending
            self.pending = None
            self.index += 1
            self.receipt = dict(index=self.index, seconds=time.monotonic() - self.started)
            if previous is not None:
                shutil.rmtree(previous)
            return self.receipt.copy()
        raise CourseError("Invalid checkpoint message fields.")

    def result(self, *, reason="completed", finished=None):
        return dict(
            seconds=(time.monotonic() if finished is None else finished) - self.started,
            budget_seconds=self.limits["wall_seconds"],
            stop_reason=reason,
            checkpoint=self.receipt.copy()
            if reason == "time_limit" and self.receipt
            else None,
        )


def checkpoint_writer(output, history_dir, api_version, limits, request):
    """Return the student's synchronous checkpoint(history=None) callback."""
    from coursekit.training_v2 import checked_files, save_history

    def checkpoint(history=None):
        history_dir.mkdir(exist_ok=True)
        # Each snapshot contains exactly its supplied histories, including an empty set.
        for path in history_dir.glob("*.npz"):
            path.unlink()
        save_history(history_dir, history)
        write_json(output.parent / "history.json", dict(api_version=api_version))
        paths = [
            *(f"artifacts/{name}" for name in checked_files(output)),
            *(f"history/{name}" for name in checked_files(history_dir)),
            "history.json",
        ]
        files = {
            name: dict(
                size=(output.parent / name).stat().st_size,
                sha256=digest(output.parent / name),
            )
            for name in paths
        }
        output_manifest(
            files,
            limits.get("output_mb", 272) * 1024**2,
            artifact_bytes=limits["artifact_bytes"],
        )
        request(dict(kind="begin", files=files))
        for name in files:
            with (output.parent / name).open("rb") as stream:
                offset = 0
                while block := stream.read(CHUNK_BYTES):
                    request(
                        dict(
                            kind="chunk",
                            path=name,
                            offset=offset,
                            data=base64.b64encode(block).decode("ascii"),
                        )
                    )
                    offset += len(block)
        return request(dict(kind="commit"))

    return checkpoint


def call_adapt(function, *args, checkpoint, **kwargs):
    """Keep existing API 2/3 functions usable without retrying student exceptions."""
    parameters = inspect.signature(function).parameters
    if "checkpoint" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        kwargs["checkpoint"] = checkpoint
    return function(*args, **kwargs)
