"""Untrusted side of the protocol. Only this process loads submitted models."""

import errno
import io
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from coursekit.common import read_toml
from coursekit.labs.blindjack import validate_images, validate_mask
from coursekit.training import load_module


def main():
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # Includes native-library prints, not just Python's sys.stdout.
    sys.stdout.reconfigure(line_buffering=True, write_through=True)

    def send(value):
        protocol.write(json.dumps(value, allow_nan=False) + "\n")

    solution = Path(sys.argv[1]).resolve()
    seed = int(sys.argv[2])
    try:
        instance = load_module(solution, "solution").Solution(
            solution / "artifacts", read_toml(solution / "config.toml"), seed
        )
        send({"id": -1, "status": "ready"})
    except BaseException as exc:
        traceback.print_exc()
        resource_failure = (
            isinstance(exc, MemoryError)
            or isinstance(exc, OSError)
            and exc.errno in {errno.EAGAIN, errno.ENOMEM}
        )
        send(
            {
                "id": -1,
                "error": "memory_or_process_limit"
                if resource_failure
                else "model_loading_failure",
            }
        )
        return
    expected = 0
    while line := sys.stdin.buffer.readline(4097):
        try:
            request = json.loads(line)
            size = request["bytes"]
            if (
                request["id"] != expected
                or type(size) is not int
                or not 0 < size <= 1024**2
            ):
                raise ValueError("Invalid request header")
            raw = sys.stdin.buffer.read(size)
            if len(raw) != size:
                raise ValueError("Truncated image payload")
            with Image.open(io.BytesIO(raw)) as image:
                if image.mode != "RGB" or image.size != (7 * 128, 128):
                    raise ValueError("Invalid observation dimensions")
                images = np.array(image).reshape(128, 7, 128, 3).transpose(1, 0, 2, 3)
            validate_images(images)
        except BaseException:
            traceback.print_exc()
            send({"id": expected, "error": "protocol_failure"})
            return
        try:
            mask = instance.select(images)
        except MemoryError:
            traceback.print_exc()
            send({"id": expected, "error": "memory_or_process_limit"})
            return
        except BaseException as exc:
            traceback.print_exc()
            resource_failure = isinstance(exc, OSError) and exc.errno in {
                errno.EAGAIN,
                errno.ENOMEM,
            }
            send(
                {
                    "id": expected,
                    "error": "memory_or_process_limit"
                    if resource_failure
                    else "runtime_exception",
                }
            )
            return
        try:
            validate_mask(mask)
        except Exception:
            traceback.print_exc()
            send({"id": expected, "error": "invalid_output"})
            return
        send({"id": expected, "mask": mask.tolist()})
        expected += 1


if __name__ == "__main__":
    main()
