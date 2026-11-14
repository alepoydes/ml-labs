"""Numeric learning data; filesystem decoding belongs to the trusted launcher.

The same request protocol works locally and across the isolated worker boundary.
Only allowlisted tables and image batches are returned, never text or filenames.
"""

import io

import numpy as np

from coursekit.common import CourseError

MAX_IMAGES = 128
MAX_BYTES = 128 * 1024**2


def image_limit(shape):
    # Leave space for NPZ headers and compression overhead even for random pixels.
    return min(MAX_IMAGES, MAX_BYTES // 2 // int(np.prod(shape)))


class ImageBatches:
    def __init__(self, request, split, count, shape):
        self.request, self.split = request, split
        self.count, self.shape = count, tuple(shape)

    def __len__(self):
        return self.count

    def __getitem__(self, indices):
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= self.count)
        ):
            raise CourseError("Expected a vector of valid image indices.")
        limit = image_limit(self.shape)
        parts = [
            self.request(
                dict(
                    kind="images",
                    split=self.split,
                    indices=indices[i : i + limit].tolist(),
                )
            )["images"]
            for i in range(0, len(indices), limit)
        ]
        return (
            np.concatenate(parts) if parts else np.empty((0, *self.shape), dtype=np.uint8)
        )


class LearningData:
    """Reusable numeric tables and on-demand pixels; no corpus paths or spellings."""

    def __init__(self, request):
        self.request = request
        self.metadata = request(dict(kind="metadata"))
        self.episode = self.metadata.pop("episode")
        self.splits = tuple(self.metadata.pop("splits"))
        self._tables = {}

    def load_split(self, split):
        if split not in self.splits:
            raise CourseError("This split is not available for learning.")
        if split not in self._tables:
            arrays = self.request(dict(kind="table", split=split))
            images = ImageBatches(
                self.request, split, int(arrays["count"][0]), arrays["shape"]
            )
            tokens = arrays["tokens"]
            questions = [row[row != 0] for row in tokens]
            self._tables[split] = images, arrays["index"], questions, arrays["answers"]
        return self._tables[split]

    def glyphs(self):
        images, _, _, labels = self.load_split("glyphs")
        return images, labels


def provider(root):
    """Create an allowlisted host-side provider before loading solution sources."""
    from pathlib import Path

    from coursekit.labs.visual_queries_data import (
        episode,
        load_glyphs,
        load_split,
        metadata,
    )

    root = Path(root)
    meta = metadata(root)
    adapting = (root / "episode.json").exists()
    task = (
        episode(root)
        if adapting
        else dict(vocab_size=meta["base_vocab_size"], episode_id=None)
    )
    base = root / "rehearsal" if adapting else root
    paths = {"base/train": (base, "base/train")}
    if adapting:
        paths["support"] = (root, "support")
    tables = {}

    def table(split):
        if split not in tables:
            if split == "glyphs":
                images, labels = load_glyphs(base)
                tables[split] = (
                    images,
                    np.arange(len(labels), dtype=np.int64),
                    [np.array([1], dtype=np.int64)] * len(labels),
                    labels,
                )
            elif split in paths:
                tables[split] = load_split(*paths[split])
            else:
                raise ValueError("Requested data is not authorized for learning.")
        return tables[split]

    def request(message):
        if not isinstance(message, dict):
            raise ValueError("Expected a learning request object")
        kind = message.get("kind")
        if kind == "metadata" and set(message) == {"kind"}:
            return {
                key: meta[key]
                for key in ("base_vocab_size", "max_question_tokens", "image_shape")
            } | dict(episode=task, splits=[*paths, "glyphs"])
        if kind not in {"table", "images"} or set(message) != (
            {"kind", "split"} if kind == "table" else {"kind", "split", "indices"}
        ):
            raise ValueError("Invalid learning data request.")
        images, index, questions, answers = table(message["split"])
        if kind == "table":
            tokens = np.zeros((len(questions), max(map(len, questions))), np.int64)
            for i, q in enumerate(questions):
                tokens[i, : len(q)] = q
            return dict(
                count=np.array([len(images)], np.int64),
                shape=np.asarray(images.shape, np.int64),
                index=index.copy(),
                tokens=tokens,
                answers=answers.copy(),
            )
        indices = message["indices"]
        if (
            not isinstance(indices, list)
            or not 1 <= len(indices) <= image_limit(images.shape)
            or any(type(i) is not int or not 0 <= i < len(images) for i in indices)
        ):
            raise ValueError("Invalid image batch request.")
        return dict(images=images[np.asarray(indices, np.int64)])

    return request


def open_data(root):
    return LearningData(provider(root))


def encode(value):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **value)
    return buffer.getvalue()


def decode(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}
