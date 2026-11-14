"""CSV joins, lazy PNG tiles and padded batches for Visual Queries."""

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from coursekit.common import CourseError, digest, safe_path


def rows(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise CourseError(f"Expected columns {fields}: {path}")
        result = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in result):
        raise CourseError(f"Malformed CSV: {path}")
    return result


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise CourseError(f"Expected integer in [{low}, {high}].")
    return value


def metadata(root):
    value = json.loads((Path(root) / "dataset.json").read_text())
    if (
        value.get("format_version") != 1
        or value.get("lab") != "visual_queries"
        or value.get("pad_token_id") != 0
    ):
        raise CourseError("Expected Visual Queries resource format 1.")
    shape, grid = value.get("image_shape"), value.get("sheet_grid")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[2] != 3
        or not isinstance(grid, list)
        or len(grid) != 2
    ):
        raise CourseError("Invalid image/sheet dimensions.")
    for number in shape[:2]:
        integer(number, 16, 1024)
    for number in grid:
        integer(number, 1, 16)
    if shape[0] * grid[0] * shape[1] * grid[1] > 4096**2:
        raise CourseError("PNG sheet exceeds the pixel budget.")
    integer(value.get("base_vocab_size"), 2, 4096)
    integer(value.get("max_added_tokens"), 1, 256)
    integer(value.get("max_question_tokens"), 1, 256)
    integer(value.get("max_batch_size"), 1, 64)
    for split in value.get("splits", []):
        safe_path(split)
    return value


def vocabulary(root):
    table = rows(Path(root) / "vocabulary.csv", ["token_id", "spelling"])
    if (
        [r["token_id"] for r in table] != [str(i) for i in range(len(table))]
        or not table
        or table[0]["spelling"] != "[PAD]"
    ):
        raise CourseError("Vocabulary IDs must start at padding zero and be contiguous.")
    words = [r["spelling"] for r in table]
    if len(set(words)) != len(words) or len(words) != metadata(root)["base_vocab_size"]:
        raise CourseError("Invalid base vocabulary.")
    return words


def episode(root, name=None):
    root = Path(root)
    path = (
        root / "adaptation" / safe_path(name) / "episode.json"
        if name is not None
        else root / "episode.json"
    )
    value = json.loads(path.read_text())
    meta = metadata(root)
    if (
        set(value)
        != {
            "format_version",
            "episode_id",
            "base_vocabulary_sha256",
            "base_vocab_size",
            "vocab_size",
        }
        or value["format_version"] != 1
    ):
        raise CourseError("Invalid adaptation metadata.")
    if (
        not isinstance(value["episode_id"], str)
        or value["base_vocabulary_sha256"] != digest(root / "vocabulary.csv")
        or value["base_vocab_size"] != meta["base_vocab_size"]
    ):
        raise CourseError("Adaptation and base vocabulary differ.")
    integer(
        value["vocab_size"],
        meta["base_vocab_size"] + 1,
        meta["base_vocab_size"] + meta["max_added_tokens"],
    )
    return value


def scope(root, split):
    meta = metadata(root)
    if split not in meta["splits"]:
        raise CourseError(f"Unknown split: {split}")
    if split.startswith("adaptation/"):
        return episode(root, split.split("/")[1])["vocab_size"]
    return episode(root)["vocab_size"] if split == "support" else meta["base_vocab_size"]


class Images:
    def __init__(self, folder, shape, grid):
        self.folder, self.shape, self.grid = Path(folder), tuple(shape), tuple(grid)
        self.index = rows(
            self.folder / "images.csv", ["image_id", "sheet", "row", "column"]
        )
        self.positions = []
        ids, positions = set(), set()
        for row in self.index:
            safe_path(row["sheet"])
            if not row["sheet"].startswith("images/") or not row["sheet"].endswith(".png"):
                raise CourseError("Images must reference PNG files under images/.")
            try:
                y, x = int(row["row"]), int(row["column"])
            except ValueError as exc:
                raise CourseError("Invalid PNG tile coordinates.") from exc
            if (
                row["row"] != str(y)
                or row["column"] != str(x)
                or not 1 <= y <= grid[0]
                or not 1 <= x <= grid[1]
            ):
                raise CourseError(
                    "PNG tile coordinates are one-based and within the sheet."
                )
            position = (row["sheet"], y - 1, x - 1)
            if not row["image_id"] or row["image_id"] in ids or position in positions:
                raise CourseError("Repeated image ID or tile.")
            ids.add(row["image_id"])
            self.positions.append(position)
            positions.add(position)
        self._sheet = lru_cache(maxsize=2)(self.read_sheet)

    def read_sheet(self, name):
        path = self.folder / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise CourseError("Symbolic links are not image resources.")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (
                self.shape[1] * self.grid[1],
                self.shape[0] * self.grid[0],
            ):
                raise CourseError("Invalid RGB PNG sheet dimensions.")
            return np.asarray(image)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, indices):
        indices = np.asarray(indices)
        if (
            indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= len(self))
        ):
            raise CourseError("Image index outside the split.")
        output = np.empty((*indices.shape, *self.shape), np.uint8)
        for target, index in zip(output.reshape(-1, *self.shape), indices.flat):
            name, row, column = self.positions[int(index)]
            h, w, _ = self.shape
            target[:] = self._sheet(name)[
                row * h : (row + 1) * h, column * w : (column + 1) * w
            ]
        return output


def load_split(root, split):
    root = Path(root)
    meta, size = metadata(root), scope(root, split)
    folder = root / safe_path(split)
    images = Images(folder, meta["image_shape"], meta["sheet_grid"])
    image_ids = {r["image_id"]: i for i, r in enumerate(images.index)}
    questions = rows(folder / "questions.csv", ["question_id", "image_id", "token_ids"])
    ids, indices, tokens, seen = [], [], [], set()
    for row in questions:
        if (
            not row["question_id"]
            or row["question_id"] in seen
            or row["image_id"] not in image_ids
        ):
            raise CourseError("Invalid question/image identity.")
        value = json.loads(row["token_ids"])
        if (
            not isinstance(value, list)
            or not 1 <= len(value) <= meta["max_question_tokens"]
        ):
            raise CourseError("Invalid question length.")
        for token in value:
            integer(token, 1, size - 1)
        ids.append(row["question_id"])
        seen.add(row["question_id"])
        indices.append(image_ids[row["image_id"]])
        tokens.append(np.array(value, dtype=np.int64))
    answers = None
    if (folder / "answers.csv").exists():
        table = rows(folder / "answers.csv", ["question_id", "answer_id"])
        mapping = {r["question_id"]: r["answer_id"] for r in table}
        if len(mapping) != len(table) or set(mapping) != set(ids):
            raise CourseError("Answers must match question IDs exactly.")
        values = []
        for name in ids:
            try:
                answer = int(mapping[name])
            except ValueError as exc:
                raise CourseError("Invalid answer ID.") from exc
            if str(answer) != mapping[name]:
                raise CourseError("Noncanonical answer ID.")
            values.append(integer(answer, 1, size - 1))
        answers = np.array(values, dtype=np.int64)
    return images, np.array(indices, dtype=np.int64), tokens, answers


def batch(data, indices):
    images, image_index, questions, _ = data
    indices = np.asarray(indices)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(questions))
    ):
        raise CourseError("Choose a nonempty vector of question indices.")
    selected = [questions[int(i)] for i in indices]
    tokens = np.zeros((len(indices), max(map(len, selected))), np.int64)
    mask = np.zeros(tokens.shape, bool)
    for row, value in enumerate(selected):
        tokens[row, : len(value)], mask[row, : len(value)] = value, True
    return images[image_index[indices]], tokens, mask


def load_glyphs(root):
    folder = Path(root) / "glyphs"
    meta = json.loads((folder / "metadata.json").read_text())
    images = Images(folder, meta["image_shape"], meta["sheet_grid"])
    table = rows(folder / "labels.csv", ["image_id", "symbol_id"])
    labels = {r["image_id"]: int(r["symbol_id"]) for r in table}
    if len(labels) != len(table) or set(labels) != {r["image_id"] for r in images.index}:
        raise CourseError("Glyph labels must match image IDs.")
    size = metadata(root)["base_vocab_size"]
    values = [integer(labels[r["image_id"]], 1, size - 1) for r in images.index]
    return images, np.array(values, dtype=np.int64)


def public_members(root):
    """Exact public resource members, derived from validated indexes."""
    root = Path(root)
    meta = metadata(root)
    vocabulary(root)
    members = {"dataset.json", "vocabulary.csv", "SOURCES.md", "index.html"}
    for split in meta["splits"]:
        images, _, _, answers = load_split(root, split)
        if answers is None:
            raise CourseError("Public practice splits must include answers.")
        members.update(
            f"{split}/{name}" for name in ("images.csv", "questions.csv", "answers.csv")
        )
        members.update(f"{split}/{row['sheet']}" for row in images.index)
    for name in meta["episodes"]:
        episode(root, name)
        members.add(f"adaptation/{name}/episode.json")
    glyphs, _ = load_glyphs(root)
    members.update(
        f"glyphs/{name}" for name in ("metadata.json", "images.csv", "labels.csv")
    )
    members.update(f"glyphs/{row['sheet']}" for row in glyphs.index)
    return members
