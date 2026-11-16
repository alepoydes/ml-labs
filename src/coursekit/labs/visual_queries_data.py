"""CSV joins, lazy PNG tiles and padded batches for Visual Queries."""

import csv
import json
import re
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
        value.get("format_version") not in (1, 2)
        or value.get("lab") != "visual_queries"
        or value.get("pad_token_id") != 0
    ):
        raise CourseError("Expected Visual Queries resource format 1 or 2.")
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


def episode_label(episodes, name):
    """Human labels follow the dataset schedule; model-facing IDs stay opaque."""
    if name is None:
        return "Base validation"
    return (
        f"Adaptation {episodes.index(name) + 1:02d}"
        if name in episodes
        else f"Adaptation {name}"
    )


def split_label(meta, split):
    if split.startswith("base/"):
        return f"Base · {split.split('/')[-1]}"
    parts = split.split("/")
    if len(parts) == 3 and parts[0] == "adaptation":
        return f"{episode_label(meta['episodes'], parts[1])} · {parts[2]}"
    return split


def read_vocabulary(path, size):
    table = rows(path, ["token_id", "spelling"])
    if (
        [r["token_id"] for r in table] != [str(i) for i in range(len(table))]
        or not table
        or table[0]["spelling"] != "[PAD]"
    ):
        raise CourseError("Vocabulary IDs must start at padding zero and be contiguous.")
    words = [r["spelling"] for r in table]
    if len(set(words)) != len(words) or len(words) != size:
        raise CourseError("Invalid vocabulary size or duplicate spelling.")
    if any(tokenize(word) != [word] for word in words[1:]):
        raise CourseError("Each spelling must represent one token.")
    return words


def vocabulary(root, name=None):
    folder = Path(root) if name is None else Path(root) / "adaptation" / safe_path(name)
    size = (
        metadata(root)["base_vocab_size"]
        if name is None
        else episode(root, name)["vocab_size"]
    )
    words = read_vocabulary(folder / "vocabulary.csv", size)
    if name is not None and words[: metadata(root)["base_vocab_size"]] != vocabulary(root):
        raise CourseError("An episode changes old token meanings.")
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
    if set(value) != {
        "format_version",
        "episode_id",
        "base_vocabulary_sha256",
        "base_vocab_size",
        "vocab_size",
    } or value["format_version"] not in (1, 2):
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
    def __init__(self, folder, shape, grid, index=None):
        self.folder, self.shape, self.grid = Path(folder), tuple(shape), tuple(grid)
        self.index = (
            index
            if index is not None
            else rows(self.folder / "images.csv", ["image_id", "sheet", "row", "column"])
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
    if meta["format_version"] == 2:
        return text_split(root, split)
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
    if metadata(root)["format_version"] == 2:
        return text_glyphs(Path(root))
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
    if meta["format_version"] == 2:
        return text_members(root)
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


EXAMPLE_FIELDS = ["sheet", "row", "column", "question", "answer"]


def tokenize(text):
    """The published alphabet is ASCII words/digits and explicit punctuation."""
    if (
        not isinstance(text, str)
        or not text.strip()
        or re.search(r"[^A-Za-z0-9?., ]", text)
    ):
        raise CourseError("Invalid question text or unsupported punctuation.")
    return re.findall(r"[A-Za-z]+|[0-9]|[?.,]", text)


def sheet_path(number):
    return f"images/{number:06d}.png"


def tile_index(table, grid):
    coordinates = []
    for row in table:
        values = []
        for key, limit in zip(("sheet", "row", "column"), (10**6 - 1, *grid)):
            try:
                value = int(row[key])
            except (ValueError, TypeError) as exc:
                raise CourseError("Expected one-based PNG coordinates.") from exc
            if str(value) != row[key]:
                raise CourseError("Noncanonical PNG coordinate.")
            values.append(integer(value, 1, limit))
        position = tuple(values)
        coordinates.append(position)
    positions = {position: i for i, position in enumerate(sorted(set(coordinates)))}
    image_index = [positions[position] for position in coordinates]
    index = [
        dict(image_id=str(i), sheet=sheet_path(s), row=str(y), column=str(x))
        for (s, y, x), i in positions.items()
    ]
    return index, np.asarray(image_index, dtype=np.int64)


def text_split(root, split):
    meta = metadata(root)
    words = (
        vocabulary(root, split.split("/")[1])
        if split.startswith("adaptation/")
        else vocabulary(root)
    )
    # A sealed host adaptation view stores the extended dictionary beside support.
    if split == "support":
        extended = read_vocabulary(root / "support/vocabulary.csv", scope(root, split))
        if extended[: len(words)] != words:
            raise CourseError("Support changes old token meanings.")
        words = extended
    dictionary = {word: i for i, word in enumerate(words)}
    table = rows(root / split / "examples.csv", EXAMPLE_FIELDS)
    if not table:
        raise CourseError("An example table must not be empty.")
    index, mapping = tile_index(table, meta["sheet_grid"])
    questions, answers = [], []
    for row in table:
        try:
            ids = [dictionary[t] for t in tokenize(row["question"])]
            answer = dictionary[row["answer"]] if row["answer"] else None
        except KeyError as exc:
            raise CourseError(f"Word outside the declared vocabulary: {exc}") from exc
        if not 1 <= len(ids) <= meta["max_question_tokens"] or 0 in ids or answer == 0:
            raise CourseError("Invalid question length or padding in text.")
        questions.append(np.asarray(ids, dtype=np.int64))
        answers.append(answer)
    if any(a is None for a in answers) and not all(a is None for a in answers):
        raise CourseError("Labels must be present for every row or no rows.")
    images = Images(root / split, meta["image_shape"], meta["sheet_grid"], index)
    return (
        images,
        mapping,
        questions,
        None if answers[0] is None else np.asarray(answers, dtype=np.int64),
    )


def text_glyphs(root):
    folder = root / "glyphs"
    meta = json.loads((folder / "metadata.json").read_text())
    table = rows(folder / "examples.csv", ["sheet", "row", "column", "symbol"])
    index, mapping = tile_index(table, meta["sheet_grid"])
    if len(set(mapping)) != len(mapping):
        raise CourseError("Repeated glyph tile.")
    dictionary = {word: i for i, word in enumerate(vocabulary(root))}
    try:
        labels = np.asarray([dictionary[r["symbol"]] for r in table], dtype=np.int64)
    except KeyError as exc:
        raise CourseError("Unknown glyph label.") from exc
    if not len(labels) or np.any(labels == 0):
        raise CourseError("Invalid glyph labels.")
    aligned = np.empty_like(labels)
    aligned[mapping] = labels
    return Images(folder, meta["image_shape"], meta["sheet_grid"], index), aligned


def text_members(root):
    meta = metadata(root)
    members = {"dataset.json", "vocabulary.csv", "SOURCES.md", "index.html"}
    for split in meta["splits"]:
        images, _, _, answers = load_split(root, split)
        if answers is None:
            raise CourseError("Public examples need answers.")
        members.add(f"{split}/examples.csv")
        members.update(f"{split}/{r['sheet']}" for r in images.index)
    for name in meta["episodes"]:
        words = vocabulary(root, name)
        if words[: meta["base_vocab_size"]] != vocabulary(root):
            raise CourseError("An episode changes old token meanings.")
        members.update(f"adaptation/{name}/{f}" for f in ("episode.json", "vocabulary.csv"))
    images, _ = load_glyphs(root)
    members.update(("glyphs/metadata.json", "glyphs/examples.csv"))
    members.update(f"glyphs/{r['sheet']}" for r in images.index)
    return members
