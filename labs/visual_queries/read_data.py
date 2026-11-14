"""Independent CSV/PNG example for humans; submitted models use numeric data."""

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def read_example(root, split, number):
    """Read a one-based CSV record, excluding the header."""
    root = Path(root)
    meta = json.loads((root / "dataset.json").read_text())
    folder = root / split
    dictionary_folder = (
        root / "adaptation" / split.split("/")[1]
        if split.startswith("adaptation/")
        else root
    )
    with (dictionary_folder / "vocabulary.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        words = {r["spelling"]: int(r["token_id"]) for r in csv.DictReader(stream)}
    with (folder / "examples.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not 1 <= number <= len(records):
        raise ValueError("Record number is outside the table.")
    record = records[number - 1]
    h, w, _ = meta["image_shape"]
    x, y = (int(record["column"]) - 1) * w, (int(record["row"]) - 1) * h
    with Image.open(folder / "images" / f"{int(record['sheet']):06d}.png") as sheet:
        image = np.array(sheet.crop((x, y, x + w, y + h)))
    tokens = np.asarray(
        [words[word] for word in re.findall(r"[A-Za-z]+|[0-9]|[?.,]", record["question"])],
        dtype=np.int64,
    )
    answer = words[record["answer"]] if record["answer"] else None
    return image, tokens, answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("split")
    parser.add_argument("number", type=int, nargs="?", default=1)
    args = parser.parse_args()
    image, tokens, answer = read_example(args.root, args.split, args.number)
    print(f"Image {image.shape}, tokens {tokens.tolist()}, answer {answer}")
