"""Read one verified example without the coursekit loader: python read_data.py ROOT SPLIT."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def read_example(root, split, question_id=None):
    root = Path(root)
    meta = json.loads((root / "dataset.json").read_text())
    folder = root / split

    def rows(name):
        with (folder / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    questions = rows("questions.csv")
    question = (
        questions[0]
        if question_id is None
        else next(r for r in questions if r["question_id"] == question_id)
    )
    tile = next(r for r in rows("images.csv") if r["image_id"] == question["image_id"])
    height, width, _ = meta["image_shape"]
    x, y = (int(tile["column"]) - 1) * width, (int(tile["row"]) - 1) * height
    with Image.open(folder / tile["sheet"]) as sheet:
        image = np.array(sheet.crop((x, y, x + width, y + height)))
    tokens = np.asarray(json.loads(question["token_ids"]), dtype=np.int64)
    answer = None
    if (folder / "answers.csv").exists():
        answer = int(
            next(
                r["answer_id"]
                for r in rows("answers.csv")
                if r["question_id"] == question["question_id"]
            )
        )
    return image, tokens, answer


if __name__ == "__main__":
    image, tokens, answer = read_example(sys.argv[1], sys.argv[2])
    print(image.shape, image.dtype, tokens.tolist(), answer)
