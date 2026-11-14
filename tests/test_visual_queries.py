import csv
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from coursekit.common import CourseError, digest, write_json
from coursekit.labs.visual_queries import (
    adaptation_view,
    evaluate_solution,
    run_benchmark,
    validate_answers,
    validate_inputs,
)
from coursekit.labs.visual_queries_data import batch, load_split, metadata, vocabulary
from coursekit.training_v2 import run


def csv_file(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


@pytest.fixture
def queries(tmp_path):
    root = tmp_path / "data"
    names = [
        "base/train",
        "base/validation",
        "adaptation/a/support",
        "adaptation/a/query",
        "adaptation/b/support",
        "adaptation/b/query",
    ]
    words = ["[PAD]", "yes", "no", "Is", "empty", "?", "A"]
    csv_file(root / "vocabulary.csv", ["token_id", "spelling"], enumerate(words))
    write_json(
        root / "dataset.json",
        dict(
            format_version=1,
            lab="visual_queries",
            resource_version="fixture",
            pad_token_id=0,
            base_vocab_size=len(words),
            image_shape=[16, 16, 3],
            sheet_grid=[1, 2],
            max_added_tokens=4,
            max_question_tokens=96,
            max_batch_size=16,
            splits=names,
            episodes=["a", "b"],
            adaptation_limits=dict(
                wall_seconds=10, memory_mb=128, gpu_memory_mb=0, artifact_bytes=100000
            ),
        ),
    )
    for split in names:
        folder = root / split
        csv_file(
            folder / "images.csv",
            ["image_id", "sheet", "row", "column"],
            [["r", "images/s.png", 1, 2], ["l", "images/s.png", 1, 1]],
        )
        pixels = np.full((16, 32, 3), 255, np.uint8)
        pixels[:, 16:] = [12, 34, 56]
        (folder / "images").mkdir()
        Image.fromarray(pixels).save(folder / "images/s.png")
        rows = [["q2", "r", "[3,4,5]"], ["q1", "l", "[4,5]"]]
        if split.endswith("query") and "/b/" in split:
            rows = rows[:1]
        csv_file(folder / "questions.csv", ["question_id", "image_id", "token_ids"], rows)
        csv_file(
            folder / "answers.csv",
            ["question_id", "answer_id"],
            [(r[0], 1 if r[0] == "q1" else 2) for r in reversed(rows)],
        )
    for name in ("a", "b"):
        write_json(
            root / f"adaptation/{name}/episode.json",
            dict(
                format_version=1,
                episode_id=name,
                base_vocabulary_sha256=digest(root / "vocabulary.csv"),
                base_vocab_size=7,
                vocab_size=8,
            ),
        )
    (root / "glyphs/images").mkdir(parents=True)
    shutil.copyfile(root / "base/train/images/s.png", root / "glyphs/images/s.png")
    csv_file(
        root / "glyphs/images.csv",
        ["image_id", "sheet", "row", "column"],
        [["g", "images/s.png", 1, 2]],
    )
    csv_file(root / "glyphs/labels.csv", ["image_id", "symbol_id"], [["g", 6]])
    write_json(
        root / "glyphs/metadata.json", dict(image_shape=[16, 16, 3], sheet_grid=[1, 2])
    )
    return root


def test_loader_independent_csv_join_and_masks(queries):
    data = load_split(queries, "base/train")
    x, t, m = batch(data, np.array([0, 1]))
    assert x.shape == (2, 16, 16, 3) and np.all(x[0] == [12, 34, 56])
    assert data[3].tolist() == [2, 1] and t.tolist() == [[3, 4, 5], [4, 5, 0]]
    assert m.tolist() == [[True] * 3, [True, True, False]]
    script = Path(__file__).parents[1] / "labs/visual_queries/read_data.py"
    spec = importlib.util.spec_from_file_location("independent_reader", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image, tokens, answer = module.read_example(queries, "base/train", "q2")
    np.testing.assert_array_equal(image, x[0])
    np.testing.assert_array_equal(tokens, t[0])
    assert answer == 2
    (queries / "base/train/answers.csv").unlink()
    assert load_split(queries, "base/train")[3] is None
    path = queries / "base/train/questions.csv"
    path.write_text(path.read_text().replace("[3,4,5]", "[true,4,5]"))
    with pytest.raises(CourseError):
        load_split(queries, "base/train")


def test_support_sealing_and_complete_independent_lifecycle(queries, tmp_path):
    view = adaptation_view(queries, "a", tmp_path / "view")
    files = {p.relative_to(view).as_posix() for p in view.rglob("*") if p.is_file()}
    assert not any("query" in p or "validation" in p or "adaptation/" in p for p in files)
    assert metadata(view)["splits"] == ["support", "rehearsal/base/train"]
    assert vocabulary(view) == vocabulary(queries)
    solution = tmp_path / "solution"
    shutil.copytree(Path(__file__).parents[1] / "labs/visual_queries/starter", solution)
    script = solution / "train.py"
    script.write_text(
        script.read_text().replace(
            "    state.update(",
            '    assert state["episode_id"] is None, "Episode state leaked from the previous adaptation"\n    state.update(',
        )
    )
    run(solution, queries, "fixture", 0, run_dir=tmp_path / "train")
    before = digest(solution / "artifacts/state.json")
    result = evaluate_solution(solution, queries, run_dir=tmp_path / "episodes")
    assert result["questions"] == 5 and result["correct"] == 2 and result["score"] == 40
    assert digest(solution / "artifacts/state.json") == before
    states = [
        json.loads(p.read_text())
        for p in (tmp_path / "episodes").rglob("state.json")
        if p.parent.name == "artifacts"
    ]
    assert {s["episode_id"] for s in states} == {"a", "b"}
    calls = []

    def incomplete(split, name, progress):
        calls.append(name)
        if name == "a":
            raise RuntimeError("broken model")
        return dict(questions=2, correct=1)

    with pytest.raises(RuntimeError):
        run_benchmark(queries, incomplete)
    assert calls == [None, "a"]


def test_prediction_boundary_rejects_coercion(queries):
    args = batch(load_split(queries, "base/train"), np.array([0]))
    validate_inputs(*args, vocab_size=7, image_shape=[16, 16, 3])
    for bad in (
        np.array([True]),
        np.array([1.0]),
        np.array([0]),
        np.array([7]),
        np.array([[1]]),
    ):
        with pytest.raises(CourseError):
            validate_answers(bad, 1, 7)
    args[2][0, 1] = False
    with pytest.raises(CourseError):
        validate_inputs(*args, vocab_size=7, image_shape=[16, 16, 3])
