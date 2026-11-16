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
from coursekit.labs.visual_queries_data import (
    batch,
    load_glyphs,
    load_split,
    metadata,
    vocabulary,
)
from coursekit.labs.visual_queries_learning import (
    LearningData,
    decode,
    encode,
    open_data,
    provider,
)
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
            format_version=2,
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
        (folder / "images").mkdir(parents=True)
        pixels = np.full((16, 32, 3), 255, np.uint8)
        pixels[:, 16:] = [12, 34, 56]
        Image.fromarray(pixels).save(folder / "images/000001.png")
        records = [[1, 1, 2, "Is empty?", "no"], [1, 1, 1, "empty?", "yes"]]
        if split.endswith("query") and "/b/" in split:
            records = records[:1]
        csv_file(
            folder / "examples.csv",
            ["sheet", "row", "column", "question", "answer"],
            records,
        )
    for name in ("a", "b"):
        csv_file(
            root / f"adaptation/{name}/vocabulary.csv",
            ["token_id", "spelling"],
            enumerate([*words, "novel"]),
        )
        write_json(
            root / f"adaptation/{name}/episode.json",
            dict(
                format_version=2,
                episode_id=name,
                base_vocabulary_sha256=digest(root / "vocabulary.csv"),
                base_vocab_size=7,
                vocab_size=8,
            ),
        )
    (root / "glyphs/images").mkdir(parents=True)
    shutil.copyfile(
        root / "base/train/images/000001.png", root / "glyphs/images/000001.png"
    )
    csv_file(
        root / "glyphs/examples.csv", ["sheet", "row", "column", "symbol"], [[1, 1, 2, "A"]]
    )
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
    image, tokens, answer = module.read_example(queries, "base/train", 1)
    np.testing.assert_array_equal(image, x[0])
    np.testing.assert_array_equal(tokens, t[0])
    assert answer == 2
    path = queries / "base/train/examples.csv"
    csv_file(
        path, ["sheet", "row", "column", "question", "answer"], [[1, 1, 1, "empty?", ""]]
    )
    assert load_split(queries, "base/train")[3] is None
    path.write_text(path.read_text().replace("empty?", "missing?"))
    with pytest.raises(CourseError):
        load_split(queries, "base/train")


def test_glyph_labels_follow_tiles_and_episode_preserves_base_tokens(queries):
    csv_file(
        queries / "glyphs/examples.csv",
        ["sheet", "row", "column", "symbol"],
        [[1, 1, 2, "A"], [1, 1, 1, "yes"]],
    )
    images, labels = load_glyphs(queries)
    assert labels.tolist() == [1, 6]
    assert np.all(images[np.array([1])][0] == [12, 34, 56])
    path = queries / "adaptation/a/vocabulary.csv"
    path.write_text(path.read_text().replace("1,yes", "1,changed"))
    with pytest.raises(CourseError, match="old token"):
        load_split(queries, "adaptation/a/support")


def test_support_sealing_and_complete_independent_lifecycle(queries, tmp_path):
    view = adaptation_view(queries, "a", tmp_path / "view")
    files = {p.relative_to(view).as_posix() for p in view.rglob("*") if p.is_file()}
    assert not any("query" in p or "validation" in p or "adaptation/" in p for p in files)
    assert metadata(view)["splits"] == ["support", "rehearsal/base/train"]
    assert vocabulary(view) == vocabulary(queries)
    solution = tmp_path / "solution"
    shutil.copytree(Path(__file__).parents[1] / "labs/visual_queries/starter", solution)
    (solution / "config.toml").write_text("example_batch_size=2\nconstant_answer_id=1\n")
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
    from coursekit.labs.visual_queries_inspection import read_record

    saved, _ = read_record(result["recording"])
    assert saved["complete"] and len(saved["runs"]) == 3
    assert sum(len(b["rows"]) for r in saved["runs"] for b in r["batches"]) == 5
    assert len({r["model"]["artifacts_hash"] for r in saved["runs"]}) == 3
    assert all("debug" not in b for r in saved["runs"] for b in r["batches"])
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


def test_browser_names_follow_schedule_without_changing_identities(queries):
    from html.parser import HTMLParser

    from coursekit.labs.visual_queries_browser import build

    meta = metadata(queries)
    meta["episodes"] = ["b", "a"]
    write_json(queries / "dataset.json", meta)

    class Payload(HTMLParser):
        reading = False
        text = ""

        def handle_starttag(self, tag, attrs):
            self.reading = tag == "script" and dict(attrs).get("id") == "dataset"

        def handle_endtag(self, tag):
            if tag == "script":
                self.reading = False

        def handle_data(self, value):
            if self.reading:
                self.text += value

    parser = Payload()
    parser.feed(build(queries).read_text())
    data = json.loads(parser.text)
    assert data["labels"]["adaptation/b/query"] == "Adaptation 01 · query"
    assert data["labels"]["adaptation/a/query"] == "Adaptation 02 · query"
    names = list(data["splits"])
    assert names.index("adaptation/b/support") < names.index("adaptation/a/support")
    assert data["splits"]["adaptation/b/query"][0][1][0].startswith(
        "adaptation/b/query/images/"
    )
    assert (
        json.loads((queries / "adaptation/b/episode.json").read_text())["episode_id"] == "b"
    )


def test_cli_archive_check_retains_predictions_and_adaptation(
    queries, tmp_path, monkeypatch, capsys
):
    from coursekit import resources, submissions
    from coursekit.cli import main
    from coursekit.labs.visual_queries_inspection import read_record

    project = tmp_path / "student"
    data = project / "data/visual_queries"
    shutil.copytree(queries, data)
    entries = [
        dict(
            path=p.relative_to(data).as_posix(),
            size=p.stat().st_size,
            sha256=digest(p),
            url=p.as_uri(),
        )
        for p in sorted(data.rglob("*"))
        if p.is_file()
    ]
    lab = project / "labs/visual_queries"
    lab.mkdir(parents=True)
    (lab / "resources.toml").write_text(
        "schema_version = 1\n"
        + "".join(
            "\n[[files]]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in entry.items())
            for entry in entries
        )
    )
    resources.sync(data, resources.manifest(lab / "resources.toml"))
    solution = project / "work/model"
    shutil.copytree(Path(__file__).parents[1] / "labs/visual_queries/starter", solution)
    (solution / "config.toml").write_text("example_batch_size=2\nconstant_answer_id=1\n")
    run(solution, data, "fixture", 0, run_dir=project / "training")
    archive = submissions.pack(solution, project / "dist")
    monkeypatch.chdir(project)
    previous = None
    for _ in range(2):
        monkeypatch.setattr("sys.argv", ["coursekit", "check", str(archive)])
        main()
        saved = project / "runs/visual_queries/template"
        result = json.loads((saved / "check.json").read_text())
        output = capsys.readouterr().out
        assert (
            "Base validation" in output
            and "Adaptation 01" in output
            and "Adaptation 02" in output
        )
        assert "Known language" in output and "New language" in output
        assert "Total" in output and "2/5" in output and "40.00%" in output
        path = Path(result["recording"])
        assert path.is_relative_to(saved / "check")
        record, _ = read_record(path)
        assert record["complete"] and result["correct"] == 2
        assert result["questions"] == 5
        if previous is not None:
            assert path != previous and previous.is_file()
        previous = path
    # Both checks retain one independent model/history snapshot per adaptation.
    snapshots = list((saved / "check").glob("*/training.json"))
    assert len(snapshots) == 4
    for snapshot in snapshots:
        assert json.loads(snapshot.read_text())["status"] == "complete"
        assert (snapshot.parent / "artifacts/state.json").is_file()
        assert (snapshot.parent / "history/adapt.npz").is_file()


def test_learned_baseline_growth_reload_and_adaptation(queries, tmp_path):
    import torch

    from coursekit.common import read_toml
    from coursekit.training import load_module

    solution = tmp_path / "learned"
    shutil.copytree(Path(__file__).parents[1] / "labs/visual_queries/baseline", solution)
    module = load_module(solution, "train")
    model_code = __import__("lib.model", fromlist=["Model"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        model = model_code.Model(7, 96).eval()
        x, t, m = batch(load_split(queries, "base/train"), np.array([0, 1]))
        args = model_code.tensors(x, t, m, "cpu")
        with torch.no_grad():
            expected = model(*args)
            grown = model_code.expand(model, 8).eval()
            torch.testing.assert_close(grown(*args)[:, :7], expected, rtol=0, atol=0)
            torch.testing.assert_close(model(*(a[:1] for a in args)), expected[:1])
        # Learn a newly appended answer class on a tiny support set.
        view = adaptation_view(queries, "a", tmp_path / "learn-view")
        csv_file(
            view / "support/examples.csv",
            ["sheet", "row", "column", "question", "answer"],
            [[1, 1, 1, "empty?", "novel"], [1, 1, 2, "Is empty?", "novel"]],
        )
        source, target = tmp_path / "base", tmp_path / "adapted"
        source.mkdir()
        target.mkdir()
        model_code.save(model, source, None)
        original = digest(source / "model.pt")
        config = read_toml(solution / "config.toml") | dict(
            adapt_epochs=12, adapt_learning_rate=0.01, rehearsal_examples=0
        )
        history = module.adapt(
            open_data(view),
            source,
            target,
            config,
            0,
            device="cpu",
            limits={"wall_seconds": 30},
        )
        assert history["adapt"]["loss"][-1] < history["adapt"]["loss"][0]
        assert digest(source / "model.pt") == original
        predictor = load_module(solution, "solution").Solution(
            target, config, 0, device="cpu"
        )
        assert predictor.episode_id == "a" and predictor.model.vocab_size == 8
        prediction = predictor.predict(x, t, m)
        np.testing.assert_array_equal(prediction, [7, 7])
        np.testing.assert_array_equal(
            predictor.predict(x[::-1], t[::-1], m[::-1]),
            prediction[::-1],
        )
        padded_t, padded_m = (
            np.pad(t[:1], ((0, 0), (0, 4))),
            np.pad(m[:1], ((0, 0), (0, 4))),
        )
        np.testing.assert_array_equal(
            predictor.predict(x[:1], padded_t, padded_m), prediction[:1]
        )
    finally:
        torch.set_num_threads(previous_threads)


def test_numeric_learning_boundary(queries):
    host = provider(queries)

    def transport(message):
        value = host(message)
        return (
            json.loads(json.dumps(value))
            if message["kind"] == "metadata"
            else decode(encode(value))
        )

    data = LearningData(transport)
    assert set(data.splits) == {"base/train", "glyphs"}
    assert "vocabulary" not in data.metadata and "spelling" not in json.dumps(data.metadata)
    local = batch(load_split(queries, "base/train"), np.array([1, 0]))
    remote = batch(data.load_split("base/train"), np.array([1, 0]))
    for a, b in zip(local, remote):
        np.testing.assert_array_equal(a, b)
    for request in [
        dict(kind="table", split="base/validation"),
        dict(kind="images", split="../secret", indices=[0]),
        dict(kind="images", split="base/train", indices=[True]),
    ]:
        with pytest.raises((CourseError, ValueError)):
            transport(request)
