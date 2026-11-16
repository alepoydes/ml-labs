# Imported pytest fixtures are intentionally rebound by injection below.
# ruff: noqa: F811
import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_gui import app, wait_for  # noqa: F401
from test_visual_queries import queries  # noqa: F401

from coursekit import recording
from coursekit.common import CourseError
from coursekit.labs.visual_queries import evaluate
from coursekit.labs.visual_queries_inspection import (
    Dataset,
    new_record,
    predict_batch,
    read_record,
)


def test_batch_snapshots_replay_and_plain_grading(queries, tmp_path):
    data = Dataset(queries, "base/validation")

    class Model:
        calls = 0

        def predict(self, images, tokens, mask):
            self.calls += 1
            self.result = np.where(images[:, 0, 0, 0] == 255, 1, 2).astype(np.int64)
            return self.result

        def debug(self, index=0):
            p = np.zeros(7)
            p[self.result[index]] = 1
            return dict(index=index, calls=self.calls, answer_probabilities=p)

    model = Model()
    numbers = np.array([1, 2])
    item = predict_batch(model, data.arguments(numbers), data.settings)
    assert model.calls == 1
    assert [v["index"] for v in item["debug"]] == [0, 1]
    assert item["answers"].tolist() == [2, 1]
    model.predict(*data.arguments(numbers[::-1]))
    assert item["debug"][1]["answer_probabilities"][1] == 1
    record = new_record()
    record["runs"] = [data.new_run()]
    record["runs"][0]["batches"] = [item | dict(rows=numbers)]
    path = tmp_path / "answers.npz"
    recording.save(path, record)
    replay, sources = read_record(path)
    assert replay["runs"][0]["batches"][0]["debug"][1]["calls"] == 1
    np.testing.assert_array_equal(sources[0].image(1), data.image(1))

    def forbidden(*args):
        raise AssertionError("Ordinary evaluation must not call debug")

    model.debug = forbidden
    saved = []
    assert evaluate(model, queries, on_batch=saved.append)["score"] == 100
    assert saved[0]["rows"].tolist() == [1, 2]
    changed = queries / "base/validation/examples.csv"
    changed.write_text(changed.read_text().replace("empty?", "Is empty?"))
    with pytest.raises(CourseError, match="missing or changed"):
        read_record(path)


def test_viewer_selects_every_batch_row_without_inference_and_preserves_replay(
    app, queries, tmp_path
):
    from PySide6.QtCore import Qt

    from coursekit.gui.visual_queries import VisualQueriesPage

    solution = tmp_path / "solution"
    shutil.copytree(Path(__file__).parents[1] / "labs/visual_queries/starter", solution)
    (solution / "artifacts").mkdir()
    (solution / "solution.py").write_text("""
import numpy as np
class Solution:
    def __init__(self, *args, **kwargs): self.calls = 0
    def predict(self, images, tokens, mask):
        self.calls += 1
        assert len(images) == 2
        return np.array([1, 2], dtype=np.int64)
    def debug(self, index=0):
        p = np.zeros(7)
        p[index + 1] = 1
        return {"answer_probabilities": p, "row": index, "calls": self.calls,
                "matrix": np.array([[index, 1], [2, 3]])}
""")
    page = VisualQueriesPage(tmp_path, data=queries)
    page.resize(1100, 750)
    page.show()

    def select_split(name):
        index = page.split.findData(name)
        assert index >= 0
        assert page.split.itemData(index, Qt.ItemDataRole.ToolTipRole) == name
        page.split.setCurrentIndex(index)
        wait_for(
            app,
            lambda: page.dataset.split == name and not page.preparation.isRunning(),
        )
        assert page.preparation.error is None
        assert page.split.currentData() == name

    try:
        wait_for(app, lambda: page.dataset is not None)
        index = page.split.findData("adaptation/b/query")
        assert page.split.itemText(index) == "Adaptation 02 · query"
        select_split("adaptation/b/query")
        select_split("base/validation")
        page.chooser.path.setText(str(solution))
        page.start.click()
        wait_for(
            app,
            lambda: (
                page.job is not None
                and not page.job.isRunning()
                and len(page.predictions) == 2
            ),
        )
        page.table.selectRow(1)
        wait_for(
            app,
            lambda: (
                page.diagnostics.snapshot.get("row") == 1 and not page.image_job.isRunning()
            ),
        )
        assert page.diagnostics.snapshot["calls"] == 1
        assert "row 2/2" in page.answer.text() and "Model: no" in page.answer.text()
        page.table.selectRow(0)
        assert page.diagnostics.snapshot["row"] == 0
        assert page.diagnostics.snapshot["calls"] == 1
        path = tmp_path / "saved.npz"
        recording.save(path, page.record)
        assert page.job.complete
        select_split("base/train")
        select_split("adaptation/b/query")
        assert page.split.currentText() == "Adaptation 02 · query"
        assert not page.predictions
        page.load_record(path)
        wait_for(app, lambda: bool(page.datasets))
        assert page.split.currentText() == "Base · validation"
        assert page.model.rowCount() == 2
        page.table.selectRow(1)
        assert page.diagnostics.snapshot["row"] == 1
        invalid = tmp_path / "invalid.npz"
        recording.save(invalid, dict(lab="other", version=1))
        page.load_record(invalid)
        wait_for(app, lambda: page.preparation.error is not None and page.start.isEnabled())
        assert page.diagnostics.snapshot["row"] == 1 and page.model.rowCount() == 2
        # Valid arrays do not make malformed model/run metadata safe to install.
        previous = page.record
        for target, key, value in (
            ("run", "model", []),
            ("model", "version", 7),
            ("run", "complete", "yes"),
            ("record", "complete", 1),
        ):
            bad = copy.deepcopy(previous)
            container = (
                bad
                if target == "record"
                else bad["runs"][0]["model"]
                if target == "model"
                else bad["runs"][0]
            )
            container[key] = value
            recording.save(invalid, bad)
            page.load_record(invalid)
            wait_for(app, lambda: not page.preparation.isRunning())
            app.processEvents()
            assert page.preparation.error is not None
            assert page.record is previous
            assert page.diagnostics.snapshot["row"] == 1
            assert page.model.rowCount() == 2
        # A blocked prediction must remain cancellable, without losing completed runs.
        (solution / "solution.py").write_text("""
import time
from pathlib import Path
class Solution:
    def __init__(self, artifacts, *args, **kwargs): self.artifacts = Path(artifacts)
    def predict(self, *args):
        (self.artifacts / "entered").touch()
        time.sleep(60)
""")
        page.start.click()
        wait_for(app, lambda: (solution / "artifacts/entered").exists())
        page.stop_button.click()
        wait_for(app, lambda: not page.job.isRunning() and page.start.isEnabled())
        assert not page.job.complete and page.job.error is None
        select_split("base/train")
        select_split("adaptation/a/support")
        assert page.split.currentText() == "Adaptation 01 · support"
    finally:
        page.shutdown()
        page.close()
