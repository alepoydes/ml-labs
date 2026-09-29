import sys

import pytest

from coursekit import cli
from coursekit.runtime import Worker
from coursekit.submissions import pack
from coursekit.training import train


@pytest.mark.parametrize("step", ["evaluate", "check"])
@pytest.mark.parametrize(
    "source,kind,detail",
    [
        (
            "raise ImportError('missing student dependency')\n",
            "model_loading_failure",
            "missing student dependency",
        ),
        (
            "class Solution:\n def __init__(self,*args):\n"
            "  print('loading artifacts')\n  raise ValueError('bad artifact shape')\n",
            "model_loading_failure",
            "bad artifact shape",
        ),
        (
            "class Solution:\n def __init__(self,*args): pass\n"
            " def select(self,images):\n  print('selecting cards')\n"
            "  raise RuntimeError('cannot predict these images')\n",
            "runtime_exception",
            "cannot predict these images",
        ),
        (
            "class Solution:\n def __init__(self,*args): pass\n"
            " def select(self,images): return [True] * 7\n",
            "invalid_output",
            "Boolean NumPy array",
        ),
    ],
)
def test_cli_reports_worker_traceback(
    student_project, solution, monkeypatch, capsys, step, source, kind, detail
):
    (solution / "solution.py").write_text(source)
    train(solution, student_project / "data/blindjack", "fixture", 0)
    path = pack(solution, student_project / "dist") if step == "check" else solution
    containers = []

    def local_worker(*args, container, **kwargs):
        # Exercise extraction and the real subprocess protocol without a daemon.
        containers.append(container)
        return Worker(*args, container=False, **kwargs)

    monkeypatch.setattr(cli, "Worker", local_worker)
    monkeypatch.chdir(student_project)
    monkeypatch.setattr(sys, "argv", ["coursekit", step, str(path)])
    run_dir = student_project / "runs/blindjack/template"
    run_dir.mkdir(parents=True)
    result = run_dir / f"{step}.json"
    result.write_text('{"score": 100}')
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    stderr = capsys.readouterr().err
    assert "Traceback (most recent call last)" in stderr
    assert detail in stderr
    assert "Worker log:" in stderr
    assert "solution.py" in stderr or kind == "invalid_output"
    log = (run_dir / f"{step}.log").read_text()
    assert detail in log and kind in log
    assert "Traceback (most recent call last)" in log
    assert containers == [step == "check"]
    assert not result.exists()
