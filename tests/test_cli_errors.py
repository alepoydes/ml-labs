import os
import subprocess
import sys

import pytest

from coursekit.submissions import pack
from coursekit.training import train


@pytest.mark.parametrize("step", ["evaluate", "check"])
@pytest.mark.parametrize(
    "source,detail,traceback",
    [
        (
            "raise ImportError('missing student dependency')\n",
            "missing student dependency",
            True,
        ),
        (
            "class Solution:\n def __init__(self,*args):\n"
            "  print('loading artifacts')\n  raise ValueError('bad artifact shape')\n",
            "bad artifact shape",
            True,
        ),
        (
            "class Solution:\n def __init__(self,*args): pass\n"
            " def select(self,images):\n  print('selecting cards')\n"
            "  raise RuntimeError('cannot predict these images')\n",
            "cannot predict these images",
            True,
        ),
        (
            "class Solution:\n def __init__(self,*args): pass\n"
            " def select(self,images): return [True] * 7\n",
            "Boolean NumPy array",
            False,
        ),
    ],
)
def test_cli_reports_original_error(
    student_project, solution, step, source, detail, traceback
):
    (solution / "solution.py").write_text(source)
    train(solution, student_project / "data/blindjack", "fixture", 0)
    path = pack(solution, student_project / "dist") if step == "check" else solution
    run_dir = student_project / "runs/blindjack/template"
    run_dir.mkdir(parents=True)
    result_path = run_dir / f"{step}.json"
    result_path.write_text('{"score": 100}')
    result = subprocess.run(
        [sys.executable, "-m", "coursekit.cli", step, str(path)],
        cwd=student_project,
        # The ordinary Python entry point needs no commands from PATH.
        env=os.environ | {"PATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert detail in result.stderr
    if traceback:
        assert "Traceback (most recent call last)" in result.stderr
        assert "solution.py" in result.stderr
    if "print(" in source:
        assert "loading artifacts" in result.stdout or "selecting cards" in result.stdout
    assert not result_path.exists()
