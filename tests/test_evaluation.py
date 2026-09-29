import os

import pytest

from coursekit.common import CourseError
from coursekit.labs.blindjack import evaluate
from coursekit.training import load_solution


def test_direct_evaluation_is_reproducible(solution, data):
    result = evaluate(load_solution(solution), data)
    assert evaluate(load_solution(solution), data) == result
    assert 45 < result["score"] < 75


@pytest.mark.parametrize(
    "implementation,error,detail",
    [
        ("return np.ones(7, dtype=int)", CourseError, "Boolean NumPy array"),
        ("raise RuntimeError('boom')", RuntimeError, "boom"),
    ],
)
def test_original_exceptions(solution, data, implementation, error, detail):
    (solution / "solution.py").write_text(
        "import numpy as np\nclass Solution:\n def __init__(self,*args): pass\n"
        " def select(self,images):\n  " + implementation + "\n"
    )
    with pytest.raises(error, match=detail):
        evaluate(load_solution(solution), data)


def test_solution_runs_in_current_process_with_live_output(solution, data, capsys):
    (solution / "solution.py").write_text(
        "import os\nimport numpy as np\nclass Solution:\n"
        " def __init__(self,*args):\n  print('loading my model')\n"
        " def select(self,images):\n"
        f"  assert os.getpid() == {os.getpid()}\n"
        "  print('my prediction')\n  return np.ones(7, dtype=bool)\n"
    )
    instance = load_solution(solution)
    assert "loading my model" in capsys.readouterr().out
    evaluate(instance, data)
    assert capsys.readouterr().out.count("my prediction") == 330
