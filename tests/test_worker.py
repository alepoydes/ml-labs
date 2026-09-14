import numpy as np
import pytest

from coursekit.labs.blindjack import evaluate
from coursekit.runtime import EvaluationError, Worker


def test_worker_reproducibility_and_logging(solution, data):
    with Worker(solution, container=False) as first:
        result = evaluate(first, data)
    with Worker(solution, container=False) as second:
        assert evaluate(second, data) == result
    assert 45 < result["score"] < 75


@pytest.mark.parametrize(
    "implementation,kind",
    [
        ("return np.ones(7, dtype=int)", "invalid_output"),
        ("raise RuntimeError('boom')", "runtime_exception"),
        ("import time; time.sleep(10)", "timeout"),
    ],
)
def test_worker_failures(solution, implementation, kind):
    (solution / "solution.py").write_text(
        "import numpy as np\nclass Solution:\n def __init__(self,*args): pass\n def select(self,images):\n  "
        + implementation
        + "\n"
    )
    with Worker(solution, container=False, limits={"hand_seconds": 0.2}) as worker:
        with pytest.raises(EvaluationError) as error:
            worker.select(np.zeros((7, 128, 128, 3), np.uint8))
        assert error.value.kind == kind
