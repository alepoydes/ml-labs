"""Prepared episode sets and local/evaluator rollouts for Line Following."""

from pathlib import Path

import numpy as np

from coursekit.common import file_hashes, fingerprint
from coursekit.labs.line_following_cases import load_cases
from coursekit.labs.line_following_env import Environment, validate_action
from coursekit.labs.line_following_results import episode_metrics


def benchmark(data: Path, split="validation"):
    return load_cases(data, split)


def data_fingerprint(data: Path) -> str:
    return fingerprint(
        {k: v for k, v in file_hashes(data).items() if not k.startswith(".")}
    )


def evaluate(
    solution,
    data: Path,
    split="validation",
    *,
    compiled=True,
    time_limit=None,
    progress=None,
):
    """Run the entire fixed set; any numerical or controller error aborts the result.

    Local solutions reset through the normal mask. The instructor proxy additionally
    creates a fresh isolated worker for each episode, with the same acting API.
    """
    episodes = benchmark(data, split)
    settings = {} if time_limit is None else {"time_limit": time_limit}
    env = Environment([episodes[0]], compiled=compiled, **settings)
    scores, times, statuses, completion = [], [], [], []
    for index, case in enumerate(episodes):
        if index:
            env.reset(cases=[case])
        if progress:
            progress(
                {
                    "case": index + 1,
                    "cases": len(episodes),
                    "phase": "preparing physics / controller",
                }
            )
        steps = 0
        try:
            while not env.done.all():
                action = solution.act(env.observe(), env.first)
                validate_action(action, 1)
                env.step(action)
                steps += 1
                if progress and (steps % 25 == 0 or env.done.all()):
                    progress(
                        {
                            "case": index + 1,
                            "cases": len(episodes),
                            "seconds": float(env.info()["time"][0]),
                        }
                    )
        except BaseException:
            if progress:
                progress(
                    {
                        "case": index + 1,
                        "cases": len(episodes),
                        "metrics": episode_metrics(env.info())
                        | {"status": "interrupted", "score": None},
                    }
                )
            raise
        info = env.info()
        scores.append(float(info["score"][0]))
        statuses.append(str(info["status"][0]))
        completion.append(float(info["completion_fraction"][0]))
        if progress:
            progress(
                {
                    "case": index + 1,
                    "cases": len(episodes),
                    "metrics": episode_metrics(info),
                }
            )
        if statuses[-1] == "finished":
            times.append(float(info["time"][0]))
    return {
        "episodes": len(episodes),
        "score": float(np.mean(scores)),
        "completion_rate": statuses.count("finished") / len(episodes),
        "mean_completion_fraction": float(np.mean(completion)),
        "fall_rate": statuses.count("fallen") / len(episodes),
        "timeout_rate": statuses.count("timeout") / len(episodes),
        "mean_success_time": float(np.mean(times)) if times else None,
    }
