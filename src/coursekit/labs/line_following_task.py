"""Prepared episode sets and local/evaluator rollouts for Line Following."""

import csv
from pathlib import Path

import numpy as np

from coursekit.common import CourseError, file_hashes, fingerprint
from coursekit.labs.line_following_course import load_courses
from coursekit.labs.line_following_env import Environment, validate_action
from coursekit.labs.line_following_results import episode_metrics


def benchmark(data: Path, split="validation"):
    if split not in {"train", "validation", "hidden"}:
        raise CourseError("Unknown episode split.")
    courses = {c.name: c for c in load_courses(data / "courses.npz")}
    episodes = []
    with (data / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["course", "seed"]:
            raise CourseError("Expected episode CSV header: course,seed.")
        for row in reader:
            if (
                len(row) != 2
                or row[0] not in courses
                or not row[1].isascii()
                or not row[1].isdecimal()
                or int(row[1]) >= 2**63
            ):
                raise CourseError("Invalid course or seed in episode CSV.")
            episodes.append((courses[row[0]], int(row[1])))
    if not episodes or len({(c.name, s) for c, s in episodes}) != len(episodes):
        raise CourseError("Expected a nonempty set of distinct episodes.")
    return episodes


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
    env = Environment([episodes[0][0]], [episodes[0][1]], compiled=compiled, **settings)
    scores, times, statuses, completion = [], [], [], []
    for index, (course, seed) in enumerate(episodes):
        if index:
            env.reset(courses=[course], seeds=[seed])
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
