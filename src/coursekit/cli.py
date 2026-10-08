import argparse
import csv
import json
import tempfile
from pathlib import Path

from filelock import FileLock, Timeout

from coursekit import resources, submissions, training
from coursekit.common import CourseError, write_json
from coursekit.labs import get_lab


def main():
    parser = argparse.ArgumentParser(
        description="Course solution workflow (paths relative to this project)."
    )
    commands = parser.add_subparsers(dest="step", required=True)
    commands.add_parser("update")
    new = commands.add_parser("new", help="Copy a lab template into work/.")
    new.add_argument("lab")
    new.add_argument(
        "variant", nargs="?", help="Name of the working copy (default: template name)."
    )
    new.add_argument(
        "--template",
        choices=["starter", "baseline"],
        default="starter",
        help="Template to copy from labs/<lab>/ (default: %(default)s).",
    )
    for step in ("train", "evaluate", "pack", "check"):
        command = commands.add_parser(step)
        command.add_argument("path", type=Path)
        if step in ("train", "evaluate", "check"):
            command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path.cwd()
    try:
        if args.step == "update":
            for path in sorted((root / "labs").glob("*/resources.toml")):
                resources.sync(root / "data" / path.parent.name, resources.manifest(path))
            return
        if args.step == "new":
            print(training.new_solution(root, args.lab, args.template, args.variant))
            return
        if args.step == "pack":
            submissions.pack(args.path, root / "dist")
            return
        if args.step == "train":
            training.train_local(args.path, root, args.seed)
            return
        with tempfile.TemporaryDirectory(prefix="course-check-") as temporary:
            solution = (
                Path(temporary) / "solution"
                if args.step == "check"
                else args.path.resolve()
            )
            info = (
                submissions.extract(args.path, solution)
                if args.step == "check"
                else submissions.metadata((solution / "submission.toml").read_bytes())
            )
            data = root / "data" / info["lab"]
            spec = resources.manifest(root / "labs" / info["lab"] / "resources.toml")
            run_dir = root / "runs" / info["lab"] / info["variant"]
            run_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_dir / f"{args.step}.json"
            result_path.unlink(missing_ok=True)
            cases = []
            case_path = run_dir / f"{args.step}.cases.json"
            csv_path = case_path.with_suffix(".csv")
            robot = info["lab"] == "line_following"
            if robot:
                case_path.unlink(missing_ok=True)
                csv_path.unlink(missing_ok=True)

            def record_case(event):
                if "metrics" in event:
                    cases.append({"case": event["case"], **event["metrics"]})

            with FileLock(str(data / ".update.lock"), timeout=0):
                resource_hash = resources.ready(data, spec)
                complete = False
                failure = None
                try:
                    instance = training.load_solution(solution, args.seed)
                    result = get_lab(info["lab"]).evaluate(
                        instance, data, **({"progress": record_case} if robot else {})
                    )
                    complete = True
                except BaseException as exc:
                    failure = {"type": type(exc).__name__, "message": str(exc)}
                    raise
                finally:
                    if robot:
                        from coursekit.labs.line_following_results import (
                            EPISODE_COLUMNS,
                            text_table,
                        )

                        write_json(
                            case_path,
                            {"complete": complete, "cases": cases, "error": failure},
                        )
                        with csv_path.open("w", encoding="utf-8", newline="") as stream:
                            fields = [*EPISODE_COLUMNS, "reference_time", "outward_time"]
                            writer = csv.DictWriter(stream, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(cases)
                        print(text_table(cases))
                        print(f"Episode diagnostics: {case_path}\nCSV: {csv_path}")
                        if not complete:
                            print(
                                "Incomplete evaluation; diagnostics have no official aggregate score."
                            )
                result["resource_hash"] = resource_hash
                write_json(result_path, result)
                print(json.dumps(result, indent=2))
                print(
                    "Local archive check passed in your Python environment."
                    if args.step == "check"
                    else "Local evaluation complete."
                )
    except (CourseError, Timeout) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
