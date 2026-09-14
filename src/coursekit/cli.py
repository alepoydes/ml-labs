import argparse
import json
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock, Timeout

from coursekit import resources, submissions, training
from coursekit.common import CourseError, write_json
from coursekit.labs import require_lab
from coursekit.labs.blindjack import evaluate
from coursekit.runtime import Worker


def main():
    parser = argparse.ArgumentParser(
        description="Course solution workflow (paths relative to this project)."
    )
    commands = parser.add_subparsers(dest="step", required=True)
    commands.add_parser("update")
    new = commands.add_parser("new")
    new.add_argument("lab")
    new.add_argument("variant")
    new.add_argument("--template", choices=["starter", "baseline"], default="starter")
    for step in ("train", "evaluate", "pack", "check"):
        command = commands.add_parser(step)
        command.add_argument("path", type=Path)
        if step in ("train", "evaluate", "check"):
            command.add_argument("--seed", type=int, default=0)
        if step == "check":
            command.add_argument("--image", default="mlcourse-reference:local")
    args = parser.parse_args()
    root = Path.cwd()
    try:
        if args.step == "update":
            for path in sorted((root / "labs").glob("*/resources.toml")):
                resources.sync(root / "data" / path.parent.name, resources.manifest(path))
            return
        if args.step == "new":
            require_lab(args.lab)
            template = root / "labs" / args.lab / args.template
            raw = (
                (template / "submission.toml")
                .read_text()
                .replace('variant = "template"', f'variant = "{args.variant}"')
            )
            submissions.metadata(raw.encode())
            destination = root / "work" / args.lab / args.variant
            if destination.exists():
                raise CourseError(f"Solution already exists: {destination}")
            shutil.copytree(
                template, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
            (destination / "submission.toml").write_text(raw)
            print(destination)
            return
        if args.step == "pack":
            submissions.pack(args.path, root / "dist")
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
            with FileLock(str(data / ".update.lock"), timeout=0):
                resource_hash = resources.ready(data, spec)
                if args.step == "train":
                    training.train(solution, data, resource_hash, args.seed)
                else:
                    with Worker(
                        solution,
                        args.seed,
                        container=args.step == "check",
                        image=getattr(args, "image", "mlcourse-reference:local"),
                    ) as worker:
                        result = evaluate(worker, data)
                    result["resource_hash"] = resource_hash
                    write_json(
                        root / "runs" / info["lab"] / info["variant"] / f"{args.step}.json",
                        result,
                    )
                    print(json.dumps(result, indent=2))
                    print(
                        "Submission check passed."
                        if args.step == "check"
                        else "Local evaluation complete (not isolated)."
                    )
    except (CourseError, OSError, ValueError, KeyError, Timeout) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
