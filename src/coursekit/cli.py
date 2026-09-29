import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from filelock import FileLock, Timeout

from coursekit import resources, submissions, training
from coursekit.common import CourseError, write_json
from coursekit.labs import require_lab
from coursekit.labs.blindjack import evaluate
from coursekit.runtime import EvaluationError, Worker


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
            variant = args.template if args.variant is None else args.variant
            raw = (
                (template / "submission.toml")
                .read_text()
                .replace('variant = "template"', f"variant = {json.dumps(variant)}")
            )
            submissions.metadata(raw.encode())
            destination = root / "work" / args.lab / variant
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
            with FileLock(str(data / ".update.lock"), timeout=0):
                resource_hash = resources.ready(data, spec)
                worker, log = None, b""
                log_path = run_dir / f"{args.step}.log"
                try:
                    with Worker(
                        solution,
                        args.seed,
                        container=args.step == "check",
                        image=getattr(args, "image", "mlcourse-reference:local"),
                    ) as worker:
                        result = evaluate(worker, data)
                except EvaluationError as exc:
                    log = f"{exc.kind}: {exc}\n".encode() + exc.log
                    if exc.log:
                        sys.stderr.write(exc.log.decode(errors="replace"))
                    raise
                finally:
                    if not log and worker is not None:
                        log = bytes(worker.log)
                    log_path.write_bytes(log)
                    print(f"Worker log: {log_path}", file=sys.stderr)
                result["resource_hash"] = resource_hash
                write_json(result_path, result)
                print(json.dumps(result, indent=2))
                print(
                    "Submission check passed."
                    if args.step == "check"
                    else "Local evaluation complete (not isolated)."
                )
    except (CourseError, OSError, ValueError, KeyError, Timeout) as exc:
        if args.step == "train" and not isinstance(exc, (CourseError, Timeout)):
            traceback.print_exc()
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
