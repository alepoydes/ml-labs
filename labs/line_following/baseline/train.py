import json
from pathlib import Path


def train(data_dir: Path, output_dir: Path, config: dict, seed: int) -> None:
    # This reference is hand tuned. Replace this procedure with learning or fitting.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "controller.json").write_text(json.dumps(config, indent=2) + "\n")
