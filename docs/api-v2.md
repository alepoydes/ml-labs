# Training several components and selecting a device

Blindjack and Line Following accept API v1 and v2.
[Visual Queries](../labs/visual_queries/README.md) uses [API 3 numeric data](api-v3.md)
with adaptation in every evaluation. The Blindjack and Line Following starters and
baselines use API 1. Choose API 2 for these labs when you need
explicit devices, separately trained components, reusable snapshots or independent
learning histories. Each lab's statement defines its observations, acting method
and score for both versions.

Run the commands below from the directory containing `pyproject.toml`,
`step_2_train`, `labs/` and `examples/`.

## Entry points

Keep `schema_version = 1` and set `api_version = 2` in `submission.toml`.
Optional named stages are declared as follows:

```toml
[training]
stages = ["encoder", "member"]
supports_adapt = true # Only when train.py implements adapt.
```

`all` is reserved: it always produces a complete solution. Names use lowercase
letters, digits, underscores and hyphens, start with a letter, and have at most
64 characters. Without this table only `all` is available. `supports_adapt` is an
optional Boolean (default false) declaring that `train.py` implements `adapt`.
It enables fine-tuning of a separate copy. Blindjack and Line Following evaluation
uses the saved model directly; adaptation is a separate development operation.

```python
def train(
    data_dir, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
):
    # Save every component needed for inference inside output_dir.
    return None


class Solution:
    def __init__(self, artifact_dir, config, seed, *, device):
        # Load saved components using the selected device.
        ...
```

`train.py` defines `train`; `solution.py` defines `Solution`. `device` is `cpu` or
an explicit `cuda:N`. Use `map_location=device` when loading PyTorch checkpoints,
move neural modules and their inputs to that device, and save state dictionaries.
Do not serialize a live model as the training return value. Several weights files,
fitted preprocessing and an ensemble combiner may share one artifact directory.

Blindjack returns a Boolean NumPy array from `select(images)`. Line Following
receives CPU observation tensors and returns CPU float32 commands from
`act(observation, reset)`. Move inputs internally when useful; robot physics stays
on the CPU. Gradients remain available in a controller call.

API v1 has no device keyword and remains a CPU contract. Version dispatch is
explicit: an exception inside a v2 constructor is not retried as a v1 call.

## A complete example

[The composite Blindjack example](../examples/composite_blindjack/README.md) learns
one small encoder, two predictors and combines their logits. It demonstrates both
a single complete training call and independent stages. Its small architecture
illustrates component ownership, checkpoint loading and prediction combination.

```sh
cp -R examples/composite_blindjack work/blindjack/composite-example
./step_2_train work/blindjack/composite-example
./step_3_evaluate work/blindjack/composite-example
./step_4_pack work/blindjack/composite-example
```

Create `work/blindjack/` first if it does not exist. Windows users can copy the
example directory in their file manager and use the `.cmd` wrappers.

For separate stages, pass `--stage encoder`, then train members with
`--stage member --input encoder=<printed-snapshot>/artifacts`. Each command prints
its unique run directory. Assembly uses `--stage all` and repeated `--input`
arguments. Input names are chosen by the solution, not by the framework.
The same operations are available from Python:

```python
from pathlib import Path
from coursekit.training import train_local, load_solution, load_training

root = Path.cwd()
solution = root / "work/blindjack/composite-example"
shared = train_local(solution, root, stage="encoder")
a = train_local(solution, root, seed=1, stage="member", inputs={"encoder": shared})
b = train_local(solution, root, seed=2, stage="member", inputs={"encoder": shared})
complete = train_local(
    solution, root, inputs={"encoder": shared, "member_1": a, "member_2": b}
)
controller = load_solution(solution, device="cpu")
history, record = load_training(solution, a.parent)
```

Every call receives a new empty output directory. Inputs are copied into the run
before training, authenticated by hashes, and checked again afterwards; do not
modify them. A named stage retains an intermediate result and does not replace
`artifacts/`. A successful `all` installs the complete result under `artifacts/`.
A failed call preserves the previous complete selection. Source/configuration
edits still require successful preparation before packaging. Artifacts must be
self-contained regular files; references to previous run directories are invalid.

Runs are stored under `runs/<lab>/<variant>/<run-id>/`. `training.json` records
operation, stage, input hashes, code, configuration, data identity, device/runtime,
seed, completion, elapsed time and output hashes. `inputs/` retains the pinned
parents; `artifacts/` retains the output. Scratch files and histories are not packed.
The existing archive size/file limits continue to apply to the complete solution.

## Independent histories and live progress

Return either `None` or a dictionary grouped by component or stage:

```python
import numpy as np

history = {
    "encoder": {"step": np.array([1, 2]), "loss": np.array([0.8, 0.5])},
    "member": {"step": np.array([1]), "accuracy": np.array([0.4])},
}
```

Each `step` is a strictly increasing, nonnegative integer vector. Other columns
are finite real numeric vectors of the same length within that group. Groups may
have different lengths. Names follow the stage-name rule. Each group is saved as
compressed NPZ under `history/`; `load_training` verifies its hashes and the run's
saved artifacts before returning `(history, record)`.

For a notebook or another caller's progress display, pass `report=callback` to
`train_local`. Student code may call the supplied reporter:

```python
if report:
    report({"stage": "encoder", "step": 2, "total": 5, "metrics": {"loss": 0.5}})
```

The runner validates monotone counters, finite metrics (at most 32) and an 8 KiB
payload limit. It forwards at most ten ordinary updates per second per stage;
completion updates are also forwarded. `total` and `metrics` are optional. These
are training diagnostics, not evaluation results. Callback exceptions propagate,
so a local caller can interrupt cooperatively. Local calls do not impose a sandbox.

## CUDA environment

CPU is the default uv dependency group. CUDA is a mutually exclusive extra;
select it explicitly on **every** CUDA command:

```sh
uv sync --locked --no-group cpu --extra cuda
uv run --locked --no-group cpu --extra cuda python -m coursekit.cli train work/blindjack/composite-example --device cuda:0
uv run --locked --no-group cpu --extra cuda python -m coursekit.cli evaluate work/blindjack/composite-example --device cuda:0
```

The ordinary numbered wrappers select the CPU environment. A `--device cuda:0`
flag alone cannot turn CPU PyTorch into a CUDA build. Returning to the wrappers or
`uv sync --locked` restores the CPU profile. The locked CUDA wheel and compatible
NVIDIA driver are required; an unavailable device raises an error without fallback.
This profile is currently exercised on Linux x86-64. The framework follows
[uv's documented PyTorch index selection](https://docs.astral.sh/uv/guides/integration/pytorch/).
Installing the project with plain pip is not the course environment installation:
use uv so that the chosen CPU group or CUDA extra is included.

Instructor device selection is a separate published assessment profile; students
cannot request more grader resources through their solution configuration. Existing
assessments remain CPU by default. CUDA availability alone does not make either
lab faster, particularly the CPU robot simulator.

## Local adaptation entry point

An adaptive solution may additionally define this function in `train.py`:

```python
def adapt(
    data_dir,
    input_dir,
    output_dir,
    config,
    seed,
    *,
    device,
    limits,
    checkpoint=None,
    report=None,
):
    # Read the complete base from input_dir; save a complete adapted state in output_dir.
    return None
```

Blindjack and Line Following evaluate saved controllers without calling `adapt`.
You can implement it to fine-tune a copy locally or through the instructor's
learning tools.
`coursekit.training.adapt_local(solution, data, base, run_dir=..., limits=...,
 device="cpu", seed=0)` is available for local development. Its histories and
progress follow `train`. It retains a new independently loadable snapshot without
installing it over the submitted base. Load it with
`load_solution(solution, artifact_dir=adapted, device="cpu")`.

After saving every component to `output_dir`, call the optional `checkpoint()`
callback to commit a complete snapshot, or `checkpoint(history)` to include numeric
histories. A time-limited adaptation can use its last committed snapshot. Code
errors and cancellation still fail the operation. Existing functions without this
keyword remain supported. See the shared
[time and checkpoint contract](api-v3.md#adaptation-time-and-checkpoints).

`limits` contains positive `wall_seconds`, `memory_mb`, `artifact_bytes`, and a
nonnegative integer `gpu_memory_mb` (zero for CPU). Local execution validates this
profile and output artifact size; it does **not** enforce wall time, host memory or
VRAM through operating-system isolation. Save these keys in a TOML file to use:

```sh
uv run --locked python -m coursekit.cli adapt work/blindjack/my-adaptive-model --data data/practice --base work/blindjack/my-adaptive-model/artifacts --runs runs/adaptation --limits practice-limits.toml
```

Only invoke local helpers on your own trusted code. Visual Queries has a separate
[numeric adaptation contract](api-v3.md); its evaluator runs independent episodes
automatically as part of every complete assessment.
