# Machine learning course

Five substantial labs connect machine learning methods with complete working
solutions. Two labs are available with pilot settings; [three more](docs/planned-labs.md)
are planned. Choose the lab assigned by your instructor:

| Lab | Task and API | Inspect before training |
| --- | --- | --- |
| [Blindjack](labs/blindjack/README.md) | Learn image values and select cards with `select(images)`. | Open `data/blindjack/index.html` after resource preparation, or run `./view_lab blindjack`. |
| [Line Following](labs/line_following/README.md) | Train an adaptive robot controller with `act(observation, reset)`. | Run `./view_lab line_following`; choose **Course** in the top row, then **Top view** and **Fit track**. No trained solution is needed to preview courses. |

The [student guide](docs/README.md) links the common development, environment and
submission instructions. Task rules and score definitions live in each lab's statement.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
student repository supplied by your instructor, and enter its root directory.
Commands use one shared locked Python environment.

Each `labs/<lab>/` provides `starter`, a minimal API example, and `baseline`, a
runnable reference. Blindjack's baseline trains an MLP; Line Following's baseline
is hand tuned and its preparation only saves controller settings. Replace or improve
it with your own learning procedure. First copy a template into `work/`, then edit,
train and evaluate that copy. Leave tracked course files unchanged so updates do
not conflict with your work.

```sh
./step_0_update_resources

# Choose the copy for your lab:
./step_1_new_solution blindjack my-model --template baseline
./step_1_new_solution line_following my-controller --template baseline
```

Read your lab's statement and inspect `train.py`, `solution.py`, and `config.toml`
in the working copy. Continue with the same commands for either lab. For the robot:

```sh
./step_2_train work/line_following/my-controller
./step_3_evaluate work/line_following/my-controller
./step_4_pack work/line_following/my-controller
./step_5_check_submission dist/line_following-my-controller-<hash>.zip
```

For Blindjack, use `work/blindjack/my-model` and the ZIP printed by its pack command.
Replace `<hash>` with the actual filename. `step_3_evaluate` tests the working
directory on public validation hands or episodes; `step_5_check_submission` tests
the packaged ZIP. Edit and repeat training and evaluation as needed. Both commands
run directly in your Python environment, with ordinary output and tracebacks.
The ZIP check validates and extracts a fresh copy; it imposes no isolation or
resource limits. No container tools are needed. See [environment](docs/environment.md)
for the requirements that apply to instructor grading.
Submit the resulting ZIP from `dist/` through the instructor's designated channel.
Do not submit the template in `labs/` or the working directory itself; packaging
does not upload anything.

On Windows use, for example, `.\step_2_train.cmd work\line_following\my-controller`.
All paths are relative to the student project root, or absolute. The wrappers use
their own directory even when called from somewhere else.

## Inspect data and behaviour

```sh
./view_lab blindjack
./view_lab line_following --solution work/line_following/my-controller
```

Blindjack provides manual play, solution inspection and human/model comparison.
Its offline `data/blindjack/index.html` also shows labeled examples and validation
hands using the canonical PNG sheets, without Python or a server. Line Following
previews every public course, then runs a prepared controller with live sensor plots,
pause and rewind. Blue marks the start/return finish; red marks the far turnaround.
Its 50 training courses progress from straight driving to mixed routes; 15 separate
validation courses cover the same stages. On a fresh checkout, **Create baseline**
copies and prepares the robot reference controller; **Start** then begins driving.
The engine prepares in the background as the viewer opens. Course selection stays
visible during a run; **Restart** and course changes do not ask to save recordings.
Both viewers save and open recordings with diagnostics. The launcher installs the
optional Qt and ModernGL dependencies. See
[graphical inspection](docs/development.md#graphical-inspection-and-manual-play).

The Blindjack baseline also includes a [marimo notebook](docs/notebook.md):

```sh
uv run --locked marimo edit work/blindjack/my-model/notebook.py
```

It reads saved training results; training is opt-in. For Line Following, use the
viewer or the [batched Python example](docs/development.md#call-the-same-functions-from-python).
There is no supplied robot notebook.

## Copy a template

The copy keeps the selected template's name unless you supply a new name.
`--template` chooses the source; the optional name chooses the destination.
Choose the command for the copy you want; an existing directory is never overwritten.

```sh
# starter -> work/blindjack/starter
./step_1_new_solution blindjack

# baseline -> work/blindjack/baseline
./step_1_new_solution blindjack --template baseline

# baseline -> work/blindjack/my-model
./step_1_new_solution blindjack my-model --template baseline
```

## Course updates and ownership

```sh
git pull --ff-only
./step_0_update_resources
```

The resource updater does not pull Git changes. It validates SHA-256 hashes, replaces
changed resources, and removes obsolete course-managed files. Interrupted updates
must be rerun before training or evaluation.
Blindjack's dataset arrives in a ZIP: the updater verifies the archive and each
extracted file, then removes the downloaded ZIP. Unchanged data are not downloaded
again. PNG sheets remain directly viewable under `data/blindjack/`. The smaller robot
course collection is bundled in `labs/line_following/resources/` and copied to
`data/line_following/` with the same size and hash checks.

Keep your variants and models under `work/`, experiments and logs under `runs/`, and
packaged archives under `dist/`. These directories are ignored by Git and reserved
for your work. `data/` is managed course storage. Updating a template never changes
an existing variant. A new-solution command refuses to overwrite it.

See [submission format](docs/submissions.md), [environment](docs/environment.md), and
[AI use](docs/ai-use.md). If resources have not yet been published, ask the instructor
for the release rather than inventing download URLs.
