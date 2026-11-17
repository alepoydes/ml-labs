# Machine learning course

This course develops machine-learning methods through complete working solutions.
Three labs are available, with [two further topics](docs/planned-labs.md) planned.
Choose the lab assigned by your instructor:

| Lab | Task and API | Inspect before training |
| --- | --- | --- |
| [Blindjack](labs/blindjack/README.md) | Learn image values and select cards with `select(images)`. | Open `data/blindjack/index.html` after resource preparation, or run `./view_lab blindjack`. |
| [Line Following](labs/line_following/README.md) | Train an adaptive robot controller with `act(observation, reset)`. | Run `./view_lab line_following`; choose **Course** in the top row, then **Top view** and **Fit track**. No trained solution is needed to preview courses. |
| [Visual Queries](labs/visual_queries/README.md) | Learn one-token answers to image questions and adapt to unknown tokens with `predict(images, tokens, mask)`. | Open `data/visual_queries/index.html` after resource preparation, or run `./view_lab visual_queries`. |

The [student guide](docs/README.md) links the common development, environment and
submission instructions. Task rules and score definitions live in each lab's statement.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
repository supplied by your instructor, and enter the project directory: the
directory containing `pyproject.toml`, `uv.lock`, `view_lab` and the numbered
`step_*` scripts. Locate these files before running the commands below; a lab's
`labs/<lab>/` directory contains its statement and templates, not the launchers.
Commands use one shared locked Python environment.
Line Following's Numba simulator needs no separate C++ compiler. Build tools for
accelerating its educational PyTorch example are [optional](docs/environment.md#optional-acceleration-of-the-pytorch-mechanics-example).

Each `labs/<lab>/` provides `starter`, a minimal API example, and `baseline`, a
runnable reference. Visual Queries provides a constant-answer starter and a small
trainable CNN/MLP with independent adaptation. Blindjack's baseline trains an MLP; Line Following's baseline
is hand tuned and its preparation only saves controller settings. Replace or improve
it with your own learning procedure. First copy a template into `work/`, then edit,
train and evaluate that copy. Leave tracked course files unchanged so updates do
not conflict with your work.

```sh
./step_0_update_resources

# Choose the copy for your lab:
./step_1_new_solution blindjack my-model --template baseline
./step_1_new_solution line_following my-controller --template baseline
./step_1_new_solution visual_queries my-answers --template baseline
```

Read your lab's statement and inspect `train.py`, `solution.py`, and `config.toml`
in the working copy. The same command sequence applies to all three labs. For the robot:

```sh
./step_2_train work/line_following/my-controller
./step_3_evaluate work/line_following/my-controller
./step_4_pack work/line_following/my-controller
./step_5_check_submission dist/line_following-my-controller-<hash>.zip
```

For another lab, substitute its working directory and the ZIP path printed by
its packaging command. Replace `<hash>` with the actual filename.
`step_3_evaluate` tests the working directory on the lab's public benchmark;
`step_5_check_submission` tests the packaged ZIP. Visual Queries evaluation includes
an initial base diagnostic and all independent adaptation episodes, with scored
episode queries and old controls after adaptation. Edit and repeat training
and evaluation as needed. Both commands
run directly in your Python environment, with ordinary output and tracebacks.
The ZIP check validates and extracts a fresh copy; it imposes no isolation or
resource limits. No container tools are needed. See [environment](docs/environment.md)
for the requirements that apply to instructor grading.
Submit the resulting ZIP from `dist/` through the instructor's designated channel.
Do not submit the template in `labs/` or the working directory itself; packaging
does not upload anything.

On Windows use, for example, `.\step_2_train.cmd work\line_following\my-controller`.
All paths are relative to the project directory, or absolute. The wrappers use
their own directory even when called from somewhere else.

## Inspect data and behaviour

```sh
./view_lab blindjack
./view_lab line_following --solution work/line_following/my-controller
./view_lab visual_queries
```

Blindjack provides manual play, solution inspection and human/model comparison.
Its offline `data/blindjack/index.html` also shows labeled examples and validation
hands using the canonical PNG sheets, without Python or a server. Line Following
previews every public course, then runs a prepared controller with live sensor plots,
pause and rewind. Blue marks the start/return finish; red marks the far turnaround.
Its [course atlas](labs/line_following/resources/index.html) shows 50 training courses
progressing from two straight episodes to compact mixed routes; 15 separate
validation courses cover the same stages. On a fresh checkout, **Built-in baseline**
is selected; **Start** prepares it in managed storage and begins driving.
The engine prepares in the background as the viewer opens. Course selection stays
visible during a run; **Restart** and course changes do not ask to save recordings.
Visual Queries displays image/question pairs, expected answers, batched model
predictions and optional answer distributions. Its offline
`data/visual_queries/index.html` also provides support/query browsing.
All three graphical tools save and open recordings with diagnostics. The launcher installs the
optional Qt and ModernGL dependencies. See
[graphical inspection](docs/development.md#graphical-inspection-and-manual-play).

The Blindjack baseline also includes a [marimo notebook](docs/notebook.md):

```sh
uv run --locked marimo edit work/blindjack/my-model/notebook.py
```

It reads saved training results; training is opt-in. For Line Following, use the
graphical simulator or the [batched Python example](docs/development.md#call-the-same-functions-from-python).
For Visual Queries, use its [data and prediction inspector](labs/visual_queries/README.md#7-inspecting-examples-and-predictions).

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
Blindjack and Visual Queries use downloadable data archives, with release locations
recorded in their `labs/<lab>/resources.toml` manifests. The current course release is
[ML Labs pilot 6](https://github.com/alepoydes/ml-labs/releases/tag/ml-labs-pilot-6).
The updater downloads each required ZIP, verifies the archive and every extracted
file, then removes the downloaded ZIP. Unchanged data are not downloaded again.
PNG sheets remain directly viewable under each lab's `data/` directory. The smaller robot
course collection is bundled in `labs/line_following/resources/` and copied to
`data/line_following/` with the same size and hash checks.

Keep your variants and models under `work/`, experiments and logs under `runs/`, and
packaged archives under `dist/`. These directories are ignored by Git and reserved
for your work. `data/` is managed course storage. Updating a template never changes
an existing variant. A new-solution command refuses to overwrite it.

See [submission format](docs/submissions.md), [environment](docs/environment.md), and
[AI use](docs/ai-use.md). If a required resource is unavailable at its configured
location, ask the instructor for the corresponding course release.
