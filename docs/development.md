# Develop your own solution

Create a working copy from `baseline` for a complete learning example, or `starter`
for the smallest API example:

```sh
./step_1_new_solution blindjack my-model --template baseline
```

Edit the files under `work/blindjack/my-model/`:

| File | What to change |
| --- | --- |
| `config.toml` | Hyperparameters passed to training and inference. |
| `train.py` | Load data, train a model, and save inference artifacts. |
| `solution.py` | Load the saved model in `Solution.__init__`; predict in `select`. |
| `lib/features.py` (baseline) | Image preprocessing shared by training and inference. |
| `lib/model.py` (baseline) | Model construction, prediction, saving, and loading. |
| `notebook.py` (baseline) | Inspect data, saved learning curves, and predictions. |

The baseline is an example, not a required model architecture. Replace its helpers
as needed while keeping the documented `train` and `Solution` interfaces.
Keep reusable code in source files and import it into the notebook.

## Call the same functions from Python

The numbered scripts are conveniences around ordinary functions. From the student
project root, start `uv run --locked python` and run:

```python
from pathlib import Path

from coursekit.training import load_solution, load_training, train_local
from coursekit.labs.blindjack import benchmark, evaluate

project = Path.cwd()
solution_dir = project / "work/blindjack/my-model"
run_dir = project / "runs/blindjack/my-model"
data_dir = project / "data/blindjack"
seed = 0

# Run explicitly when you want to train, after step_0_update_resources:
# train_local(solution_dir, project, seed)

history, saved = load_training(solution_dir, run_dir)
print(saved["config"])
print(history.keys())

solution = load_solution(solution_dir, seed)
images, labels, hands = benchmark(data_dir)
mask = solution.select(images[hands[0]])
print(mask, labels[hands[0]])

result = evaluate(solution, data_dir)
print(result)
```

`train_local` verifies resources, calls your `train.py`, and records completed
training and any numeric history returned by `train`. `load_training` reads those
saved diagnostics; `load_solution` constructs your current `Solution` with its
artifacts and current configuration. `evaluate` calls `select` for each validation
hand, checks the returned mask, and computes the score. All calls run in the same
Python process. The evaluation function returns a dictionary; the numbered scripts
also save that dictionary under `runs/`.

You can also call functions from your own modules directly during an experiment.
Use `train_local` or `step_2_train` for the completed run you intend to package,
so its artifacts have matching provenance. Changed source or configuration
requires retraining before packaging, but saved learning curves remain viewable.

## Debug and repeat

Add `print()` or `breakpoint()` in `train.py`, `solution.py`, or a helper. Choose
this project's `.venv` interpreter in your editor and run a small script containing
the calls above, with the student project root as its working directory. Exceptions
show the original file, line, and traceback. No special debug flag is required.

For plots and intermediate values use the [marimo notebook](notebook.md).
After editing imported modules, rerun the loading cells or restart the notebook.
Training is commented out by default; rerunning analysis reads saved files.
Create a new variant to retain a separate experiment. When ready, run
`step_4_pack` and `step_5_check_submission`, then submit the ZIP.

## Graphical inspection and manual play

Run `./view_lab blindjack` or `./view_lab line_following` (Windows: `view_lab.cmd`).
The launcher installs the locked optional Qt and ModernGL dependencies. Select a
working solution directory; the viewer uses its existing artifacts. It never trains
implicitly. Model calls run as ordinary Python in a computation thread, with original
tracebacks in the terminal. Qt owns the main thread.
Use the window buttons to minimize, maximize or close. F11 enters fullscreen and
Escape restores the preceding state; these actions are also in the Window menu.

Blindjack offers **Play**, **Inspect solution**, and **Human vs model**. Click a card
image, its checkbox, or Alt+1…Alt+7 to select it. **Previous**, **Next hand**, and
**Next unplayed** keep navigation separate from committing an answer; unfinished
selections survive navigation. In comparison mode, load a trained solution, run the
model, then commit your cards before seeing either the labels or its answer. Each
hand contributes its first committed answer. The selected outline is your choice;
each revealed value also says whether the model selected it.

**Save session** keeps results, diagnostics, unfinished selections and the current
hand; **Open session** restores them without running the model. Replacing or closing
an unsaved session asks before discarding it. Verifying resources preserves the
current session. If you edit the solution path, load that solution before running it. Model errors appear in a
compact message with expandable details as well as the original terminal traceback.

In Line Following, **Run setup** contains the solution, course, robot seed and batch
size. Course preparation runs in the background; failed preparation can be retried.
**New run** starts paused and folds setup away. **Resume live**, **Pause live**,
**Step 1 ms**, and **Step control** operate the current simulation. **Play recording**
uses recorded frames, with a speed selector, previous/next frame and **Next control**
at an actual recorded controller call. Space toggles playback. Drag the timeline to
seek; Home/End jump to its ends. Reviewing history pauses the live run. Resume live
returns to its latest state without repeating old controller calls.

Batch runs use consecutive course entries and seeds; **View robot** selects the row.
Top view, follow camera, wheel zoom and dragging to pan help inspect the line, chassis
and rear sphere; **Fit track** restores the full-course view. The RGB strip shows the
latest actual optical reading. Unsaved motion is protected when replacing or closing
a run. Saving while live stores the frames captured so far; subsequent frames need
another save. Malformed recordings leave the currently open recording intact.

An optional `debug(index=0)` method may return a dictionary of text, finite scalars,
or small numeric vectors/matrices. The viewer copies it after inference, lists values,
and plots selected arrays or scalar history. Select a field in **Diagnostics**;
the same field stays selected as new snapshots arrive. **Save plot** exports the
current figure as PNG or SVG. Empty snapshots have an explicit empty state. Debug
methods must not change controller behavior.
The judge never calls it. Keep each array at most two-dimensional and 4096 elements,
with at most 64 fields. Debug failures are visible inspection errors.

Both viewers save compressed NPZ recordings, including copied controller diagnostics.
Opening a recording executes no solution code. Blindjack recordings refer to the
verified public resource set and reuse its PNG sheets. Robot recordings include
course geometry, physical parameters, poses, observations, actions and diagnostics;
they are local inspection artifacts, not resumable simulator checkpoints or submissions.

Robot recordings now use the spline geometry format (version 2). Old polyline
recordings are rejected with an explicit version message. Recreate them after
updating course resources; no old recording is silently assigned a new curve.

The offline `data/blindjack/index.html` provides a separate practice path without Qt:
filter labeled examples by value, then switch to **Play validation hands**. All seven
cards are visible together on a 1024-pixel desktop. Committed answers and the average
score remain in the current page session when navigating or switching activities;
reload or **New practice session** starts over. Source-image and CSV links use the
canonical sheets, without duplicate images. A newly generated dataset includes the
current browser; existing verified datasets retain their original browser until a
resource update.
