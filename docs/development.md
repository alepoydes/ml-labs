# Develop your own solution

The workflow is shared by [Blindjack](../labs/blindjack/README.md) and
[Line Following](../labs/line_following/README.md). Choose `baseline` for a runnable
reference or `starter` for the smallest API example. Create the copy for your lab:

```sh
./step_1_new_solution blindjack my-model --template baseline
./step_1_new_solution line_following my-controller --template baseline
```

Edit the files under `work/<lab>/<variant>/`. Leave the templates in `labs/` unchanged.

| File | What to change |
| --- | --- |
| `config.toml` | Hyperparameters passed to training and inference. |
| `train.py` | Implement reproducible learning or fitting and save inference artifacts. |
| `solution.py` | Load artifacts in `Solution.__init__`; implement the lab's acting method. |
| `lib/` (optional) | Helpers shared by training, inference and analysis. |
| `submission.toml` | Variant metadata and an accurate AI-use declaration. |

| Lab | Acting method | Supplied baseline |
| --- | --- | --- |
| Blindjack | `select(images)` returns a Boolean mask for seven cards. | A trained MLP; `lib/features.py`, `lib/model.py` and a marimo notebook. |
| Line Following | `act(observation, reset)` returns batched left/right motor commands. | A hand-tuned line follower; preparation saves `controller.json`. No supplied notebook. |

Replace the baseline as needed while keeping the documented `train` and `Solution`
interfaces. The robot baseline does not perform learning; implement your own training
procedure, for example controller fitting or reinforcement learning. Keep reusable
code in source files and import it into analysis scripts or notebooks.

## Call the same functions from Python

The numbered scripts are conveniences around ordinary functions. After resource
preparation and a completed `step_2_train` run, start `uv run --locked python` from
the [project directory](../README.md#start-here). These calls work for either lab:

```python
from pathlib import Path

from coursekit.training import load_solution, load_training, train_local
from coursekit.labs import get_lab

project = Path.cwd()
lab, variant = "line_following", "my-controller"  # Or "blindjack", "my-model".
solution_dir = project / "work" / lab / variant
run_dir = project / "runs" / lab / variant
data_dir = project / "data" / lab
seed = 0

# Run explicitly when you want to train, after step_0_update_resources:
# train_local(solution_dir, project, seed)

history, saved = load_training(solution_dir, run_dir)
print(saved["config"])
print(history.keys())

solution = load_solution(solution_dir, seed)
# A full validation run; invoke explicitly when ready:
# result = get_lab(lab).evaluate(solution, data_dir)
# print(result)
```

`train_local` verifies resources, calls your `train.py`, and records completed
training and any numeric history returned by `train`. `load_training` reads those
saved diagnostics; `load_solution` constructs your current `Solution` with its
artifacts and current configuration. Numeric history may be empty, as it is for the
hand-tuned robot baseline. `get_lab(lab).evaluate` runs every validation hand or
episode and returns aggregate metrics. All calls run in the same Python process;
the numbered evaluation scripts also save the result dictionary under `runs/`.
Line Following scripts additionally print episode tables and save companion
`.cases.json` and `.cases.csv` diagnostics, including accepted distance, actual path
and failure outcome. See the [result-file description](../labs/line_following/README.md#development-and-submission-workflow).

For one Blindjack decision:

```python
from pathlib import Path
from coursekit.labs.blindjack import benchmark
from coursekit.training import load_solution

solution = load_solution(Path("work/blindjack/my-model"))
images, labels, hands = benchmark(Path("data/blindjack"))
mask = solution.select(images[hands[0]])
print(mask, labels[hands[0]])
```

For one batched robot control interval:

```python
from pathlib import Path
from coursekit.labs.line_following_env import Environment
from coursekit.labs.line_following_task import benchmark
from coursekit.training import load_solution

cases = benchmark(Path("data/line_following"), "train")[:2]
env = Environment(cases)
controller = load_solution(Path("work/line_following/my-controller"))
action = controller.act(env.observe(), env.first)
observation, reward, terminated, truncated, info = env.step(action)
done = terminated | truncated
if done.any():
    observation = env.reset(done)
```

`compiled=False` selects the PyTorch autodiff reference for study and comparison; normal evaluation uses the
Numba engine. Reuse an environment to amortise startup; changing the batch length
does not require a new specialization with matching array types and layouts.
The controller must reset the indicated rows of memory when `reset` is true. Memory
persists through the turnaround. Use `info` to analyse training, but the evaluated
controller receives only observations and the reset mask. See the lab statement for
the [complete robot API](../labs/line_following/README.md#controller-and-environment-api).

You can also call functions from your own modules directly during an experiment.
Use `train_local` or `step_2_train` for the completed run you intend to package,
so its artifacts have matching provenance. Changed source or configuration
requires retraining before packaging, but saved learning curves remain viewable.

## Debug and repeat

Add `print()` or `breakpoint()` in `train.py`, `solution.py`, or a helper. Choose
this project's `.venv` interpreter in your editor and run a small script containing
the calls above, with the [project directory](../README.md#start-here) as its working
directory. Exceptions show the original file, line, and traceback. No special debug
flag is required.

Blindjack includes a [marimo notebook](notebook.md) with opt-in training and saved
learning curves. After editing imported modules, rerun the loading cells or restart
the notebook. For the robot, start with the graphical simulator's sensor plots and
recordings, or use the batched environment from Python. You can write your own
notebooks for either lab; none is required in a submission.
Create a new variant to retain a separate experiment. When ready, run
`step_4_pack` and `step_5_check_submission`, then submit the ZIP.

## Graphical inspection and manual play

Run `./view_lab blindjack` or `./view_lab line_following` (Windows: `view_lab.cmd`).
The launcher installs the locked optional Qt and ModernGL dependencies. Select a
working solution directory; the viewer uses its existing artifacts. It never trains
implicitly. Blindjack calls the model in a computation thread. Line Following runs
the model and shared physics in a cancellable local process; this is ordinary local
execution, not an instructor sandbox. Original tracebacks remain available. Qt owns
the main thread.
Use the native window buttons or the **Window** menu to minimize, maximize or close.
F11 enters fullscreen and Escape restores the preceding state. The desktop decides
which title-bar buttons to show. Student windows remember their size and window mode
for each lab; Ctrl+W also closes the window.

### Blindjack

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
current session. If you edit the solution path, load that solution before running it.
Model errors appear in a compact message with expandable details as well as the
original terminal traceback.

The offline `data/blindjack/index.html` provides a separate practice path without Qt:
filter labeled examples by value, then switch to **Play validation hands**. Committed
answers and the average score remain in the current page session when navigating
or switching activities; reload or **New practice session** starts over. Source-image
and CSV links use the canonical sheets. Existing verified datasets retain their
original browser until a resource update.

### Line Following

To browse courses, run `./view_lab line_following` and select **Course** in
the top row or use its previous/next arrows. The preview needs no trained solution
and prepares the bundled courses automatically. Use **Top view** and **Fit track**
for a plan view. Blue marks the start/return finish; red marks the far turnaround.
On a fresh checkout, **Next run** selects **Built-in baseline**. **Start** prepares
the installed reference in managed storage under `runs/line_following/builtin/`;
opening the viewer alone performs no training. Select **Working solution** to use
your own directory; this choice and path persist between openings. Under **Setup…**,
**Prepare solution** runs that copy's `train.py`, using the same function as the
numbered scripts. **New copy from baseline** creates and prepares a fresh numbered
working copy, preserving existing code. Working copies need preparation metadata to
start; subsequent source/configuration edits are identified as stale preparation
and remain runnable for local debugging. Parameter integrity is checked when loading.
See the [data description](../labs/line_following/README.md#data-and-course-collection)
for files and fixed experiment cases, and the
[episode rules](../labs/line_following/README.md#rules-of-the-episode) for marker
dimensions and route coverage.
There are 50 training courses and 15 separate validation courses. Begin with the two
straight episodes, then progress through compact curves, turn sequences, hairpins,
sharp corners, crossings and mixed layouts. The
[offline atlas](../labs/line_following/resources/index.html) shows all courses with
lengths and fixed case identifiers; it is also copied to `data/line_following/index.html`.

The solution, fixed case and batch size stay visible during a run.
The engine and course geometry prepare in the background on opening the viewer,
without executing a student controller. **Start** begins moving automatically as
soon as preparation is complete. **Cancel preparation** cancels a queued start;
closing also cancels background preparation. The status shows preparation and
simulation separately. First use compiles the Numba functions; later windows load
saved functions. A different batch length reuses the same array-type specialization. **Restart** reloads the
solution, case and batch settings. Course changes start a fresh run automatically.
Ordinary restarts reuse the warm process; engine/runtime updates invalidate its cache.

**Pause / Continue** controls simulation; Space does the same while a run is active.
While paused, **Step 1 ms** and **Step control** advance the simulation. **Stop** ends
it and retains received frames. **Play recording** uses recorded frames, with a
speed selector, previous/next frame and **Next control** at an actual recorded
controller call. Space toggles playback when viewing a finished run or saved file.
Drag the timeline to seek; Home/End jump to its ends. Reviewing history pauses the
run. **Continue** returns to its latest state without repeating old controller calls.

Batch runs use consecutive fixed cases; **View robot** selects the row.
The graphical and command-line runners use the same stored robot, initial state
and noise for a case. Open **Setup…** to disable **Sensor noise** only for diagnostic
comparisons.
**Running controller** identifies the actual loaded solution, independently of
**Next run**. **Details…** shows the path, code/configuration/parameter
fingerprints and preparation status captured by the worker. Recordings retain this
identity; historical missing identities remain unknown. **Restart** creates a fresh
controller instance and episode, while **Prepare solution** prepares saved parameters
again and **New copy from baseline** starts a separate working variant.
Top view, follow camera, wheel zoom and dragging to pan help inspect the line, chassis
and rear sphere; **Fit track** restores the full-course view. In the large view,
the RGB strip in the scene's upper-left corner shows the latest actual optical
reading. Use **Save…**
explicitly before replacing or closing a run you want to keep; these operations do
not show save prompts.
Saving while running stores the frames captured so far; subsequent frames need another save. Malformed recordings leave the currently open recording intact.

The viewer opens with a large 3D scene. Double-click the scene to show diagnostics:
the same live scene fills the top band to the right of the solution and course
controls, without the RGB strip; plots fill the area below. On a wide window, the
episode status appears beside **Fit track**. On a narrow window it moves below the
camera controls.
Double-click the preview to return to the large scene; its tooltip explains the
gesture. **Sensor area** overlays the 16 measured ground sectors through the chassis.
Use **Top view**, **Follow** and the mouse wheel to
inspect the 94×8 mm strip between the wheels. This strip also defines route
coverage: the centreline must pass through it in route order.

### Diagnostics and recordings

An optional `debug(index=0)` method may return a dictionary of text, finite scalars,
or small numeric vectors/matrices. The viewer copies it after inference. The
**Diagnostics** table contains exactly those fields. In the robot viewer, open its
drawer from the left edge of the plots or through **Panels → Diagnostic fields**.
The drawer slides over the plots and can be returned behind the edge. Click a row
to inspect its current value; checking its box also
selects that row and adds it to the chosen plot. Expand a vector to see and choose
individual components. The inspector above
the table shows complete vector values; a missing value is identified explicitly.
Colours identify curves and remain stable across updates. Click a plot panel to
choose which plot the field checkboxes edit. **Add plot** is below the plots; each
plot has its own remove button. The **Panels** menu can hide or reveal a plot
without deleting its field selections.

**Time series** plots scalar and component histories with a legend. **Heatmap**
shows selected components of one vector against time, with a value colour scale.
**X–Y** pairs exactly two scalar channels at matching observation times; **Swap
X/Y** exchanges the axes. Its highlighted point is the current sample. **Value**
shows the current vector or matrix, also useful for Blindjack decisions without
a time history. Text remains in the value inspector.

Click a time plot or an X–Y trajectory to pause and inspect the nearest recorded
controller call. Robot pose, sensor readings and all diagnostic panels follow
the same playback position. **Follow time** scrolls the chosen time window;
**Fit all** fits the available history. Pan/zoom freezes the time interval
across time plots. Missing fields or changed array dimensions leave gaps.
The student robot viewer retains plot settings, panel visibility and the selected
course across restarts. It opens in the large 3D view. The playback timeline sits
beside **Speed**. **Save plot** exports the
figure and legend as PNG or SVG. Empty snapshots have an explicit empty state.
Debug methods must not change
controller behavior.
The judge never calls it. Keep each array at most two-dimensional and 4096 elements,
with at most 64 fields. Debug failures are visible inspection errors.

For Line Following, the optional `coursekit.controller.Controller` base supplies
sensor diagnostics. Initialise it with `super().__init__()` and call
`super().act(observation, reset)` before processing inputs. It copies the current
observations without retaining autograd graphs. `super().debug(index)` exports
`sensor.time`, `sensor.gyro`, `sensor.accel`, `sensor.encoders` and the 16-element
`sensor.red`, `sensor.green`, `sensor.blue` vectors. Extend that dictionary with
your estimates and actions, as the baseline does. Controllers without this base
remain valid. Explicit instructor diagnostic runs collect the same debug fields
inside the isolated worker; ordinary grading does not call `debug()`.

Both viewers save compressed NPZ recordings, including copied controller diagnostics.
Opening a recording executes no solution code. Blindjack recordings refer to the
verified public resource set and reuse its PNG sheets. Robot recordings include
course geometry, physical parameters, poses, observations, actions and diagnostics;
they are local inspection artifacts, not resumable simulator checkpoints or submissions.

Robot recordings use version 4, including case identities, sensor-noise settings,
sensor/paint geometry and the case/coverage rule versions. Both viewers reject older recordings or incompatible settings before
replacing the open run. Use the original coursekit environment for historical
recordings, or create new recordings with the current environment. Course archives
remain in spline format version 2.
