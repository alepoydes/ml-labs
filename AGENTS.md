# Student project

Line Following uses the public Numba engine for actual runs. Keep PyTorch autodiff
as a teaching reference and check analytic derivatives against it. System build
tools for the optional PyTorch compiler must stay optional in installation guides.
Keep physical steps, coverage and terminal handling in the same native loop.

Write solutions in `work/<lab>/<variant>/`. Course updates own tracked files;
never store a solution, dataset, or model in those files. Use the shared locked
environment. Keep code short and explain the mathematical idea and invariants.
`labs/<lab>/starter/` demonstrates the solution format; `baseline/` is a runnable
reference (trainable for Blindjack, hand tuned for Line Following). Copy a template
with `step_1_new_solution` before running or
editing it. The working copy keeps the template's name unless a name is supplied.
Include reproducible training, all inference artifacts, and an honest AI-use
declaration. Submitted execution uses the lab's explicit CPU/CUDA profile without network access. Visual Queries requires v3
and adaptation; it supplies a constant-answer starter and a small CNN/MLP baseline.
Keep the baseline's spatial cells and word order, image-grouped internal holdout,
fresh-base adaptation and input/output vocabulary growth. Measure actual learning
and saved-model reloads on tiny fixtures; full accuracy pilots stay out of pytest.
During model pilots, compare real images with erased images; a score improvement
from answer/question priors alone does not establish visual learning.
All local commands and notebooks run directly in Python without isolation.
Packaging requires the exact artifact files recorded by completed training;
prepare again after changing them. Check documented batch shapes and constant
answer IDs against the current resource metadata and vocabulary.
Keep solution code in ordinary source files shared by training, evaluation, and
the notebook. Use plain assignments and commented training calls in notebooks,
without mode-selection widgets. See `docs/development.md` for direct entry points.
Shared guides cover all runnable labs. Label lab-specific APIs, limits and examples,
and provide visual entry points before requiring training.
Lab statements introduce the task and rules before data, APIs, workflow and graphical
tools. Define the project directory by its launchers and configuration files. Keep
data formats independently readable, and verify examples and cross-document links.
Write for a first-time reader: define task terms before using them and describe
model choices positively. Keep migration notes and historical measurements in
clearly scoped sections; do not assume access to design discussions or instructor
documentation. Check shared guides against every available lab.

Images are stored once in lossless PNG sheets. Labels use CSV with the PNG filename
and one-based row/column. Hands use CSV with hand/card numbers and PNG coordinates,
all one-based; values are not duplicated. Other numeric arrays use compressed NPZ with pickling
disabled. Preserve zero-based image indices and label alignment when loading sheets.
Open `data/blindjack/index.html` to explore examples and play validation hands offline.
Run `uv run --locked pytest` for public package tests.
Keep routine workflow checks short: use minimal synthetic episodes instead of
loading the full course collection or driving complete benchmark routes. Full
controller sweeps belong in explicit pilots.

GUI worker completions must enter a queued QObject slot before updating widgets.
Keep one QApplication alive throughout GUI tests and finish workers before
destroying their windows.
Preserve dataset labels, selection identifiers and tooltips after completed or
stopped model runs and when returning from recording replay.
Check course changes and consecutive robot runs, including fresh controller state,
warm compiler reuse and cancellation. Measure simulation throughput separately
from compilation and include sensor, coverage, recording and GUI overhead.
Include selected diagnostic plots and same-size rapid source/helper edits in restart
checks. Native compilation checks must cover initial tensor layouts, resets and
mixed running/terminal batches, not just graph capture.
Check saved kernel reuse in a fresh process with compilation forbidden and runtime
guards enabled. Keep executable compiler artifacts in the local user cache.
Robot setup stays visible. Prepare the engine at startup without invoking a solution.
Distinguish built-in baseline from working solutions and the next selection from the
loaded controller. Preserve working copies and recorded identities across updates.
Start/Restart and course changes begin moving automatically. Check queued starts,
cancellation and close during preparation. Save recordings only on explicit request;
do not interrupt routine switching with save/discard confirmations.

Optics, route coverage and the GUI sensor overlay share the same projected
sensor rectangle. Check its extent against the inner wheel edges and reject
progress for lines visible only as paint edges or covered only by the chassis.
Rendering the chassis body must not define the judge geometry.
Record sensor/paint geometry and the coverage rule version with robot runs. Reject
incompatible recordings before replacing the current view. Label the RGB strip in
its documented body-coordinate order, independently of the camera orientation.

Robot experiments are fixed Case data, including model, initial state and sensor
noise. Environment.reset repeats them exactly; sample extra training cases explicitly.
Keep GUI, CLI and assessment case selection consistent. Validate whole recovery
manoeuvres and restored route progress on both legs, with and without sensor noise.

Visual Queries keeps readable CSV text outside submitted code. Learning receives
only numeric tables and image batches; instructor containers never mount the corpus.
Adaptation examples must demonstrate the optional checkpoint callback: save every
component before publishing, including expanded input/output vocabularies. A time
limit may select only a committed complete snapshot. Code errors and cancellation
remain failures; local direct calls cannot preempt blocked learning code.
Charge numeric setup and student imports to one adaptation deadline locally and
in isolated execution; pass only the remaining time to adapt and never reset it
after initialization. Verify expiry before the first checkpoint.
