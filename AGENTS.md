# Student project

Write solutions in `work/<lab>/<variant>/`. Course updates own tracked files;
never store a solution, dataset, or model in those files. Use the shared locked
environment. Keep code short and explain the mathematical idea and invariants.
`labs/<lab>/starter/` demonstrates the solution format; `baseline/` is a runnable
reference (trainable for Blindjack, hand tuned for Line Following). Copy a template
with `step_1_new_solution` before running or
editing it. The working copy keeps the template's name unless a name is supplied.
Include reproducible training, all inference artifacts, and an honest AI-use
declaration. Submitted inference must run on CPU without network access.
All local commands and notebooks run directly in Python without isolation.
Keep solution code in ordinary source files shared by training, evaluation, and
the notebook. Use plain assignments and commented training calls in notebooks,
without mode-selection widgets. See `docs/development.md` for direct entry points.
Shared guides cover both runnable labs. Label lab-specific APIs, limits and examples,
and provide visual entry points before requiring training.

Images are stored once in lossless PNG sheets. Labels use CSV with the PNG filename
and one-based row/column. Hands use CSV with hand/card numbers and PNG coordinates,
all one-based; values are not duplicated. Other numeric arrays use compressed NPZ with pickling
disabled. Preserve zero-based image indices and label alignment when loading sheets.
Open `data/blindjack/index.html` to explore examples and play validation hands offline.
Run `uv run --locked pytest` for public package tests.

GUI worker completions must enter a queued QObject slot before updating widgets.
Keep one QApplication alive throughout GUI tests and finish workers before
destroying their windows.
Check course changes and consecutive robot runs, including fresh controller state,
warm compiler reuse and cancellation. Measure simulation throughput separately
from compilation and include sensor, coverage, recording and GUI overhead.
Include selected diagnostic plots and same-size rapid source/helper edits in restart
checks. Native compilation checks must cover initial tensor layouts, resets and
mixed running/terminal batches, not just graph capture.
Check saved kernel reuse in a fresh process with compilation forbidden and runtime
guards enabled. Keep executable compiler artifacts in the local user cache.
Robot setup stays visible. Prepare the engine at startup without invoking a solution;
Start/Restart and course changes begin moving automatically. Check queued starts,
cancellation and close during preparation. Save recordings only on explicit request;
do not interrupt routine switching with save/discard confirmations.
