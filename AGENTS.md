# Student project

Write solutions in `work/<lab>/<variant>/`. Course updates own tracked files;
never store a solution, dataset, or model in those files. Use the shared locked
environment. Keep code short and explain the mathematical idea and invariants.
`labs/<lab>/starter/` demonstrates the solution format; `baseline/` is a naive
trainable solution. Copy a template with `step_1_new_solution` before running or
editing it. The working copy keeps the template's name unless a name is supplied.
Include reproducible training, all inference artifacts, and an honest AI-use
declaration. Submitted inference must run on CPU without network access.
All local commands and notebooks run directly in Python without isolation.
Keep solution code in ordinary source files shared by training, evaluation, and
the notebook. Use plain assignments and commented training calls in notebooks,
without mode-selection widgets. See `docs/development.md` for direct entry points.

Images are stored once in lossless PNG sheets. Labels use CSV with the PNG filename
and one-based row/column. Hands use CSV with hand/card numbers and PNG coordinates,
all one-based; values are not duplicated. Other numeric arrays use compressed NPZ with pickling
disabled. Preserve zero-based image indices and label alignment when loading sheets.
Open `data/blindjack/index.html` to explore examples and play validation hands offline.
Run `uv run --locked pytest` for public package tests.

GUI worker completions must enter a queued QObject slot before updating widgets.
Keep one QApplication alive throughout GUI tests and finish workers before
destroying their windows.
