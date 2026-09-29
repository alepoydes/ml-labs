# Explore a solution with marimo

The baseline includes a small Python notebook. Create a working copy and open it
from the student project root, using the same locked environment as the scripts:

```sh
./step_1_new_solution blindjack my-model --template baseline
uv run --locked marimo edit work/blindjack/my-model/notebook.py
```

On Windows the first command is `.\step_1_new_solution.cmd ...`; the marimo command
is the same. Prepare resources with `step_0_update_resources` before running cells.
Existing working copies are not updated automatically: create a new baseline copy
to get this example and transfer your changes as needed.

## Read, train, and inspect

The notebook uses ordinary assignments and Python calls. Read it in your editor
or run cells in marimo. The first cells load `config.toml`, show labeled images and
the model's input features, and construct an untrained model for inspection.
`sample_indices` uses zero-based image indices; `hand_number` is one-based, matching
the data browser and `hands.csv`.

Training is commented out by default:

```python
# training.train_local(solution_dir, project_dir, seed)
history, saved = training.load_training(solution_dir, run_dir)
```

For the first run, uncomment the training call and run that cell. Comment it again
when training finishes, so rerunning analysis only reads the saved result. You can
also train with `./step_2_train work/blindjack/my-model --seed 0`; both entry points
verify resources, call your `train.py`, and record the same provenance.
If no result exists, the notebook gives these instructions instead of training.

Edit hyperparameters in `config.toml` and set the training seed in the notebook
(or with `--seed` in the script). Each training call starts a fresh model. Resuming
optimizer state is not implemented. marimo's runtime settings can disable automatic
execution while editing expensive cells. See [execution controls](https://docs.marimo.io/guides/expensive_notebooks/).

Subsequent cells load the saved model and show:

- Training loss and accuracy on the MLP's **internal training holdout**. This is
  distinct from the public validation benchmark.
- Public validation class predictions, accuracy, and a confusion matrix.
- Misclassified images with predicted labels and class probabilities.
- A chosen seven-card hand, the selection made by `Solution.select()`, and its reward.

You can inspect intermediate variables and add your own analysis cells. Model
construction, prediction, and serialization live in `lib/model.py`; preprocessing
lives in `lib/features.py`; the learning procedure lives in `train.py`.
`solution.py` calls the same prediction code. Change these files to develop your
model, then rerun the loading cells (or restart the notebook). No training loop or
model implementation needs to be copied into notebook cells.

## Saved results

| Path | Contents |
| --- | --- |
| `work/blindjack/my-model/artifacts/model.joblib` | Selected baseline model and input resolution. |
| `work/blindjack/my-model/artifacts/provenance.json` | Completed training provenance used by packaging. |
| `runs/blindjack/my-model/training.npz` | Numeric learning curves, loaded with pickling disabled. |
| `runs/blindjack/my-model/training.json` | Configuration, seed, provenance, and hashes matching the history to the model. |

These files describe the latest training run of a variant. Create another variant
to keep a separate experiment. At the start of retraining, completion metadata is
invalidated; interrupted or failed training cannot be packaged as complete.
The loader rejects changed source/configuration and mismatched history or model
files, with instructions to retrain. Saved seed/configuration are shown explicitly;
changing the notebook's seed alone does not change an already saved model.

The notebook can analyze results after a restart, with no in-memory training state.
It can also execute without a browser:

```sh
uv run --locked python work/blindjack/my-model/notebook.py
```

This prints diagnostics; open marimo to view the figures. Figures use the canonical
PNG sheets and are not saved as a duplicate image dataset. The notebook and the
files under `runs/` are not included in submissions. Finish with the ordinary
evaluation, packaging, and archive-check commands.

## Errors in evaluation

`step_3_evaluate` and `step_5_check_submission` print the worker's traceback on
failure, including errors while importing the solution or loading its model.
They save bounded worker output to `runs/<lab>/<variant>/evaluate.log` or
`check.log`, and print the path. Output from successful runs is also saved there.
Each command replaces its previous log and result; a failed attempt leaves no
successful result JSON for that command.
