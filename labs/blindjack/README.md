# Blindjack

Recognize a hidden integer from an image and select cards whose sum approaches 21.
Each image has a value determined by an unknown deterministic mapping. Learn this
mapping from labeled examples.

A hand contains seven RGB images representing seven distinct values sampled from
the integers 0 through 10. Order has no numerical meaning. Select any subset,
including none or all seven cards. For the true selected sum `s`, your reward is
`s` when `s <= 21`, and zero otherwise. Over `M` hands:

```text
score = 100 * sum(rewards) / (21 * M)
```

Every allowed hand admits a subset summing to 21, so the maximum score is 100.
Different masks can be equally good. Selecting a zero-valued card has no effect.

## Templates, working copies, and submissions

Paths below are relative to the [project directory](../../README.md#start-here):

| Path | Purpose |
| --- | --- |
| `labs/blindjack/starter/` | Minimal example of the solution format and API. |
| `labs/blindjack/baseline/` | Naive, trainable solution supplied by the instructor. |
| `work/blindjack/<name>/` | Your working copy: edit code, train, and test here. |
| `dist/blindjack-<name>-<hash>.zip` | Packaged solution to check and submit. |

Before running a template, copy it into `work/`. Keep the tracked course files,
including the templates in `labs/`, unchanged to avoid conflicts with course
updates. Your ignored working copies are independent of those templates.

From the [project directory](../../README.md#start-here), choose a template and
optionally name the copy:

```sh
./step_1_new_solution blindjack
# labs/blindjack/starter -> work/blindjack/starter

./step_1_new_solution blindjack --template baseline
# labs/blindjack/baseline -> work/blindjack/baseline

./step_1_new_solution blindjack my-model --template baseline
# labs/blindjack/baseline -> work/blindjack/my-model
```

Without an explicit name, the copy keeps the template's name. Without `--template`,
the source is `starter`. Existing working directories are never overwritten.

For a working copy named `my-model`:

```sh
./step_2_train work/blindjack/my-model
./step_3_evaluate work/blindjack/my-model
./step_4_pack work/blindjack/my-model
./step_5_check_submission dist/blindjack-my-model-<hash>.zip
```

`step_3_evaluate` tests the working copy on public validation data.
`step_4_pack` prints the archive filename; use it in `step_5_check_submission` to
test a fresh extraction in your Python environment. Submit that ZIP through the
instructor's designated channel.

## Interface

```python
class Solution:
    def __init__(
        self, artifact_dir, config, seed
    ): ...  # Load artifacts once; initialize your RNG with seed.

    def select(self, images): ...  # Return np.ndarray, shape (7,), dtype bool.
```

`images` is a `numpy.uint8` array with shape `(7,128,128,3)`, RGB channels and values
0 through 255. `True` selects a card. The same instance handles all hands in an
evaluation. Model probabilities and the selection algorithm are your choices.

Training implements `train(data_dir: Path, output_dir: Path, config: dict, seed: int)`
in `train.py`. It must build or adjust a recognizer from labeled data. Conventional
learners, neural networks, learned analytic features, calibration, and discovery of
an exact rule are all acceptable. Save the trained model and any learned
preprocessing parameters in `work/blindjack/<name>/artifacts/`. Keep helper code in
that working copy's `lib/` directory. Libraries are provided by the shared course
environment. Final inference runs on CPU without external services.

## Data

Prepare the dataset with `./step_0_update_resources` from the project directory.
The [resource instructions](../../README.md#course-updates-and-ownership) link to
the pilot release and explain how updates are verified.

`data/blindjack/` contains `train/`, `validation/`, and `dataset.json`. Each split has
lossless `images-0000.png`, `images-0001.png`, … sheets and `labels.csv`:

```csv
sheet,row,column,label
images-0000.png,1,1,4
images-0000.png,1,2,0
```

These are illustrative rows. `sheet` names the PNG in the same directory; `row`
and `column` start at **1**, from the top left. `label` is the image's value, 0–10.
Open the actual CSV in a text editor or spreadsheet alongside the PNG sheets.

Validation also has `hands.csv`, with one row per card:

```csv
hand,card,sheet,row,column
1,1,images-0000.png,3,18
1,2,images-0000.png,4,11
```

`hand` identifies the deal, `card` its position (1–7), and the remaining columns
locate the image in a validation PNG. All four numeric columns start at 1. Records
follow hand order, then card order, with exactly seven records per hand. Values
are not duplicated: look them up by the same coordinates in `labels.csv`.
The loader rejects missing or reordered cards, invalid coordinates (including
padding), and repeated values within a hand. It returns zero-based image indices
of shape `(M,7)` in memory.
Validation has one third as many images as training.

Each sheet has 32 columns and at most 32 rows of original 128×128 images. Read tiles
left to right, row by row, then continue with the next sheet. The metadata gives
the valid image count; white cells beyond that count are padding, not examples.
CSV records follow that order. Their positions (excluding the header) define the
zero-based image indices used in code. The loader checks each CSV
coordinate and the label count, so reordered or missing rows cannot silently shift
labels. No duplicate image arrays are stored; both labels and hands use plain CSV.

### Explore and play without code

Open **`data/blindjack/index.html`** in a browser after resource preparation. No
server or network is needed. It provides:

- All training and validation examples, with sheet coordinates and optional values.
- Every fixed validation hand, with all seven cards shown together.
- Card selection and a **Check selection** button to reveal values and calculate
  your reward. Answers start hidden for each new validation hand.

Use Previous/Next or enter a page/hand number. Try to infer the rule from training
examples and solve hands while your model trains. This is practice on public data,
not a submission or a hidden evaluation. The page crops the original PNG sheets
in the browser; only small navigation and answer metadata are embedded in HTML.

### Load data for training

```python
from coursekit.labs.blindjack import load_split

images, labels = load_split(data_dir, "train")
batch = images[[0, 1, 2]]
```

The loader caches at most two decoded sheets. The public validation benchmark is
fixed and labeled to support development. It is not the final examination set;
rearranging the same images into more hands does not test new recognition examples.

## Starter and baseline

The starter selects three random distinct positions using the supplied seed. Its
expected score is approximately 60.5483. Its empty training function demonstrates
the interface, not substantive learning.

The baseline trains a small scikit-learn MLP on reduced, normalized pixels. It
predicts each image's most likely value and greedily considers cards from largest
predicted value to smallest. It skips a card that does not fit and continues.
Recognition, uncertainty handling, and selection all leave room for improvement.

The baseline's `notebook.py` demonstrates loading and visualizing data, plotting
saved learning curves, predicting classes, and testing a hand. Its training call
is commented out by default; uncomment it to train from scratch, or use
`step_2_train`. The notebook and scripts share `train.py` and `lib/` code.
See the [notebook workflow](../../docs/notebook.md).

Training may return a dictionary of finite numeric NumPy arrays for diagnostics
(the baseline returns `epoch`, `loss`, and `validation_accuracy`). The course
training entry point stores them in `runs/blindjack/<name>/training.npz`, with
configuration and provenance in `training.json`. Returning `None` remains supported.

If you change source or configuration in your working copy, rerun training before
packaging so provenance matches your submitted procedure.

## Augmentation and evaluation

Translation, rotation, and scaling can preserve the value when the complete pattern
and its relevant details remain intact; use white padding. Exact quarter-turn
rotations preserve the original pixel grid. Arbitrary angles, translations with
cropping, and downscaling require careful validation and can destroy information.
No general safe scaling range has yet been calibrated. Reflections and arbitrary
color changes are not declared label-preserving.

The current [reference limits](../../docs/environment.md) are pilot settings.
An incomplete run has no score: averaging only completed hands would be misleading.
Points measure absolute performance. Cohort rank does not determine your grade.
