# Visual Queries

An image contains handwritten symbols. Your program receives that image and a
question, then returns **one answer token**. The task combines visual recognition,
language interpretation, selection by attributes, spatial reasoning and learning
new vocabulary from examples. Each question is independent: previous requests
provide no conversational context.

Use machine learning. Choose your own architecture and training procedure; a
transformer is not required. Train from the supplied resources, without pretrained
models, external language corpora or outside image collections. Your submitted code
must support the supplied execution device, including CUDA on the instructor's
server. CPU is supported for local development and the lifecycle example.

This release provides the task, data, loader, constant-answer starter, a small
learned CNN/MLP baseline and evaluator. The starter demonstrates the interface;
it does not analyze images or questions and always answers `yes`. The baseline
learns from images and ordered token embeddings and supports vocabulary growth.

## 1. Images and meanings

Images are RGB, 256 × 256 pixels, on a white background. They contain up to nine
nonoverlapping handwritten symbols, freely positioned and mildly rotated (at most
7 degrees). The alphabet is `0123456789ABCDEFGHJKLMNPQRSTUVWXYZ`: digits and
uppercase letters, with `I` and `O` omitted. Different handwritten versions of the
same class remain possible. Symbols are entirely inside the image, with at least
3 pixels of border and 4 pixels between their ink bounding boxes.

The position of a symbol means the centre of its visible ink bounding box. A
symbol may contain disconnected strokes. Image coordinates increase rightwards
and downwards. Region boundaries divide each axis into thirds:

- The centre is the middle third on both axes.
- Corners combine the corresponding outer thirds.
- An edge is its outer-third strip, including both corners.
- `left of`, `right of`, `above` and `below` compare centres. They do not mean
  “nearest”; diagonal positions can satisfy two relations. An object is never
  spatially related to itself.

Questions involving positions exclude relevant comparisons within 0.035 of the
image width/height of equality or a region boundary. There is no visible grid.

Size is the larger dimension of the final ink bounding box:

| Word | Size in pixels |
| --- | ---: |
| tiny | 16–20 |
| small | 26–30 |
| normal | 36–40 |
| big | 46–50 |
| huge | 56–60 |

Relative size compares these categories, rather than differences within one
category. Selecting “the largest” requires a unique largest category. Visible
ink is defined before blending with the background, at opacity at least 16/255;
lower-opacity pixels are removed. Antialiased edge colors do not form extra objects.

Base color categories have RGB representatives red (175,30,45), green (25,125,55),
blue (30,65,180), and yellow (170,140,10). Actual ink varies slightly around these
representatives. The categories remain separated; new vocabulary never changes
the meanings of existing words.

## 2. Questions and answers

A question can ask about emptiness, identity, color, size, position, combinations
of attributes, relative positions, exclusion, alternatives, comparisons, shared
properties and references within the question. Examples include:

- `Is the image empty?` — `yes` or `no`.
- `What is this symbol?` — a symbol token; the image contains one symbol.
- `What color is the symbol?` / `What size is the symbol?`
- `Is the top left corner empty?`
- `Which symbol is in the center?`
- `Which symbol is small and red?`
- `Which symbol is to the left of the symbol A?`
- `Is there a symbol which is red or blue?`
- `Is there a red symbol which is not tiny?`
- `Consider the blue symbol H. Which other symbol has its color?`
- `Is every red symbol above a blue symbol?`
- `Consider the symbol A. Find the symbol below it. What color is the symbol
  to the right of this symbol?`

A definite description must select exactly one object. Each supplied question
has one unambiguous answer, without `unknown`, `none` or multi-token responses.
An existential statement asks whether at least one matching object exists. A
universal statement has a nonempty subject set and requires a matching witness
for every subject; different subjects may have different witnesses. `or` is
inclusive. Negation applies to its stated predicate. References such as `it`
refer only to objects introduced in the same question.

The base dictionary is `vocabulary.csv`. Spelling is case-sensitive and punctuation
has its own IDs. ID 0 is padding, followed by contiguous word/symbol IDs. Answers
use this same dictionary; they are word IDs, not positions in a separate list of
answer classes. The base training set includes examples of every question family.

## 3. Adaptation is part of evaluation

Evaluation includes base questions and independent adaptation episodes. In each
episode new token IDs are appended after the base dictionary, without gaps. Their
spellings and grammatical classes are not supplied to solution code. Infer their meanings from
labeled **support** examples. New tokens can affect attributes, relations, logical
expressions or references, and can also appear as answers. Some episodes introduce
one new element, others combine several. Existing IDs keep their base meanings.

`data.episode["vocab_size"]` specifies the active vocabulary size. Use it rather than guessing
from the largest ID in a minibatch. The same added ID can mean different things in
different episodes. Expand both input and output representations when necessary.

Each episode starts from the submitted base artifacts. The runner invokes `adapt`,
saves a separate complete result, loads it in a fresh solution, then asks the query
questions. Adapted state is never carried into the next episode. During official
adaptation, only current support examples and the public base-training/glyph
rehearsal data are available. Query images, labels and other episodes are absent.

Public query answers are available for local assessment. They must not be used as
training labels for that episode's adaptation. Public episodes may be used to
develop the adaptation algorithm. Keep a separate validation discipline when
choosing models and hyperparameters.

## 4. What to implement

Copy the starter into a working directory and implement:

1. `train.py:train(...)`: train one or several components and save a complete base
   state in `output_dir`. Named stages and component inputs are available.
2. `train.py:adapt(...)`: initialize from immutable `input_dir`, learn from the
   provided numeric support data, and save a complete state in a separate `output_dir`.
3. `solution.py:Solution(...)` and `predict(...)`: load saved components and return
   an answer ID for each row.

Keep `training.supports_adapt = true` and `api_version = 3` in `submission.toml`.
Document your method and AI assistance honestly. Artifacts must contain all models
and preprocessing state needed for prediction **and further training**, including
active vocabulary size and episode identity. Do not assume a single model file.
The framework controls `provenance.json`; do not write that file yourself.

The complete signatures are:

```python
def train(
    data, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
): ...


def adapt(data, input_dir, output_dir, config, seed, *, device, limits, report=None): ...


class Solution:
    def __init__(self, artifact_dir, config, seed, *, device): ...
    def predict(self, images, question_tokens, question_mask): ...
```

`data` is the numeric learning interface described in [API 3](../../docs/api-v3.md).
Artifact paths are `pathlib.Path` objects. `config` is the parsed `config.toml`; `device` is an
explicit device string. `train` and `adapt` can report bounded progress dictionaries
and return histories grouped by stage. See the small starter and the shared
[development guide](../../docs/development.md) for complete lifecycle conventions.
The starter also accepts the full public dataset for manual fine-tuning.

| Prediction value | CPU NumPy representation |
| --- | --- |
| `images` | `uint8`, `(B,256,256,3)`, RGB |
| `question_tokens` | `int64`, `(B,L)` |
| `question_mask` | `bool`, `(B,L)`, nonempty true prefix |
| Return value | `int64`, `(B,)`, one non-padding active token ID per row |

The pilot supports 1–16 rows and up to 96 tokens per question. Masked positions
contain 0; valid positions contain nonzero IDs. Transfer tensors to the supplied
device inside your implementation and return CPU arrays. Batch sizes and question
order can change. Predictions must not depend on other rows or earlier queries;
caching image features is permitted, learning from query statistics is not.

The public pilot adaptation budget is 300 seconds, 4096 MiB RAM and 64 MiB saved
artifacts per episode. Local Python calls do not enforce RAM/time isolation.
The instructor publishes the actual execution profile and GPU capacity used for
assessment; available VRAM is supplied in `limits["gpu_memory_mb"]`. The local
CPU profile sets that field to zero. GPU performance has no implied CPU equivalent.

## 5. Data and loading

The project directory is the directory containing `pyproject.toml`, `labs/`,
`step_0_update_resources` and `view_lab`. In the instructor checkout this directory
is named `public`; in a standalone student checkout it is the repository root.
Run the commands below from that directory.

```sh
./step_0_update_resources
```

The pilot contains 10,000 base-training questions, 6,800 labeled source glyphs,
1,890 base-validation questions, and eight adaptation episodes with 300 support
and 120 query questions each. The complete local benchmark has 2,850 questions,
570 at each difficulty level. Multiple questions can share a scene.

Resources appear under `data/visual_queries/`. The updater verifies the complete
archive member set, sizes and hashes, extracts files, and removes the downloaded
archive. See [the data format](DATA.md) for every CSV field, sheet coordinate,
metadata key. Text is readable for human inspection; the launcher converts it to
tokens before calling a solution. Files can be read without coursekit. The following
example is an inspection script; model training uses `data.load_split(...)`.

```python
from pathlib import Path
import numpy as np
from coursekit.labs.visual_queries_data import load_split, batch, load_glyphs

root = Path("data/visual_queries")
images, image_index, questions, answers = load_split(root, "base/train")
rows = np.arange(2)
x, tokens, mask = batch((images, image_index, questions, answers), rows)
y = answers[rows]
glyph_images, glyph_labels = load_glyphs(root)
print(x.shape, tokens.shape, mask.shape, y)
```

Images are lazy PNG tiles; `image_index` maps question rows to image rows. Multiple
questions can refer to one image. `answers` follows CSV question order and is `None` for an unlabeled split. Glyph labels use the base
symbol IDs. The optional glyph pool contains only base-training source exemplars.

[read_data.py](read_data.py) independently implements the CSV/JSON/Pillow reading
example for already verified data. Run it with:

```sh
uv run --locked python labs/visual_queries/read_data.py data/visual_queries base/train
```

## 6. Training, checking and submission

On Windows use the corresponding `.cmd` launchers.

```sh
./step_1_new_solution visual_queries my_model
./step_2_train work/visual_queries/my_model
./step_3_evaluate work/visual_queries/my_model
./step_4_pack work/visual_queries/my_model
./step_5_check_submission dist/<archive-name-printed-by-step-4>.zip
```

Use the actual ZIP path printed by the packaging command; its name includes the
submission identity. The numbered scripts use CPU PyTorch. For CUDA, select its
dependency profile on every command, as well as the model's execution device:

```sh
uv run --locked --no-group cpu --extra cuda python -m coursekit.cli train work/visual_queries/my_model --device cuda:0
uv run --locked --no-group cpu --extra cuda python -m coursekit.cli evaluate work/visual_queries/my_model --device cuda:0
uv run --locked --no-group cpu --extra cuda python -m coursekit.cli check dist/<archive-name-printed-by-step-4>.zip --device cuda:0
```

Adding `--device cuda:0` to a numbered script does not select CUDA dependencies.
See the shared [environment guide](../../docs/environment.md). Evaluation and archive checking always execute
the complete base-plus-adaptation protocol. Adaptation snapshots are retained
separately; evaluation does not replace the submitted base artifacts.
Both commands retain predictions, adaptation artifacts and learning histories
under `runs/visual_queries/<variant>/evaluate/` or `check/`. Each adaptation has
its own snapshot and each prediction recording has a unique filename; the latest
summary is `evaluate.json` or `check.json` beside those directories. Unpacking a
submission for checking does not make these outputs temporary.

The score is `100 * correct_answers / total_questions`. Every question has equal
weight. Across the complete benchmark, the five difficulty levels have exactly
equal counts. Episodes can have different sizes; the evaluator sums counts instead
of averaging episode percentages. Support examples are not scored. Valid wrong
answers earn no credit for that question. A runtime/protocol failure or cancellation
leaves an incomplete attempt, without a score computed from its successful prefix.
Saved evaluation metrics also count questions using known and new language. New
input words, new answer concepts and new episode constructions belong to the new
group. This does not mean an exact previously seen question or image. Instructor
reports additionally distinguish published training/validation, adaptation support
and queries. Training/support diagnostics do not contribute to the official score.

To start from the trainable example, use
`./step_1_new_solution visual_queries --template baseline`, then train and evaluate
`work/visual_queries/baseline`. See the [baseline description](baseline/README.md)
for the architecture, internal holdout, settings and adaptation procedure. Its
numeric training histories describe training/internal validation, while
`step_3_evaluate` runs the separate complete public benchmark.

## 7. Inspecting examples

From the project directory containing `view_lab` and `step_0_update_resources`, run:

```sh
./view_lab visual_queries
```

The Qt inspector opens without a trained solution. It shows an image and its
question, optional expected answers, and a searchable table. Select a prepared
working solution to predict complete batches. **New run** loads a fresh model and
predicts the first batch; **Next batch** advances once, while **Run remaining**
continues until paused or finished. Every row can be selected, including rows
of a shorter final batch. Selection never invokes the model again.

```sh
./view_lab visual_queries --solution work/visual_queries/baseline --device cpu
./view_lab visual_queries --recording runs/visual_queries/predictions.npz
```

For CUDA inference in the inspector, select both extras explicitly:

```sh
uv run --locked --no-group cpu --extra cuda --extra gui python -m coursekit.gui visual_queries --solution work/visual_queries/my_model --device cuda:0
```

Use **Weights…** or `--artifacts PATH` for a specific saved component bundle,
including an adapted model, and choose its matching episode split. This viewer
does not train or adapt a model automatically. The loaded code/weight identities
remain visible independently of the next selected solution. **Save recording…**
retains answers, batch boundaries and exported diagnostics. Reopening needs the
same verified corpus but no model execution; `--data PATH` selects another corpus
for a new run. Errors-only and same-image filters help examine failures.

`step_3_evaluate` and `step_5_check_submission` save uniquely named
`predictions-*.npz` files beside their adaptation runs and include the path in
their results. These records contain the
actual evaluated answers for every phase, without requesting model diagnostics.
Open them with **Open recording…** to inspect errors without another evaluation.

Models may export an optional `debug(index=0)` snapshot for any row of their last
`predict` batch. The baseline demonstrates `answer_probabilities`, indexed by
active token ID. The viewer displays the most likely answers or the full
distribution. Other scalars, vectors and matrices appear in the shared **Model
debug** panel; choose **Value** to plot a vector or a matrix heatmap. An absent
distribution is reported as unavailable. See the [diagnostic contract](../../docs/api-v3.md#prediction-diagnostics).
Inference should use evaluation mode for dropout and batch normalization.

Alternatively, open `data/visual_queries/index.html`. The offline browser shows canonical PNG images, questions, optional answers and
support/query navigation. Practice questions and answers are shown as readable text.
It does not need a trained solution or an internet connection after preparation.
The image path and one-based sheet position are shown for independent inspection.

Optional ideas for model development are in [HINTS.md](HINTS.md); they are not
additional task requirements. Source attribution is packaged as
`data/visual_queries/SOURCES.md`.
