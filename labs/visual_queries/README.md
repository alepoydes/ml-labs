# Visual Queries

Develop a machine-learning system that answers questions about images of
handwritten symbols. Each request consists of an image and an English question;
the answer is **one token**, such as `yes`, `A`, `red` or `small`. Questions are
independent. All references needed to answer a question occur within that question.

The task combines visual recognition, language understanding, selection by
attributes and spatial reasoning. It also includes **adaptation**: learning new
words and language constructions from a small set of labeled examples during
evaluation.

Choose the model architecture and training procedure. A transformer is one option
for combining question and image features; convolutional encoders, multilayer
perceptrons and models with separately trained components are also possible. Train
from the supplied resources. Pretrained models, external language corpora and
outside image collections are excluded. Your code must support the execution
device selected for assessment, including CUDA. CPU execution is available for
local development.

## 1. Images and spatial conventions

Images are RGB, 256 × 256 pixels, on a white background. An image contains zero
to nine nonoverlapping handwritten symbols from the alphabet
`0123456789ABCDEFGHJKLMNPQRSTUVWXYZ`: digits and uppercase letters, with `I` and
`O` omitted. Symbols have varied handwriting, positions and sizes, with rotations
of at most 7 degrees. Each symbol fits entirely inside the image, with at least
3 pixels of border and 4 pixels between ink bounding boxes.

A symbol's position is the centre of its visible ink bounding box. A symbol may
contain disconnected strokes. Image coordinates increase rightwards and
downwards. The following conventions apply to questions about position:

- Each axis is divided into thirds. The **center** is the middle third on both
  axes; a **corner** combines the corresponding outer thirds.
- An **edge** is its outer-third strip, including both corners.
- `left of`, `right of`, `above` and `below` compare centres. A diagonally placed
  symbol can satisfy two relations. These relations do not imply nearest-neighbour
  selection, and a symbol is never spatially related to itself.

Questions avoid ambiguous positional comparisons: relevant centres are separated
from equality and region boundaries by at least 0.035 of the image width or height.
The image itself has no grid.

Size is the larger dimension of the final ink bounding box:

| Word | Size in pixels |
| --- | ---: |
| tiny | 16–20 |
| small | 26–30 |
| normal | 36–40 |
| big | 46–50 |
| huge | 56–60 |

Relative-size questions compare these categories. Selecting “the largest” requires
a unique largest category among the relevant objects. Visible ink is defined
before blending with the background, at opacity at least 16/255; lower-opacity
pixels are removed. Antialiased edges belong to the same symbol.

The base color categories have RGB representatives red (175,30,45), green
(25,125,55), blue (30,65,180) and yellow (170,140,10). Actual ink varies slightly
around these representatives while remaining within distinct color categories.

## 2. Questions, vocabulary and answers

Questions range from recognition of a single symbol to combinations of attributes,
relative positions, logical conditions and references to objects already mentioned
in the question. Examples include:

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

Each supplied question has one unambiguous answer. A definite description, such as
“the blue symbol H”, selects exactly one object. An existential statement asks
whether at least one matching object exists. A universal statement has a nonempty
subject set and requires a matching witness for every subject; different subjects
may have different witnesses. `or` is inclusive. Negation applies to its stated
predicate. Pronouns such as `it` refer to objects introduced in the same question.
Answers contain exactly one token; `unknown`, `none` and multi-token responses are
outside the answer format.

The base vocabulary is listed in `vocabulary.csv`. Each entry has an integer
**token ID**. Spelling is case-sensitive, punctuation has separate entries, and
ID 0 is reserved for padding. The remaining IDs are consecutive. Questions and
answers use the same vocabulary: a prediction is the ID of the answer token.
The base training set contains examples of every question family.

## 3. Adaptation and scoring

An **adaptation episode** is a separate learning task with two sets of examples:

- **Support** contains images, questions and correct answers for learning the
  episode's new words or constructions.
- **Query** contains separate examples on which the adapted model is evaluated.
  Their answers are withheld from the model.

New token IDs are appended to the base vocabulary without gaps. Existing IDs
retain their meanings. New tokens can describe attributes, spatial relations,
logical expressions or references, and can also occur as answers. Their spellings
and grammatical classes are withheld from solution code; the adaptation procedure
must learn their meanings from support examples. The same appended ID can have
different meanings in different episodes.

Every evaluation follows this sequence:

1. Load the submitted **base state**: the saved models and preprocessing state
   produced by training. Answer the base-validation questions.
2. For each episode, start again from that same base state and run `adapt` on its
   support set. Save a complete adapted state in a separate directory.
3. Load the adapted state in a fresh solution instance and answer that episode's
   query questions.

Episodes are independent: learning in one episode is not carried into another.
During adaptation, the learning interface provides the current support set,
public base-training examples and the optional labeled glyph pool. Query data and
other episodes are unavailable through that interface. Rehearsal on base-training
examples is allowed.

The score is

```text
100 * correct_answers / total_questions
```

Every question has equal weight. The evaluator adds integer correct/total counts
across base validation and all episode queries. The complete benchmark contains
equal numbers of questions at five difficulty levels. Support examples and
training diagnostics do not contribute to the score. A wrong answer earns zero
for that question. A runtime failure, protocol violation or cancellation leaves
an incomplete attempt with no score for its completed prefix.

Reaching the adaptation time limit has a separate outcome: if your code has
committed a complete checkpoint, the runner stops learning and evaluates that
model on the full query set. Its answers receive ordinary credit. Without a
committed checkpoint, time expiry is an adaptation failure. See the
[checkpoint API](../../docs/api-v3.md#adaptation-time-and-checkpoints).

The instructor uses separate assessment examples and vocabulary additions.
Assessment includes adaptation for every submission. The execution device and
resource profile are announced for that assessment. The public adaptation budget
is 300 seconds, 4096 MiB RAM and 64 MiB saved artifacts per episode; available GPU
memory is supplied as `limits["gpu_memory_mb"]` (zero for CPU). Local Python runs
do not enforce time or memory isolation, so local elapsed time alone does not
establish that a model meets the assessment limits.

## 4. Supplied data

Run commands from the **project directory**: the directory containing
`pyproject.toml`, `uv.lock`, `labs/`, `step_0_update_resources` and `view_lab`.
The lab directory `labs/visual_queries/` contains this statement and templates.

```sh
./step_0_update_resources
```

Resources are prepared under `data/visual_queries/`. The updater verifies the
archive member set, sizes and hashes, extracts the files and removes the downloaded
archive. The supplied collection contains:

| Partition | Examples | Use |
| --- | ---: | --- |
| `base/train` | 10,000 questions | Train the base model. |
| `base/validation` | 1,890 questions | Evaluate the base model. |
| Each episode's `support` | 300 questions | Learn that episode's additions. |
| Each episode's `query` | 120 questions | Evaluate the adapted model. |
| `glyphs` | 6,800 labeled symbol images | Optionally train a visual component. |

There are eight public episodes. The local benchmark therefore contains 2,850
scored questions: 570 at each difficulty level. Multiple questions can refer to
the same scene. Scene partitions use separate groups of source handwriting
samples, with all questions about a generated scene kept together. Empty images
can be identical across partitions.

Public episodes cover new colors, spatial expressions, logical constructions,
references and combinations of additions. They are displayed as **Adaptation 01**
through **Adaptation 08**, in the order listed in `dataset.json["episodes"]`.
The underlying directories have opaque episode IDs; the split menu shows the
actual path in its tooltip.

All public question tables include readable answers for local assessment.
Keep query answers out of that episode's adaptation training. You may use public
episodes to develop the adaptation procedure; reserving entire episodes for final
local testing helps assess how well the procedure generalizes to new additions.
For internal validation during base training or adaptation, split the authorized
training or support examples yourself, keeping questions about the same image
together. See [modelling hints](HINTS.md) for further suggestions.

### Loading examples

Images are stored in lossless PNG sheets; questions, answers and sheet coordinates
are stored in readable CSV tables. [The data-format reference](DATA.md) defines
every column, coordinate convention and metadata field, so you can write an
independent reader. The built-in loader provides lazy image access and numeric
question/answer arrays. This inspection script loads two question rows:

```python
from pathlib import Path

import numpy as np
from coursekit.labs.visual_queries_data import batch, load_glyphs, load_split

root = Path("data/visual_queries")
loaded = load_split(root, "base/train")
images, image_index, questions, answers = loaded
rows = np.arange(2)
x, tokens, mask = batch(loaded, rows)
y = answers[rows]
glyph_images, glyph_labels = load_glyphs(root)
print(x.shape, tokens.shape, mask.shape, y)
```

`image_index` maps question rows to image rows. `answers` follows CSV question
order; it is `None` for an unlabeled table. Glyph labels are base-vocabulary
symbol IDs. The optional glyph pool contains base-training source exemplars.

[read_data.py](read_data.py) demonstrates independent CSV/JSON/Pillow loading of
already verified data. Its optional final argument is a one-based CSV record
number, defaulting to 1:

```sh
uv run --locked python labs/visual_queries/read_data.py data/visual_queries base/train 1
```

These filesystem loaders are for inspection scripts. Training and adaptation
receive the numeric data interface described next.

## 5. Implementing a solution

The **starter** is a complete interface example that always answers `yes`.
The **baseline** is a small trainable CNN/MLP model combining image features with
ordered question embeddings. Both demonstrate the full training, adaptation and
prediction lifecycle. Copy one into `work/` and implement:

1. `train.py:train(...)`: train one or several components and save the complete
   base state in `output_dir`.
2. `train.py:adapt(...)`: load the base state from `input_dir`, learn from the
   support data and save a complete adapted state in a separate `output_dir`.
   Leave the input state unchanged.
3. `solution.py:Solution(...)` and `predict(...)`: load the saved state and return
   one answer ID per question.

The entry-point signatures are:

```python
def train(
    data, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
): ...


def adapt(
    data, input_dir, output_dir, config, seed, *, device, limits,
    checkpoint=None, report=None,
): ...


class Solution:
    def __init__(self, artifact_dir, config, seed, *, device): ...
    def predict(self, images, question_tokens, question_mask): ...
```

`data` provides numeric tables and image batches through [API 3](../../docs/api-v3.md).
During base training, use `data.load_split("base/train")` and optionally
`data.glyphs()`. During adaptation, load new examples with
`data.load_split("support")`; `base/train` and `glyphs()` remain available for
rehearsal. Token spellings and corpus files are outside this learning interface.

`config` is the parsed `config.toml`; artifact directories are `pathlib.Path`
objects. `device` is an explicit device string such as `cpu` or `cuda:0`.
`data.episode["vocab_size"]` gives the active vocabulary size, including all added
IDs. Extend both input and output representations as needed. Save the active
vocabulary size and episode identity with the models.

Saved **artifacts** include every model and preprocessing parameter needed for
prediction and further training. A solution may contain several model files.
Named stages and component inputs support separate training; `train` and `adapt`
can report progress and return histories grouped by stage. The starter also
supports manual continuation on base-training data. See the
[development guide](../../docs/development.md) and [component-training API](../../docs/api-v2.md)
for these shared conventions.

Use the supplied `checkpoint()` callback after saving a complete state to retain
it if adaptation reaches its time limit. `checkpoint(history)` also retains the
current numeric histories. The starter demonstrates this call; the baseline saves
an initial expanded model and each completed adaptation epoch. Track the remaining
time from `limits["wall_seconds"]` and leave time for saving and transfer.

Keep `api_version = 3` and `training.supports_adapt = true` in `submission.toml`.
Document your method and [AI assistance](../../docs/ai-use.md). The runner writes
`provenance.json` to identify completed training; leave that file under its control.

### Prediction batches

| Value | CPU NumPy representation |
| --- | --- |
| `images` | `uint8`, `(B,256,256,3)`, RGB |
| `question_tokens` | `int64`, `(B,L)` |
| `question_mask` | `bool`, `(B,L)`, nonempty true prefix |
| Return value | `int64`, `(B,)`, one non-padding active token ID per row |

Batch sizes range from 1 to 16; questions contain at most 96 tokens. Masked
positions contain 0 and valid positions contain nonzero IDs. Transfer tensors to
the supplied device within your implementation and return CPU arrays. Use
evaluation mode for dropout and batch normalization during prediction.

Batch size and question order may change. Each prediction must depend only on
that row's image and question and the loaded model state. Caching image features
is permitted; learning from query statistics is excluded.

## 6. Training, checking and submission

On Windows, use the corresponding `.cmd` launchers. These commands create a
working copy of the starter, prepare its artifacts, evaluate it and package it:

```sh
./step_1_new_solution visual_queries my_model
./step_2_train work/visual_queries/my_model
./step_3_evaluate work/visual_queries/my_model
./step_4_pack work/visual_queries/my_model
./step_5_check_submission dist/<archive-name-printed-by-step-4>.zip
```

Use the actual ZIP path printed by step 4; its name includes the submission
identity. To start from the trainable baseline, create it with
`./step_1_new_solution visual_queries --template baseline`, then use
`work/visual_queries/baseline` in the remaining commands. Its
[description](baseline/README.md) explains the architecture and training settings.

The numbered scripts select CPU PyTorch. For CUDA, select both the dependency
profile and the model device on every command:

```sh
uv run --locked --no-group cpu --extra cuda python -m coursekit.cli train work/visual_queries/my_model --device cuda:0
uv run --locked --no-group cpu --extra cuda python -m coursekit.cli evaluate work/visual_queries/my_model --device cuda:0
uv run --locked --no-group cpu --extra cuda python -m coursekit.cli check dist/<archive-name-printed-by-step-4>.zip --device cuda:0
```

Passing `--device cuda:0` to a numbered script alone does not select CUDA
dependencies. See the [environment guide](../../docs/environment.md).

### Reading evaluation results

Evaluation and archive checking both run base validation and every independent
adaptation. The base artifacts remain unchanged. Both commands print a table with
correct/total counts and accuracy for **Base validation**, each numbered adaptation
and **Total**. It also separates questions using known and new language. New input
words, new answer concepts and new episode constructions count as new language;
this distinction concerns language coverage, not whether an exact question or image
has appeared before. A dash indicates a group with no questions.
Adaptation rows also show elapsed seconds against the available budget, the stop
reason and the evaluated model: final output or a numbered committed checkpoint.

The latest summary is saved as `evaluate.json` or `check.json` under
`runs/visual_queries/<variant>/`. Predictions, independent adaptation snapshots
and learning histories are retained under the accompanying `evaluate/` or `check/`
directory. Each prediction recording has a unique filename printed in the results.
Checking an archive retains these outputs after its temporary extraction is
removed. The complete JSON also preserves technical episode IDs.

Training histories describe learning on training and internal-validation examples.
The evaluation table describes the separate public benchmark. Only base validation
and adaptation queries contribute points; support diagnostics are kept separate.
The instructor's reports additionally distinguish published training, validation,
support and query statistics. The evaluator reports each adapted model's result;
it does not automatically run a second, unadapted model for comparison.

## 7. Inspecting examples and predictions

Run the graphical inspector from the project directory:

```sh
./view_lab visual_queries
```

It opens without a trained solution and shows images, readable questions and a
searchable table. Expected answers can be revealed for inspection. To run a
prepared working solution, select its directory or supply it on the command line:

```sh
./view_lab visual_queries --solution work/visual_queries/baseline --device cpu
```

**New run** loads a fresh model and predicts the first complete batch. **Next
batch** advances once; **Run remaining** continues until paused or finished.
Select any row to inspect its saved prediction, including rows in a short final
batch. Selecting a row never reruns inference. Errors-only and same-image filters
help examine failures.

Use **Weights…** or `--artifacts PATH` to select a saved component bundle. For an
adapted model, choose the matching episode split. The inspector uses existing
artifacts; training and adaptation run through the commands in section 6. The
loaded code and weight identities are shown separately from the next selected
solution. `--data PATH` selects another corpus for inspection.

For CUDA inference, include both the CUDA and GUI extras:

```sh
uv run --locked --no-group cpu --extra cuda --extra gui python -m coursekit.gui visual_queries --solution work/visual_queries/my_model --device cuda:0
```

### Recordings and model diagnostics

**Save recording…** retains predictions, batch boundaries and exported diagnostics.
**Open recording…** replays them against the same verified corpus without executing
the model. To open a recording from evaluation or archive checking, use its printed
`predictions-*.npz` path:

```sh
./view_lab visual_queries --recording <path-to-predictions.npz>
```

Evaluation recordings contain the actual answers from every evaluated phase.
Ordinary evaluation does not request model diagnostics.

A model can optionally export `debug(index=0)` for any row of its last prediction
batch. The baseline demonstrates `answer_probabilities`, a vector indexed by active
token ID. The inspector shows the most likely answers or the full distribution;
if probabilities are absent, it reports them as unavailable. Other exported
scalars, vectors and matrices appear in **Model debug**. Choose **Value** to plot a
vector or a matrix heatmap. See the [diagnostic contract](../../docs/api-v3.md#prediction-diagnostics).

### Offline data browser

Open `data/visual_queries/index.html` to browse canonical PNG images, questions and
answers, with support/query navigation. It works without a model or internet access
after resource preparation. Each example shows its PNG path and one-based sheet
position for independent inspection. Source attribution is included in
`data/visual_queries/SOURCES.md`.
