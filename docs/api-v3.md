# Numeric learning data for Visual Queries

Visual Queries uses `api_version = 3` in `submission.toml`, with
`training.supports_adapt = true`. Blindjack and Line Following keep their existing
API 1/2 contracts. Version 3 retains the component stages, artifact snapshots,
progress reporting, histories and explicit device selection of [API 2](api-v2.md).
Its first argument to `train` and `adapt` is a numeric data object instead of a
filesystem path. Existing Visual Queries API 2 solutions need migration.

```python
def train(
    data, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
): ...


def adapt(data, input_dir, output_dir, config, seed, *, device, limits, report=None): ...
```

`output_dir`, `input_dir`, and named `inputs` remain paths to model artifacts.
The solution controls its models, optimizer, sampling and training loop. The
framework supplies no training algorithm and does not require one neural network.

`data.metadata` contains only `base_vocab_size`, `max_question_tokens` and
`image_shape`. `data.episode` contains `vocab_size` and an opaque `episode_id`
(`None` outside an episode); episode calls also retain numeric sizes and the base
vocabulary fingerprint. No spelling or grammatical class is supplied.
`data.splits` lists authorized table names:

- Scratch training and manual continuation: `base/train`, `glyphs`.
- Episode adaptation: `support`, `base/train`, `glyphs`.

All tables are reusable; repeated loading does not consume a stream. Query and
validation tables are unavailable through this interface. Reserve your own
internal holdout from authorized training rows, keeping questions sharing an
image together. Practice adaptation episodes can be run through the standard
runner to develop your method; their queries are evaluation data.

```python
import numpy as np
from coursekit.labs.visual_queries_data import batch

loaded = data.load_split(
    "support" if data.episode["episode_id"] is not None else "base/train"
)
images, image_index, questions, answers = loaded
rows = np.arange(min(16, len(questions)))
x, tokens, mask = batch(loaded, rows)
y = answers[rows]
# Convert to device tensors and run your own forward/backward/optimizer steps.
```

`image_index` is a zero-based in-memory index identifying shared images, not a
filename or CSV identifier. `questions` contains unpadded int64 arrays; `answers`
is an aligned int64 vector. `images[indices]` returns `uint8` RGB crops for a
one-dimensional integer index array. Its `shape` is one image's `(H,W,3)`;
`len(images)` is the number of distinct images. Request order and duplicates are
preserved. Large image requests are split into bounded transport batches. Avoid
loading every decoded image at once when the resource or memory budget is large.

`data.glyphs()` returns a reusable image collection and aligned numeric symbol
labels for optional visual pretraining. All values crossing the learning boundary
are numeric arrays or explicit structural metadata. Question text, dictionaries,
PNG paths and instructor annotations are not available to submitted code.

The instructor's trusted loader reads and tokenizes the corpus outside the
isolated solution process. The process requests allowlisted numeric tables and
image batches; it receives no corpus filesystem mount. Inference still uses
`predict(images, question_tokens, question_mask)` and returns CPU int64 answer IDs.
Answers remain on the host during inference. Each episode loads the original
submitted base and saves an independent complete adapted snapshot.

Local commands use the same numeric interface directly in Python. This is a
convenience API, not a security boundary against your own local code: readable
practice files are on your computer. Instructor isolation supplies the actual
boundary for private examples and word mappings.

To migrate an API 2 Visual Queries solution, replace `load_split(data_dir, name)`
with `data.load_split(name)`, `metadata(data_dir)` with `data.metadata`, and
`episode(data_dir)` with `data.episode`. Replace the old rehearsal split path with
`base/train`. Remove vocabulary/file access and use the supplied vocabulary sizes.
Update the declared API, retrain or explicitly assemble compatible components,
then pack a new archive with matching provenance. Old snapshots and reports remain
historical evidence; they are not silently relabeled as newly trained solutions.
