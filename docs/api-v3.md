# Numeric learning API for Visual Queries

[Visual Queries](../labs/visual_queries/README.md) uses API 3 for training,
adaptation and batched prediction. Set `api_version = 3` and
`training.supports_adapt = true` in `submission.toml`.

The runner supplies a numeric data object to learning code and paths to model
artifacts. Your implementation owns the model architecture, optimizer, sampling
and training loop. It can train one model or several components. Named stages,
input snapshots, progress reporting and grouped histories follow the
[component-training conventions](api-v2.md).

## Learning entry points

Define these functions in `train.py`:

```python
def train(
    data, output_dir, config, seed, *, device, inputs=None, stage="all", report=None
): ...


def adapt(data, input_dir, output_dir, config, seed, *, device, limits, report=None): ...
```

`output_dir`, `input_dir` and named `inputs` are paths to model artifacts. Load
PyTorch checkpoints on the supplied `device` and transfer tensors to it explicitly.
Every adaptation starts from the submitted base state in `input_dir` and saves a
complete, independently loadable state in `output_dir`. Leave input artifacts
unchanged. The [task statement](../labs/visual_queries/README.md#3-adaptation-and-scoring)
defines the evaluation sequence and adaptation budget.

## Available data

`data.metadata` contains `base_vocab_size`, `max_question_tokens` and `image_shape`.
`data.episode` provides the active `vocab_size` and an opaque `episode_id`.
Outside an adaptation episode, `episode_id` is `None`. During an episode, the
metadata also identifies the base vocabulary by its size and fingerprint. Token
spellings and grammatical classes are withheld from this interface.

`data.splits` lists the authorized table names:

| Operation | Available data |
| --- | --- |
| Training from scratch or manual continuation | `base/train`, `glyphs` |
| Episode adaptation | `support`, `base/train`, `glyphs` |

Tables are reusable: repeated loading returns the same data. Validation and query
tables are reserved for evaluation. To form an internal validation set, divide the
authorized training or support rows while keeping all questions about an image
together. The standard evaluation runner executes the public adaptation episodes
for local testing.

Within `train` or `adapt`, a batch can be loaded as follows:

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
# Convert to device tensors and run your forward/backward/optimizer steps.
```

`image_index` is a zero-based in-memory index identifying shared images.
`questions` contains unpadded int64 arrays, and `answers` is an aligned int64
vector. The `batch` helper resolves each question's image and pads questions
within the batch.

`images[indices]` returns uint8 RGB crops for a one-dimensional integer index
array, preserving order and duplicates. `images.shape` is one image's `(H,W,3)`;
`len(images)` is the number of distinct images. Large requests are divided into
bounded transport batches. Keep decoded-image storage within your memory budget.

`data.glyphs()` returns a reusable image collection and aligned numeric symbol
labels for optional visual pretraining.

### Data access during assessment

The trusted loader reads and tokenizes corpus files outside the isolated solution
process. It serves authorized numeric tables and bounded image batches, with no
corpus directory mounted in that process. Question text, vocabulary tables, PNG
paths and scene annotations remain with the loader. Reference answers accompany
training/support rows; the prediction call receives only its input arrays.

Local commands use the same numeric interface directly in Python. The readable
public corpus remains available on your computer for inspection, while the
assessment process enforces access to its authorized data.

## Prediction and diagnostics

Define `Solution(artifact_dir, config, seed, *, device)` in `solution.py`. Its
`predict(images, question_tokens, question_mask)` method returns CPU int64 answer
IDs of shape `(B,)`. See the task statement for
[input shapes and batch rules](../labs/visual_queries/README.md#prediction-batches).

### Prediction diagnostics

An optional `debug(index=0)` method exports diagnostics for row `index` of the
**last successful prediction batch**, where `0 <= index < B`. The inspector calls
it for every row before another prediction, then retains detached copies.
Selecting a row displays the saved prediction and diagnostics. In `debug`, leave
weights and BatchNorm statistics unchanged and reuse the stored prediction result.
Call `model.eval()` when preparing inference.

Return a dictionary with at most 64 short field names. Values can be finite numeric
scalars, vectors or matrices of at most 4,096 elements, or strings up to 4,096
characters. NumPy arrays and detached CPU/CUDA tensors are accepted locally;
isolated execution uses the same validated numeric transport.

The optional `answer_probabilities` field is a vector of length `vocab_size`,
indexed by token ID (up to 4,352 entries, including appended tokens). Values must
lie in [0,1], sum to one within tolerance 1e-5, and have zero probability at index
0. If your model uses private class indices, convert them to vocabulary IDs before
exporting. The answer returned by `predict` is the one used for evaluation.
For the supplied baseline, which excludes padding from its logits:

```python
def debug(self, index=0):
    return {
        "answer_probabilities": self.last_logits[index].float().softmax(-1),
    }
```

Retain `last_logits` from the actual batched `predict` call, detached from autograd.
The inspector translates token IDs into readable labels. Additional fields can
show features or small attention matrices; choose **Value** in the shared
diagnostic plot to display a vector or matrix. Ordinary evaluation calls only
`predict`; diagnostics and probability distributions are optional.

## Migrating an API 2 Visual Queries solution

API 3 retains component training and explicit device selection, and replaces
filesystem data access with the numeric learning object. In an older solution,
replace `load_split(data_dir, name)` with `data.load_split(name)`,
`metadata(data_dir)` with `data.metadata`, and `episode(data_dir)` with
`data.episode`. Use `base/train` for rehearsal. Remove vocabulary/file access from
learning code and use the supplied vocabulary sizes.

Update the API declaration, retrain or explicitly assemble compatible components,
and package a new archive with matching provenance. Existing snapshots and reports
retain their original training identities. Blindjack and Line Following use their
[API 1/2 contracts](api-v2.md).
