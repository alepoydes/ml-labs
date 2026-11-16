# Submission format

Develop and test a working copy under `work/<lab>/<name>/`, created from a template
in `labs/` with `step_1_new_solution`. Keep tracked course files unchanged so they
can be updated without conflicts. `step_4_pack` packages the working copy into
`dist/`; submit the resulting ZIP, not the template or working directory.

One ZIP contains one variant of one lab, with files directly at its root:

```text
submission.toml
config.toml
train.py
solution.py
lib/                      optional local helpers
artifacts/                saved models, preprocessing and state needed for further training
    provenance.json       written after successful training
package.json              file hashes, written by packaging
```

Use portable relative filenames (letters, digits, underscores, hyphens and dots).
Links, special files, duplicate paths, traversal, and encrypted members are rejected.
Datasets, environments, caches, logs, and unrelated variants are excluded. Do not use
standalone NPY files; image resources use PNG and other numeric arrays compressed NPZ.

`submission.toml` declares `schema_version = 1`, the lab name (`blindjack`,
`line_following` or `visual_queries`), an API version, a variant name and
[AI usage](ai-use.md). Blindjack and Line Following accept API 1 or 2;
Visual Queries requires API 3 and `training.supports_adapt = true`.
Student identity comes from the instructor's submission
directory, not this file. The packer preserves the resource fingerprint of actual
training even if newer data have been downloaded. It verifies source and config
hashes against training provenance, and includes hashes of every packaged file.

Training provenance records lab, API, source/configuration/resource hashes, seed,
Python and relevant library versions, and completion status. Failed training does
not leave successful provenance. It is reproducibility metadata, not an automatic
proof of substantive learning or authorship.

The training API determines how learning data and model artifacts are supplied.
[API 1](development.md#call-the-same-functions-from-python) uses
`train(data_dir, output_dir, config, seed)` and
`Solution(artifact_dir, config, seed)`. [API 2](api-v2.md) adds explicit devices,
named training stages, input snapshots and grouped histories. [API 3](api-v3.md)
uses those component conventions with numeric learning data and episode adaptation.
Each lab defines its prediction or control method:

| Lab | Required acting method | What persists during evaluation |
| --- | --- | --- |
| [Blindjack](../labs/blindjack/README.md) | `select(images)` returns a seven-card Boolean mask. | One instance handles the hand set. |
| [Line Following](../labs/line_following/README.md) | `act(observation, reset)` returns float32 PyTorch commands of shape `(B, 2)`. | Keep independent memory for each batch row; reset it on `reset`, not on turnaround. |
| [Visual Queries](../labs/visual_queries/README.md) | `predict(images, question_tokens, question_mask)` returns CPU int64 answer IDs of shape `(B,)`. | Load the base or current adapted state; each question is independent. |

Save learned weights and other fitted parameters under `artifacts/`. Mutable robot
episode memory belongs to the controller instance; a GUI recording is not a trained
model or a simulator checkpoint. The optional `debug(index=0)` supplies graphical
diagnostics; ordinary grading does not call it. The hand-tuned robot baseline's preparation saves
settings; packaging it successfully does not demonstrate learning.

The local archive check validates the ZIP and evaluates a fresh extraction in your
normal Python environment. Keep model paths relative to `artifact_dir`, not to
your original working directory. The check does not restrict filesystem or network
access. Instructor grading has separate [environment requirements](environment.md).
