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
artifacts/                inference model, fitted parameters or controller settings
    provenance.json       written after successful training
package.json              file hashes, written by packaging
```

Use portable relative filenames (letters, digits, underscores, hyphens and dots).
Links, special files, duplicate paths, traversal, and encrypted members are rejected.
Datasets, environments, caches, logs, and unrelated variants are excluded. Do not use
standalone NPY files; image resources use PNG and other numeric arrays compressed NPZ.

`submission.toml` declares schema/API version 1, `lab = "blindjack"` or
`lab = "line_following"`, a variant name,
and [AI usage](ai-use.md). Student identity comes from the instructor's submission
directory, not this file. The packer preserves the resource fingerprint of actual
training even if newer data have been downloaded. It verifies source and config
hashes against training provenance, and includes hashes of every packaged file.

Training provenance records lab, API, source/configuration/resource hashes, seed,
Python and relevant library versions, and completion status. Failed training does
not leave successful provenance. It is reproducibility metadata, not an automatic
proof of substantive learning or authorship.

Both labs implement `train(data_dir, output_dir, config, seed)` and construct
`Solution(artifact_dir, config, seed)`. Their acting interfaces differ:

| Lab | Required acting method | What persists during evaluation |
| --- | --- | --- |
| [Blindjack](../labs/blindjack/README.md) | `select(images)` returns a seven-card Boolean mask. | One instance handles the hand set. |
| [Line Following](../labs/line_following/README.md) | `act(observation, reset)` returns float32 PyTorch commands of shape `(B, 2)`. | Keep independent memory for each batch row; reset it on `reset`, not on turnaround. |

Save learned weights and other fitted parameters under `artifacts/`. Mutable robot
episode memory belongs to the controller instance; a GUI recording is not a trained
model or a simulator checkpoint. The optional `debug(index=0)` is for local inspection
and is never called by the judge. The hand-tuned robot baseline's preparation saves
settings; packaging it successfully does not demonstrate learning.

The local archive check validates the ZIP and evaluates a fresh extraction in your
normal Python environment. Keep model paths relative to `artifact_dir`, not to
your original working directory. The check does not restrict filesystem or network
access. Instructor grading has separate [environment requirements](environment.md).
