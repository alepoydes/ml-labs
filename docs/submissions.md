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
artifacts/                model files and preprocessing state
    provenance.json       written after successful training
package.json              file hashes, written by packaging
```

Use portable relative filenames (letters, digits, underscores, hyphens and dots).
Links, special files, duplicate paths, traversal, and encrypted members are rejected.
Datasets, environments, caches, logs, and unrelated variants are excluded. Do not use
standalone NPY files; image resources use PNG and other numeric arrays compressed NPZ.

`submission.toml` declares schema/API version 1, `lab = "blindjack"`, a variant name,
and [AI usage](ai-use.md). Student identity comes from the instructor's submission
directory, not this file. The packer preserves the resource fingerprint of actual
training even if newer data have been downloaded. It verifies source and config
hashes against training provenance, and includes hashes of every packaged file.

Training provenance records lab, API, source/configuration/resource hashes, seed,
Python and relevant library versions, and completion status. Failed training does
not leave successful provenance. It is reproducibility metadata, not an automatic
proof of substantive learning or authorship.

The fresh archive check evaluates only extracted files in the reference environment.
Keep model paths relative to `artifact_dir`, not to your original working directory.
Model deserialization takes place inside the worker. Do not load untrusted models
manually in the instructor environment.
