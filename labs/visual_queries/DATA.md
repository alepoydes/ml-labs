# Visual Queries data format

Resource format 2 uses UTF-8 CSV with headers and standard quoting. Questions and
answers are ordinary English text. The launcher tokenizes them before calling a
solution; submitted code receives numeric arrays through [API 3](../../docs/api-v3.md).
The original text and vocabulary files are for people and the trusted loader.

```text
dataset.json
vocabulary.csv
SOURCES.md
index.html
base/train/{examples.csv,images/*.png}
base/validation/{examples.csv,images/*.png}
adaptation/<episode_id>/{episode.json,vocabulary.csv}
adaptation/<episode_id>/support/{examples.csv,images/*.png}
adaptation/<episode_id>/query/{examples.csv,images/*.png}
glyphs/{metadata.json,examples.csv,images/*.png}
```

Each question table has exactly these columns:

```csv
sheet,row,column,question,answer
1,2,3,What color is the symbol?,red
1,2,3,Is the symbol small?,yes
2,1,4,Is the image empty?,no
```

`sheet`, `row`, and `column` are one-based integers. Sheet 1 is
`images/000001.png`, relative to that split. Scene sheets contain a 4 × 4 grid of
256 × 256 RGB tiles. The pixel origin of a tile is
`((column-1)*width, (row-1)*height)`. Several questions may reference the same tile;
the image is stored once. Unused cells on the last sheet are ignored. Support and
query are packed separately, so support sheets contain no query images.

There are no separate image or question IDs and no answer join. A question's
human-readable reference is its split and one-based CSV record number, excluding
the header. Saved results also identify the immutable resource version/hash;
reordering a table changes that identity. All public tables have answers. A table
used for independent unlabeled inspection may leave every answer cell empty;
partially labeled tables are invalid. During assessment, the inference worker
receives no answer column.

`glyphs/examples.csv` has columns `sheet,row,column,symbol`, with one label per
referenced tile. Glyph sheets use a 16 × 16 grid of 28 × 28 RGB tiles. Their exact
shape/grid are in `glyphs/metadata.json`. The optional glyph pool contains only
base-training exemplars. Scene partitions use distinct source groups; this is not
a claim of independent writers or exhaustive removal of similar handwriting.

`vocabulary.csv` has `token_id,spelling`, consecutive IDs starting with `[PAD]` at
0. Question and answer IDs share this vocabulary. Each episode's table contains
the complete vocabulary: the unchanged base prefix followed by new entries.
Public practice spellings are readable. Future instructor spellings and their
ID assignments remain private and are never sent to solution code.

Tokenization preserves case, separates each digit, and recognizes maximal ASCII
letter sequences plus the individual punctuation tokens `?`, `.` and `,`.
Spaces separate words. Other characters are rejected. Every token must be
in the relevant dictionary. An answer is exactly one non-padding dictionary entry;
`A` and `a` are different entries. There is no external tokenizer.

`dataset.json` contains:

- `format_version` (2), `lab` (`visual_queries`), `resource_version`;
- `pad_token_id` (0), `base_vocab_size`, `max_added_tokens`;
- `image_shape` `[256,256,3]`, `sheet_grid` `[4,4]`;
- `max_question_tokens` (96), `max_batch_size` (16);
- `splits` and opaque `episodes`;
- `scored_questions_per_level`, equal across the five levels;
- `adaptation_limits`: time, RAM, GPU memory, artifact and transport budgets;
- `observable_rules`: alphabet, background, sizes, rotations, spacing, ink
  threshold, positional guards and base color representatives.

`episode.json` has exactly `format_version`, `episode_id`,
`base_vocabulary_sha256`, `base_vocab_size`, and `vocab_size`. The hash refers to
the actual base dictionary bytes. If the sizes are V and V+K, the new IDs are
V through V+K−1. IDs have meaning only within the current episode.

For independent human inspection, [read_data.py](read_data.py) reads a numbered
record with CSV/JSON/Pillow. The built-in `load_split(root, split)` returns
`images, image_index, questions, answers`: lazy RGB tiles, an int64 question-to-image
index, variable-length int64 token arrays, and an aligned int64 answer vector (or
`None` for an unlabeled table). `batch(loaded, indices)` returns uint8 RGB images,
zero-padded int64 questions and Boolean masks. Normalization belongs to the model.
The loader validates coordinates, dictionary membership, dimensions and paths.

These filesystem helpers are for launchers and inspection scripts. In `train`
and `adapt`, use `data.load_split(...)` and `data.glyphs()` instead. The same
numeric batch helper works with those tables. The instructor supplies only the
current support, public base training and glyph pool, through a bounded data
channel. The corpus directory is not mounted in the solution's container.
