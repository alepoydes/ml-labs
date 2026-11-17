# Visual Queries data format

This reference describes the files prepared under `data/visual_queries/` by
`step_0_update_resources`. It complements the [task statement](README.md).
Resource format 2 stores images in lossless PNG sheets and questions and answers
in UTF-8 CSV tables with headers and standard quoting.

The files are readable independently of coursekit. During training, adaptation
and prediction, the runner converts text to token IDs and supplies numeric arrays
through [API 3](../../docs/api-v3.md). Dictionaries and readable questions remain
with the loader.

This dataset directory is created by resource preparation and is ignored by Git.
Its `index.html` and PNG sheets arrive with the data archive; look under
`data/visual_queries/` after running the updater. The tracked `labs/visual_queries/`
directory contains the task description, templates, reader example and resource
manifest.

## Directory layout

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

`base/train` supplies base-model training examples; `base/validation` supplies
the initial diagnostic and old control questions assigned to episodes. Each adaptation directory contains a labeled
**support** set for learning its additions and a separate **query** set for
checking the adapted model. Every episode starts from the same submitted base
state. The optional `glyphs` directory contains labeled images of individual
handwritten symbols for visual pretraining. `index.html` is an offline data
browser, and `SOURCES.md` records source attribution.

The displayed labels **Adaptation 01**, **Adaptation 02**, and so on follow
`dataset.json["episodes"]` order. Use the corresponding episode ID in file paths.
These labels identify independent tasks, not successive stages of one training run.

## Questions and PNG coordinates

Each `examples.csv` question table has exactly these columns:

```csv
sheet,row,column,question,answer
1,2,3,What color is the symbol?,red
1,2,3,Is the symbol small?,yes
2,1,4,Is the image empty?,no
```

`sheet`, `row` and `column` are one-based integers. Sheet 1 is
`images/000001.png`, relative to that split. Scene sheets contain a 4 × 4 grid of
512 × 512 RGB tiles. For tile width `width` and height `height`, the tile's
upper-left pixel is

```text
x = (column - 1) * width
y = (row - 1) * height
```

Several questions can reference one tile; its image is stored once. Unused cells
on the last sheet are ignored. Support and query images are packed separately.

Every CSV record contains the complete question and answer, so a separate join
is unnecessary. Identify an example for debugging by its split and one-based CSV
record number, excluding the header. Saved results also retain the immutable
resource version and hash; reordering a table changes that identity.

All public question tables include answers. A custom table for unlabeled
inspection can leave every answer cell empty; partially labeled tables are
invalid. The prediction call receives only images, question tokens and masks.
Reference answers are used by the evaluator.

## Labeled glyphs

`glyphs/examples.csv` has columns `sheet,row,column,symbol`, with one label per
referenced tile. Glyph sheets use a 16 × 16 grid of 28 × 28 RGB tiles. The exact
shape and grid are recorded in `glyphs/metadata.json`.
Each glyph has opaque black ink on white, with a 24-pixel maximum ink-box dimension
centered in its tile. Scene glyphs use colored ink and the size bands in the task
statement. Keep these scale and margin conventions in mind when reusing an encoder
trained on the glyph pool.

The glyph pool contains base-training source exemplars. Scene partitions use
distinct source groups; writer identities and similarity between handwriting
samples are not guaranteed to be disjoint. Glyph labels use the base vocabulary's
symbol IDs when loaded into numeric arrays.

## Vocabulary and tokenization

`vocabulary.csv` has columns `token_id,spelling`, with consecutive IDs starting
at 0 for `[PAD]`. Questions and answers share this vocabulary. Each episode's
vocabulary contains the unchanged base prefix followed by its new entries.
Public spellings are readable for inspection. Assessment additions have private
word-to-ID mappings; these are withheld from solution code.

Tokenization preserves case and recognizes:

- maximal sequences of ASCII letters;
- each digit as an individual token;
- each punctuation character `?`, `.` or `,` as an individual token.

Spaces separate words. Other characters are rejected, and every token must occur
in the relevant vocabulary. An answer is exactly one non-padding entry.
For example, `Is the image empty?` is tokenized as `Is`, `the`, `image`, `empty`,
`?`. `A` and `a` are distinct spellings. The loader uses these rules directly,
without an external tokenizer.

## Metadata

`dataset.json` describes the collection:

| Fields | Meaning |
| --- | --- |
| `format_version`, `lab`, `resource_version` | Format version 2, lab name `visual_queries` and resource version. |
| `pad_token_id`, `base_vocab_size`, `max_added_tokens` | Padding ID 0, base vocabulary size and maximum additions per episode. |
| `image_shape`, `sheet_grid` | Scene tile shape `[512,512,3]` and sheet grid `[4,4]`. |
| `max_question_tokens`, `max_batch_size` | Limits of 96 tokens and 16 question rows. |
| `splits`, `episodes` | Available split paths and ordered episode IDs. |
| `assessment_version`, `retention_rows` | Schedule 2; episode IDs mapped to one-based `base/validation` CSV record numbers. The lists partition every base-control row exactly once. Missing schedule metadata denotes legacy schedule 1. |
| `scored_questions_per_level` | Equal question count at each of the five difficulty levels. |
| `adaptation_limits` | Time, RAM, GPU memory, saved-artifact and data-transport budgets. |
| `observable_rules` | Alphabet, background, size ranges, rotations, spacing, opaque ink style, positional guards and base color representatives. |

`episode.json` has exactly five fields: `format_version`, `episode_id`,
`base_vocabulary_sha256`, `base_vocab_size` and `vocab_size`. The hash identifies
the actual base vocabulary file bytes. If the vocabulary grows from V to V+K,
the appended IDs are V through V+K−1. An appended ID's meaning is specific to
its episode.

## Built-in and independent readers

[read_data.py](read_data.py) demonstrates reading one record using CSV, JSON and
Pillow. The [loading example](README.md#loading-examples) uses the built-in helpers:

- `load_split(root, split)` returns `images, image_index, questions, answers`.
  These are lazy RGB tiles, an int64 question-to-image index, variable-length
  int64 question-token arrays and an aligned int64 answer vector (`None` for an
  unlabeled table).
- `batch(loaded, indices)` returns uint8 RGB images, zero-padded int64 questions
  and Boolean masks. `indices` selects question rows. Pixel normalization is
  the model's responsibility.
- `load_glyphs(root)` returns a lazy image collection and aligned numeric symbol
  labels.

In memory, `image_index` is zero-based. The loader validates sheet coordinates,
dimensions, paths and dictionary membership.

These filesystem helpers are for inspection and launcher code. Inside `train`
and `adapt`, use `data.load_split(...)` and `data.glyphs()`; the same `batch`
helper works with their numeric tables. The learning interface supplies only
base training and glyphs, plus the current support set during adaptation. An
isolated assessment process receives bounded numeric batches without access to
the corpus directory.
