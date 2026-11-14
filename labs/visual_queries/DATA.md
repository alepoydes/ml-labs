# Visual Queries data format

All text files are UTF-8. CSV files have a header and standard CSV quoting. JSON
arrays embedded in CSV are quoted by the CSV writer. Join records by IDs, never by
assuming that image, question and answer rows have the same order.

```text
dataset.json
vocabulary.csv
SOURCES.md
index.html
base/train/{images.csv,questions.csv,answers.csv,images/*.png}
base/validation/{images.csv,questions.csv,answers.csv,images/*.png}
adaptation/<episode_id>/episode.json
adaptation/<episode_id>/support/{images.csv,questions.csv,answers.csv,images/*.png}
adaptation/<episode_id>/query/{images.csv,questions.csv,answers.csv,images/*.png}
glyphs/{metadata.json,images.csv,labels.csv,images/*.png}
```

Scene PNG sheets contain a 4 × 4 grid of 256 × 256 RGB tiles. Glyph sheets have
16 × 16 tiles of 28 × 28 pixels. Each indexed image has one canonical tile. Final
sheets can have unused white tiles. Images are not also stored in numeric arrays.

| File | Columns |
| --- | --- |
| `vocabulary.csv` | `token_id,spelling`: consecutive IDs from 0; `[PAD]` at 0 |
| `images.csv` | `image_id,sheet,row,column`: PNG path relative to the split, one-based row/column |
| `questions.csv` | `question_id,image_id,token_ids`: unpadded JSON integer array |
| `answers.csv` | `question_id,answer_id`: exactly one nonzero active token ID per question |
| `glyphs/labels.csv` | `image_id,symbol_id`: base dictionary symbol ID |

IDs and PNG paths are identifiers only. Question/image IDs are unique within a
split. Repeated images across several questions share an image ID. Public answers
are provided for every split, including validation and adaptation queries. An
unlabeled evaluation view omits `answers.csv`; the inference API receives neither
IDs nor sheets, only the requested image crops and token arrays.

`dataset.json` contains:

- `format_version` (1), `lab` (`visual_queries`), `resource_version`;
- `pad_token_id` (0), `base_vocab_size`, `max_added_tokens`;
- `image_shape` `[256,256,3]`, `sheet_grid` `[4,4]`;
- `max_question_tokens` (96), `max_batch_size` (16);
- `splits`: available relative split paths; `episodes`: opaque episode IDs;
- `scored_questions_per_level`: the equal count used in each of five levels;
- `adaptation_limits`: pilot time/RAM/artifact budgets and transfer/output bounds;
- `observable_rules`: alphabet, background, size intervals, rotation range, spacing,
  ink threshold, positional guards and base color representatives.

`episode.json` has exactly `format_version`, `episode_id`,
`base_vocabulary_sha256`, `base_vocab_size` and `vocab_size`. If the latter two are
V and V+K, the new IDs are V through V+K−1. No supplemental spelling table is supplied.
The hash refers to the actual bytes of the base `vocabulary.csv`.

`glyphs/metadata.json` supplies `image_shape` and `sheet_grid` for that pool. Source
glyph classes remain distinct. Held-out scene partitions use different source
exemplars/group assignments from the glyph pool; this is not a claim of independent
writers or exhaustive removal of visually similar handwriting.

During adaptation the root changes to:

```text
dataset.json
vocabulary.csv
episode.json
support/{images.csv,questions.csv,answers.csv,images/*.png}
rehearsal/dataset.json
rehearsal/vocabulary.csv
rehearsal/base/train/{images.csv,questions.csv,answers.csv,images/*.png}
rehearsal/glyphs/{metadata.json,images.csv,labels.csv,images/*.png}
```

Use `load_split(data_dir, "support")`, `load_split(data_dir / "rehearsal",
"base/train")` and `load_glyphs(data_dir / "rehearsal")`. The current episode is
read with `episode(data_dir)`. Support and query images are packed separately;
support sheets contain no query tiles. This view excludes validation, other
practice episodes and query files. The runner authenticates the complete file set.

[read_data.py](read_data.py) shows independent loading. Convert `(row,column)` to
pixel offsets as `((column-1)*width, (row-1)*height)`. RGB channels are last and
remain uint8; normalization is the model's responsibility. The built-in
`batch(loaded, indices)` pads with zero and returns a Boolean mask. The loader
validates IDs, integer ranges, dimensions and paths before returning arrays.
