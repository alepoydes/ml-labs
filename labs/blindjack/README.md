# Blindjack

Recognize a hidden integer from an image and select cards whose sum approaches 21.
Each image has a value determined by an unknown deterministic mapping. Learn this
mapping from labeled examples.

A hand contains seven RGB images representing seven distinct values sampled from
the integers 0 through 10. Order has no numerical meaning. Select any subset,
including none or all seven cards. For the true selected sum `s`, your reward is
`s` when `s <= 21`, and zero otherwise. Over `M` hands:

```text
score = 100 * sum(rewards) / (21 * M)
```

Every allowed hand admits a subset summing to 21, so the maximum score is 100.
Different masks can be equally good. Selecting a zero-valued card has no effect.

## Interface

```python
class Solution:
    def __init__(
        self, artifact_dir, config, seed
    ): ...  # Load artifacts once; initialize your RNG with seed.

    def select(self, images): ...  # Return np.ndarray, shape (7,), dtype bool.
```

`images` is a `numpy.uint8` array with shape `(7,128,128,3)`, RGB channels and values
0 through 255. `True` selects a card. The same instance handles all hands in an
evaluation. Model probabilities and the selection algorithm are your choices.

Training implements `train(data_dir: Path, output_dir: Path, config: dict, seed: int)`
in `train.py`. It must build or adjust a recognizer from labeled data. Conventional
learners, neural networks, learned analytic features, calibration, and discovery of
an exact rule are all acceptable. Include every inference dependency in `artifacts/`
or local code. Final inference runs on CPU without external services.

## Data

`data/blindjack/` contains `train/`, `validation/`, and `dataset.json`. Each split has
lossless `images-0000.png`, `images-0001.png`, … sheets and `labels.npz` with the array
`labels`. Validation also has `hands.npz` with integer indices of shape `(M,7)`.
Validation has one third as many images as training.

Each sheet has 32 columns and at most 32 rows of original 128×128 images. Read tiles
left to right, row by row, then continue with the next sheet. The metadata gives
the valid image count; white cells beyond that count are padding, not examples.
Labels and hand indices use that same zero-based order. No duplicate image arrays
are stored. NPZ arrays are loaded with pickling disabled.

```python
from coursekit.labs.blindjack import load_split

images, labels = load_split(data_dir, "train")
batch = images[[0, 1, 2]]
```

The loader caches at most two decoded sheets. The public validation benchmark is
fixed and labeled to support development. It is not the final examination set;
rearranging the same images into more hands does not test new recognition examples.

## Starter and baseline

The starter selects three random distinct positions using the supplied seed. Its
expected score is approximately 60.5483. Its empty training function demonstrates
the interface, not substantive learning.

The baseline trains a small scikit-learn MLP on reduced, normalized pixels. It
predicts each image's most likely value and greedily considers cards from largest
predicted value to smallest. It skips a card that does not fit and continues.
Recognition, uncertainty handling, and selection all leave room for improvement.

Follow the commands in the project README to copy, train, evaluate, and package
either template. If you change source or configuration, rerun training before
packaging so provenance matches your submitted procedure.

## Augmentation and evaluation

Translation, rotation, and scaling can preserve the value when the complete pattern
and its relevant details remain intact; use white padding. Exact quarter-turn
rotations preserve the original pixel grid. Arbitrary angles, translations with
cropping, and downscaling require careful validation and can destroy information.
No general safe scaling range has yet been calibrated. Reflections and arbitrary
color changes are not declared label-preserving.

The current [reference limits](../../docs/environment.md) are pilot settings.
An incomplete run has no score: averaging only completed hands would be misleading.
Points measure absolute performance. Cohort rank does not determine your grade.
