# Composite training example

This is a small runnable Blindjack solution using [API v2](../../docs/api-v2.md).
Copy this directory into `work/blindjack/composite-example` before changing it.
It uses the ordinary public Blindjack training data and the common loaders.

`train.py` learns an autoencoder representation of reduced RGB images, retains its
encoder, and fits two eleven-class prediction heads. `solution.py` averages their
logits and selects cards greedily. The decoder is used only in training. This
example demonstrates component ownership and checkpoint loading; it is not a
recommended architecture or a score benchmark.

- `all` without inputs trains the encoder and both heads in one call.
- `encoder` saves an intermediate `encoder.pt`.
- `member` requires the named `encoder` input and saves that encoder plus `head.pt`.
  Train two members with different seeds to create independent predictors.
- `all` with `encoder`, `member_1` and `member_2` inputs checks encoder identity and
  assembles a complete solution without retraining its components.

`adapt` reloads the encoder and both heads, keeps the encoder fixed, fine-tunes
each head, and saves a complete new artifact set. Its input snapshot remains
unchanged. The metadata declares `supports_adapt = true`.

See the linked API guide for executable CLI and Python workflows, CPU/CUDA
selection, histories, packaging and failure recovery. The templates under
`labs/blindjack/` provide simpler API 1 examples.
