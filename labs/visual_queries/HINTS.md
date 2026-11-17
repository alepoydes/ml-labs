# Modelling hints for Visual Queries

These suggestions are starting points for experiments. The
[task statement](README.md) defines the requirements; the architecture, component
boundaries and learning schedule are your choices.

## Establish a reliable experiment

Start with the constant-answer starter to check data loading, artifact saving,
reloading and prediction. A saved model should reproduce the same answers after
reloading. Then use the trainable baseline to obtain a first learned result.
Keep training examples, internal validation examples and the public benchmark
separate when comparing changes.

Many questions share an image. Form internal validation folds by image so that
related questions stay together. For adaptation, use a held-out part of support
for checkpoint selection, and use queries only to assess the resulting model.
Count new answer labels on both sides of the split: random image selection can
leave a rare new class absent from validation. Preserve image groups when
adjusting the split.
Reserving complete public episodes provides a further check of the adaptation
procedure's ability to handle unfamiliar additions.

## Visual representations

The labeled glyph pool can train a recognition encoder before learning from
whole scenes. Useful visual features must also preserve color, size and position.
An autoencoder offers another way to learn compact features; its reconstruction
objective alone does not ensure that features separate individual objects.

A whole-image encoder needs spatial information. For example, global average
pooling can discard positions that a question asks about. Spatial feature maps,
position encodings and explicit object representations offer different ways to
retain this information. Compare their complexity and performance on the simpler
question families before adding more components.

## Combining images and language

Token embeddings map integer IDs to learned vectors. A language encoder combines
those vectors using word positions and the padding mask. The baseline uses an
ordered, fixed-length multilayer perceptron. A transformer is another option:
self-attention exchanges information between features, while cross-attention lets
question features select visual features.

These mechanisms provide trainable operations. Correct object selection,
reference binding and logical scope must still be learned from examples. You can
train a small joint model or train components separately and then combine them.
Check component performance and feature scales when connecting independently
trained parts.

## Learning new vocabulary

Each adaptation begins with the same base model. Allocate representations for
appended input IDs and for any new answer IDs. Initialize them reproducibly, learn
from support examples and save the full adapted state, including vocabulary sizes
and mappings needed by your model. Added IDs have episode-specific meanings.

Rehearsal on base-training examples can reduce forgetting while learning the new
examples. Compare support-training and support-validation loss and accuracy to
choose the learning rate and update budget. Check predictions for new answers as
well as retention of familiar answers.

## Interpreting results

Inspect errors on recognition, attributes, selection, spatial relations and longer
requests separately. The evaluation table's known/new-language counts help assess
adaptation. Compare a question-only or answer-frequency predictor with the full
model to estimate how much it relies on visual information.

Changing batch order or size should preserve each question's prediction. Use
recorded examples, optional answer distributions and exported model diagnostics
to investigate failures. Attention plots show where features interact; test the
answers themselves to assess whether the intended relation was learned.
