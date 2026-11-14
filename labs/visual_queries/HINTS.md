# Optional modelling hints

Begin by making the lifecycle reliable: loading a batch, saving state, reloading it
and obtaining the same predictions. Keep development queries separate from the
support labels used for an adaptation episode. A constant predictor is useful for
checking this machinery, but supplies no learned understanding.

The visual and linguistic parts can be developed separately. The labeled glyph
pool can support a small recognition encoder; an autoencoder can learn a compact
visual representation, although reconstruction alone does not guarantee separate
objects. Preserve spatial information when encoding whole images: global average
pooling can discard precisely the positions requested by a question.

Token embeddings map IDs to learned vectors. A language encoder combines them with
position information and the padding mask. Cross-attention lets query features
select visual features; self-attention lets features exchange information. These
operations do not by themselves enforce correct reference binding or logical scope.
A small joint model, or several explicitly trained components, are reasonable
experiments. The task does not prescribe a particular decomposition.

For adaptation, allocate representations for the appended IDs and allow new answer
classes as well as new input words. Initialize consistently from the base model,
optimize on support examples, and consider mixing public rehearsal examples to
reduce forgetting. Keep vocabulary-dependent dimensions and mappings in the saved
state. Do not infer an episode's identity from a fixed added token number.

Measure errors on simple recognition, attributes, selection, relations and longer
requests separately. Compare against a question-only predictor or answer-frequency
predictor to understand shortcuts. Reordered batches should give the same answer
for each individual question. Attention pictures can help debug a model, but do
not prove that its reasoning is correct.
