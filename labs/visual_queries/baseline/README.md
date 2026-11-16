# CNN/MLP baseline

This baseline combines a convolutional image encoder with a multilayer perceptron
for question encoding and answer prediction. All components learn from labeled
image/question/answer examples. Copy it from the directory containing
`pyproject.toml`, `labs/` and the `step_*` launchers:

```sh
./step_1_new_solution visual_queries --template baseline
./step_2_train work/visual_queries/baseline
./step_3_evaluate work/visual_queries/baseline
```

For CUDA, use the [explicit CUDA commands](../README.md#6-training-checking-and-submission)
with `--no-group cpu --extra cuda` and `--device cuda:0`, replacing `my_model`
with `baseline`. The numbered scripts select CPU dependencies even when passed
`--device cuda:0`.
CPU uses the same source and artifacts. Training optimizes the model weights over
several epochs. Progress is printed after each epoch; numeric
histories and provenance are saved by the shared runner. Packaging and checking
use the ordinary `step_4_pack` and `step_5_check_submission` commands.

## Model

Four `Conv2d(3×3) → ReLU → MaxPool2d(2)` blocks use 16, 32, 64 and 64 channels.
For a 512×512 input they produce a 32×32 feature map. Pooling to 8×8, flattening
and a linear layer produce 256 image features. Spatial cells remain distinct;
RGB channels preserve color information. White background is shifted to zero.

Each word ID has a learned 32-dimensional embedding. Questions are zero-padded to
the dataset's maximum length. Their ordered embeddings are flattened and passed
through two 128-unit layers. Different token positions have different weights,
so exchanging the subject and object can change the prediction. Padding has a
fixed zero embedding. This is an ordered, fixed-length MLP.

The concatenated image/question features pass through a 256-unit layer and an
output layer with one logit per active vocabulary token. ReLU and small dropout
layers are used. Layer normalization of the image
and question vectors balances their scales before fusion, without sharing
statistics between examples. Training uses AdamW and cross
entropy directly on logits. Prediction takes the largest logit, equivalent to
maximizing the predicted answer probability. Padding is excluded as an answer.

## Training and adaptation

The default training budget is at most 20 epochs, with early stopping after five
epochs without improved internal validation accuracy. Ten percent of training
images form the internal holdout: all questions sharing an image stay together.
The selected checkpoint has the best internal accuracy. The supplied public
validation and adaptation queries are never read by training or adaptation.

Groups of nearby image indices and the questions within each group are shuffled
at every epoch. The launcher retains the underlying PNG access. This
keeps decoding bounded by the number of sheets rather than the number of
questions, without retaining the entire image collection in RAM. Mini-batches can
therefore contain related examples; this is a simple memory/decoding tradeoff.

Each adaptation loads the given base state, appends randomly initialized input
embedding and output-layer rows, and retains every old weight. It trains all
parameters for up to 12 epochs at a smaller learning rate. Up to 128 public
base-training examples are cached for rehearsal; one rehearsal example is added
per four support examples. The soft time budget leaves room for saving before
the executor's hard deadline. The initial expanded model is saved and committed
before learning, and every completed adaptation epoch commits another full model
with its numeric history using `checkpoint(history)`. At a hard timeout the runner
evaluates the last committed model, including all its component files. Manual fine-tuning on the full public root uses
`base/train` without vocabulary growth. Both paths save complete reloadable states.

`config.toml` contains the training settings. `lib/model.py` defines construction,
vocabulary growth and portable weight-only checkpoints; `train.py` owns learning;
`solution.py` owns inference. CPU NumPy batches are transferred to the explicitly
requested device. Prediction uses evaluation mode and has no state between queries.

This is an introductory architecture. Flattening the whole scene and question
does not provide explicit object selection, relation reasoning or reference
binding. Measure its errors and adaptation results before choosing improvements.

## Historical measurement

A seed-0 run using API 2 and resource format 1 obtained
1,142/2,850 correct answers, or **40.07 points**, including all eight independent
adaptations. Training took 252 seconds and the complete evaluation 69 seconds on
an RTX 5060 Ti with PyTorch 2.14.1+cu130. Early stopping selected epoch 9 and
finished after epoch 14. Peak PyTorch-allocated training memory was about 555 MiB;
this excludes driver/runtime memory and is not a certified VRAM budget.

Replacing query images with white images reduced the score to 28.32. This is an
input ablation of the trained model, not a separately trained question-only model.
It confirms that images affect useful predictions, without establishing reliable
symbol recognition or reasoning. These measurements describe that earlier run;
measure the current API 3 implementation on your own hardware when choosing a
training budget. Driver memory and assessment-process overhead require separate
measurement.
