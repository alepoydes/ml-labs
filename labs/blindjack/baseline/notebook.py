"""Explore your baseline. Training is opt-in; analysis reads saved results."""

import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

    from coursekit import resources, training
    from coursekit.common import read_toml
    from coursekit.labs.blindjack import benchmark, load_split, reward

    return (
        ConfusionMatrixDisplay,
        Path,
        accuracy_score,
        benchmark,
        load_split,
        np,
        plt,
        read_toml,
        resources,
        reward,
        training,
    )


@app.cell
def _(Path, read_toml, resources):
    # Edit assignments here; keep model/training changes in train.py and lib/.
    solution_dir = Path(__file__).resolve().parent
    if solution_dir.parent.parent.name != "work":
        raise ValueError("Copy the baseline with step_1_new_solution before running it.")
    project_dir = solution_dir.parents[2]
    data_dir = project_dir / "data/blindjack"
    variant = read_toml(solution_dir / "submission.toml")["variant"]
    run_dir = project_dir / "runs/blindjack" / variant
    config = read_toml(solution_dir / "config.toml")
    seed = 0
    hand_number = 1  # One-based, matching hands.csv and the data browser.
    sample_indices = [0, 1, 2, 3, 4, 5]  # Zero-based image indices.
    resource_hash = resources.ready(
        data_dir, resources.manifest(project_dir / "labs/blindjack/resources.toml")
    )
    print("Configuration:", config, "Training seed:", seed)
    return (
        config,
        data_dir,
        hand_number,
        project_dir,
        resource_hash,
        run_dir,
        sample_indices,
        seed,
        solution_dir,
    )


@app.cell
def _(config, data_dir, load_split, np, plt, sample_indices, seed, solution_dir, training):
    train_images, train_labels = load_split(data_dir, "train")
    training_source = training.load_module(solution_dir, "train")
    untrained_model = training_source.create_model(config, seed)
    examples = train_images[np.asarray(sample_indices)]
    features = training_source.pixels(examples, config["resolution"])
    print("Images:", examples.shape, examples.dtype, "Features:", features.shape)
    print("Training class counts:", np.bincount(train_labels, minlength=11))
    print(untrained_model)
    _fig, _axes = plt.subplots(2, len(examples), squeeze=False)
    for _i, _image in enumerate(examples):
        _axes[0, _i].imshow(_image)
        _axes[0, _i].set_title(f"Label {train_labels[sample_indices[_i]]}")
        _axes[1, _i].imshow(features[_i].reshape(config["resolution"], -1, 3) + 1)
        for _ax in _axes[:, _i]:
            _ax.axis("off")
    _fig.suptitle("Original images / model inputs (white restored for display)")
    plt.close(_fig)
    _fig
    return


@app.cell
def _(project_dir, run_dir, seed, solution_dir, training):
    # Uncomment to train FROM SCRATCH, then comment again before further analysis.
    # Edit config.toml for hyperparameters; this uses the same path as step_2_train.
    # training.train_local(solution_dir, project_dir, seed)
    history, saved = training.load_training(solution_dir, run_dir)
    return history, saved


@app.cell
def _(history, plt, resource_hash, saved):
    print("Saved configuration:", saved["config"])
    print("Saved seed:", saved["provenance"]["seed"])
    if saved["provenance"]["resource_hash"] != resource_hash:
        print("Data changed since training; these curves describe the saved training run.")
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    _axes[0].plot(history["epoch"], history["loss"])
    _axes[0].set(title="Training loss", xlabel="Epoch")
    _axes[1].plot(history["epoch"], history["validation_accuracy"])
    _axes[1].set(title="Internal training holdout accuracy", xlabel="Epoch", ylim=(0, 1))
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(data_dir, load_split, np, saved, solution_dir, training):
    # The saved model and current inference source are also used by CLI evaluation.
    source = training.load_module(solution_dir, "solution")
    from lib.model import predict_classes, predict_probabilities

    solution = source.Solution(
        solution_dir / "artifacts", saved["config"], saved["provenance"]["seed"]
    )
    validation_images, validation_labels = load_split(data_dir, "validation")
    # Predict in small batches, retaining the canonical sheets as the image storage.
    predictions = np.concatenate(
        [
            predict_classes(
                solution.model, validation_images[_indices], solution.resolution
            )
            for _indices in np.array_split(
                np.arange(len(validation_images)),
                max(1, (len(validation_images) + 63) // 64),
            )
        ]
    )
    return (
        predict_classes,
        predict_probabilities,
        predictions,
        solution,
        validation_images,
        validation_labels,
    )


@app.cell
def _(ConfusionMatrixDisplay, accuracy_score, np, plt, predictions, validation_labels):
    print("Public validation accuracy:", accuracy_score(validation_labels, predictions))
    _display = ConfusionMatrixDisplay.from_predictions(
        validation_labels, predictions, labels=np.arange(11)
    )
    plt.close(_display.figure_)
    _display.figure_
    return


@app.cell
def _(
    np,
    plt,
    predict_probabilities,
    predictions,
    solution,
    validation_images,
    validation_labels,
):
    wrong = np.flatnonzero(predictions != validation_labels)[:6]
    if len(wrong):
        _images = validation_images[wrong]
        _probabilities = predict_probabilities(solution.model, _images, solution.resolution)
        print("Image indices:", wrong, "Probability columns:", solution.model.classes_)
        print(_probabilities.round(3))
        _fig, _axes = plt.subplots(1, len(wrong), squeeze=False)
        for _ax, _index, _image in zip(_axes[0], wrong, _images):
            _ax.imshow(_image)
            _ax.set_title(
                f"True {validation_labels[_index]}, predicted {predictions[_index]}"
            )
            _ax.axis("off")
        plt.close(_fig)
    else:
        _fig = "No classification errors on public validation."
    _fig
    return


@app.cell
def _(benchmark, data_dir, hand_number, plt, reward, solution):
    _images, _labels, _hands = benchmark(data_dir)
    if not 1 <= hand_number <= len(_hands):
        raise ValueError(f"hand_number must be between 1 and {len(_hands)}.")
    _indices = _hands[hand_number - 1]
    _cards, _values = _images[_indices], _labels[_indices]
    mask = solution.select(_cards)
    print("Selected sum:", int(_values[mask].sum()), "Reward:", reward(_values, mask))
    _fig, _axes = plt.subplots(1, 7, figsize=(12, 2))
    for _ax, _card, _value, _selected in zip(_axes, _cards, _values, mask):
        _ax.imshow(_card)
        _ax.set_title(f"{_value}: {'selected' if _selected else 'skip'}")
        _ax.axis("off")
    plt.close(_fig)
    _fig
    return (mask,)


if __name__ == "__main__":
    app.run()
