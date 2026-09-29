import joblib
from sklearn.neural_network import MLPClassifier

from lib.features import pixels


def create_model(config, seed):
    return MLPClassifier(
        hidden_layer_sizes=(config["hidden_units"],),
        early_stopping=True,
        random_state=seed,
        max_iter=config["max_iter"],
        learning_rate_init=config["learning_rate"],
        n_iter_no_change=config.get("patience", 10),
        activation=config.get("activation", "relu"),
    )


def save_model(output_dir, resolution, model):
    joblib.dump((resolution, model), output_dir / "model.joblib")


def load_model(artifact_dir):
    return joblib.load(artifact_dir / "model.joblib")


def predict_probabilities(model, images, resolution):
    return model.predict_proba(pixels(images, resolution))


def predict_classes(model, images, resolution):
    probabilities = predict_probabilities(model, images, resolution)
    return model.classes_[probabilities.argmax(axis=1)]
