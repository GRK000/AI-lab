from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from ai_lab.neural_core import DenseLayer, DropoutLayer, NeuralNetwork
from ai_lab.visualization import plot_training_history

PLOTS_DIR = REPO_ROOT / "artifacts" / "plots"


def _load_mnist_from_tensorflow() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from tensorflow.keras.datasets import mnist  # type: ignore
    except Exception as error:
        raise RuntimeError("TensorFlow/Keras is not available.") from error

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def _load_mnist_from_keras() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from keras.datasets import mnist  # type: ignore
    except Exception as error:
        raise RuntimeError("Standalone Keras is not available.") from error

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def _load_mnist_from_sklearn() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.datasets import fetch_openml  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
    except Exception as error:
        raise RuntimeError("scikit-learn is not available.") from error

    data = fetch_openml("mnist_784", version=1, as_frame=False)
    X = np.asarray(data.data, dtype=np.float32).reshape(-1, 28, 28)
    y = np.asarray(data.target, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=10000,
        random_state=42,
        stratify=y,
    )
    return X_train, y_train, X_test, y_test


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loaders = (
        ("tensorflow.keras.datasets.mnist", _load_mnist_from_tensorflow),
        ("keras.datasets.mnist", _load_mnist_from_keras),
        ("sklearn.fetch_openml('mnist_784')", _load_mnist_from_sklearn),
    )

    errors: list[str] = []
    for loader_name, loader in loaders:
        try:
            X_train, y_train, X_test, y_test = loader()
            print(f"Loaded MNIST from {loader_name}")
            return X_train, y_train, X_test, y_test
        except Exception as error:
            errors.append(f"- {loader_name}: {error}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(
        "Could not load MNIST from any supported source.\n"
        "Tried:\n"
        f"{joined_errors}\n\n"
        "Install one of these options:\n"
        "- tensorflow\n"
        "- keras\n"
        "- scikit-learn\n"
    )


def preprocess_images(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1).astype(np.float32, copy=False)
    return flat / 255.0


def build_model(hidden_units: tuple[int, ...], dropout_rate: float) -> NeuralNetwork:
    layers: list[Any] = []

    for units in hidden_units:
        layers.append(DenseLayer(units=units, activation="relu"))
        if dropout_rate > 0.0:
            layers.append(DropoutLayer(rate=dropout_rate))

    layers.append(DenseLayer(units=10, activation="softmax"))

    return NeuralNetwork(
        layers=layers,
        problem_type="multiclass",
        learning_rate=0.001,
        optimizer="adam",
        max_epochs=12,
        batch_size=128,
        random_state=42,
    )


def preview_predictions(
    model: NeuralNetwork,
    X_test_images: np.ndarray,
    X_test_flat: np.ndarray,
    y_test: np.ndarray,
    save_path: Path,
    num_samples: int = 8,
) -> None:
    predictions = model.predict(X_test_flat[:num_samples])
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle("MNIST sample predictions")

    for axis, image, truth, prediction in zip(
        axes.flat,
        X_test_images[:num_samples],
        y_test[:num_samples],
        predictions,
        strict=False,
    ):
        axis.imshow(image, cmap="gray")
        axis.set_title(f"y={truth} | pred={prediction}")
        axis.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("\n=== MNIST dense demo ===")
    print("Task: multiclass image classification")
    architecture = " -> ".join(
        str(layer.units) for layer in model.layers if hasattr(layer, "units")
    )
    print("Architecture:", architecture)
    print("Optimizer: adam")
    print("Train samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])
    print("Input features:", X_train.shape[1])
    print("Train accuracy:", round(train_score, 4))
    print("Test accuracy:", round(test_score, 4))
    print("Final loss:", round(float(model.history_[-1].loss), 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST demo using the dense neural network implemented in this repository.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=12000,
        help="Maximum number of training samples to use.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2000,
        help="Maximum number of test samples to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
        help="Dropout rate for hidden layers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_train_images, y_train, X_test_images, y_test = load_mnist()

    train_size = min(args.train_size, X_train_images.shape[0])
    test_size = min(args.test_size, X_test_images.shape[0])

    X_train_images = X_train_images[:train_size]
    y_train = y_train[:train_size]
    X_test_images = X_test_images[:test_size]
    y_test = y_test[:test_size]

    X_train = preprocess_images(X_train_images)
    X_test = preprocess_images(X_test_images)

    model = build_model(hidden_units=(128, 64), dropout_rate=args.dropout)
    model.max_epochs = args.epochs
    model.fit(X_train, y_train)

    history_path = PLOTS_DIR / "mnist_training_history.png"
    preview_path = PLOTS_DIR / "mnist_sample_predictions.png"

    fig = plot_training_history(
        model.history_,
        title="MNIST training history",
        save_path=history_path,
    )
    plt.close(fig)
    preview_predictions(model, X_test_images, X_test, y_test, preview_path)

    print_summary(model, X_train, y_train, X_test, y_test)
    print("Saved:", history_path)
    print("Saved:", preview_path)


if __name__ == "__main__":
    main()
