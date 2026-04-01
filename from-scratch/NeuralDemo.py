from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_core import DenseLayer, DropoutLayer, NeuralNetwork, Neuron
from perceptron import Perceptron
from visualization import (
    plot_binary_model_comparison,
    plot_optimizer_histories,
    plot_training_history,
)


PLOTS_DIR = Path("artifacts/plots")


def _or_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.15, 0.10],
            [0.20, 0.85],
            [0.85, 0.15],
            [0.90, 0.90],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 0, 0, 0, 1])
    return X, y


def _xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0])
    return X, y


def _regression_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.linspace(-1.5, 1.5, 16, dtype=np.float32).reshape(-1, 1)
    y = (2.5 * X.reshape(-1) - 0.75).astype(np.float32)
    return X, y


def _multiclass_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [1.1, -0.1],
            [0.0, 1.0],
            [0.2, 0.8],
            [-0.1, 1.1],
            [-1.0, -0.2],
            [-0.8, -0.4],
            [-1.2, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return X, y


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _close_figure(fig: plt.Figure) -> None:
    plt.close(fig)


def demo_perceptron_vs_neuron() -> None:
    X, y = _or_dataset()

    perceptron = Perceptron(learning_rate=0.2, max_epochs=100, random_state=7)
    neuron = Neuron(
        problem_type="binary",
        learning_rate=0.25,
        optimizer="adam",
        max_epochs=800,
        batch_size=4,
        random_state=7,
    )

    perceptron.fit(X, y)
    neuron.fit(X, y)

    comparison_path = PLOTS_DIR / "perceptron_vs_neuron.png"
    neuron_history_path = PLOTS_DIR / "neuron_training_history.png"
    perceptron_history_path = PLOTS_DIR / "perceptron_training_history.png"

    _close_figure(
        plot_binary_model_comparison(
            X,
            y,
            perceptron,
            neuron,
            save_path=comparison_path,
        )
    )
    _close_figure(
        plot_training_history(
            neuron.history_,
            title="Neuron training history",
            save_path=neuron_history_path,
        )
    )
    _close_figure(
        plot_training_history(
            perceptron.history_,
            title="Perceptron training history",
            save_path=perceptron_history_path,
        )
    )

    _print_section("Perceptron vs Neuron")
    print("Dataset: binary OR-like classification")
    print("Perceptron accuracy:", round(perceptron.score(X, y), 4))
    print("Neuron accuracy:", round(neuron.score(X, y), 4))
    print("Saved:", comparison_path)
    print("Saved:", neuron_history_path)
    print("Saved:", perceptron_history_path)


def demo_optimizer_showcase() -> None:
    X, y = _regression_dataset()

    optimizer_configs = {
        "sgd": {"learning_rate": 0.05, "max_epochs": 600},
        "momentum": {"learning_rate": 0.03, "max_epochs": 450},
        "adam": {"learning_rate": 0.05, "max_epochs": 250},
        "adamw": {
            "learning_rate": 0.05,
            "max_epochs": 250,
            "optimizer_kwargs": {"weight_decay": 0.01},
        },
        "adamax": {"learning_rate": 0.05, "max_epochs": 250},
    }

    histories: dict[str, list[object]] = {}
    scores: list[tuple[str, float, float]] = []

    for optimizer_name, config in optimizer_configs.items():
        model = NeuralNetwork(
            layers=[DenseLayer(1, "identity")],
            problem_type="regression",
            learning_rate=config["learning_rate"],
            optimizer=optimizer_name,
            optimizer_kwargs=config.get("optimizer_kwargs"),
            max_epochs=config["max_epochs"],
            batch_size=8,
            random_state=3,
        )
        model.fit(X, y)
        histories[optimizer_name.upper()] = list(model.history_)
        scores.append(
            (
                optimizer_name.upper(),
                float(model.history_[-1].loss),
                float(model.score(X, y)),
            )
        )

    comparison_path = PLOTS_DIR / "optimizer_regression_comparison.png"
    _close_figure(
        plot_optimizer_histories(
            histories,
            field="loss",
            title="Optimizer comparison on linear regression",
            save_path=comparison_path,
        )
    )

    _print_section("Optimizer Showcase")
    print("Dataset: one-dimensional regression")
    for optimizer_name, final_loss, score in sorted(scores, key=lambda item: item[1]):
        print(
            f"{optimizer_name}: final_loss={final_loss:.6f} | r2={score:.4f}"
        )
    print("Saved:", comparison_path)


def demo_dropout_multiclass() -> None:
    X, y = _multiclass_dataset()

    network = NeuralNetwork(
        layers=[
            DenseLayer(units=12, activation="tanh"),
            DropoutLayer(rate=0.20),
            DenseLayer(units=3, activation="softmax"),
        ],
        problem_type="multiclass",
        learning_rate=0.03,
        optimizer="adam",
        max_epochs=1200,
        batch_size=9,
        random_state=19,
    )
    network.fit(X, y)

    history_path = PLOTS_DIR / "multiclass_dropout_history.png"
    _close_figure(
        plot_training_history(
            network.history_,
            title="Multiclass network with dropout",
            save_path=history_path,
        )
    )

    _print_section("Multiclass Network With Dropout")
    print("Dataset: three synthetic classes")
    print("Predictions:", network.predict(X))
    print("Accuracy:", round(network.score(X, y), 4))
    print("Saved:", history_path)


def demo_xor_network() -> None:
    X, y = _xor_dataset()

    network = NeuralNetwork(
        layers=[
            DenseLayer(units=8, activation="tanh"),
            DenseLayer(units=1, activation="sigmoid"),
        ],
        problem_type="binary",
        learning_rate=0.03,
        optimizer="adam",
        max_epochs=2500,
        batch_size=4,
        random_state=21,
    )
    network.fit(X, y)

    history_path = PLOTS_DIR / "xor_network_history.png"
    _close_figure(
        plot_training_history(
            network.history_,
            title="XOR neural network history",
            save_path=history_path,
        )
    )

    _print_section("XOR Neural Network")
    print("Dataset: XOR")
    print("Predictions:", network.predict(X))
    print("Accuracy:", round(network.score(X, y), 4))
    print("Saved:", history_path)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("AI Lab demo suite")
    print("Artifacts directory:", PLOTS_DIR)

    demo_perceptron_vs_neuron()
    demo_optimizer_showcase()
    demo_dropout_multiclass()
    demo_xor_network()


if __name__ == "__main__":
    main()
