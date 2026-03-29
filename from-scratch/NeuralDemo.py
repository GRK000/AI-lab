from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_core import DenseLayer, NeuralNetwork, Neuron
from perceptron import Perceptron
from visualization import plot_binary_model_comparison, plot_training_history


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


def demo_perceptron_vs_neuron() -> None:
    X, y = _or_dataset()

    perceptron = Perceptron(learning_rate=0.2, max_epochs=100, random_state=7)
    neuron = Neuron(
        problem_type="binary",
        learning_rate=0.25,
        max_epochs=800,
        batch_size=4,
        random_state=7,
    )

    perceptron.fit(X, y)
    neuron.fit(X, y)

    comparison_path = PLOTS_DIR / "perceptron_vs_neuron.png"
    neuron_history_path = PLOTS_DIR / "neuron_training_history.png"
    perceptron_history_path = PLOTS_DIR / "perceptron_training_history.png"

    fig = plot_binary_model_comparison(
        X,
        y,
        perceptron,
        neuron,
        save_path=comparison_path,
    )
    plt.close(fig)

    fig = plot_training_history(
        neuron.history_,
        title="Neuron training history",
        save_path=neuron_history_path,
    )
    plt.close(fig)

    fig = plot_training_history(
        perceptron.history_,
        title="Perceptron training history",
        save_path=perceptron_history_path,
    )
    plt.close(fig)

    print("=== Perceptron vs Neuron ===")
    print("Perceptron accuracy:", round(perceptron.score(X, y), 4))
    print("Neuron accuracy:", round(neuron.score(X, y), 4))
    print(f"Saved: {comparison_path}")
    print(f"Saved: {neuron_history_path}")
    print(f"Saved: {perceptron_history_path}")


def demo_xor_network() -> None:
    X, y = _xor_dataset()

    network = NeuralNetwork(
        layers=[
            DenseLayer(units=8, activation="tanh"),
            DenseLayer(units=1, activation="sigmoid"),
        ],
        problem_type="binary",
        learning_rate=0.2,
        max_epochs=4000,
        batch_size=4,
        random_state=21,
    )
    network.fit(X, y)

    history_path = PLOTS_DIR / "xor_network_history.png"
    fig = plot_training_history(
        network.history_,
        title="XOR neural network history",
        save_path=history_path,
    )
    plt.close(fig)

    print("\n=== XOR Neural Network ===")
    print("Predictions:", network.predict(X))
    print("Accuracy:", round(network.score(X, y), 4))
    print(f"Saved: {history_path}")


if __name__ == "__main__":
    demo_perceptron_vs_neuron()
    demo_xor_network()
