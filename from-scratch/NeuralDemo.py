from __future__ import annotations

import numpy as np

from neural_core import DenseLayer, NeuralNetwork, Neuron


def _demo_binary_neuron() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1])

    neuron = Neuron(
        problem_type="binary",
        learning_rate=0.4,
        max_epochs=1500,
        batch_size=4,
        random_state=7,
    )
    neuron.fit(X, y)

    print("=== Binary Neuron (OR) ===")
    print("Weights:", np.round(neuron.weights_, 4))
    print("Bias:", round(neuron.bias_, 4))
    print("Probabilities:", np.round(neuron.predict_proba(X), 4))
    print("Predictions:", neuron.predict(X))
    print("Accuracy:", round(neuron.score(X, y), 4))


def _demo_xor_network() -> None:
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

    print("\n=== Neural Network (XOR) ===")
    print("Predictions:", network.predict(X))
    print("Accuracy:", round(network.score(X, y), 4))


if __name__ == "__main__":
    _demo_binary_neuron()
    _demo_xor_network()
