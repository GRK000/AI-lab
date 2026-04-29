from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FROM_SCRATCH = REPO_ROOT / "from-scratch"
if str(FROM_SCRATCH) not in sys.path:
    sys.path.insert(0, str(FROM_SCRATCH))

from neural_core import DenseLayer, NeuralNetwork, Neuron  # noqa: E402
from perceptron import Perceptron  # noqa: E402


@dataclass(slots=True)
class BenchmarkResult:
    model_name: str
    train_score: float
    predictions: np.ndarray


def binary_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.1, 0.9],
            [0.9, 0.2],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 0, 0, 0, 1])
    return X, y


def xor_dataset() -> tuple[np.ndarray, np.ndarray]:
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


def own_perceptron_result() -> BenchmarkResult:
    X, y = binary_dataset()
    model = Perceptron(learning_rate=0.2, max_epochs=100, random_state=7)
    model.fit(X, y)
    return BenchmarkResult(
        model_name="Own Perceptron",
        train_score=model.score(X, y),
        predictions=model.predict(X),
    )


def own_neuron_result() -> BenchmarkResult:
    X, y = binary_dataset()
    model = Neuron(
        problem_type="binary",
        learning_rate=0.2,
        max_epochs=1500,
        batch_size=4,
        random_state=7,
    )
    model.fit(X, y)
    return BenchmarkResult(
        model_name="Own Neuron",
        train_score=model.score(X, y),
        predictions=model.predict(X),
    )


def own_network_result() -> BenchmarkResult:
    X, y = xor_dataset()
    model = NeuralNetwork(
        layers=[DenseLayer(8, "tanh"), DenseLayer(1, "sigmoid")],
        problem_type="binary",
        learning_rate=0.2,
        max_epochs=4000,
        batch_size=4,
        random_state=21,
    )
    model.fit(X, y)
    return BenchmarkResult(
        model_name="Own NeuralNetwork",
        train_score=model.score(X, y),
        predictions=model.predict(X),
    )
