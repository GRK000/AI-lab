from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ai_lab.neural_core import DenseLayer, NeuralNetwork, r2_score


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    model: str
    dataset: str
    metric: str
    score: float
    seconds: float


def run_all() -> list[BenchmarkResult]:
    results = [
        _benchmark_xor(),
        _benchmark_linear_regression(),
        _benchmark_multiclass_synthetic(),
    ]
    results.extend(_optional_sklearn_comparison())
    return results


def format_results(results: list[BenchmarkResult]) -> str:
    rows = [
        ("Model", "Dataset", "Metric", "Score", "Seconds"),
        ("-" * 24, "-" * 18, "-" * 12, "-" * 8, "-" * 8),
    ]
    rows.extend(
        (
            result.model,
            result.dataset,
            result.metric,
            f"{result.score:.4f}",
            f"{result.seconds:.3f}",
        )
        for result in results
    )
    return "\n".join(
        f"{model:<24} {dataset:<18} {metric:<12} {score:>8} {seconds:>8}"
        for model, dataset, metric, score, seconds in rows
    )


def main() -> None:
    print(format_results(run_all()))


def _benchmark_xor() -> BenchmarkResult:
    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0])
    model = NeuralNetwork(
        layers=[DenseLayer(8, "tanh"), DenseLayer(1, "sigmoid")],
        problem_type="binary",
        learning_rate=0.2,
        max_epochs=2500,
        batch_size=4,
        random_state=21,
    )
    started = perf_counter()
    model.fit(X, y)
    seconds = perf_counter() - started
    return BenchmarkResult("NeuralNetwork", "XOR", "accuracy", model.score(X, y), seconds)


def _benchmark_linear_regression() -> BenchmarkResult:
    X = np.linspace(-1.0, 1.0, 80, dtype=np.float32).reshape(-1, 1)
    y = (3.0 * X.reshape(-1) - 0.5).astype(np.float32)
    model = NeuralNetwork(
        layers=[DenseLayer(1, "identity")],
        problem_type="regression",
        learning_rate=0.05,
        max_epochs=500,
        batch_size=16,
        random_state=3,
    )
    started = perf_counter()
    model.fit(X, y)
    seconds = perf_counter() - started
    return BenchmarkResult(
        "NeuralNetwork",
        "linear_regression",
        "r2",
        r2_score(y, model.predict(X)),
        seconds,
    )


def _benchmark_multiclass_synthetic() -> BenchmarkResult:
    rng = np.random.default_rng(42)
    centers = np.array([[1.5, 0.0], [-1.5, 0.0], [0.0, 1.8]], dtype=np.float32)
    X_parts = [center + 0.25 * rng.normal(size=(40, 2)) for center in centers]
    X = np.vstack(X_parts).astype(np.float32)
    y = np.repeat(np.arange(3), 40)
    model = NeuralNetwork(
        layers=[DenseLayer(12, "tanh"), DenseLayer(3, "softmax")],
        problem_type="multiclass",
        learning_rate=0.05,
        optimizer="adam",
        max_epochs=350,
        batch_size=24,
        random_state=5,
    )
    started = perf_counter()
    model.fit(X, y)
    seconds = perf_counter() - started
    return BenchmarkResult(
        "NeuralNetwork",
        "synthetic_3class",
        "accuracy",
        model.score(X, y),
        seconds,
    )


def _optional_sklearn_comparison() -> list[BenchmarkResult]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError:
        return []

    X = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.8, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 0, 1, 1])
    model = LogisticRegression(random_state=0)
    started = perf_counter()
    model.fit(X, y)
    seconds = perf_counter() - started
    return [
        BenchmarkResult(
            "sklearn LogisticRegression",
            "binary_toy",
            "accuracy",
            float(model.score(X, y)),
            seconds,
        )
    ]


if __name__ == "__main__":
    main()
