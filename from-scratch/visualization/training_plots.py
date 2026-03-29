from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def _save_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def _history_series(history: list[Any], field: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []

    for snapshot in history:
        if hasattr(snapshot, field):
            epochs.append(int(getattr(snapshot, "epoch")))
            values.append(float(getattr(snapshot, field)))

    return epochs, values


def plot_training_history(
    history: list[Any],
    *,
    title: str = "Training history",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot every metric available in the provided training history.

    The function accepts histories from both the classical perceptron and the
    differentiable models implemented in this repository.
    """

    if not history:
        raise ValueError("history cannot be empty.")

    fig, axis = plt.subplots(figsize=(9, 5))

    plotted_anything = False
    for field, label in (
        ("loss", "Loss"),
        ("metric", "Metric"),
        ("accuracy", "Accuracy"),
        ("errors", "Errors"),
        ("mean_margin", "Mean margin"),
        ("weight_norm", "Weight norm"),
    ):
        epochs, values = _history_series(history, field)
        if values:
            axis.plot(epochs, values, linewidth=2, label=label)
            plotted_anything = True

    if not plotted_anything:
        raise ValueError("history does not expose plottable fields.")

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.25)
    axis.legend()

    _save_figure(fig, save_path)
    return fig


def _plot_dataset(axis: plt.Axes, X: np.ndarray, y: np.ndarray, title: str) -> None:
    axis.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.8,
        s=70,
    )
    axis.set_title(title)
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.grid(alpha=0.2)


def _decision_surface(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32, copy=False)
    zz = np.asarray(model.predict(grid)).reshape(xx.shape)
    return xx, yy, zz


def plot_binary_model_comparison(
    X: ArrayLike,
    y: ArrayLike,
    perceptron: Any,
    neuron: Any,
    *,
    title: str = "Perceptron vs neuron",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Compare a classical perceptron and a differentiable neuron on a 2D problem.
    """

    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y).reshape(-1)

    if X_array.ndim != 2 or X_array.shape[1] != 2:
        raise ValueError("X must be a 2D array with exactly 2 features.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14)

    for axis, model, model_name in (
        (axes[0, 0], perceptron, "Perceptron"),
        (axes[0, 1], neuron, "Neuron"),
    ):
        xx, yy, zz = _decision_surface(model, X_array)
        axis.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
        _plot_dataset(axis, X_array, y_array, model_name)

    epochs, values = _history_series(perceptron.history_, "accuracy")
    if values:
        axes[1, 0].plot(epochs, values, color="#1f77b4", linewidth=2)
    axes[1, 0].set_title("Perceptron accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(alpha=0.25)

    loss_epochs, losses = _history_series(neuron.history_, "loss")
    metric_epochs, metrics = _history_series(neuron.history_, "metric")
    if losses:
        axes[1, 1].plot(loss_epochs, losses, color="#d62728", linewidth=2, label="Loss")
    if metrics:
        axes[1, 1].plot(metric_epochs, metrics, color="#2ca02c", linewidth=2, label="Metric")
    axes[1, 1].set_title("Neuron training")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(alpha=0.25)
    if losses or metrics:
        axes[1, 1].legend()

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
