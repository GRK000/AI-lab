from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .common import FloatArray
from .layers import DenseLayer


@dataclass(frozen=True, slots=True)
class GradientCheckResult:
    max_absolute_error: float
    mean_absolute_error: float
    checked_parameters: int

    @property
    def passed(self) -> bool:
        return self.max_absolute_error < 1e-3


def check_dense_layer_gradients(
    layer: DenseLayer,
    X: ArrayLike,
    upstream_gradient: ArrayLike,
    *,
    epsilon: float = 1e-4,
) -> GradientCheckResult:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")

    X_array = np.asarray(X, dtype=np.float32)
    upstream = np.asarray(upstream_gradient, dtype=np.float32)
    initial_weights = layer.weights.copy()
    initial_bias = layer.bias.copy()

    layer.forward(X_array, training=True)
    layer.backward(upstream, learning_rate=1.0)
    analytical_weights = initial_weights - layer.weights
    analytical_bias = initial_bias - layer.bias

    numerical_weights = np.zeros_like(initial_weights)
    numerical_bias = np.zeros_like(initial_bias)

    def scalar_loss(weights: FloatArray, bias: FloatArray) -> float:
        layer.weights_ = weights.astype(np.float32, copy=True)
        layer.bias_ = bias.astype(np.float32, copy=True)
        output = layer.forward(X_array, training=False)
        return float(np.sum(output * upstream))

    for row in range(initial_weights.shape[0]):
        for col in range(initial_weights.shape[1]):
            plus = initial_weights.copy()
            minus = initial_weights.copy()
            plus[row, col] += epsilon
            minus[row, col] -= epsilon
            numerical_weights[row, col] = (
                scalar_loss(plus, initial_bias) - scalar_loss(minus, initial_bias)
            ) / (2.0 * epsilon)

    for col in range(initial_bias.shape[1]):
        plus = initial_bias.copy()
        minus = initial_bias.copy()
        plus[0, col] += epsilon
        minus[0, col] -= epsilon
        numerical_bias[0, col] = (
            scalar_loss(initial_weights, plus) - scalar_loss(initial_weights, minus)
        ) / (2.0 * epsilon)

    layer.weights_ = initial_weights
    layer.bias_ = initial_bias

    errors = np.concatenate(
        (
            np.abs(analytical_weights - numerical_weights).reshape(-1),
            np.abs(analytical_bias - numerical_bias).reshape(-1),
        )
    )
    return GradientCheckResult(
        max_absolute_error=float(np.max(errors)),
        mean_absolute_error=float(np.mean(errors)),
        checked_parameters=int(errors.size),
    )


def finite_difference_parameter_gradient(
    loss_fn: Any,
    parameter: FloatArray,
    *,
    epsilon: float = 1e-4,
) -> FloatArray:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")

    gradient = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        parameter[index] = original + epsilon
        plus = float(loss_fn())
        parameter[index] = original - epsilon
        minus = float(loss_fn())
        parameter[index] = original
        gradient[index] = (plus - minus) / (2.0 * epsilon)
    return gradient
