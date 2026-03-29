from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .activations import ACTIVATIONS
from .common import ActivationName, FloatArray


@dataclass(slots=True)
class DenseLayer:
    """
    Fully connected layer.

    Internally it stores many neurons at once in matrix form:
    `weights_[i, j]` is the connection from input i to neuron j.
    """

    units: int
    activation: ActivationName = "relu"
    input_dim_: int | None = field(init=False, default=None)
    dtype_: Any | None = field(init=False, default=None)
    weights_: FloatArray | None = field(init=False, default=None)
    bias_: FloatArray | None = field(init=False, default=None)
    _last_input: FloatArray | None = field(init=False, default=None)
    _last_linear: FloatArray | None = field(init=False, default=None)
    _last_output: FloatArray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.units <= 0:
            raise ValueError("units must be > 0.")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation!r}.")

    @property
    def weights(self) -> FloatArray:
        if self.weights_ is None:
            raise RuntimeError("Layer parameters are not initialized yet.")
        return self.weights_

    @property
    def bias(self) -> FloatArray:
        if self.bias_ is None:
            raise RuntimeError("Layer parameters are not initialized yet.")
        return self.bias_

    def build(self, input_dim: int, rng: np.random.Generator, dtype: Any) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")

        self.input_dim_ = int(input_dim)
        self.dtype_ = dtype

        if self.activation == "relu":
            scale = np.sqrt(2.0 / input_dim)
        else:
            scale = np.sqrt(1.0 / input_dim)

        self.weights_ = rng.normal(
            loc=0.0,
            scale=scale,
            size=(input_dim, self.units),
        ).astype(dtype, copy=False)
        self.bias_ = np.zeros((1, self.units), dtype=dtype)

    def forward(self, inputs: FloatArray, training: bool = False) -> FloatArray:
        linear_output = inputs @ self.weights + self.bias
        activated_output = ACTIVATIONS[self.activation].forward(linear_output).astype(
            inputs.dtype,
            copy=False,
        )

        if training:
            self._last_input = inputs
            self._last_linear = linear_output
            self._last_output = activated_output

        return activated_output

    def activation_derivative(self) -> FloatArray:
        if self._last_linear is None or self._last_output is None:
            raise RuntimeError("No forward pass cached for this layer.")
        return ACTIVATIONS[self.activation].derivative(
            self._last_output,
            self._last_linear,
        ).astype(self._last_output.dtype, copy=False)

    def backward(
        self,
        delta: FloatArray,
        learning_rate: float,
        l2_lambda: float = 0.0,
    ) -> FloatArray:
        if self._last_input is None:
            raise RuntimeError("No forward pass cached for this layer.")

        grad_input = delta @ self.weights.T
        grad_weights = self._last_input.T @ delta
        grad_bias = np.sum(delta, axis=0, keepdims=True)

        if l2_lambda > 0.0:
            grad_weights += l2_lambda * self.weights

        self.weights_ = self.weights - learning_rate * grad_weights.astype(
            self.weights.dtype,
            copy=False,
        )
        self.bias_ = self.bias - learning_rate * grad_bias.astype(
            self.bias.dtype,
            copy=False,
        )

        return grad_input.astype(self.weights.dtype, copy=False)
