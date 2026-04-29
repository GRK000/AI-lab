from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .activations import ACTIVATIONS
from .common import ActivationName, FloatArray, Optimizer


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

    def build(self, input_dim: int, rng: np.random.Generator, dtype: Any) -> int:
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
        return self.units

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "DenseLayer",
            "units": self.units,
            "activation": self.activation,
        }

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
        grad_output: FloatArray,
        learning_rate: float | None = None,
        optimizer: Optimizer | None = None,
        l2_lambda: float = 0.0,
        apply_activation_derivative: bool = True,
    ) -> FloatArray:
        if self._last_input is None:
            raise RuntimeError("No forward pass cached for this layer.")

        delta = grad_output
        if apply_activation_derivative:
            delta = grad_output * self.activation_derivative()

        grad_input = delta @ self.weights.T
        grad_weights = self._last_input.T @ delta
        grad_bias = np.sum(delta, axis=0, keepdims=True)

        if l2_lambda > 0.0:
            grad_weights += l2_lambda * self.weights

        cast_grad_weights = grad_weights.astype(self.weights.dtype, copy=False)
        cast_grad_bias = grad_bias.astype(self.bias.dtype, copy=False)

        if optimizer is not None:
            self.weights_ = optimizer.update(
                f"{id(self)}.weights",
                self.weights,
                cast_grad_weights,
            )
            self.bias_ = optimizer.update(
                f"{id(self)}.bias",
                self.bias,
                cast_grad_bias,
            )
        else:
            if learning_rate is None:
                raise ValueError(
                    "learning_rate is required when no optimizer instance is provided."
                )
            self.weights_ = self.weights - learning_rate * cast_grad_weights
            self.bias_ = self.bias - learning_rate * cast_grad_bias

        return grad_input.astype(self.weights.dtype, copy=False)


@dataclass(slots=True)
class DropoutLayer:
    """
    Inverted dropout layer.

    During training it randomly drops activations and rescales survivors so
    inference can run as a pure identity transform.
    """

    rate: float
    input_dim_: int | None = field(init=False, default=None)
    dtype_: Any | None = field(init=False, default=None)
    _rng: np.random.Generator | None = field(init=False, default=None)
    _mask: FloatArray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate < 1.0:
            raise ValueError("rate must be in [0, 1).")

    def build(self, input_dim: int, rng: np.random.Generator, dtype: Any) -> int:
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")

        self.input_dim_ = int(input_dim)
        self.dtype_ = dtype
        self._rng = rng
        return input_dim

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "DropoutLayer",
            "rate": self.rate,
        }

    def forward(self, inputs: FloatArray, training: bool = False) -> FloatArray:
        if self.input_dim_ is None or self._rng is None:
            raise RuntimeError("Layer parameters are not initialized yet.")

        if inputs.shape[1] != self.input_dim_:
            raise ValueError(f"Expected {self.input_dim_} features, got {inputs.shape[1]}.")

        if not training or self.rate == 0.0:
            self._mask = None
            return inputs

        keep_probability = 1.0 - self.rate
        self._mask = (
            (self._rng.random(inputs.shape) < keep_probability).astype(inputs.dtype)
            / keep_probability
        )
        return (inputs * self._mask).astype(inputs.dtype, copy=False)

    def backward(
        self,
        grad_output: FloatArray,
        learning_rate: float | None = None,
        optimizer: Optimizer | None = None,
        l2_lambda: float = 0.0,
        apply_activation_derivative: bool = True,
    ) -> FloatArray:
        del learning_rate, optimizer, l2_lambda, apply_activation_derivative

        if self._mask is None:
            return grad_output
        return (grad_output * self._mask).astype(grad_output.dtype, copy=False)


@dataclass(slots=True)
class FlattenLayer:
    input_dim_: int | None = field(init=False, default=None)
    input_shape_: tuple[int, ...] | None = field(init=False, default=None)

    def build(self, input_dim: int, rng: np.random.Generator, dtype: Any) -> int:
        del rng, dtype
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        self.input_dim_ = int(input_dim)
        return input_dim

    def forward(self, inputs: FloatArray, training: bool = False) -> FloatArray:
        del training
        self.input_shape_ = tuple(inputs.shape)
        if inputs.ndim == 2:
            return inputs
        return inputs.reshape(inputs.shape[0], -1)

    def backward(
        self,
        grad_output: FloatArray,
        learning_rate: float | None = None,
        optimizer: Optimizer | None = None,
        l2_lambda: float = 0.0,
        apply_activation_derivative: bool = True,
    ) -> FloatArray:
        del learning_rate, optimizer, l2_lambda, apply_activation_derivative
        if self.input_shape_ is None:
            raise RuntimeError("No forward pass cached for this layer.")
        return grad_output.reshape(self.input_shape_)

    def get_config(self) -> dict[str, Any]:
        return {"type": "FlattenLayer"}


@dataclass(slots=True)
class BatchNormLayer:
    momentum: float = 0.9
    epsilon: float = 1e-5
    input_dim_: int | None = field(init=False, default=None)
    dtype_: Any | None = field(init=False, default=None)
    gamma_: FloatArray | None = field(init=False, default=None)
    beta_: FloatArray | None = field(init=False, default=None)
    running_mean_: FloatArray | None = field(init=False, default=None)
    running_var_: FloatArray | None = field(init=False, default=None)
    _last_centered: FloatArray | None = field(init=False, default=None)
    _last_inv_std: FloatArray | None = field(init=False, default=None)
    _last_normalized: FloatArray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.")

    @property
    def gamma(self) -> FloatArray:
        if self.gamma_ is None:
            raise RuntimeError("Layer parameters are not initialized yet.")
        return self.gamma_

    @property
    def beta(self) -> FloatArray:
        if self.beta_ is None:
            raise RuntimeError("Layer parameters are not initialized yet.")
        return self.beta_

    def build(self, input_dim: int, rng: np.random.Generator, dtype: Any) -> int:
        del rng
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        self.input_dim_ = int(input_dim)
        self.dtype_ = dtype
        self.gamma_ = np.ones((1, input_dim), dtype=dtype)
        self.beta_ = np.zeros((1, input_dim), dtype=dtype)
        self.running_mean_ = np.zeros((1, input_dim), dtype=dtype)
        self.running_var_ = np.ones((1, input_dim), dtype=dtype)
        return input_dim

    def forward(self, inputs: FloatArray, training: bool = False) -> FloatArray:
        if self.running_mean_ is None or self.running_var_ is None:
            raise RuntimeError("Layer parameters are not initialized yet.")
        if inputs.ndim != 2:
            raise ValueError("BatchNormLayer expects a 2D input.")

        if training:
            batch_mean = np.mean(inputs, axis=0, keepdims=True)
            batch_var = np.var(inputs, axis=0, keepdims=True)
            self.running_mean_ = (
                self.momentum * self.running_mean_ + (1.0 - self.momentum) * batch_mean
            )
            self.running_var_ = (
                self.momentum * self.running_var_ + (1.0 - self.momentum) * batch_var
            )
            centered = inputs - batch_mean
            inv_std = 1.0 / np.sqrt(batch_var + self.epsilon)
            normalized = centered * inv_std
            self._last_centered = centered
            self._last_inv_std = inv_std
            self._last_normalized = normalized
        else:
            normalized = (inputs - self.running_mean_) / np.sqrt(
                self.running_var_ + self.epsilon
            )

        return (self.gamma * normalized + self.beta).astype(inputs.dtype, copy=False)

    def backward(
        self,
        grad_output: FloatArray,
        learning_rate: float | None = None,
        optimizer: Optimizer | None = None,
        l2_lambda: float = 0.0,
        apply_activation_derivative: bool = True,
    ) -> FloatArray:
        del l2_lambda, apply_activation_derivative
        if (
            self._last_centered is None
            or self._last_inv_std is None
            or self._last_normalized is None
        ):
            raise RuntimeError("No forward pass cached for this layer.")

        n_samples = grad_output.shape[0]
        grad_gamma = np.sum(grad_output * self._last_normalized, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        grad_normalized = grad_output * self.gamma
        grad_var = np.sum(
            grad_normalized
            * self._last_centered
            * -0.5
            * np.power(self._last_inv_std, 3),
            axis=0,
            keepdims=True,
        )
        grad_mean = (
            np.sum(-grad_normalized * self._last_inv_std, axis=0, keepdims=True)
            + grad_var * np.mean(-2.0 * self._last_centered, axis=0, keepdims=True)
        )
        grad_input = (
            grad_normalized * self._last_inv_std
            + grad_var * 2.0 * self._last_centered / n_samples
            + grad_mean / n_samples
        )

        cast_grad_gamma = grad_gamma.astype(self.gamma.dtype, copy=False)
        cast_grad_beta = grad_beta.astype(self.beta.dtype, copy=False)
        if optimizer is not None:
            self.gamma_ = optimizer.update(f"{id(self)}.gamma", self.gamma, cast_grad_gamma)
            self.beta_ = optimizer.update(f"{id(self)}.beta", self.beta, cast_grad_beta)
        else:
            if learning_rate is None:
                raise ValueError(
                    "learning_rate is required when no optimizer instance is provided."
                )
            self.gamma_ = self.gamma - learning_rate * cast_grad_gamma
            self.beta_ = self.beta - learning_rate * cast_grad_beta

        return grad_input.astype(grad_output.dtype, copy=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "BatchNormLayer",
            "momentum": self.momentum,
            "epsilon": self.epsilon,
        }


@dataclass(slots=True)
class Conv2DLayer:
    filters: int
    kernel_size: int | tuple[int, int]
    stride: int = 1
    padding: int = 0
    input_shape_: tuple[int, int, int] | None = field(init=False, default=None)
    kernels_: FloatArray | None = field(init=False, default=None)
    bias_: FloatArray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.filters <= 0:
            raise ValueError("filters must be > 0.")
        if self.stride <= 0:
            raise ValueError("stride must be > 0.")
        if self.padding < 0:
            raise ValueError("padding must be >= 0.")

    def build(
        self,
        input_shape: tuple[int, int, int],
        rng: np.random.Generator,
        dtype: Any = np.float32,
    ) -> tuple[int, int, int]:
        height, width, channels = input_shape
        kernel_height, kernel_width = self._kernel_shape()
        if min(height, width, channels) <= 0:
            raise ValueError("input_shape dimensions must be > 0.")
        if kernel_height <= 0 or kernel_width <= 0:
            raise ValueError("kernel_size must be > 0.")

        out_height = (height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_width) // self.stride + 1
        if out_height <= 0 or out_width <= 0:
            raise ValueError("kernel_size, stride and padding produce an empty output.")

        scale = np.sqrt(2.0 / (kernel_height * kernel_width * channels))
        self.input_shape_ = input_shape
        self.kernels_ = rng.normal(
            0.0,
            scale,
            size=(self.filters, kernel_height, kernel_width, channels),
        ).astype(dtype, copy=False)
        self.bias_ = np.zeros((self.filters,), dtype=dtype)
        return out_height, out_width, self.filters

    def forward(self, inputs: FloatArray) -> FloatArray:
        if self.kernels_ is None or self.bias_ is None:
            raise RuntimeError("Layer parameters are not initialized yet.")
        if inputs.ndim != 4:
            raise ValueError(
                "Conv2DLayer expects input shaped as (batch, height, width, channels)."
            )

        batch, height, width, _channels = inputs.shape
        kernel_height, kernel_width = self._kernel_shape()
        padded = np.pad(
            inputs,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
        )
        out_height = (height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_width) // self.stride + 1
        output = np.zeros((batch, out_height, out_width, self.filters), dtype=inputs.dtype)

        for row in range(out_height):
            for col in range(out_width):
                window = padded[
                    :,
                    row * self.stride : row * self.stride + kernel_height,
                    col * self.stride : col * self.stride + kernel_width,
                    :,
                ]
                output[:, row, col, :] = np.tensordot(
                    window,
                    self.kernels_,
                    axes=((1, 2, 3), (1, 2, 3)),
                )

        output += self.bias_
        return output.astype(inputs.dtype, copy=False)

    def _kernel_shape(self) -> tuple[int, int]:
        if isinstance(self.kernel_size, int):
            return self.kernel_size, self.kernel_size
        return self.kernel_size


@dataclass(slots=True)
class MaxPool2DLayer:
    pool_size: int | tuple[int, int] = 2
    stride: int | None = None

    def forward(self, inputs: FloatArray) -> FloatArray:
        if inputs.ndim != 4:
            raise ValueError(
                "MaxPool2DLayer expects input shaped as (batch, height, width, channels)."
            )
        pool_height, pool_width = self._pool_shape()
        stride = self.stride or pool_height
        if pool_height <= 0 or pool_width <= 0 or stride <= 0:
            raise ValueError("pool_size and stride must be > 0.")

        batch, height, width, channels = inputs.shape
        out_height = (height - pool_height) // stride + 1
        out_width = (width - pool_width) // stride + 1
        if out_height <= 0 or out_width <= 0:
            raise ValueError("pool_size and stride produce an empty output.")

        output = np.zeros((batch, out_height, out_width, channels), dtype=inputs.dtype)
        for row in range(out_height):
            for col in range(out_width):
                window = inputs[
                    :,
                    row * stride : row * stride + pool_height,
                    col * stride : col * stride + pool_width,
                    :,
                ]
                output[:, row, col, :] = np.max(window, axis=(1, 2))
        return output

    def _pool_shape(self) -> tuple[int, int]:
        if isinstance(self.pool_size, int):
            return self.pool_size, self.pool_size
        return self.pool_size
