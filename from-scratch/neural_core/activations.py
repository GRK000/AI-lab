from __future__ import annotations

import numpy as np

from .common import Activation, FloatArray


def _identity(x: FloatArray) -> FloatArray:
    return x


def _identity_derivative(output: FloatArray, linear: FloatArray) -> FloatArray:
    del output
    return np.ones_like(linear)


def _sigmoid(x: FloatArray) -> FloatArray:
    result = np.empty_like(x)
    positive_mask = x >= 0.0

    result[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))
    exp_values = np.exp(x[~positive_mask])
    result[~positive_mask] = exp_values / (1.0 + exp_values)
    return result


def _sigmoid_derivative(output: FloatArray, linear: FloatArray) -> FloatArray:
    del linear
    return output * (1.0 - output)


def _tanh(x: FloatArray) -> FloatArray:
    return np.tanh(x)


def _tanh_derivative(output: FloatArray, linear: FloatArray) -> FloatArray:
    del linear
    return 1.0 - np.square(output)


def _relu(x: FloatArray) -> FloatArray:
    return np.maximum(x, 0.0)


def _relu_derivative(output: FloatArray, linear: FloatArray) -> FloatArray:
    del output
    return (linear > 0.0).astype(linear.dtype, copy=False)


def _softmax(x: FloatArray) -> FloatArray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _softmax_derivative(output: FloatArray, linear: FloatArray) -> FloatArray:
    del output, linear
    raise ValueError(
        "softmax uses a coupled derivative. Use it only in the output layer "
        "with categorical_crossentropy."
    )


ACTIVATIONS: dict[str, Activation] = {
    "identity": Activation(_identity, _identity_derivative),
    "sigmoid": Activation(_sigmoid, _sigmoid_derivative),
    "tanh": Activation(_tanh, _tanh_derivative),
    "relu": Activation(_relu, _relu_derivative),
    "softmax": Activation(_softmax, _softmax_derivative),
}
