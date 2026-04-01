from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]
ProblemType = Literal["regression", "binary", "multiclass"]
ActivationName = Literal["identity", "sigmoid", "tanh", "relu", "softmax"]
LossName = Literal["mse", "binary_crossentropy", "categorical_crossentropy"]
OptimizerName = Literal[
    "sgd",
    "momentum",
    "rmsprop",
    "adadelta",
    "adam",
    "adamw",
    "adamax",
]

EPSILON = 1e-7


@dataclass(slots=True)
class TrainingSnapshot:
    epoch: int
    loss: float
    metric: float


@dataclass(frozen=True, slots=True)
class Activation:
    forward: Callable[[FloatArray], FloatArray]
    derivative: Callable[[FloatArray, FloatArray], FloatArray]


class Optimizer(Protocol):
    learning_rate: float

    def begin_step(self) -> None: ...

    def reset_state(self) -> None: ...

    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray: ...


class Layer(Protocol):
    def build(self, input_dim: int, rng: np.random.Generator, dtype: Any) -> int: ...

    def forward(self, inputs: FloatArray, training: bool = False) -> FloatArray: ...

    def backward(
        self,
        grad_output: FloatArray,
        learning_rate: float | None = None,
        optimizer: Optimizer | None = None,
        l2_lambda: float = 0.0,
        apply_activation_derivative: bool = True,
    ) -> FloatArray: ...
