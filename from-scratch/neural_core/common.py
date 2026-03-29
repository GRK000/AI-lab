from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]
ProblemType = Literal["regression", "binary", "multiclass"]
ActivationName = Literal["identity", "sigmoid", "tanh", "relu", "softmax"]
LossName = Literal["mse", "binary_crossentropy", "categorical_crossentropy"]

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
