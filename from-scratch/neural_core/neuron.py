from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .common import ActivationName, FloatArray, LossName, OptimizerName
from .layers import DenseLayer
from .network import NeuralNetwork
from .optimizers import BaseOptimizer


class Neuron(NeuralNetwork):
    """
    Single differentiable neuron.

    It is ideal for linear regression and binary classification. For deeper
    or multiclass problems, use NeuralNetwork with several DenseLayer blocks.
    """

    def __init__(
        self,
        problem_type: Literal["regression", "binary"] = "regression",
        activation: ActivationName | None = None,
        learning_rate: float = 0.05,
        optimizer: OptimizerName | BaseOptimizer = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        max_epochs: int = 1000,
        batch_size: int | None = 32,
        shuffle: bool = True,
        l2_lambda: float = 0.0,
        random_state: int | None = None,
        loss: LossName | None = None,
        dtype: Any = np.float32,
    ) -> None:
        if problem_type not in {"regression", "binary"}:
            raise ValueError("Neuron supports only regression or binary classification.")

        final_activation = activation or (
            "identity" if problem_type == "regression" else "sigmoid"
        )

        super().__init__(
            layers=[DenseLayer(units=1, activation=final_activation)],
            problem_type=problem_type,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            max_epochs=max_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            l2_lambda=l2_lambda,
            random_state=random_state,
            loss=loss,
            dtype=dtype,
        )

    @property
    def weights_(self) -> FloatArray:
        return self.layers[0].weights.reshape(-1)

    @property
    def bias_(self) -> float:
        return float(self.layers[0].bias.reshape(-1)[0])
