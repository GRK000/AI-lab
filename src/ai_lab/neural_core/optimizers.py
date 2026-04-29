from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .common import FloatArray, OptimizerName


class BaseOptimizer:
    def __init__(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")

        self.learning_rate = float(learning_rate)
        self._step = 0
        self._state: dict[str, dict[str, FloatArray]] = {}

    def begin_step(self) -> None:
        self._step += 1

    def reset_state(self) -> None:
        self._step = 0
        self._state.clear()

    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray:
        raise NotImplementedError

    def _state_array(
        self,
        parameter_key: str,
        state_name: str,
        parameter: FloatArray,
    ) -> FloatArray:
        parameter_state = self._state.setdefault(parameter_key, {})
        if state_name not in parameter_state:
            parameter_state[state_name] = np.zeros_like(parameter)
        return parameter_state[state_name]

    def _cast_like(self, reference: FloatArray, values: FloatArray) -> FloatArray:
        return np.asarray(values, dtype=reference.dtype)


class SGDOptimizer(BaseOptimizer):
    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray:
        del parameter_key
        updated = parameter - self.learning_rate * gradient
        return self._cast_like(parameter, updated)


class MomentumOptimizer(BaseOptimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        super().__init__(learning_rate)
        self.momentum = float(momentum)

    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray:
        velocity = self._state_array(parameter_key, "velocity", parameter)
        velocity *= self.momentum
        velocity -= self.learning_rate * gradient
        updated = parameter + velocity
        return self._cast_like(parameter, updated)


class RMSpropOptimizer(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        rho: float = 0.9,
        epsilon: float = 1e-7,
    ) -> None:
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1).")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.")
        super().__init__(learning_rate)
        self.rho = float(rho)
        self.epsilon = float(epsilon)

    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray:
        avg_squared = self._state_array(parameter_key, "avg_squared", parameter)
        avg_squared *= self.rho
        avg_squared += (1.0 - self.rho) * np.square(gradient)
        updated = parameter - self.learning_rate * gradient / (
            np.sqrt(avg_squared) + self.epsilon
        )
        return self._cast_like(parameter, updated)


class AdadeltaOptimizer(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        rho: float = 0.95,
        epsilon: float = 1e-6,
    ) -> None:
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1).")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.")
        super().__init__(learning_rate)
        self.rho = float(rho)
        self.epsilon = float(epsilon)

    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray:
        avg_squared_grad = self._state_array(parameter_key, "avg_squared_grad", parameter)
        avg_squared_update = self._state_array(
            parameter_key,
            "avg_squared_update",
            parameter,
        )

        avg_squared_grad *= self.rho
        avg_squared_grad += (1.0 - self.rho) * np.square(gradient)

        rms_update = np.sqrt(avg_squared_update + self.epsilon)
        rms_grad = np.sqrt(avg_squared_grad + self.epsilon)
        delta = -self.learning_rate * (rms_update / rms_grad) * gradient

        avg_squared_update *= self.rho
        avg_squared_update += (1.0 - self.rho) * np.square(delta)

        updated = parameter + delta
        return self._cast_like(parameter, updated)


class AdamOptimizer(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1).")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1).")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.")
        super().__init__(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray:
        m = self._state_array(parameter_key, "m", parameter)
        v = self._state_array(parameter_key, "v", parameter)

        m *= self.beta1
        m += (1.0 - self.beta1) * gradient
        v *= self.beta2
        v += (1.0 - self.beta2) * np.square(gradient)

        bias_correction1 = 1.0 - self.beta1**self._step
        bias_correction2 = 1.0 - self.beta2**self._step
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2

        updated = parameter - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return self._cast_like(parameter, updated)


class AdamWOptimizer(AdamOptimizer):
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0.")
        super().__init__(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        self.weight_decay = float(weight_decay)

    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray:
        updated = super().update(parameter_key, parameter, gradient)
        if self.weight_decay == 0.0:
            return updated

        decayed = updated - self.learning_rate * self.weight_decay * parameter
        return self._cast_like(parameter, decayed)


class AdamaxOptimizer(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1).")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1).")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.")
        super().__init__(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

    def update(
        self,
        parameter_key: str,
        parameter: FloatArray,
        gradient: FloatArray,
    ) -> FloatArray:
        m = self._state_array(parameter_key, "m", parameter)
        u = self._state_array(parameter_key, "u", parameter)

        m *= self.beta1
        m += (1.0 - self.beta1) * gradient
        u *= self.beta2
        u[:] = np.maximum(u, np.abs(gradient))

        bias_correction = 1.0 - self.beta1**self._step
        step_size = self.learning_rate / bias_correction
        updated = parameter - step_size * m / (u + self.epsilon)
        return self._cast_like(parameter, updated)


OPTIMIZER_REGISTRY: dict[str, type[BaseOptimizer]] = {
    "sgd": SGDOptimizer,
    "momentum": MomentumOptimizer,
    "rmsprop": RMSpropOptimizer,
    "adadelta": AdadeltaOptimizer,
    "adam": AdamOptimizer,
    "adamw": AdamWOptimizer,
    "adamax": AdamaxOptimizer,
}


def make_optimizer(
    optimizer: OptimizerName | BaseOptimizer,
    learning_rate: float,
    optimizer_kwargs: Mapping[str, Any] | None = None,
) -> BaseOptimizer:
    if isinstance(optimizer, BaseOptimizer):
        if optimizer_kwargs:
            raise ValueError(
                "optimizer_kwargs cannot be used when passing an optimizer instance."
            )
        optimizer.reset_state()
        return optimizer

    normalized_name = optimizer.lower()
    optimizer_cls = OPTIMIZER_REGISTRY.get(normalized_name)
    if optimizer_cls is None:
        available = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise ValueError(f"Unsupported optimizer: {optimizer!r}. Available: {available}.")

    kwargs = dict(optimizer_kwargs or {})
    return optimizer_cls(learning_rate=learning_rate, **kwargs)
