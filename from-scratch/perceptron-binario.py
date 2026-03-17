from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.int8]


@dataclass(slots=True)
class EpochSnapshot:
    epoch: int
    errors: int
    accuracy: float
    mean_margin: float
    weight_norm: float


class Perceptron:
    """
    Binary perceptron implemented from scratch.

    Design goals:
    - Fast enough for medium tabular data through vectorized updates.
    - Clean API that can evolve into dense layers and full neural networks.
    - Reproducible training through controlled initialization.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_epochs: int = 100,
        fit_intercept: bool = True,
        shuffle: bool = True,
        random_state: int | None = None,
        dtype: Any = np.float32,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0.")

        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.fit_intercept = bool(fit_intercept)
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.dtype = dtype

        self.weights_: FloatArray | None = None
        self.bias_: float = 0.0
        self.classes_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None
        self.history_: list[EpochSnapshot] = []
        self.epochs_trained_: int = 0

        self._rng = np.random.default_rng(random_state)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "Perceptron":
        X_array = self._prepare_features(X)
        y_raw = self._prepare_targets(y)
        self._validate_sample_count(X_array, y_raw)
        y_signed = self._set_or_validate_classes(y_raw)

        self._initialize_parameters(X_array.shape[1], reset=True)
        self.history_.clear()
        self.epochs_trained_ = 0

        features = X_array
        targets = y_signed

        for epoch in range(1, self.max_epochs + 1):
            if self.shuffle:
                order = self._rng.permutation(features.shape[0])
                features = features[order]
                targets = targets[order]

            self._train_one_epoch(features, targets)
            snapshot = self._snapshot(epoch, features, targets)
            self.history_.append(snapshot)
            self.epochs_trained_ = epoch

            if snapshot.errors == 0:
                break

        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        classes: Iterable[Any] | None = None,
    ) -> "Perceptron":
        X_array = self._prepare_features(X)
        y_raw = self._prepare_targets(y)
        self._validate_sample_count(X_array, y_raw)

        if self.classes_ is None:
            if classes is None:
                inferred = np.unique(y_raw)
                if inferred.size != 2:
                    raise ValueError(
                        "partial_fit needs both classes on first call or an explicit classes iterable."
                    )
                self.classes_ = inferred
            else:
                provided = np.asarray(list(classes))
                if provided.size != 2:
                    raise ValueError("classes must contain exactly two labels.")
                self.classes_ = np.unique(provided)

        y_signed = self._encode_targets(y_raw)
        self._initialize_parameters(X_array.shape[1], reset=False)

        self._train_one_epoch(X_array, y_signed)
        self.epochs_trained_ += 1
        self.history_.append(self._snapshot(self.epochs_trained_, X_array, y_signed))
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        X_array = self._prepare_features(X, validate_only=True)
        return self._linear_output(X_array)

    def margin(self, X: ArrayLike) -> FloatArray:
        scores = self.decision_function(X)
        norm = float(np.linalg.norm(self._weights))
        scale = norm if norm > 0 else 1.0
        return scores / scale

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        scores = self.decision_function(X)
        indices = (scores >= 0.0).astype(np.intp)
        return self._classes[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_true = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        return float(np.mean(y_pred == y_true))

    @property
    def _weights(self) -> FloatArray:
        if self.weights_ is None:
            raise RuntimeError("The perceptron is not fitted yet.")
        return self.weights_

    @property
    def _classes(self) -> NDArray[Any]:
        if self.classes_ is None:
            raise RuntimeError("Class labels are not initialized yet.")
        return self.classes_

    def _prepare_features(self, X: ArrayLike, validate_only: bool = False) -> FloatArray:
        X_array = np.asarray(X, dtype=self.dtype)

        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        elif X_array.ndim != 2:
            raise ValueError("X must be a 2D array or a single sample.")

        X_array = np.ascontiguousarray(X_array, dtype=self.dtype)

        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if validate_only:
            if self.n_features_in_ is None:
                raise RuntimeError("The perceptron is not fitted yet.")
            if X_array.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {X_array.shape[1]}."
                )

        return X_array

    def _prepare_targets(self, y: ArrayLike) -> NDArray[Any]:
        y_array = np.asarray(y).reshape(-1)
        if y_array.size == 0:
            raise ValueError("y cannot be empty.")
        return y_array

    def _validate_sample_count(self, X: FloatArray, y: NDArray[Any]) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

    def _set_or_validate_classes(self, y: NDArray[Any]) -> IntArray:
        unique_classes = np.unique(y)
        if unique_classes.size != 2:
            raise ValueError("Perceptron is a binary classifier and needs exactly two classes.")

        self.classes_ = unique_classes
        return self._encode_targets(y)

    def _encode_targets(self, y: NDArray[Any]) -> IntArray:
        classes = self._classes
        unknown_mask = ~np.isin(y, classes)
        if np.any(unknown_mask):
            unknown = np.unique(y[unknown_mask])
            raise ValueError(f"Unknown class labels found: {unknown!r}")

        encoded = np.where(y == classes[0], -1, 1).astype(np.int8, copy=False)
        return encoded

    def _initialize_parameters(self, n_features: int, reset: bool) -> None:
        if not reset and self.weights_ is not None:
            if self.n_features_in_ != n_features:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {n_features}."
                )
            return

        self.n_features_in_ = n_features
        self.weights_ = self._rng.normal(loc=0.0, scale=0.01, size=n_features).astype(
            self.dtype,
            copy=False,
        )
        self.bias_ = 0.0

    def _linear_output(self, X: FloatArray) -> FloatArray:
        return X @ self._weights + self.bias_

    def _train_one_epoch(self, X: FloatArray, y: IntArray) -> int:
        scores = self._linear_output(X)
        predictions = np.where(scores >= 0.0, 1, -1).astype(np.int8, copy=False)
        mistake_mask = predictions != y
        errors = int(np.count_nonzero(mistake_mask))

        if errors == 0:
            return 0

        mistaken_X = X[mistake_mask]
        mistaken_y = y[mistake_mask].astype(self.dtype, copy=False)

        step = self.learning_rate / errors
        update = step * (mistaken_y @ mistaken_X)
        self.weights_ = self._weights + update.astype(self.dtype, copy=False)

        if self.fit_intercept:
            self.bias_ += float(step * mistaken_y.sum())

        return errors

    def _snapshot(self, epoch: int, X: FloatArray, y: IntArray) -> EpochSnapshot:
        scores = self._linear_output(X)
        predictions = np.where(scores >= 0.0, 1, -1).astype(np.int8, copy=False)
        errors = int(np.count_nonzero(predictions != y))
        margins = y.astype(self.dtype, copy=False) * scores
        return EpochSnapshot(
            epoch=epoch,
            errors=errors,
            accuracy=float(1.0 - errors / X.shape[0]),
            mean_margin=float(np.mean(margins)),
            weight_norm=float(np.linalg.norm(self._weights)),
        )


if __name__ == "__main__":
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.2, max_epochs=50, random_state=7)
    model.fit(X, y)

    print("Weights:", model.weights_)
    print("Bias:", model.bias_)
    print("Predictions:", model.predict(X))
    print("Accuracy:", model.score(X, y))
