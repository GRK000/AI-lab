from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .common import (
    EPSILON,
    FloatArray,
    Layer,
    LossName,
    OptimizerName,
    ProblemType,
    TrainingSnapshot,
)
from .layers import BatchNormLayer, DenseLayer, DropoutLayer, FlattenLayer
from .optimizers import BaseOptimizer, make_optimizer


class NeuralNetwork:
    """
    Minimal dense neural network implemented from scratch.

    Supported problem types:
    - regression: last layer usually uses identity + MSE
    - binary: last layer usually uses sigmoid + binary_crossentropy
    - multiclass: last layer usually uses softmax + categorical_crossentropy
    """

    def __init__(
        self,
        layers: Sequence[Layer],
        problem_type: ProblemType,
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
        validation_split: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        if not layers:
            raise ValueError("At least one layer is required.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be > 0 or None.")
        if l2_lambda < 0.0:
            raise ValueError("l2_lambda must be >= 0.")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1).")
        if patience <= 0:
            raise ValueError("patience must be > 0.")
        if min_delta < 0.0:
            raise ValueError("min_delta must be >= 0.")

        self.layers = list(layers)
        self.problem_type = problem_type
        self.learning_rate = float(learning_rate)
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.max_epochs = int(max_epochs)
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.l2_lambda = float(l2_lambda)
        self.random_state = random_state
        self.dtype = dtype
        self.validation_split = float(validation_split)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.callbacks = list(callbacks or [])

        self.loss_name = loss or self._default_loss(problem_type)
        self.metric_name = "mae" if problem_type == "regression" else "accuracy"

        self.n_features_in_: int | None = None
        self.n_outputs_: int | None = None
        self.classes_: NDArray[Any] | None = None
        self.history_: list[TrainingSnapshot] = []
        self.epochs_trained_: int = 0

        self._rng = np.random.default_rng(random_state)
        self._optimizer = make_optimizer(
            optimizer,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
        )
        self._validate_architecture()

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> NeuralNetwork:
        X_array = self._prepare_features(X)
        if self.problem_type != "regression":
            self.classes_ = None
        y_array = self._prepare_targets(y, fit=True)
        self._validate_sample_count(X_array, y_array)
        X_array, y_array, X_val, y_val = self._split_validation_data(
            X_array,
            y_array,
            validation_data,
        )
        self._build_network(X_array.shape[1], reset=True)
        self._validate_output_configuration(y_array)

        self.history_.clear()
        self.epochs_trained_ = 0
        self._optimizer.reset_state()
        for callback in self.callbacks:
            callback.on_train_begin(self)

        n_samples = X_array.shape[0]
        effective_batch_size = n_samples if self.batch_size is None else min(
            self.batch_size,
            n_samples,
        )
        best_loss = float("inf")
        best_parameters: list[dict[str, FloatArray]] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            if self.shuffle:
                indices = self._rng.permutation(n_samples)
                epoch_X = X_array[indices]
                epoch_y = y_array[indices]
            else:
                epoch_X = X_array
                epoch_y = y_array

            for start in range(0, n_samples, effective_batch_size):
                end = start + effective_batch_size
                batch_X = epoch_X[start:end]
                batch_y = epoch_y[start:end]

                predictions = self._forward(batch_X, training=True)
                self._train_on_batch(batch_y, predictions)

            epoch_predictions = self._forward(X_array, training=False)
            train_loss = self._loss(epoch_predictions, y_array)
            train_metric = self._metric(epoch_predictions, y_array)
            val_loss = None
            val_metric = None
            monitored_loss = train_loss

            if X_val is not None and y_val is not None:
                validation_predictions = self._forward(X_val, training=False)
                val_loss = self._loss(validation_predictions, y_val)
                val_metric = self._metric(validation_predictions, y_val)
                monitored_loss = val_loss

            snapshot = TrainingSnapshot(
                epoch=epoch,
                loss=train_loss,
                metric=train_metric,
                val_loss=val_loss,
                val_metric=val_metric,
            )
            self.history_.append(snapshot)
            self.epochs_trained_ = epoch
            logs = {
                "epoch": epoch,
                "loss": train_loss,
                "metric": train_metric,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "learning_rate": self.learning_rate,
            }

            if self.early_stopping:
                if monitored_loss < best_loss - self.min_delta:
                    best_loss = monitored_loss
                    best_parameters = self._copy_trainable_parameters()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.restore_best_weights and best_parameters is not None:
                        self._restore_trainable_parameters(best_parameters)
                    break

            if any(callback.on_epoch_end(self, logs) for callback in self.callbacks):
                break

        for callback in self.callbacks:
            callback.on_train_end(self)
        return self

    def save(self, path: str | Path) -> None:
        if self.n_features_in_ is None or self.n_outputs_ is None:
            raise RuntimeError("Cannot save an unfitted model.")
        if not isinstance(self.optimizer, str):
            raise ValueError("Only models using a built-in optimizer name can be saved.")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "layers": [self._layer_config(layer) for layer in self.layers],
            "problem_type": self.problem_type,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "l2_lambda": self.l2_lambda,
            "random_state": self.random_state,
            "loss": self.loss_name,
            "dtype": np.dtype(self.dtype).name,
            "validation_split": self.validation_split,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "restore_best_weights": self.restore_best_weights,
            "n_features_in": self.n_features_in_,
            "n_outputs": self.n_outputs_,
            "classes": None if self.classes_ is None else self.classes_.tolist(),
        }

        arrays: dict[str, FloatArray] = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                arrays[f"layer_{index}_weights"] = layer.weights
                arrays[f"layer_{index}_bias"] = layer.bias
            if isinstance(layer, BatchNormLayer):
                arrays[f"layer_{index}_gamma"] = layer.gamma
                arrays[f"layer_{index}_beta"] = layer.beta
                if layer.running_mean_ is not None and layer.running_var_ is not None:
                    arrays[f"layer_{index}_running_mean"] = layer.running_mean_
                    arrays[f"layer_{index}_running_var"] = layer.running_var_

        np.savez_compressed(target, metadata=json.dumps(metadata), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            layers = [cls._layer_from_config(config) for config in metadata["layers"]]
            model = cls(
                layers=layers,
                problem_type=metadata["problem_type"],
                learning_rate=metadata["learning_rate"],
                optimizer=metadata["optimizer"],
                optimizer_kwargs=metadata["optimizer_kwargs"],
                max_epochs=metadata["max_epochs"],
                batch_size=metadata["batch_size"],
                shuffle=metadata["shuffle"],
                l2_lambda=metadata["l2_lambda"],
                random_state=metadata["random_state"],
                loss=metadata["loss"],
                dtype=np.dtype(metadata["dtype"]).type,
                validation_split=metadata["validation_split"],
                early_stopping=metadata["early_stopping"],
                patience=metadata["patience"],
                min_delta=metadata["min_delta"],
                restore_best_weights=metadata["restore_best_weights"],
            )
            model._build_network(int(metadata["n_features_in"]), reset=True)
            model.n_outputs_ = int(metadata["n_outputs"])
            model.classes_ = (
                None
                if metadata["classes"] is None
                else np.asarray(metadata["classes"])
            )

            for index, layer in enumerate(model.layers):
                if isinstance(layer, DenseLayer):
                    layer.weights_ = np.asarray(
                        data[f"layer_{index}_weights"],
                        dtype=model.dtype,
                    )
                    layer.bias_ = np.asarray(
                        data[f"layer_{index}_bias"],
                        dtype=model.dtype,
                    )
                if isinstance(layer, BatchNormLayer):
                    layer.gamma_ = np.asarray(data[f"layer_{index}_gamma"], dtype=model.dtype)
                    layer.beta_ = np.asarray(data[f"layer_{index}_beta"], dtype=model.dtype)
                    layer.running_mean_ = np.asarray(
                        data[f"layer_{index}_running_mean"],
                        dtype=model.dtype,
                    )
                    layer.running_var_ = np.asarray(
                        data[f"layer_{index}_running_var"],
                        dtype=model.dtype,
                    )

        return model

    def forward(self, X: ArrayLike) -> FloatArray:
        X_array = self._prepare_features(X, validate_only=True)
        return self._forward(X_array, training=False)

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        if self.problem_type == "regression":
            raise RuntimeError("predict_proba is only available for classification models.")

        outputs = self.forward(X)

        if self.problem_type == "binary":
            positive = outputs.reshape(-1, 1)
            negative = 1.0 - positive
            return np.hstack((negative, positive)).astype(self.dtype, copy=False)

        return outputs

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        outputs = self.forward(X)

        if self.problem_type == "regression":
            return self._format_regression_output(outputs)

        if self.problem_type == "binary":
            indices = (outputs.reshape(-1) >= 0.5).astype(np.intp, copy=False)
            return self._classes[indices]

        indices = np.argmax(outputs, axis=1)
        return self._classes[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        if self.problem_type == "regression":
            y_true = self._prepare_targets(y, fit=False)
            y_pred = np.asarray(self.predict(X), dtype=self.dtype)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            ss_res = float(np.sum(np.square(y_true - y_pred)))
            centered = y_true - np.mean(y_true, axis=0, keepdims=True)
            ss_tot = float(np.sum(np.square(centered)))

            if ss_tot == 0.0:
                return 1.0 if ss_res == 0.0 else 0.0

            return float(1.0 - ss_res / ss_tot)

        predicted_labels = self.predict(X)
        target_labels = self._prepare_label_targets(y)
        if predicted_labels.shape[0] != target_labels.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        return float(np.mean(predicted_labels == target_labels))

    @property
    def _classes(self) -> NDArray[Any]:
        if self.classes_ is None:
            raise RuntimeError("Classes are not initialized yet.")
        return self.classes_

    def _validate_architecture(self) -> None:
        for layer in self.layers[:-1]:
            if isinstance(layer, DropoutLayer):
                continue
            if getattr(layer, "activation", None) == "softmax":
                raise ValueError("softmax is only supported in the output layer.")

        if isinstance(self.layers[-1], DropoutLayer):
            raise ValueError("DropoutLayer cannot be used as the output layer.")

    def _default_loss(self, problem_type: ProblemType) -> LossName:
        if problem_type == "regression":
            return "mse"
        if problem_type == "binary":
            return "binary_crossentropy"
        return "categorical_crossentropy"

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
                raise RuntimeError("The model is not fitted yet.")
            if X_array.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {X_array.shape[1]}."
                )

        return X_array

    def _prepare_targets(self, y: ArrayLike, fit: bool) -> FloatArray:
        if self.problem_type == "regression":
            y_array = np.asarray(y, dtype=self.dtype)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            elif y_array.ndim != 2:
                raise ValueError("Regression targets must be 1D or 2D.")
            return np.ascontiguousarray(y_array, dtype=self.dtype)

        if self.problem_type == "binary":
            labels = self._prepare_label_targets(y)

            if fit:
                unique_labels = np.unique(labels)
                if unique_labels.size != 2:
                    raise ValueError("Binary classification requires exactly two classes.")
                self.classes_ = unique_labels
            elif self.classes_ is None:
                raise RuntimeError("The model is not fitted yet.")

            known_labels = set(self._classes.tolist())
            unknown_labels = [
                label for label in labels.tolist() if label not in known_labels
            ]
            if unknown_labels:
                raise ValueError(f"Unknown labels found: {np.unique(unknown_labels)!r}")

            encoded = (labels == self._classes[1]).astype(self.dtype, copy=False)
            return encoded.reshape(-1, 1)

        y_array = np.asarray(y)

        if y_array.ndim == 2:
            if y_array.shape[1] == 0:
                raise ValueError("Multiclass targets cannot be empty.")

            if fit and self.classes_ is None:
                self.classes_ = np.arange(y_array.shape[1])
            elif self.classes_ is not None and y_array.shape[1] != self._classes.size:
                raise ValueError(
                    f"Expected {self._classes.size} output columns, got {y_array.shape[1]}."
                )

            return np.ascontiguousarray(y_array, dtype=self.dtype)

        labels = self._prepare_label_targets(y)
        if fit:
            self.classes_ = np.unique(labels)
        elif self.classes_ is None:
            raise RuntimeError("The model is not fitted yet.")

        label_to_index = {label: index for index, label in enumerate(self._classes.tolist())}

        try:
            indices = np.array(
                [label_to_index[label] for label in labels.tolist()],
                dtype=np.intp,
            )
        except KeyError as error:
            raise ValueError(f"Unknown label found: {error.args[0]!r}") from error

        encoded = np.zeros((labels.shape[0], self._classes.size), dtype=self.dtype)
        encoded[np.arange(labels.shape[0]), indices] = 1.0
        return encoded

    def _prepare_label_targets(self, y: ArrayLike) -> NDArray[Any]:
        y_array = np.asarray(y)

        if y_array.ndim == 2 and self.problem_type == "multiclass":
            if y_array.shape[1] == 0:
                raise ValueError("Multiclass targets cannot be empty.")
            if self.classes_ is None:
                classes = np.arange(y_array.shape[1])
            else:
                classes = self._classes
            indices = np.argmax(y_array, axis=1)
            return classes[indices]

        labels = y_array.reshape(-1)
        if labels.size == 0:
            raise ValueError("y cannot be empty.")
        return labels

    def _validate_sample_count(self, X: FloatArray, y: FloatArray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

    def _split_validation_data(
        self,
        X: FloatArray,
        y: FloatArray,
        validation_data: tuple[ArrayLike, ArrayLike] | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
        if validation_data is not None and self.validation_split > 0.0:
            raise ValueError("Use either validation_data or validation_split, not both.")

        if validation_data is not None:
            X_val = self._prepare_features(validation_data[0])
            y_val = self._prepare_targets(validation_data[1], fit=False)
            self._validate_sample_count(X_val, y_val)
            return X, y, X_val, y_val

        if self.validation_split == 0.0:
            return X, y, None, None

        n_samples = X.shape[0]
        validation_count = int(round(n_samples * self.validation_split))
        if validation_count <= 0 or validation_count >= n_samples:
            raise ValueError("validation_split leaves no data for training or validation.")

        if self.shuffle:
            indices = self._rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        validation_indices = indices[:validation_count]
        training_indices = indices[validation_count:]
        return (
            X[training_indices],
            y[training_indices],
            X[validation_indices],
            y[validation_indices],
        )

    def _build_network(self, input_dim: int, reset: bool) -> None:
        if not reset and self.n_features_in_ is not None:
            if self.n_features_in_ != input_dim:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {input_dim}."
                )
            return

        self.n_features_in_ = input_dim
        current_dim = input_dim

        for layer in self.layers:
            current_dim = layer.build(current_dim, self._rng, self.dtype)

        self.n_outputs_ = current_dim

    def _validate_output_configuration(self, y: FloatArray) -> None:
        if self.n_outputs_ is None:
            raise RuntimeError("The network output dimension is not initialized yet.")

        output_units = self.n_outputs_
        target_units = y.shape[1]

        if output_units != target_units:
            raise ValueError(
                f"Output layer has {output_units} units but the targets require {target_units}."
            )

        output_layer = self.layers[-1]
        if not isinstance(output_layer, DenseLayer):
            raise ValueError("The output layer must be a DenseLayer.")

        output_activation = output_layer.activation

        if self.loss_name == "binary_crossentropy" and output_activation != "sigmoid":
            raise ValueError("binary_crossentropy requires a sigmoid output layer.")

        if (
            self.loss_name == "categorical_crossentropy"
            and output_activation != "softmax"
        ):
            raise ValueError("categorical_crossentropy requires a softmax output layer.")

    def _forward(self, X: FloatArray, training: bool) -> FloatArray:
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def _train_on_batch(self, y_true: FloatArray, y_pred: FloatArray) -> None:
        delta, is_preactivation_delta = self._output_delta(y_true, y_pred)
        self._optimizer.begin_step()

        for index in range(len(self.layers) - 1, -1, -1):
            apply_activation_derivative = not (
                index == len(self.layers) - 1 and is_preactivation_delta
            )

            delta = self.layers[index].backward(
                delta,
                optimizer=self._optimizer,
                l2_lambda=self.l2_lambda,
                apply_activation_derivative=apply_activation_derivative,
            )

    def _output_delta(self, y_true: FloatArray, y_pred: FloatArray) -> tuple[FloatArray, bool]:
        batch_size = y_true.shape[0]

        if self.loss_name == "binary_crossentropy":
            return (y_pred - y_true) / batch_size, True

        if self.loss_name == "categorical_crossentropy":
            return (y_pred - y_true) / batch_size, True

        diff = y_pred - y_true
        scale = 2.0 / (y_true.shape[0] * y_true.shape[1])
        return scale * diff, False

    def _loss(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        if self.loss_name == "mse":
            return float(np.mean(np.square(y_pred - y_true)))

        clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)

        if self.loss_name == "binary_crossentropy":
            loss = -(
                y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)
            )
            return float(np.mean(loss))

        loss = -np.sum(y_true * np.log(clipped), axis=1)
        return float(np.mean(loss))

    def _metric(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        if self.problem_type == "regression":
            return float(np.mean(np.abs(y_pred - y_true)))

        if self.problem_type == "binary":
            predicted = (y_pred.reshape(-1) >= 0.5).astype(np.int8, copy=False)
            truth = (y_true.reshape(-1) >= 0.5).astype(np.int8, copy=False)
            return float(np.mean(predicted == truth))

        predicted = np.argmax(y_pred, axis=1)
        truth = np.argmax(y_true, axis=1)
        return float(np.mean(predicted == truth))

    def _format_regression_output(self, outputs: FloatArray) -> FloatArray:
        if outputs.shape[1] == 1:
            return outputs.reshape(-1)
        return outputs

    def _copy_trainable_parameters(self) -> list[dict[str, FloatArray]]:
        parameters: list[dict[str, FloatArray]] = []
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                parameters.append(
                    {
                        "weights": layer.weights.copy(),
                        "bias": layer.bias.copy(),
                    }
                )
        return parameters

    def _restore_trainable_parameters(
        self,
        parameters: list[dict[str, FloatArray]],
    ) -> None:
        parameter_index = 0
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.weights_ = parameters[parameter_index]["weights"].copy()
                layer.bias_ = parameters[parameter_index]["bias"].copy()
                parameter_index += 1

    @staticmethod
    def _layer_config(layer: Layer) -> dict[str, Any]:
        if hasattr(layer, "get_config"):
            return layer.get_config()
        raise TypeError(f"Layer {type(layer).__name__!r} is not serializable.")

    @staticmethod
    def _layer_from_config(config: dict[str, Any]) -> Layer:
        layer_type = config["type"]
        if layer_type == "DenseLayer":
            return DenseLayer(
                units=int(config["units"]),
                activation=config["activation"],
            )
        if layer_type == "DropoutLayer":
            return DropoutLayer(rate=float(config["rate"]))
        if layer_type == "FlattenLayer":
            return FlattenLayer()
        if layer_type == "BatchNormLayer":
            return BatchNormLayer(
                momentum=float(config["momentum"]),
                epsilon=float(config["epsilon"]),
            )
        raise ValueError(f"Unsupported layer type: {layer_type!r}.")
