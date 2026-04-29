from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: NDArray[np.int_]
    classes: NDArray[Any]


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    truth = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    _validate_label_shapes(truth, predicted)
    return float(np.mean(truth == predicted))


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: ArrayLike | None = None,
) -> NDArray[np.int_]:
    truth = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    _validate_label_shapes(truth, predicted)

    classes = (
        np.asarray(labels)
        if labels is not None
        else np.unique(np.concatenate((truth, predicted)))
    )
    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    matrix = np.zeros((classes.size, classes.size), dtype=np.int_)

    for actual, guess in zip(truth.tolist(), predicted.tolist(), strict=True):
        if actual not in class_to_index or guess not in class_to_index:
            raise ValueError("labels must include every class present in y_true and y_pred.")
        matrix[class_to_index[actual], class_to_index[guess]] += 1

    return matrix


def precision_recall_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    true_positive = np.diag(matrix)
    predicted_positive = np.sum(matrix, axis=0)
    actual_positive = np.sum(matrix, axis=1)

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive != 0.0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive),
        where=actual_positive != 0.0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0.0,
    )
    return precision, recall, f1


def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: ArrayLike | None = None,
) -> ClassificationReport:
    truth = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    classes = (
        np.asarray(labels)
        if labels is not None
        else np.unique(np.concatenate((truth, predicted)))
    )
    precision, recall, f1 = precision_recall_f1(truth, predicted, labels=classes)

    return ClassificationReport(
        accuracy=accuracy_score(truth, predicted),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        confusion_matrix=confusion_matrix(truth, predicted, labels=classes),
        classes=classes,
    )


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    residual = float(np.sum(np.square(truth - predicted)))
    centered = truth - np.mean(truth)
    total = float(np.sum(np.square(centered)))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return float(1.0 - residual / total)


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(np.abs(truth - predicted)))


def _validate_label_shapes(y_true: NDArray[Any], y_pred: NDArray[Any]) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
