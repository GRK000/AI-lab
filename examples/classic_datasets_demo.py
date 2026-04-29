from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from ai_lab.neural_core import DenseLayer, NeuralNetwork, classification_report, r2_score


def main() -> None:
    try:
        from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as error:
        raise SystemExit(
            "This demo requires scikit-learn. Install it with:\n"
            'python -m pip install -e ".[examples]"'
        ) from error

    _run_iris(load_iris, train_test_split, StandardScaler)
    _run_breast_cancer(load_breast_cancer, train_test_split, StandardScaler)
    _run_diabetes(load_diabetes, train_test_split, StandardScaler)


def _run_iris(load_iris: object, train_test_split: object, scaler_cls: object) -> None:
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )
    scaler = scaler_cls()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    model = NeuralNetwork(
        layers=[DenseLayer(12, "tanh"), DenseLayer(3, "softmax")],
        problem_type="multiclass",
        learning_rate=0.03,
        optimizer="adam",
        max_epochs=300,
        batch_size=16,
        validation_split=0.2,
        random_state=7,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print("\nIris")
    print("Accuracy:", round(report.accuracy, 4))
    print("Macro F1:", round(report.macro_f1, 4))


def _run_breast_cancer(
    load_breast_cancer: object,
    train_test_split: object,
    scaler_cls: object,
) -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )
    scaler = scaler_cls()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    model = NeuralNetwork(
        layers=[DenseLayer(16, "relu"), DenseLayer(1, "sigmoid")],
        problem_type="binary",
        learning_rate=0.005,
        optimizer="adam",
        max_epochs=250,
        batch_size=32,
        validation_split=0.2,
        random_state=9,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print("\nBreast cancer")
    print("Accuracy:", round(report.accuracy, 4))
    print("Macro F1:", round(report.macro_f1, 4))


def _run_diabetes(load_diabetes: object, train_test_split: object, scaler_cls: object) -> None:
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
    )
    scaler = scaler_cls()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    model = NeuralNetwork(
        layers=[DenseLayer(16, "relu"), DenseLayer(1, "identity")],
        problem_type="regression",
        learning_rate=0.001,
        optimizer="adam",
        max_epochs=500,
        batch_size=32,
        validation_split=0.2,
        random_state=11,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nDiabetes regression")
    print("R2:", round(r2_score(y_test, predictions), 4))


if __name__ == "__main__":
    main()
