from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ai_lab.neural_core import DenseLayer, DropoutLayer, NeuralNetwork


class NeuralNetworkTests(unittest.TestCase):
    def test_invalid_optimizer_name_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            NeuralNetwork(
                layers=[DenseLayer(1, "identity")],
                problem_type="regression",
                optimizer="not-an-optimizer",
            )

    def test_binary_network_solves_xor(self) -> None:
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 1, 1, 0])

        model = NeuralNetwork(
            layers=[DenseLayer(8, "tanh"), DenseLayer(1, "sigmoid")],
            problem_type="binary",
            learning_rate=0.2,
            max_epochs=4000,
            batch_size=4,
            random_state=21,
        )
        model.fit(X, y)

        self.assertGreaterEqual(model.score(X, y), 1.0)
        self.assertLess(model.history_[-1].loss, model.history_[0].loss)

    def test_regression_network_fits_linear_mapping(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)

        model = NeuralNetwork(
            layers=[DenseLayer(1, "identity")],
            problem_type="regression",
            learning_rate=0.05,
            max_epochs=1500,
            batch_size=4,
            random_state=3,
        )
        model.fit(X, y)

        self.assertGreater(model.score(X, y), 0.999)

    def test_multiclass_network_handles_integer_labels(self) -> None:
        X = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
                [-1.0, 0.0],
                [-0.8, -0.2],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1, 2, 2])

        model = NeuralNetwork(
            layers=[DenseLayer(6, "tanh"), DenseLayer(3, "softmax")],
            problem_type="multiclass",
            learning_rate=0.1,
            max_epochs=2500,
            batch_size=6,
            random_state=5,
        )
        model.fit(X, y)

        self.assertGreaterEqual(model.score(X, y), 1.0)
        self.assertEqual(model.predict_proba(X).shape, (6, 3))

    def test_dropout_layer_cannot_be_used_as_output_layer(self) -> None:
        with self.assertRaises(ValueError):
            NeuralNetwork(
                layers=[DenseLayer(4, "tanh"), DropoutLayer(0.2)],
                problem_type="binary",
            )

    def test_network_supports_dropout_hidden_layer(self) -> None:
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.15, 0.10],
                [0.20, 0.85],
                [0.85, 0.15],
                [0.90, 0.90],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 1, 0, 0, 0, 1])

        model = NeuralNetwork(
            layers=[
                DenseLayer(12, "tanh"),
                DropoutLayer(0.25),
                DenseLayer(1, "sigmoid"),
            ],
            problem_type="binary",
            learning_rate=0.03,
            optimizer="adam",
            max_epochs=1500,
            batch_size=4,
            random_state=11,
        )
        model.fit(X, y)

        self.assertGreaterEqual(model.score(X, y), 1.0)
        self.assertLess(model.history_[-1].loss, model.history_[0].loss)

    def test_supported_optimizers_fit_simple_regression(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)

        optimizer_configs = {
            "sgd": {"learning_rate": 0.05, "max_epochs": 1500},
            "momentum": {"learning_rate": 0.03, "max_epochs": 1000},
            "rmsprop": {"learning_rate": 0.01, "max_epochs": 1200},
            "adadelta": {"learning_rate": 1.0, "max_epochs": 1800},
            "adam": {"learning_rate": 0.05, "max_epochs": 600},
            "adamw": {
                "learning_rate": 0.05,
                "max_epochs": 700,
                "optimizer_kwargs": {"weight_decay": 0.01},
            },
            "adamax": {"learning_rate": 0.05, "max_epochs": 700},
        }

        for optimizer_name, config in optimizer_configs.items():
            with self.subTest(optimizer=optimizer_name):
                model = NeuralNetwork(
                    layers=[DenseLayer(1, "identity")],
                    problem_type="regression",
                    learning_rate=config["learning_rate"],
                    optimizer=optimizer_name,
                    optimizer_kwargs=config.get("optimizer_kwargs"),
                    max_epochs=config["max_epochs"],
                    batch_size=4,
                    random_state=3,
                )
                model.fit(X, y)

                self.assertGreater(model.score(X, y), 0.999)
                self.assertLess(model.history_[-1].loss, model.history_[0].loss)

    def test_predict_proba_is_not_available_for_regression(self) -> None:
        model = NeuralNetwork(
            layers=[DenseLayer(1, "identity")],
            problem_type="regression",
            random_state=0,
        )
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)
        model.fit(X, y)
        with self.assertRaises(RuntimeError):
            model.predict_proba(X)

    def test_validation_split_tracks_validation_metrics(self) -> None:
        X = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(-1, 1)
        y = (2.0 * X.reshape(-1) + 1.0).astype(np.float32)

        model = NeuralNetwork(
            layers=[DenseLayer(1, "identity")],
            problem_type="regression",
            learning_rate=0.05,
            max_epochs=5,
            batch_size=4,
            validation_split=0.25,
            random_state=4,
        )
        model.fit(X, y)

        self.assertIsNotNone(model.history_[-1].val_loss)
        self.assertIsNotNone(model.history_[-1].val_metric)

    def test_early_stopping_stops_when_monitored_loss_does_not_improve_enough(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)

        model = NeuralNetwork(
            layers=[DenseLayer(1, "identity")],
            problem_type="regression",
            learning_rate=0.01,
            max_epochs=50,
            batch_size=4,
            early_stopping=True,
            patience=2,
            min_delta=1_000_000.0,
            random_state=3,
        )
        model.fit(X, y)

        self.assertEqual(model.epochs_trained_, 3)

    def test_save_and_load_preserve_predictions(self) -> None:
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 1, 1, 0])

        model = NeuralNetwork(
            layers=[DenseLayer(8, "tanh"), DenseLayer(1, "sigmoid")],
            problem_type="binary",
            learning_rate=0.2,
            max_epochs=1000,
            batch_size=4,
            random_state=21,
        )
        model.fit(X, y)

        path = Path("artifacts") / "test-models" / "xor_model.npz"
        model.save(path)
        loaded = NeuralNetwork.load(path)

        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
        np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X))


if __name__ == "__main__":
    unittest.main()
