from __future__ import annotations

import unittest

import numpy as np

from _bootstrap import FROM_SCRATCH  # noqa: F401
from neural_core import DenseLayer, NeuralNetwork


class NeuralNetworkTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
