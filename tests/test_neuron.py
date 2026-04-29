from __future__ import annotations

import unittest

import numpy as np
from _bootstrap import FROM_SCRATCH  # noqa: F401
from neural_core import Neuron


class NeuronTests(unittest.TestCase):
    def test_binary_neuron_solves_or_problem(self) -> None:
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

        model = Neuron(
            problem_type="binary",
            learning_rate=0.4,
            max_epochs=1500,
            batch_size=4,
            random_state=7,
        )
        model.fit(X, y)

        self.assertGreaterEqual(model.score(X, y), 1.0)
        self.assertEqual(model.predict_proba(X).shape, (4, 2))

    def test_regression_neuron_fits_simple_line(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)

        model = Neuron(
            problem_type="regression",
            learning_rate=0.05,
            max_epochs=1500,
            batch_size=4,
            random_state=3,
        )
        model.fit(X, y)

        self.assertGreater(model.score(X, y), 0.999)
        self.assertEqual(model.weights_.shape, (1,))


if __name__ == "__main__":
    unittest.main()
