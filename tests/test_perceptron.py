from __future__ import annotations

import unittest

import numpy as np

from ai_lab.perceptron import Perceptron


class PerceptronTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.y_or = np.array([0, 0, 0, 1])

    def test_fit_solves_or_problem(self) -> None:
        model = Perceptron(learning_rate=0.2, max_epochs=50, random_state=7)
        model.fit(self.X, self.y_or)
        self.assertGreaterEqual(model.score(self.X, self.y_or), 1.0)
        self.assertGreater(len(model.history_), 0)

    def test_partial_fit_handles_incremental_training(self) -> None:
        model = Perceptron(learning_rate=0.2, max_epochs=10, random_state=7)
        model.partial_fit(self.X[:2], self.y_or[:2], classes=[0, 1])
        model.partial_fit(self.X[2:], self.y_or[2:])
        self.assertEqual(model.n_features_in_, 2)
        self.assertEqual(model.classes_.shape[0], 2)

    def test_margin_matches_number_of_samples(self) -> None:
        model = Perceptron(learning_rate=0.2, max_epochs=50, random_state=7)
        model.fit(self.X, self.y_or)
        margin = model.margin(self.X)
        self.assertEqual(margin.shape, (4,))


if __name__ == "__main__":
    unittest.main()
