from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, FROM_SCRATCH  # noqa: F401
from neural_core import Neuron
from perceptron import Perceptron
from visualization import plot_binary_model_comparison, plot_training_history


class VisualizationTests(unittest.TestCase):
    output_dir = ROOT / "artifacts" / "test-plots"

    def setUp(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.y = np.array([0, 0, 0, 1])

    def test_training_history_plot_is_saved(self) -> None:
        model = Neuron(
            problem_type="binary",
            learning_rate=0.4,
            max_epochs=50,
            batch_size=4,
            random_state=7,
        )
        model.fit(self.X, self.y)

        output = self.output_dir / "history.png"
        fig = plot_training_history(model.history_, save_path=output)
        self.assertTrue(output.exists())
        fig.clf()

    def test_binary_model_comparison_plot_is_saved(self) -> None:
        perceptron = Perceptron(learning_rate=0.2, max_epochs=50, random_state=7)
        neuron = Neuron(
            problem_type="binary",
            learning_rate=0.4,
            max_epochs=500,
            batch_size=4,
            random_state=7,
        )
        perceptron.fit(self.X, self.y)
        neuron.fit(self.X, self.y)

        output = self.output_dir / "comparison.png"
        fig = plot_binary_model_comparison(
            self.X,
            self.y,
            perceptron,
            neuron,
            save_path=output,
        )
        self.assertTrue(output.exists())
        fig.clf()


if __name__ == "__main__":
    unittest.main()
