from __future__ import annotations

import unittest

import numpy as np

from _bootstrap import FROM_SCRATCH  # noqa: F401
from neural_core.layers import DenseLayer


class DenseLayerTests(unittest.TestCase):
    def test_build_creates_weight_and_bias_shapes(self) -> None:
        layer = DenseLayer(units=3, activation="relu")
        layer.build(input_dim=2, rng=np.random.default_rng(0), dtype=np.float32)
        self.assertEqual(layer.weights.shape, (2, 3))
        self.assertEqual(layer.bias.shape, (1, 3))

    def test_forward_returns_expected_shape(self) -> None:
        layer = DenseLayer(units=4, activation="tanh")
        layer.build(input_dim=2, rng=np.random.default_rng(1), dtype=np.float32)
        X = np.array([[0.2, 0.4], [0.5, 0.8]], dtype=np.float32)
        output = layer.forward(X, training=True)
        self.assertEqual(output.shape, (2, 4))

    def test_backward_updates_parameters_and_returns_input_gradient(self) -> None:
        layer = DenseLayer(units=2, activation="relu")
        layer.build(input_dim=3, rng=np.random.default_rng(2), dtype=np.float32)
        X = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float32)
        layer.forward(X, training=True)
        previous_weights = layer.weights.copy()
        delta = np.ones((2, 2), dtype=np.float32) * 0.5
        grad_input = layer.backward(delta, learning_rate=0.1)
        self.assertEqual(grad_input.shape, (2, 3))
        self.assertFalse(np.allclose(previous_weights, layer.weights))


if __name__ == "__main__":
    unittest.main()
