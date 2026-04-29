from __future__ import annotations

import unittest

import numpy as np
from _bootstrap import FROM_SCRATCH  # noqa: F401
from neural_core.layers import DenseLayer, DropoutLayer


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

    def test_backward_matches_numerical_gradient_for_identity_layer(self) -> None:
        layer = DenseLayer(units=2, activation="identity")
        layer.build(input_dim=3, rng=np.random.default_rng(4), dtype=np.float32)

        X = np.array([[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]], dtype=np.float32)
        upstream = np.array([[0.7, -0.3], [0.2, 0.5]], dtype=np.float32)

        initial_weights = layer.weights.copy()
        initial_bias = layer.bias.copy()

        layer.forward(X, training=True)
        layer.backward(upstream, learning_rate=1.0)

        analytical_weight_grad = initial_weights - layer.weights
        analytical_bias_grad = initial_bias - layer.bias

        epsilon = 1e-4
        numerical_weight_grad = np.zeros_like(initial_weights)
        numerical_bias_grad = np.zeros_like(initial_bias)

        def scalar_loss(weights: np.ndarray, bias: np.ndarray) -> float:
            layer.weights_ = weights.astype(np.float32, copy=True)
            layer.bias_ = bias.astype(np.float32, copy=True)
            output = layer.forward(X, training=False)
            return float(np.sum(output * upstream))

        for row in range(initial_weights.shape[0]):
            for col in range(initial_weights.shape[1]):
                plus = initial_weights.copy()
                minus = initial_weights.copy()
                plus[row, col] += epsilon
                minus[row, col] -= epsilon
                numerical_weight_grad[row, col] = (
                    scalar_loss(plus, initial_bias) - scalar_loss(minus, initial_bias)
                ) / (2.0 * epsilon)

        for col in range(initial_bias.shape[1]):
            plus = initial_bias.copy()
            minus = initial_bias.copy()
            plus[0, col] += epsilon
            minus[0, col] -= epsilon
            numerical_bias_grad[0, col] = (
                scalar_loss(initial_weights, plus) - scalar_loss(initial_weights, minus)
            ) / (2.0 * epsilon)

        np.testing.assert_allclose(
            analytical_weight_grad,
            numerical_weight_grad,
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            analytical_bias_grad,
            numerical_bias_grad,
            rtol=1e-3,
            atol=1e-3,
        )


class DropoutLayerTests(unittest.TestCase):
    def test_dropout_training_and_inference_follow_inverted_dropout_contract(self) -> None:
        layer = DropoutLayer(rate=0.5)
        layer.build(input_dim=4, rng=np.random.default_rng(0), dtype=np.float32)

        X = np.ones((4, 4), dtype=np.float32)
        training_output = layer.forward(X, training=True)
        backward_output = layer.backward(np.ones_like(X))
        inference_output = layer.forward(X, training=False)

        self.assertTrue(np.any(training_output == 0.0))
        np.testing.assert_array_equal(training_output, backward_output)
        np.testing.assert_allclose(inference_output, X)

    def test_dropout_rate_must_be_in_valid_range(self) -> None:
        with self.assertRaises(ValueError):
            DropoutLayer(rate=1.0)


if __name__ == "__main__":
    unittest.main()
