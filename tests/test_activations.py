from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose

from _bootstrap import FROM_SCRATCH  # noqa: F401
from neural_core.activations import ACTIVATIONS


class ActivationTests(unittest.TestCase):
    def test_identity_forward_and_derivative(self) -> None:
        X = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        output = ACTIVATIONS["identity"].forward(X)
        derivative = ACTIVATIONS["identity"].derivative(output, X)
        assert_allclose(output, X)
        assert_allclose(derivative, np.ones_like(X))

    def test_sigmoid_forward_and_derivative(self) -> None:
        X = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)
        output = ACTIVATIONS["sigmoid"].forward(X)
        derivative = ACTIVATIONS["sigmoid"].derivative(output, X)
        expected = 1.0 / (1.0 + np.exp(-X))
        assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
        assert_allclose(derivative, output * (1.0 - output), rtol=1e-6, atol=1e-6)

    def test_tanh_derivative_matches_formula(self) -> None:
        X = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        output = ACTIVATIONS["tanh"].forward(X)
        derivative = ACTIVATIONS["tanh"].derivative(output, X)
        assert_allclose(derivative, 1.0 - np.square(output), rtol=1e-6, atol=1e-6)

    def test_relu_derivative(self) -> None:
        X = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
        output = ACTIVATIONS["relu"].forward(X)
        derivative = ACTIVATIONS["relu"].derivative(output, X)
        assert_allclose(output, np.array([[0.0, 0.0, 2.0]], dtype=np.float32))
        assert_allclose(derivative, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

    def test_softmax_rows_sum_to_one(self) -> None:
        X = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
        output = ACTIVATIONS["softmax"].forward(X)
        assert_allclose(output.sum(axis=1), np.ones(2, dtype=np.float32), atol=1e-6)

    def test_softmax_derivative_requires_crossentropy_shortcut(self) -> None:
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        output = ACTIVATIONS["softmax"].forward(X)
        with self.assertRaises(ValueError):
            ACTIVATIONS["softmax"].derivative(output, X)


if __name__ == "__main__":
    unittest.main()
