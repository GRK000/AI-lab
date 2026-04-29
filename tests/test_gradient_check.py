from __future__ import annotations

import unittest

import numpy as np
from _bootstrap import FROM_SCRATCH  # noqa: F401
from neural_core import DenseLayer, check_dense_layer_gradients


class GradientCheckTests(unittest.TestCase):
    def test_dense_layer_gradient_check_passes_for_identity_layer(self) -> None:
        layer = DenseLayer(units=2, activation="identity")
        layer.build(input_dim=3, rng=np.random.default_rng(4), dtype=np.float32)
        X = np.array([[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]], dtype=np.float32)
        upstream = np.array([[0.7, -0.3], [0.2, 0.5]], dtype=np.float32)

        result = check_dense_layer_gradients(layer, X, upstream)

        self.assertTrue(result.passed)
        self.assertEqual(result.checked_parameters, 8)


if __name__ == "__main__":
    unittest.main()
