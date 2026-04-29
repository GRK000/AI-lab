from __future__ import annotations

import unittest

import numpy as np
from _bootstrap import FROM_SCRATCH  # noqa: F401
from neural_core import (
    DenseLayer,
    EarlyStopping,
    HistoryLogger,
    LearningRateScheduler,
    NeuralNetwork,
)


class CallbackTests(unittest.TestCase):
    def test_history_logger_records_epoch_logs(self) -> None:
        logger = HistoryLogger()
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
        model = NeuralNetwork(
            layers=[DenseLayer(1, "identity")],
            problem_type="regression",
            max_epochs=3,
            batch_size=4,
            callbacks=[logger],
            random_state=0,
        )

        model.fit(X, y)

        self.assertEqual(len(logger.records), 3)
        self.assertIn("loss", logger.records[-1])

    def test_callback_early_stopping_can_stop_training(self) -> None:
        stopper = EarlyStopping(patience=2, min_delta=1_000_000.0)
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
        model = NeuralNetwork(
            layers=[DenseLayer(1, "identity")],
            problem_type="regression",
            learning_rate=0.01,
            max_epochs=50,
            batch_size=4,
            callbacks=[stopper],
            random_state=3,
        )

        model.fit(X, y)

        self.assertEqual(model.epochs_trained_, 3)

    def test_learning_rate_scheduler_updates_optimizer(self) -> None:
        scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.5)
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)
        model = NeuralNetwork(
            layers=[DenseLayer(1, "identity")],
            problem_type="regression",
            learning_rate=0.1,
            max_epochs=2,
            batch_size=2,
            callbacks=[scheduler],
            random_state=0,
        )

        model.fit(X, y)

        self.assertAlmostEqual(model.learning_rate, 0.025)
        self.assertAlmostEqual(model._optimizer.learning_rate, 0.025)


if __name__ == "__main__":
    unittest.main()
