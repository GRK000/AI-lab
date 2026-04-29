from __future__ import annotations

import unittest

import numpy as np
from _bootstrap import FROM_SCRATCH  # noqa: F401
from neural_core import classification_report, confusion_matrix, r2_score


class MetricsTests(unittest.TestCase):
    def test_confusion_matrix_counts_predictions(self) -> None:
        matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
        np.testing.assert_array_equal(matrix, np.array([[1, 1], [0, 2]]))

    def test_classification_report_computes_macro_scores(self) -> None:
        report = classification_report([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
        self.assertEqual(report.accuracy, 0.75)
        self.assertGreater(report.macro_f1, 0.7)

    def test_r2_score_matches_perfect_regression(self) -> None:
        self.assertEqual(r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
