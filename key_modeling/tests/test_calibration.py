import unittest
import numpy as np

from key_modeling.calibration_utils import (
    multiclass_nll,
    probabilities_from_candidate_scores,
    top1_ece,
)


class CalibrationTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        scores = np.array([[0.8, 0.2, 0.1], [0.4, 0.5, 0.3]], dtype=float)
        p = probabilities_from_candidate_scores(scores, temperature=1.7)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_temperature_preserves_ranking(self):
        scores = np.array([[0.7, 0.6, 0.2, 0.1]], dtype=float)
        p1 = probabilities_from_candidate_scores(scores, temperature=0.5)
        p2 = probabilities_from_candidate_scores(scores, temperature=3.0)
        self.assertEqual(list(np.argsort(-p1[0])), list(np.argsort(-p2[0])))
        self.assertEqual(list(np.argsort(-p1[0])), list(np.argsort(-scores[0])))

    def test_nll_prefers_more_probability_on_truth(self):
        y = np.array([0, 1])
        good = np.array([[0.9, 0.1], [0.1, 0.9]])
        bad = np.array([[0.6, 0.4], [0.4, 0.6]])
        self.assertLess(multiclass_nll(good, y), multiclass_nll(bad, y))

    def test_ece_is_nonnegative(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        ece, rows = top1_ece(p, y, n_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
