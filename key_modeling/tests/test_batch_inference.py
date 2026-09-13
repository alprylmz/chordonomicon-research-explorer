from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from key_modeling.infer_chordonomicon_keys import _flatten_chart, _normalized_entropy, _summaries_from_probabilities


class BatchInferenceTests(unittest.TestCase):
    def test_chart_sections_are_flattened(self):
        sections, chords = _flatten_chart("<verse_1> G D Em C <chorus_1> G C D G")
        self.assertEqual(len(sections), 2)
        self.assertEqual(len(chords), 8)

    def test_entropy_bounds(self):
        uniform = np.full((1, 24), 1.0 / 24.0)
        peaked = np.zeros((1, 24))
        peaked[0, 0] = 1.0
        self.assertAlmostEqual(float(_normalized_entropy(uniform)[0]), 1.0, places=6)
        self.assertLess(float(_normalized_entropy(peaked)[0]), 1e-6)

    def test_probability_summary(self):
        probs = np.zeros((1, 24), dtype=float)
        probs[0, 14] = 0.7  # G major: tonic 7 -> index 14
        probs[0, 9] = 0.2   # E minor: tonic 4 -> index 9
        probs[0, 0] = 0.1
        rows = _summaries_from_probabilities([1], ["song"], [4], [1], probs)
        self.assertEqual(rows[0][2], "G major")
        self.assertEqual(rows[0][6], "E minor")
        self.assertAlmostEqual(rows[0][5], 0.7)
        self.assertAlmostEqual(rows[0][8], 0.5)


if __name__ == "__main__":
    unittest.main()
