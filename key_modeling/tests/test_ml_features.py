from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chord_utils import parse_progression
from key_modeling.ml_features import FEATURE_NAMES, candidate_feature_values, candidate_index, candidate_from_index


class MLFeatureTests(unittest.TestCase):
    def test_candidate_index_roundtrip(self):
        for tonic in range(12):
            for mode in ("major", "minor"):
                i = candidate_index(tonic, mode)
                self.assertEqual((tonic, mode), candidate_from_index(i))

    def test_feature_length(self):
        chords = parse_progression("G D Em C")
        vals = candidate_feature_values(chords, 7, "major")
        self.assertEqual(len(FEATURE_NAMES), len(vals))

    def test_transposition_invariance(self):
        # C G Am F in C major and D A Bm G in D major are the same harmonic shape.
        c = parse_progression("C G Am F")
        d = parse_progression("D A Bm G")
        fc = candidate_feature_values(c, 0, "major")
        fd = candidate_feature_values(d, 2, "major")
        for a, b in zip(fc, fd):
            self.assertAlmostEqual(a, b, places=7)

    def test_minor_dominant_feature(self):
        chords = parse_progression("Am Dm E7 Am")
        vals = dict(zip(FEATURE_NAMES, candidate_feature_values(chords, 9, "minor")))
        self.assertGreater(vals["transition_7_to_0_rate"], 0)
        self.assertGreater(vals["dominant_strong_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
