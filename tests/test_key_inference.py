import unittest

from chord_utils import parse_progression
from key_inference import infer_keys


class KeyInferenceBaselineTests(unittest.TestCase):
    def top(self, text):
        return infer_keys(parse_progression(text), top_k=3)

    def test_clear_g_major(self):
        ranked = self.top("G D Em C G")
        self.assertEqual(ranked[0].name, "G major")

    def test_clear_c_major_cadence(self):
        ranked = self.top("C Am Dm G7 C")
        self.assertEqual(ranked[0].name, "C major")

    def test_clear_a_minor_harmonic_dominant(self):
        ranked = self.top("Am Dm E7 Am")
        self.assertEqual(ranked[0].name, "A minor")

    def test_clear_e_minor_harmonic_dominant(self):
        ranked = self.top("Em Am B7 Em")
        self.assertEqual(ranked[0].name, "E minor")

    def test_relative_major_minor_example_stays_uncertain(self):
        ranked = self.top("Am F C G")
        names = {r.name for r in ranked}
        self.assertIn("C major", names)
        # A minor remains a plausible high-ranking candidate even though the
        # hand-built baseline currently prefers C major for this sequence.
        all_ranked = infer_keys(parse_progression("Am F C G"), top_k=6)
        self.assertIn("A minor", {r.name for r in all_ranked})

    def test_roman_numerals(self):
        ranked = self.top("G D Em C G")
        self.assertEqual(ranked[0].roman_progression, "I – V – vi – IV – I")


if __name__ == "__main__":
    unittest.main()
