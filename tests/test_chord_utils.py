import unittest

from chord_utils import (
    absolute_progression,
    parse_chord,
    parse_genres,
    parse_progression,
    parse_sections,
    pretty_relative_shape,
    relative_shape,
)


class ChordUtilsTests(unittest.TestCase):
    def test_chordonomicon_sharp_notation(self):
        chord = parse_chord("Csmin")
        self.assertIsNotNone(chord)
        self.assertEqual(chord.absolute_token, "C#m")

    def test_slash_chord(self):
        chord = parse_chord("A/Cs")
        self.assertEqual(chord.absolute_token, "A/C#")

    def test_relative_transposition_equivalence(self):
        a = parse_progression("Am F C G")
        b = parse_progression("Em C G D")
        self.assertEqual(relative_shape(a), relative_shape(b))
        self.assertEqual(pretty_relative_shape(relative_shape(a)), "0m – ♭6 – ♭3 – ♭7")

    def test_sections(self):
        text = "<intro_1> E D A/Cs <verse_1> E D A/Cs <chorus_1> G A E"
        sections = parse_sections(text)
        self.assertEqual([s[0] for s in sections], ["intro", "verse", "chorus"])
        self.assertEqual(absolute_progression(sections[0][2]), "E D A/C#")

    def test_genres(self):
        raw = "'alternative metal' 'alternative rock' 'nu metal'"
        self.assertEqual(parse_genres(raw), ["alternative metal", "alternative rock", "nu metal"])


if __name__ == "__main__":
    unittest.main()
