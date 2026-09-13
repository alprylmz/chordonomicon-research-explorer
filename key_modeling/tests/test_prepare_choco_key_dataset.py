import unittest

from key_modeling.prepare_choco_key_dataset import (
    parse_harte, parse_key, time_scalar, choose_chord_annotation, choose_key_annotations,
    extract_timed_chords, extract_keys, segment_examples, stable_split,
)


class TestChoCoPreparation(unittest.TestCase):
    def test_key_modes(self):
        self.assertEqual(parse_key("C:major")[:2], (0, "major"))
        self.assertEqual(parse_key("A:minor")[:2], (9, "minor"))
        self.assertEqual(parse_key("Bb:major")[:2], (10, "major"))
        self.assertEqual(parse_key("A:aeolian")[:2], (9, "minor"))
        self.assertIsNone(parse_key("D:dorian"))
        self.assertIsNone(parse_key("C"))

    def test_harte(self):
        self.assertEqual(parse_harte("F:min/5").token, "Fm")
        self.assertEqual(parse_harte("C#:maj7/3").token, "C#maj7")
        self.assertEqual(parse_harte("Bb:7").token, "A#7")
        self.assertIsNone(parse_harte("N"))

    def test_measure_beat(self):
        self.assertEqual(time_scalar("12:3"), 12003.0)
        self.assertEqual(time_scalar({"measure": 2, "beat": 1.5}), 2001.5)

    def test_annotation_selection_and_segmentation(self):
        anns = [
            {"namespace": "chord", "sandbox": {"notation": "Harte"}, "data": [
                {"time": 0.0, "duration": 1.0, "value": "C:maj"},
                {"time": 1.0, "duration": 1.0, "value": "G:maj"},
                {"time": 2.0, "duration": 1.0, "value": "A:min"},
                {"time": 3.0, "duration": 1.0, "value": "F:maj"},
                {"time": 4.0, "duration": 1.0, "value": "A:min"},
                {"time": 5.0, "duration": 1.0, "value": "E:7"},
                {"time": 6.0, "duration": 1.0, "value": "A:min"},
                {"time": 7.0, "duration": 1.0, "value": "D:min"},
            ]},
            {"namespace": "key_mode", "data": [
                {"time": 0.0, "duration": 4.0, "value": "C:major"},
                {"time": 4.0, "duration": 4.0, "value": "A:minor"},
            ]},
        ]
        ca = choose_chord_annotation(anns)
        ka = choose_key_annotations(anns)
        chords = extract_timed_chords(ca)
        keys = extract_keys(ka)
        segs, method = segment_examples(chords, keys)
        self.assertEqual(method, "timed-local")
        self.assertEqual([x[0].label for x in segs], ["C major", "A minor"])
        self.assertEqual([[c.token for c in x[1]] for x in segs], [["C", "G", "Am", "F"], ["Am", "E7", "Am", "Dm"]])

    def test_split_is_piece_stable(self):
        self.assertEqual(stable_split("piece123"), stable_split("piece123"))
        self.assertIn(stable_split("piece123"), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
