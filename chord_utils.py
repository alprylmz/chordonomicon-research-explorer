from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PC_NAME = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
REL_NAME = {0: "0", 1: "♭2", 2: "2", 3: "♭3", 4: "3", 5: "4", 6: "♭5", 7: "5", 8: "♭6", 9: "6", 10: "♭7", 11: "7"}

SECTION_RE = re.compile(r"^<([^>]+)>$")
SECTION_NUMBER_RE = re.compile(r"_(\d+)$")
CHORD_RE = re.compile(r"^([A-G])([#bs]?)([^/]*)?(?:/([A-G])([#bs]?))?$")
QUOTED_GENRE_RE = re.compile(r"'([^']+)'")


def _accidental_offset(acc: str) -> int:
    if acc in ("#", "s"):
        return 1
    if acc == "b":
        return -1
    return 0


def normalize_suffix(suffix: str) -> str:
    s = (suffix or "").strip()
    low = s.lower()
    replacements = {
        "": "",
        "maj": "",
        "major": "",
        "min": "m",
        "minor": "m",
        "min7": "m7",
        "minor7": "m7",
        "maj7": "maj7",
        "major7": "maj7",
        "min9": "m9",
        "minor9": "m9",
        "maj9": "maj9",
        "major9": "maj9",
    }
    if low in replacements:
        return replacements[low]
    # Chordonomicon commonly uses forms such as dim, aug, sus2, add9, 7, 9, 6.
    return low.replace("min", "m").replace("major", "maj")


@dataclass(frozen=True)
class Chord:
    root: int
    suffix: str = ""
    bass: Optional[int] = None

    @property
    def absolute_token(self) -> str:
        out = f"{PC_NAME[self.root]}{self.suffix}"
        if self.bass is not None:
            out += f"/{PC_NAME[self.bass]}"
        return out

    def transpose(self, semitones: int) -> "Chord":
        return Chord(
            root=(self.root + semitones) % 12,
            suffix=self.suffix,
            bass=((self.bass + semitones) % 12 if self.bass is not None else None),
        )


def parse_chord(token: str) -> Optional[Chord]:
    token = token.strip().strip(",;|")
    if not token or token.upper() in {"N", "NC", "X", "?"}:
        return None
    m = CHORD_RE.match(token)
    if not m:
        return None
    letter, accidental, suffix, bass_letter, bass_acc = m.groups()
    root = (ROOT_PC[letter] + _accidental_offset(accidental)) % 12
    bass = None
    if bass_letter:
        bass = (ROOT_PC[bass_letter] + _accidental_offset(bass_acc or "")) % 12
    return Chord(root=root, suffix=normalize_suffix(suffix or ""), bass=bass)


def parse_progression(text: str) -> List[Chord]:
    tokens = [x for x in re.split(r"[\s,–—-]+", text.strip()) if x]
    out: List[Chord] = []
    for token in tokens:
        chord = parse_chord(token)
        if chord is None:
            raise ValueError(f"Could not parse chord token: {token}")
        out.append(chord)
    return out


def section_type(label: str) -> str:
    label = label.strip().lower()
    return SECTION_NUMBER_RE.sub("", label)


def parse_sections(chord_text: str) -> List[Tuple[str, str, List[Chord]]]:
    """Return (section_type, section_label, parsed_chords)."""
    tokens = (chord_text or "").split()
    sections: List[Tuple[str, str, List[Chord]]] = []
    current_label = "unknown_1"
    current: List[Chord] = []

    def flush() -> None:
        nonlocal current
        if current:
            sections.append((section_type(current_label), current_label, current))
            current = []

    for raw in tokens:
        tag = SECTION_RE.match(raw)
        if tag:
            flush()
            current_label = tag.group(1)
            continue
        chord = parse_chord(raw)
        if chord is not None:
            # Keep the dataset's sequence, but collapse accidental immediate duplicates
            # because they do not add progression information for this explorer.
            if not current or chord != current[-1]:
                current.append(chord)
    flush()
    return sections


def parse_genres(raw: str) -> List[str]:
    if not raw:
        return []
    quoted = QUOTED_GENRE_RE.findall(raw)
    if quoted:
        return [x.strip().lower() for x in quoted if x.strip()]
    parts = re.split(r"[,;|]", raw)
    return [x.strip().strip("'\"").lower() for x in parts if x.strip()]


def relative_shape(chords: Sequence[Chord]) -> str:
    if not chords:
        return ""
    anchor = chords[0].root
    return " ".join(relative_token(ch, anchor) for ch in chords)


def relative_token(chord: Chord, anchor_root: int) -> str:
    root_delta = (chord.root - anchor_root) % 12
    token = f"{root_delta}:{chord.suffix or 'maj'}"
    if chord.bass is not None:
        bass_delta = (chord.bass - anchor_root) % 12
        token += f"/{bass_delta}"
    return token


def pretty_relative_token(token: str) -> str:
    head, *bass_parts = token.split("/")
    root_str, suffix = head.split(":", 1)
    root = int(root_str) % 12
    suffix_display = "" if suffix == "maj" else suffix
    out = f"{REL_NAME[root]}{suffix_display}"
    if bass_parts:
        out += f"/{REL_NAME[int(bass_parts[0]) % 12]}"
    return out


def pretty_relative_shape(shape: str) -> str:
    return " – ".join(pretty_relative_token(x) for x in shape.split() if x)


def absolute_progression(chords: Sequence[Chord]) -> str:
    return " ".join(ch.absolute_token for ch in chords)


def relative_to_absolute(token: str, anchor_root: int) -> Chord:
    head, *bass_parts = token.split("/")
    root_str, suffix = head.split(":", 1)
    root = (anchor_root + int(root_str)) % 12
    bass = None
    if bass_parts:
        bass = (anchor_root + int(bass_parts[0])) % 12
    return Chord(root=root, suffix=("" if suffix == "maj" else suffix), bass=bass)


def shape_from_absolute_text(text: str) -> Tuple[str, str]:
    chords = parse_progression(text)
    return absolute_progression(chords), relative_shape(chords)
