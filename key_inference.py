from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple

from chord_utils import Chord, PC_NAME


MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

MAJOR_QUALITY = {
    0: "major",
    2: "minor",
    4: "minor",
    5: "major",
    7: "major",
    9: "minor",
    11: "diminished",
}
MINOR_QUALITY = {
    0: "minor",
    2: "diminished",
    3: "major",
    5: "minor",
    7: "minor",  # Natural-minor v; major V is handled as harmonic-minor evidence.
    8: "major",
    10: "major",
}

# Conventional scale-degree labels used only after a candidate key has been proposed.
MAJOR_ROMAN = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}
MINOR_ROMAN = {0: "i", 2: "ii°", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}
GENERIC_DEGREE = {
    0: "1",
    1: "♭2",
    2: "2",
    3: "♭3",
    4: "3",
    5: "4",
    6: "♭5",
    7: "5",
    8: "♭6",
    9: "6",
    10: "♭7",
    11: "7",
}


@dataclass(frozen=True)
class KeyResult:
    tonic: int
    mode: str
    name: str
    score: float
    heuristic_share: float
    roman_numerals: Tuple[str, ...]
    components: Dict[str, float]
    evidence: Tuple[str, ...]

    @property
    def roman_progression(self) -> str:
        return " – ".join(self.roman_numerals)


def quality_family(suffix: str) -> str:
    """Collapse Chordonomicon suffixes into broad triadic quality families."""
    s = (suffix or "").lower().strip()
    if s in {"dim", "dim7", "o", "°"} or s.startswith("dim"):
        return "diminished"
    if "m7b5" in s or "ø" in s or "half" in s:
        return "half_diminished"
    if s.startswith("aug") or "+" in s:
        return "augmented"
    if s.startswith("sus"):
        return "suspended"
    # Dominant extensions are major-triad chords with a minor seventh.
    if s in {"7", "9", "11", "13", "7b9", "7#9", "7b5", "7#5"}:
        return "dominant"
    if s.startswith("maj") or s in {"6", "add9", "add2", "69", ""}:
        return "major"
    if s.startswith("m") and not s.startswith("maj"):
        return "minor"
    return "other"


def _quality_match(expected: str, actual: str, degree: int, mode: str) -> float:
    """Return a 0..1 compatibility score for broad chord quality."""
    if expected == actual:
        return 1.0
    if expected == "major" and actual == "dominant":
        # A dominant seventh is a strong fit on V and a plausible color elsewhere.
        return 1.0 if degree == 7 else 0.70
    if expected == "major" and actual == "suspended":
        return 0.65
    if expected == "minor" and actual == "suspended":
        return 0.55
    if expected == "diminished" and actual == "half_diminished":
        return 0.90
    if expected == "diminished" and actual == "minor":
        return 0.25
    if actual == "other":
        return 0.35
    return 0.0


def _is_tonic_quality(family: str, mode: str) -> bool:
    return family == ("major" if mode == "major" else "minor")


def _is_dominant_quality(family: str, mode: str) -> bool:
    if mode == "major":
        return family in {"major", "dominant", "suspended"}
    # In minor, major/dominant V is stronger tonal evidence than natural-minor v.
    return family in {"major", "dominant", "minor", "suspended"}


def _borrowed_fit(degree: int, family: str, mode: str) -> float:
    """Moderate credit for common mixture/modal-borrowing sonorities.

    This prevents one colorful chord from destroying an otherwise plausible key.
    It intentionally gives less credit than a diatonic chord.
    """
    if mode == "major":
        common = {
            (5, "minor"),       # iv
            (10, "major"),     # bVII
            (8, "major"),      # bVI
            (3, "major"),      # bIII
            (1, "major"),      # bII / Neapolitan color
        }
    else:
        common = {
            (7, "major"),      # harmonic-minor V
            (7, "dominant"),   # V7
            (11, "diminished"),
            (11, "half_diminished"),
            (5, "major"),      # Dorian IV
            (1, "major"),      # bII / Phrygian color
        }
    return 0.62 if (degree, family) in common else 0.0


def _candidate_components(chords: Sequence[Chord], tonic: int, mode: str) -> Tuple[Dict[str, float], List[str]]:
    n = max(1, len(chords))
    scale = set(MAJOR_SCALE if mode == "major" else MINOR_SCALE)
    expected_map = MAJOR_QUALITY if mode == "major" else MINOR_QUALITY

    root_fit_total = 0.0
    quality_fit_total = 0.0
    borrowed_total = 0.0
    tonic_hits = 0
    dominant_hits = 0
    chromatic_hits = 0
    evidence: List[str] = []

    rel_degrees: List[int] = []
    families: List[str] = []

    for chord in chords:
        degree = (chord.root - tonic) % 12
        family = quality_family(chord.suffix)
        rel_degrees.append(degree)
        families.append(family)

        if degree in scale:
            root_fit_total += 1.0
            quality_fit_total += _quality_match(expected_map[degree], family, degree, mode)
        else:
            borrowed = _borrowed_fit(degree, family, mode)
            if borrowed:
                borrowed_total += borrowed
                # A borrowed chord still provides partial pitch-class/key compatibility.
                root_fit_total += 0.45
                quality_fit_total += 0.45
            else:
                chromatic_hits += 1

        if degree == 0 and _is_tonic_quality(family, mode):
            tonic_hits += 1
        if degree == 7 and _is_dominant_quality(family, mode):
            dominant_hits += 1

    diatonicity = root_fit_total / n
    quality_match = quality_fit_total / n
    tonic_presence = min(1.0, tonic_hits / max(1.0, n / 3.0))
    dominant_presence = min(1.0, dominant_hits / max(1.0, n / 4.0))
    borrowed_color = borrowed_total / n
    chromatic_penalty = chromatic_hits / n

    # Cadential evidence. Multiple cadences help, but diminishing returns avoid
    # making a repeated two-chord loop completely dominate the score.
    cadence_raw = 0.0
    modal_cadence_raw = 0.0
    for i in range(len(chords) - 1):
        d1, d2 = rel_degrees[i], rel_degrees[i + 1]
        f1, f2 = families[i], families[i + 1]

        if d2 == 0 and _is_tonic_quality(f2, mode):
            if d1 == 7 and f1 in {"major", "dominant", "suspended"}:
                cadence_raw += 1.0 if mode == "major" else 1.18
            elif d1 == 11 and f1 in {"diminished", "half_diminished"}:
                cadence_raw += 0.82
            elif d1 == 5:
                cadence_raw += 0.34  # plagal motion
            elif mode == "minor" and d1 == 10 and f1 == "major":
                modal_cadence_raw += 0.36  # bVII -> i modal resolution

        # Predominant -> dominant -> tonic patterns.
        if i + 2 < len(chords):
            d3 = rel_degrees[i + 2]
            f3 = families[i + 2]
            if d2 == 7 and d3 == 0 and _is_tonic_quality(f3, mode):
                if mode == "major" and d1 in {2, 5}:
                    cadence_raw += 0.40
                if mode == "minor" and d1 in {2, 5, 8}:
                    cadence_raw += 0.40

    cadence = 1.0 - math.exp(-cadence_raw)
    modal_cadence = 1.0 - math.exp(-modal_cadence_raw)

    # Boundary evidence is intentionally modest: starting on a chord does not
    # prove that chord is tonic, but it is useful when relative major/minor keys tie.
    boundary = 0.0
    if chords:
        first_d = rel_degrees[0]
        first_f = families[0]
        last_d = rel_degrees[-1]
        last_f = families[-1]
        if first_d == 0 and _is_tonic_quality(first_f, mode):
            boundary += 0.42
        if last_d == 0 and _is_tonic_quality(last_f, mode):
            boundary += 0.58
        elif last_d == 7 and _is_dominant_quality(last_f, mode):
            boundary += 0.22  # plausible half-cadential ending
        boundary = min(1.0, boundary)

    # Characteristic mode evidence separates relative major/minor when possible.
    characteristic = 0.0
    if mode == "minor":
        for degree, family in zip(rel_degrees, families):
            if degree == 7 and family in {"major", "dominant"}:
                characteristic += 0.70  # harmonic-minor dominant
            if degree == 11 and family in {"diminished", "half_diminished"}:
                characteristic += 0.55  # raised leading-tone chord
        characteristic = min(1.0, characteristic)
    else:
        # Major-mode dominant and leading-tone sonorities provide mild evidence.
        for degree, family in zip(rel_degrees, families):
            if degree == 7 and family in {"major", "dominant"}:
                characteristic += 0.22
            if degree == 11 and family in {"diminished", "half_diminished"}:
                characteristic += 0.28
        characteristic = min(1.0, characteristic)

    components = {
        "diatonicity": diatonicity,
        "quality_match": quality_match,
        "tonic_presence": tonic_presence,
        "dominant_presence": dominant_presence,
        "cadence": cadence,
        "modal_cadence": modal_cadence,
        "boundary": boundary,
        "characteristic": characteristic,
        "borrowed_color": borrowed_color,
        "chromatic_penalty": chromatic_penalty,
    }

    exact_diatonic = sum(1 for d in rel_degrees if d in scale)
    evidence.append(f"{exact_diatonic}/{len(chords)} chord roots are diatonic to the candidate scale")
    if tonic_hits:
        evidence.append(f"tonic-quality chord appears {tonic_hits} time{'s' if tonic_hits != 1 else ''}")
    if dominant_hits:
        evidence.append(f"dominant-root chord appears {dominant_hits} time{'s' if dominant_hits != 1 else ''}")
    if cadence_raw > 0:
        evidence.append("contains tonal cadence evidence toward the candidate tonic")
    if mode == "minor" and characteristic > 0.2:
        evidence.append("contains harmonic-minor/leading-tone evidence")
    if borrowed_total > 0:
        evidence.append("contains common borrowed/modal-mixture color")
    if chromatic_hits:
        evidence.append(f"{chromatic_hits} chord root{'s are' if chromatic_hits != 1 else ' is'} outside the candidate scale")

    return components, evidence


def _score_components(c: Dict[str, float]) -> float:
    # Explicit heuristic baseline. These weights are deliberately interpretable
    # and will later be evaluated/calibrated on ChoCo rather than treated as truth.
    return (
        3.10 * c["diatonicity"]
        + 1.65 * c["quality_match"]
        + 0.95 * c["tonic_presence"]
        + 0.55 * c["dominant_presence"]
        + 2.20 * c["cadence"]
        + 0.45 * c["modal_cadence"]
        + 0.70 * c["boundary"]
        + 0.95 * c["characteristic"]
        + 0.35 * c["borrowed_color"]
        - 1.20 * c["chromatic_penalty"]
    )


def _extension_marker(suffix: str) -> str:
    s = (suffix or "").lower()
    if "maj7" in s:
        return "maj7"
    if "m7b5" in s or "ø" in s:
        return "7"
    if "dim7" in s:
        return "7"
    if s == "7" or s.startswith("7"):
        return "7"
    if s in {"9", "11", "13"}:
        return s
    if s.startswith("m7"):
        return "7"
    return ""


def roman_numeral(chord: Chord, tonic: int, mode: str) -> str:
    degree = (chord.root - tonic) % 12
    family = quality_family(chord.suffix)

    # Canonical diatonic spelling when quality agrees with the key.
    if mode == "major" and degree in MAJOR_ROMAN:
        expected = MAJOR_QUALITY[degree]
        if _quality_match(expected, family, degree, mode) >= 0.65:
            base = MAJOR_ROMAN[degree]
            if degree == 7 and family == "dominant":
                base = "V"
            ext = _extension_marker(chord.suffix)
            return base + ext

    if mode == "minor":
        # Harmonic-minor dominant and leading-tone chord.
        if degree == 7 and family in {"major", "dominant"}:
            return "V" + _extension_marker(chord.suffix)
        if degree == 11 and family in {"diminished", "half_diminished"}:
            return "vii°" + _extension_marker(chord.suffix)
        if degree in MINOR_ROMAN:
            expected = MINOR_QUALITY[degree]
            if _quality_match(expected, family, degree, mode) >= 0.55:
                return MINOR_ROMAN[degree] + _extension_marker(chord.suffix)

    # Generic chromatic degree label. Case reflects broad chord quality.
    degree_label = GENERIC_DEGREE[degree]
    numeral_core = {
        "1": "I",
        "♭2": "♭II",
        "2": "II",
        "♭3": "♭III",
        "3": "III",
        "4": "IV",
        "♭5": "♭V",
        "5": "V",
        "♭6": "♭VI",
        "6": "VI",
        "♭7": "♭VII",
        "7": "VII",
    }[degree_label]
    if family in {"minor", "half_diminished"}:
        numeral_core = numeral_core.lower()
    if family in {"diminished", "half_diminished"}:
        numeral_core += "°" if "°" not in numeral_core else ""
    if family == "augmented":
        numeral_core += "+"
    return numeral_core + _extension_marker(chord.suffix)


def _softmax(scores: Sequence[float], temperature: float = 0.72) -> List[float]:
    if not scores:
        return []
    t = max(0.05, float(temperature))
    m = max(scores)
    exps = [math.exp((s - m) / t) for s in scores]
    total = sum(exps)
    return [x / total for x in exps]


def infer_keys(
    chords: Sequence[Chord],
    *,
    top_k: int = 5,
    temperature: float = 0.72,
) -> List[KeyResult]:
    """Rank 24 major/minor candidate keys using an explainable theory baseline.

    `heuristic_share` is a normalized score, NOT a calibrated probability. The
    baseline is intentionally transparent so it can later be validated against
    ChoCo and compared with supervised ML models.
    """
    if not chords:
        return []

    raw: List[KeyResult] = []
    for tonic in range(12):
        for mode in ("major", "minor"):
            components, evidence = _candidate_components(chords, tonic, mode)
            score = _score_components(components)
            romans = tuple(roman_numeral(ch, tonic, mode) for ch in chords)
            raw.append(
                KeyResult(
                    tonic=tonic,
                    mode=mode,
                    name=f"{PC_NAME[tonic]} {mode}",
                    score=score,
                    heuristic_share=0.0,
                    roman_numerals=romans,
                    components=components,
                    evidence=tuple(evidence),
                )
            )

    shares = _softmax([r.score for r in raw], temperature=temperature)
    ranked = [replace(r, heuristic_share=p) for r, p in zip(raw, shares)]
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[: max(1, int(top_k))]


def inference_summary(chords: Sequence[Chord], top_k: int = 3) -> Dict[str, object]:
    ranked = infer_keys(chords, top_k=max(2, top_k))
    if not ranked:
        return {"candidates": [], "ambiguous": True, "separation": 0.0}
    share_separation = ranked[0].heuristic_share - ranked[1].heuristic_share if len(ranked) > 1 else ranked[0].heuristic_share
    score_gap = ranked[0].score - ranked[1].score if len(ranked) > 1 else ranked[0].score
    # This is intentionally a heuristic ambiguity flag, not a statistical CI.
    # Raw score gaps are more honest here than softmax shares because the shares
    # are not calibrated yet. ChoCo validation will replace this rule later.
    ambiguous = score_gap < 1.35
    return {
        "candidates": ranked[:top_k],
        "ambiguous": ambiguous,
        "separation": share_separation,
        "score_gap": score_gap,
    }
