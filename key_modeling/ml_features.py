from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from chord_utils import Chord, parse_chord
from key_inference import _candidate_components, _score_components, quality_family

CANDIDATES: list[tuple[int, str]] = [
    (tonic, mode)
    for tonic in range(12)
    for mode in ("major", "minor")
]

THEORY_COMPONENT_NAMES = [
    "diatonicity",
    "quality_match",
    "tonic_presence",
    "dominant_presence",
    "cadence",
    "modal_cadence",
    "boundary",
    "characteristic",
    "borrowed_color",
    "chromatic_penalty",
]

FEATURE_NAMES = (
    THEORY_COMPONENT_NAMES
    + ["theory_score", "candidate_is_minor", "log1p_n_chords", "unique_root_fraction"]
    + [f"root_fraction_deg_{d}" for d in range(12)]
    + [f"first_degree_{d}" for d in range(12)]
    + [f"last_degree_{d}" for d in range(12)]
    + [
        "tonic_quality_fraction",
        "dominant_strong_fraction",
        "relative_partner_root_fraction",
        "transition_7_to_0_rate",
        "transition_5_to_0_rate",
        "transition_11_to_0_rate",
        "transition_2_to_7_rate",
        "transition_8_to_7_rate",
        "transition_10_to_0_rate",
        "transition_0_to_7_rate",
        "transition_to_tonic_rate",
        "transition_from_tonic_rate",
    ]
)


@dataclass(frozen=True)
class CandidateFeatureRow:
    tonic: int
    mode: str
    values: Tuple[float, ...]


def parse_dataset_progression(text: str) -> list[Chord]:
    """Parse Step-2 whitespace-delimited chord strings safely.

    ChoCo/Harte interval-set qualities can contain commas inside one chord token,
    so this must not use the generic human-input progression splitter.
    """
    out: list[Chord] = []
    for raw in (text or "").split():
        chord = parse_chord(raw.strip())
        if chord is None:
            raise ValueError(f"Could not parse chord token: {raw}")
        out.append(chord)
    if not out:
        raise ValueError("No chords found")
    return out


def candidate_index(tonic: int, mode: str) -> int:
    return int(tonic) * 2 + (1 if mode == "minor" else 0)


def candidate_from_index(index: int) -> tuple[int, str]:
    index = int(index)
    if index < 0 or index >= 24:
        raise ValueError(f"candidate index out of range: {index}")
    return index // 2, "minor" if index % 2 else "major"


def _transition_rates(rel_degrees: Sequence[int]) -> dict[str, float]:
    if len(rel_degrees) < 2:
        denom = 1.0
        pairs: list[tuple[int, int]] = []
    else:
        denom = float(len(rel_degrees) - 1)
        pairs = list(zip(rel_degrees[:-1], rel_degrees[1:]))

    def rate(a: int | None = None, b: int | None = None) -> float:
        n = 0
        for x, y in pairs:
            if a is not None and x != a:
                continue
            if b is not None and y != b:
                continue
            n += 1
        return n / denom

    return {
        "transition_7_to_0_rate": rate(7, 0),
        "transition_5_to_0_rate": rate(5, 0),
        "transition_11_to_0_rate": rate(11, 0),
        "transition_2_to_7_rate": rate(2, 7),
        "transition_8_to_7_rate": rate(8, 7),
        "transition_10_to_0_rate": rate(10, 0),
        "transition_0_to_7_rate": rate(0, 7),
        "transition_to_tonic_rate": rate(None, 0),
        "transition_from_tonic_rate": rate(0, None),
    }


def candidate_feature_values(chords: Sequence[Chord], tonic: int, mode: str) -> tuple[float, ...]:
    if not chords:
        raise ValueError("Need at least one chord")

    components, _ = _candidate_components(chords, tonic, mode)
    theory_score = _score_components(components)

    n = float(len(chords))
    rel_degrees = [(ch.root - tonic) % 12 for ch in chords]
    families = [quality_family(ch.suffix) for ch in chords]

    root_counts = [0] * 12
    for d in rel_degrees:
        root_counts[d] += 1
    root_fracs = [x / n for x in root_counts]

    first_onehot = [0.0] * 12
    last_onehot = [0.0] * 12
    first_onehot[rel_degrees[0]] = 1.0
    last_onehot[rel_degrees[-1]] = 1.0

    tonic_family = "minor" if mode == "minor" else "major"
    tonic_quality_hits = sum(
        1 for d, fam in zip(rel_degrees, families)
        if d == 0 and fam == tonic_family
    )
    dominant_strong_hits = sum(
        1 for d, fam in zip(rel_degrees, families)
        if d == 7 and fam in {"major", "dominant", "suspended"}
    )
    relative_partner_degree = 3 if mode == "minor" else 9

    tr = _transition_rates(rel_degrees)

    values: list[float] = []
    values.extend(float(components[name]) for name in THEORY_COMPONENT_NAMES)
    values.extend([
        float(theory_score),
        1.0 if mode == "minor" else 0.0,
        math.log1p(n),
        len(set(ch.root for ch in chords)) / 12.0,
    ])
    values.extend(root_fracs)
    values.extend(first_onehot)
    values.extend(last_onehot)
    values.extend([
        tonic_quality_hits / n,
        dominant_strong_hits / n,
        root_fracs[relative_partner_degree],
        tr["transition_7_to_0_rate"],
        tr["transition_5_to_0_rate"],
        tr["transition_11_to_0_rate"],
        tr["transition_2_to_7_rate"],
        tr["transition_8_to_7_rate"],
        tr["transition_10_to_0_rate"],
        tr["transition_0_to_7_rate"],
        tr["transition_to_tonic_rate"],
        tr["transition_from_tonic_rate"],
    ])

    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(f"feature length mismatch: {len(values)} != {len(FEATURE_NAMES)}")
    return tuple(values)


def feature_matrix_for_progression(chords: Sequence[Chord]) -> list[CandidateFeatureRow]:
    return [
        CandidateFeatureRow(tonic, mode, candidate_feature_values(chords, tonic, mode))
        for tonic, mode in CANDIDATES
    ]
