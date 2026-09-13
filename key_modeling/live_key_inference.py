from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from chord_utils import Chord, PC_NAME
from key_inference import roman_numeral
from key_modeling.calibration_utils import probabilities_from_candidate_scores
from key_modeling.ml_features import candidate_from_index, feature_matrix_for_progression


DEFAULT_MODEL_PATH = Path("data/key_models/mode_balance_experiment/selected_mode_balanced_histgb.joblib")
DEFAULT_CALIBRATOR_PATH = Path("data/key_models/mode_balance_experiment/mode_balanced_temperature_calibrator.json")


@dataclass(frozen=True)
class KeyCandidate:
    tonic: int
    mode: str
    name: str
    probability: float


@dataclass(frozen=True)
class LiveKeyAnalysis:
    candidates: tuple[KeyCandidate, ...]
    roman_numerals: tuple[str, ...]
    probability_margin: float
    normalized_entropy: float
    n_chords: int

    @property
    def top(self) -> KeyCandidate:
        return self.candidates[0]

    @property
    def second(self) -> KeyCandidate | None:
        return self.candidates[1] if len(self.candidates) > 1 else None

    @property
    def roman_progression(self) -> str:
        return " – ".join(self.roman_numerals)


class KeyPredictor:
    """Small runtime wrapper around the frozen Step-6 balanced HistGB model.

    Candidate ranking is produced by the supervised 24-key ranker. The model's
    binary candidate probabilities are transformed with the frozen ChoCo-fitted
    temperature and then normalized over all 24 candidate keys.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        calibrator_path: str | Path = DEFAULT_CALIBRATOR_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.calibrator_path = Path(calibrator_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        if not self.calibrator_path.exists():
            raise FileNotFoundError(self.calibrator_path)

        self.model = joblib.load(self.model_path)
        cal = json.loads(self.calibrator_path.read_text(encoding="utf-8"))
        self.temperature = float(cal["temperature"])

    def analyze(self, chords: Sequence[Chord], *, top_k: int = 3) -> LiveKeyAnalysis:
        if not chords:
            raise ValueError("Need at least one chord")

        rows = feature_matrix_for_progression(chords)
        X = np.asarray([row.values for row in rows], dtype=np.float32)
        raw = np.asarray(self.model.predict_proba(X)[:, 1], dtype=float).reshape(1, 24)
        probs = probabilities_from_candidate_scores(raw, temperature=self.temperature)[0]
        order = np.argsort(-probs)

        candidates: list[KeyCandidate] = []
        for idx in order[: max(2, int(top_k))]:
            tonic, mode = candidate_from_index(int(idx))
            candidates.append(
                KeyCandidate(
                    tonic=int(tonic),
                    mode=str(mode),
                    name=f"{PC_NAME[int(tonic)]} {mode}",
                    probability=float(probs[int(idx)]),
                )
            )

        top_idx = int(order[0])
        second_idx = int(order[1])
        top_tonic, top_mode = candidate_from_index(top_idx)
        romans = tuple(roman_numeral(chord, int(top_tonic), str(top_mode)) for chord in chords)

        p = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(p * np.log(p)) / math.log(len(p)))

        return LiveKeyAnalysis(
            candidates=tuple(candidates),
            roman_numerals=romans,
            probability_margin=float(probs[top_idx] - probs[second_idx]),
            normalized_entropy=entropy,
            n_chords=len(chords),
        )


def artifact_stamp(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrator_path: str | Path = DEFAULT_CALIBRATOR_PATH,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    model_path = Path(model_path)
    calibrator_path = Path(calibrator_path)
    if not model_path.exists() or not calibrator_path.exists():
        return None
    m = model_path.stat()
    c = calibrator_path.stat()
    return ((m.st_size, m.st_mtime_ns), (c.st_size, c.st_mtime_ns))
