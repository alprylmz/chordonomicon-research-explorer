from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import joblib
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install ML dependencies with: python -m pip install -r key_modeling/requirements_ml.txt") from exc

from chord_utils import PC_NAME, parse_progression
from key_modeling.calibration_utils import probabilities_from_candidate_scores
from key_modeling.ml_features import candidate_from_index, feature_matrix_for_progression


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank 24 keys using the Step-3 model, with Step-4 calibrated probabilities when available.")
    ap.add_argument("progression", help='Example: "Am F C G"')
    ap.add_argument("--model", default="data/key_models/selected_model.joblib")
    ap.add_argument("--calibrator", default="data/key_models/temperature_calibrator.json")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    chords = parse_progression(args.progression)
    rows = feature_matrix_for_progression(chords)
    X = np.asarray([r.values for r in rows], dtype=np.float32)
    model = joblib.load(args.model)
    raw = np.asarray(model.predict_proba(X)[:, 1], dtype=float).reshape(1, 24)

    calibration_path = Path(args.calibrator)
    if calibration_path.exists():
        cal = json.loads(calibration_path.read_text(encoding="utf-8"))
        temperature = float(cal["temperature"])
        probs = probabilities_from_candidate_scores(raw, temperature=temperature)[0]
        probability_label = "calibrated_probability"
        note = "Probabilities use Step-4 out-of-fold temperature calibration."
    else:
        clipped = np.clip(raw[0], 1e-12, None)
        probs = clipped / clipped.sum()
        probability_label = "normalized_display_share"
        note = "No Step-4 calibrator found; shares are ranking displays, not calibrated probabilities."

    raw1 = raw[0]
    order = np.argsort(-raw1)
    out = []
    for idx in order[: max(1, args.top_k)]:
        tonic, mode = candidate_from_index(int(idx))
        row = {
            "key": f"{PC_NAME[tonic]} {mode}",
            "ranking_score": float(raw1[idx]),
            probability_label: float(probs[idx]),
        }
        out.append(row)

    print(json.dumps({
        "progression": args.progression,
        "candidates": out,
        "note": note,
    }, indent=2))


if __name__ == "__main__":
    main()
