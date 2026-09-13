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
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import GroupKFold
except ImportError as exc:
    raise SystemExit("Install ML dependencies with: python -m pip install -r key_modeling/requirements_ml.txt") from exc

from key_modeling.calibration_utils import (
    coverage_accuracy,
    multiclass_nll,
    probabilities_from_candidate_scores,
    probability_metrics,
)
from key_modeling.train_key_models import binary_rows, fit_model

SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def load_metadata(path: str) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def score_examples(model, X3_subset: np.ndarray) -> np.ndarray:
    n = X3_subset.shape[0]
    flat = X3_subset.reshape(n * 24, X3_subset.shape[2])
    raw = model.predict_proba(flat)[:, 1]
    return np.asarray(raw, dtype=float).reshape(n, 24)


def fit_temperature(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    def objective(log_t: float) -> float:
        t = float(np.exp(log_t))
        probs = probabilities_from_candidate_scores(scores, temperature=t)
        return multiclass_nll(probs, y_true)

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-5})
    temperature = float(np.exp(result.x))
    return temperature, {
        "success": bool(result.success),
        "objective_nll": float(result.fun),
        "nfev": int(result.nfev),
        "log_temperature": float(result.x),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4: out-of-fold temperature calibration for the selected 24-key candidate ranker.")
    ap.add_argument("--features", default="data/key_ml_features.npz")
    ap.add_argument("--metadata", default="data/key_ml_examples.jsonl")
    ap.add_argument("--step3-report", default="data/key_models/step3_model_report.json")
    ap.add_argument("--model", default="data/key_models/selected_model.joblib", help="Final train+validation selected model from Step 3")
    ap.add_argument("--outdir", default="data/key_models")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    d = np.load(args.features)
    X3 = d["X"]
    y = d["y"].astype(int)
    split = d["split"].astype(int)
    metadata = load_metadata(args.metadata)
    if len(metadata) != len(X3):
        raise SystemExit(f"Metadata rows ({len(metadata)}) != examples ({len(X3)})")

    step3 = json.loads(Path(args.step3_report).read_text(encoding="utf-8"))
    model_name = step3["selected_model"]
    if model_name not in {"logistic", "histgb"}:
        raise SystemExit(f"Unsupported selected model: {model_name}")

    trainval_idx = np.flatnonzero((split == SPLIT_CODE["train"]) | (split == SPLIT_CODE["validation"]))
    test_idx = np.flatnonzero(split == SPLIT_CODE["test"])
    groups = np.asarray([str(metadata[i].get("piece_id", i)) for i in trainval_idx], dtype=object)

    unique_groups = len(set(groups.tolist()))
    folds = min(args.folds, unique_groups)
    if folds < 2:
        raise SystemExit("Need at least two unique piece groups for OOF calibration")

    print(f"Selected model family: {model_name}")
    print(f"OOF calibration examples: {len(trainval_idx):,}; unique pieces: {unique_groups:,}; folds: {folds}")

    Xtv = X3[trainval_idx]
    ytv = y[trainval_idx]
    oof_scores = np.zeros((len(trainval_idx), 24), dtype=np.float64)
    oof_seen = np.zeros(len(trainval_idx), dtype=bool)

    gkf = GroupKFold(n_splits=folds)
    for fold_i, (tr_local, va_local) in enumerate(gkf.split(np.zeros(len(trainval_idx)), ytv, groups), start=1):
        print(f"Fold {fold_i}/{folds}: train={len(tr_local):,}, holdout={len(va_local):,}")
        fold_mask = np.zeros(len(trainval_idx), dtype=bool)
        fold_mask[tr_local] = True
        X_train, y_train = binary_rows(Xtv, ytv, fold_mask)
        model = fit_model(model_name, X_train, y_train, args.seed + fold_i)
        oof_scores[va_local] = score_examples(model, Xtv[va_local])
        oof_seen[va_local] = True

    if not np.all(oof_seen):
        raise RuntimeError("Some train+validation examples did not receive OOF predictions")

    temperature, fit_info = fit_temperature(oof_scores, ytv)
    oof_uncal = probabilities_from_candidate_scores(oof_scores, temperature=1.0)
    oof_cal = probabilities_from_candidate_scores(oof_scores, temperature=temperature)

    final_model = joblib.load(args.model)
    test_scores = score_examples(final_model, X3[test_idx])
    test_uncal = probabilities_from_candidate_scores(test_scores, temperature=1.0)
    test_cal = probabilities_from_candidate_scores(test_scores, temperature=temperature)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    calibrator = {
        "method": "temperature_scaling",
        "score_transform": "binary_candidate_probability -> logit -> divide by temperature -> softmax over 24 keys",
        "temperature": temperature,
        "model_family": model_name,
        "fit_data": "5-fold (or requested folds) group-out-of-fold predictions over original train+validation; grouped by piece_id",
        "folds": folds,
        "seed": args.seed,
        "ranking_preserved": True,
        "note": "A single positive temperature rescales confidence without changing the candidate ordering.",
    }
    (outdir / "temperature_calibrator.json").write_text(json.dumps(calibrator, indent=2), encoding="utf-8")

    report = {
        "selected_model": model_name,
        "calibration_method": calibrator,
        "fit_optimizer": fit_info,
        "oof_train_plus_validation": {
            "uncalibrated_temperature_1": probability_metrics(oof_uncal, ytv, n_bins=args.bins),
            "calibrated": probability_metrics(oof_cal, ytv, n_bins=args.bins),
        },
        "final_untouched_test": {
            "uncalibrated_temperature_1": probability_metrics(test_uncal, y[test_idx], n_bins=args.bins),
            "calibrated": probability_metrics(test_cal, y[test_idx], n_bins=args.bins),
            "coverage_accuracy": coverage_accuracy(test_cal, y[test_idx]),
        },
        "important_note": "Top-1/top-k ranking accuracy should be unchanged by temperature scaling. Step 4 evaluates probability quality (NLL, Brier, ECE) on the untouched test set.",
    }
    (outdir / "step4_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nCALIBRATION SUMMARY")
    print(json.dumps({
        "temperature": temperature,
        "oof_uncalibrated": {k: report["oof_train_plus_validation"]["uncalibrated_temperature_1"][k] for k in ("nll", "brier", "ece", "mean_top1_confidence")},
        "oof_calibrated": {k: report["oof_train_plus_validation"]["calibrated"][k] for k in ("nll", "brier", "ece", "mean_top1_confidence")},
        "test_uncalibrated": {k: report["final_untouched_test"]["uncalibrated_temperature_1"][k] for k in ("top1_accuracy", "nll", "brier", "ece", "mean_top1_confidence")},
        "test_calibrated": {k: report["final_untouched_test"]["calibrated"][k] for k in ("top1_accuracy", "nll", "brier", "ece", "mean_top1_confidence")},
    }, indent=2))
    print(f"Wrote {outdir / 'temperature_calibrator.json'}")
    print(f"Wrote {outdir / 'step4_calibration_report.json'}")


if __name__ == "__main__":
    main()
