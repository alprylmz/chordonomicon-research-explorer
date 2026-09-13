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
    raise SystemExit(
        "Install ML dependencies with: python -m pip install -r key_modeling/requirements_ml.txt"
    ) from exc

from key_modeling.calibration_utils import (
    coverage_accuracy,
    multiclass_nll,
    probabilities_from_candidate_scores,
    probability_metrics,
)
from key_modeling.train_mode_balanced_histgb import (
    candidate_scores,
    fit_histgb,
    make_binary_training_rows,
    rank_metrics,
)

SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def load_metadata(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fit_temperature(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    def objective(log_t: float) -> float:
        temperature = float(np.exp(log_t))
        probs = probabilities_from_candidate_scores(scores, temperature=temperature)
        return multiclass_nll(probs, y_true)

    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    temperature = float(np.exp(result.x))
    return temperature, {
        "success": bool(result.success),
        "objective_nll": float(result.fun),
        "nfev": int(result.nfev),
        "log_temperature": float(result.x),
    }


def topk_metrics_from_probs(probs: np.ndarray, y_true: np.ndarray) -> dict:
    order = np.argsort(-probs, axis=1)
    ranks = np.empty(len(y_true), dtype=np.int16)
    for i in range(len(y_true)):
        ranks[i] = int(np.where(order[i] == int(y_true[i]))[0][0]) + 1
    return {
        **{f"top{k}_accuracy": float(np.mean(ranks <= k)) for k in (1, 2, 3, 5)},
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def mode_specific_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
) -> dict:
    pred = np.argmax(probs, axis=1)
    true_mode = y_true % 2
    pred_mode = pred % 2
    out: dict[str, dict] = {}

    for mode_idx, label in ((0, "major"), (1, "minor")):
        mask = true_mode == mode_idx
        subset_probs = probs[mask]
        subset_y = y_true[mask]
        pm = probability_metrics(subset_probs, subset_y, n_bins=n_bins)
        pm.update(topk_metrics_from_probs(subset_probs, subset_y))
        pm["count"] = int(np.sum(mask))
        pm["mode_recall"] = float(np.mean(pred_mode[mask] == mode_idx))
        pm["mean_true_key_probability"] = float(
            np.mean(subset_probs[np.arange(len(subset_y)), subset_y])
        )
        pm["coverage_accuracy"] = coverage_accuracy(subset_probs, subset_y)
        out[label] = pm

    out["balanced_mode_accuracy"] = float(
        (out["major"]["mode_recall"] + out["minor"]["mode_recall"]) / 2.0
    )
    out["balanced_true_mode_key_accuracy"] = float(
        (out["major"]["top1_accuracy"] + out["minor"]["top1_accuracy"]) / 2.0
    )
    out["major_minor_key_accuracy_gap"] = float(
        out["major"]["top1_accuracy"] - out["minor"]["top1_accuracy"]
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Calibrate the Step-6 mode-balanced HistGradientBoosting model using "
            "grouped out-of-fold predictions with the SAME true-mode weighting used to train it."
        )
    )
    ap.add_argument("--features", default="data/key_ml_features.npz")
    ap.add_argument("--metadata", default="data/key_ml_examples.jsonl")
    ap.add_argument(
        "--mode-balance-report",
        default="data/key_models/mode_balance_experiment/mode_balance_report.json",
    )
    ap.add_argument(
        "--model",
        default="data/key_models/mode_balance_experiment/selected_mode_balanced_histgb.joblib",
    )
    ap.add_argument(
        "--outdir",
        default="data/key_models/mode_balance_experiment",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    data = np.load(args.features)
    X3 = data["X"]
    y = data["y"].astype(int)
    split = data["split"].astype(int)
    metadata = load_metadata(args.metadata)
    if len(metadata) != len(X3):
        raise SystemExit(f"Metadata rows ({len(metadata)}) != examples ({len(X3)})")

    balance_report = json.loads(Path(args.mode_balance_report).read_text(encoding="utf-8"))
    alpha = float(balance_report["selected"]["alpha"])

    trainval_idx = np.flatnonzero(
        (split == SPLIT_CODE["train"]) | (split == SPLIT_CODE["validation"])
    )
    test_idx = np.flatnonzero(split == SPLIT_CODE["test"])
    Xtv = X3[trainval_idx]
    ytv = y[trainval_idx]
    groups = np.asarray(
        [str(metadata[i].get("piece_id", i)) for i in trainval_idx],
        dtype=object,
    )

    unique_groups = len(set(groups.tolist()))
    folds = min(args.folds, unique_groups)
    if folds < 2:
        raise SystemExit("Need at least two unique piece groups for OOF calibration")

    print(f"Mode-balanced alpha: {alpha:g}")
    print(
        f"OOF calibration examples: {len(trainval_idx):,}; "
        f"unique pieces: {unique_groups:,}; folds: {folds}"
    )

    oof_scores = np.zeros((len(trainval_idx), 24), dtype=np.float64)
    oof_seen = np.zeros(len(trainval_idx), dtype=bool)
    fold_weighting: list[dict] = []

    gkf = GroupKFold(n_splits=folds)
    for fold_i, (tr_local, va_local) in enumerate(
        gkf.split(np.zeros(len(trainval_idx)), ytv, groups),
        start=1,
    ):
        print(f"Fold {fold_i}/{folds}: train={len(tr_local):,}, holdout={len(va_local):,}")
        X_train, y_train, weights, weight_info = make_binary_training_rows(
            Xtv[tr_local], ytv[tr_local], alpha
        )
        model = fit_histgb(X_train, y_train, weights, args.seed + fold_i)
        oof_scores[va_local] = candidate_scores(model, Xtv[va_local])
        oof_seen[va_local] = True
        fold_weighting.append({"fold": fold_i, **weight_info})

    if not np.all(oof_seen):
        raise RuntimeError("Some train+validation examples did not receive OOF predictions")

    temperature, optimizer = fit_temperature(oof_scores, ytv)
    oof_uncal = probabilities_from_candidate_scores(oof_scores, temperature=1.0)
    oof_cal = probabilities_from_candidate_scores(oof_scores, temperature=temperature)

    final_model = joblib.load(args.model)
    test_scores = candidate_scores(final_model, X3[test_idx])
    test_uncal = probabilities_from_candidate_scores(test_scores, temperature=1.0)
    test_cal = probabilities_from_candidate_scores(test_scores, temperature=temperature)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    calibrator = {
        "method": "temperature_scaling",
        "score_transform": "binary_candidate_probability -> logit -> divide by temperature -> softmax over 24 keys",
        "temperature": temperature,
        "model_family": "histgb",
        "mode_balance_alpha": alpha,
        "fit_data": (
            "grouped out-of-fold predictions over original train+validation; "
            "each OOF fold was refit with the same true-mode weighting as the selected Step-6 model"
        ),
        "folds": folds,
        "seed": args.seed,
        "ranking_preserved": True,
        "note": (
            "A single positive temperature rescales confidence without changing candidate ordering. "
            "The ChoCo test split has already been inspected previously, so its metrics here are post-hoc confirmation."
        ),
    }
    calibrator_path = outdir / "mode_balanced_temperature_calibrator.json"
    calibrator_path.write_text(json.dumps(calibrator, indent=2), encoding="utf-8")

    report = {
        "experiment": "temperature calibration of selected mode-balanced HistGB",
        "selected_alpha": alpha,
        "calibration_method": calibrator,
        "fit_optimizer": optimizer,
        "fold_weighting": fold_weighting,
        "oof_train_plus_validation": {
            "uncalibrated": {
                "overall": probability_metrics(oof_uncal, ytv, n_bins=args.bins),
                "by_true_mode": mode_specific_metrics(oof_uncal, ytv, n_bins=args.bins),
            },
            "calibrated": {
                "overall": probability_metrics(oof_cal, ytv, n_bins=args.bins),
                "by_true_mode": mode_specific_metrics(oof_cal, ytv, n_bins=args.bins),
            },
        },
        "posthoc_test": {
            "uncalibrated": {
                "overall": probability_metrics(test_uncal, y[test_idx], n_bins=args.bins),
                "ranking": rank_metrics(test_scores, y[test_idx]),
                "by_true_mode": mode_specific_metrics(test_uncal, y[test_idx], n_bins=args.bins),
            },
            "calibrated": {
                "overall": probability_metrics(test_cal, y[test_idx], n_bins=args.bins),
                "ranking": rank_metrics(test_scores, y[test_idx]),
                "by_true_mode": mode_specific_metrics(test_cal, y[test_idx], n_bins=args.bins),
            },
        },
        "method_note": (
            "Temperature scaling does not change Top-1/Top-k ranking. The key question is whether NLL/Brier/ECE "
            "and confidence-vs-accuracy alignment improve overall and within true major/minor groups. "
            "Because the ChoCo test split was already inspected earlier, test metrics are post-hoc, not a pristine final holdout."
        ),
    }
    report_path = outdir / "mode_balanced_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMODE-BALANCED CALIBRATION SUMMARY")
    compact = {
        "alpha": alpha,
        "temperature": temperature,
        "oof_uncalibrated": {
            k: report["oof_train_plus_validation"]["uncalibrated"]["overall"][k]
            for k in ("top1_accuracy", "nll", "brier", "ece", "mean_top1_confidence")
        },
        "oof_calibrated": {
            k: report["oof_train_plus_validation"]["calibrated"]["overall"][k]
            for k in ("top1_accuracy", "nll", "brier", "ece", "mean_top1_confidence")
        },
        "test_major": {
            k: report["posthoc_test"]["calibrated"]["by_true_mode"]["major"][k]
            for k in ("top1_accuracy", "mode_recall", "ece", "mean_top1_confidence", "confidence_accuracy_gap")
        },
        "test_minor": {
            k: report["posthoc_test"]["calibrated"]["by_true_mode"]["minor"][k]
            for k in ("top1_accuracy", "mode_recall", "ece", "mean_top1_confidence", "confidence_accuracy_gap")
        },
    }
    print(json.dumps(compact, indent=2))
    print(f"Wrote: {calibrator_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
