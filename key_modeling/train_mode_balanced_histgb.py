from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError as exc:
    raise SystemExit(
        "ML dependencies missing. Install with: python -m pip install -r key_modeling/requirements_ml.txt"
    ) from exc

from chord_utils import PC_NAME
from key_modeling.ml_features import candidate_from_index

SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def candidate_scores(model, X3_subset: np.ndarray) -> np.ndarray:
    n = X3_subset.shape[0]
    flat = X3_subset.reshape(n * 24, X3_subset.shape[2])
    s = model.predict_proba(flat)[:, 1]
    return np.asarray(s, dtype=float).reshape(n, 24)


def make_binary_training_rows(
    X3_subset: np.ndarray,
    y_class_subset: np.ndarray,
    mode_balance_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Flatten 24 candidates/example and return combined row weights.

    There are two independent imbalances:
      1. candidate target imbalance: 1 positive vs 23 negatives per example;
      2. example-mode imbalance: many more major-labeled examples than minor-labeled examples.

    The original Step-3 HistGB corrected (1) only. This experiment additionally
    corrects (2) at the *example* level. Every one of an example's 24 candidate
    rows receives the same true-mode weight, so we do not directly reward or
    punish candidate keys merely because the candidate itself is major/minor.

    alpha=0.0 -> original weighting
    alpha=0.5 -> square-root inverse-frequency weighting
    alpha=1.0 -> full inverse-frequency weighting
    """
    if not 0.0 <= mode_balance_alpha <= 1.0:
        raise ValueError("mode_balance_alpha must be between 0 and 1")

    n = X3_subset.shape[0]
    X = X3_subset.reshape(n * 24, X3_subset.shape[2])
    y = np.zeros(n * 24, dtype=np.uint8)
    y[np.arange(n) * 24 + y_class_subset.astype(int)] = 1

    # Original Step-3 binary candidate-class balancing.
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    binary_weight = np.where(
        y == 1,
        (pos + neg) / (2.0 * pos),
        (pos + neg) / (2.0 * neg),
    ).astype(float)

    # Additional true-mode balancing at the example level.
    true_modes = (y_class_subset.astype(int) % 2).astype(int)  # 0 major, 1 minor
    major_n = max(1, int(np.sum(true_modes == 0)))
    minor_n = max(1, int(np.sum(true_modes == 1)))
    total = major_n + minor_n
    raw_major = total / (2.0 * major_n)
    raw_minor = total / (2.0 * minor_n)

    major_w = raw_major ** mode_balance_alpha
    minor_w = raw_minor ** mode_balance_alpha
    example_w = np.where(true_modes == 0, major_w, minor_w).astype(float)
    # Normalize to mean 1 so alpha changes relative weighting, not global scale.
    example_w /= float(np.mean(example_w))
    row_mode_weight = np.repeat(example_w, 24)
    combined = binary_weight * row_mode_weight

    info = {
        "alpha": float(mode_balance_alpha),
        "training_examples_major": major_n,
        "training_examples_minor": minor_n,
        "raw_inverse_frequency_major": float(raw_major),
        "raw_inverse_frequency_minor": float(raw_minor),
        "normalized_example_weight_major": float(np.unique(example_w[true_modes == 0])[0]),
        "normalized_example_weight_minor": float(np.unique(example_w[true_modes == 1])[0]),
        "minor_to_major_example_weight_ratio": float(
            np.unique(example_w[true_modes == 1])[0] / np.unique(example_w[true_modes == 0])[0]
        ),
    }
    return X, y, combined, info


def fit_histgb(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed: int):
    model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=220,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=seed,
    )
    return model.fit(X, y, sample_weight=sample_weight)


def rank_metrics(scores: np.ndarray, y_true: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1)
    pred = order[:, 0]
    n = len(y_true)
    ranks = np.empty(n, dtype=np.int16)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == int(y_true[i]))[0][0]) + 1

    true_mode = y_true % 2
    pred_mode = pred % 2
    true_tonic = y_true // 2
    pred_tonic = pred // 2

    report = {
        **{f"top{k}_accuracy": float(np.mean(ranks <= k)) for k in (1, 2, 3, 5)},
        "mrr": float(np.mean(1.0 / ranks)),
        "tonic_accuracy": float(np.mean(true_tonic == pred_tonic)),
        "mode_accuracy": float(np.mean(true_mode == pred_mode)),
    }

    by_true_mode = {}
    mode_recall = {}
    for mode_idx, label in ((0, "major"), (1, "minor")):
        m = true_mode == mode_idx
        if not np.any(m):
            continue
        key_acc = float(np.mean(pred[m] == y_true[m]))
        mode_acc = float(np.mean(pred_mode[m] == mode_idx))
        by_true_mode[label] = {
            "count": int(np.sum(m)),
            "key_top1_accuracy": key_acc,
            "mode_recall": mode_acc,
            "mean_rank": float(np.mean(ranks[m])),
            "mrr": float(np.mean(1.0 / ranks[m])),
        }
        mode_recall[label] = mode_acc

    report["by_true_mode"] = by_true_mode
    report["mode_recall"] = mode_recall
    report["balanced_mode_accuracy"] = float(np.mean(list(mode_recall.values())))
    report["balanced_true_mode_key_accuracy"] = float(
        np.mean([x["key_top1_accuracy"] for x in by_true_mode.values()])
    )
    report["major_minor_key_accuracy_gap"] = float(
        by_true_mode["major"]["key_top1_accuracy"] - by_true_mode["minor"]["key_top1_accuracy"]
    )

    per_key = {}
    for cls in range(24):
        m = y_true == cls
        if not np.any(m):
            continue
        tonic, mode = candidate_from_index(cls)
        per_key[f"{PC_NAME[tonic]} {mode}"] = float(np.mean(pred[m] == cls))
    report["macro_key_recall"] = float(np.mean(list(per_key.values()))) if per_key else None
    report["per_key_recall"] = per_key
    report["rank_histogram"] = {str(k): int(v) for k, v in sorted(Counter(map(int, ranks)).items())}
    return report


def selection_tuple(report: dict) -> tuple[float, float, float, float]:
    """Choose a mode-balanced variant on validation only.

    Primary target is average key Top-1 across true major/minor groups. Then use
    overall Top-1, macro key recall, and MRR as tie-breakers.
    """
    return (
        report["balanced_true_mode_key_accuracy"],
        report["top1_accuracy"],
        report["macro_key_recall"],
        report["mrr"],
    )


def parse_alphas(text: str) -> list[float]:
    vals = []
    for token in text.split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise ValueError("Need at least one alpha")
    for x in vals:
        if not 0.0 <= x <= 1.0:
            raise ValueError("alphas must be in [0,1]")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 6: compare mode-balanced HistGB variants without overwriting the production model."
    )
    ap.add_argument("--features", default="data/key_ml_features.npz")
    ap.add_argument("--outdir", default="data/key_models/mode_balance_experiment")
    ap.add_argument("--alphas", default="0,0.5,1.0", help="Mode-balance strengths; 0 reproduces original weighting")
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()

    d = np.load(args.features)
    X3 = d["X"]
    y_class = d["y"].astype(int)
    split = d["split"].astype(int)

    train_mask = split == SPLIT_CODE["train"]
    val_mask = split == SPLIT_CODE["validation"]
    test_mask = split == SPLIT_CODE["test"]
    alphas = parse_alphas(args.alphas)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    validation = {}
    weighting = {}
    train_X3 = X3[train_mask]
    train_yc = y_class[train_mask]

    for alpha in alphas:
        tag = f"alpha_{alpha:g}".replace(".", "p")
        print(f"\nTraining validation candidate: {tag}")
        X, y, w, winfo = make_binary_training_rows(train_X3, train_yc, alpha)
        model = fit_histgb(X, y, w, args.seed + int(round(alpha * 1000)))
        scores = candidate_scores(model, X3[val_mask])
        metrics = rank_metrics(scores, y_class[val_mask])
        validation[tag] = metrics
        weighting[tag] = winfo
        print(json.dumps({tag: {k: metrics[k] for k in (
            "top1_accuracy", "balanced_true_mode_key_accuracy", "balanced_mode_accuracy",
            "macro_key_recall", "mrr", "major_minor_key_accuracy_gap", "by_true_mode"
        )}}, indent=2))

    selected_tag = max(validation, key=lambda k: selection_tuple(validation[k]))
    selected_alpha = alphas[list(validation.keys()).index(selected_tag)]
    print(f"\nSelected on validation only: {selected_tag} (alpha={selected_alpha:g})")

    # Refit only the selected variant on train+validation, then report test once.
    trainval_mask = train_mask | val_mask
    X, y, w, final_weighting = make_binary_training_rows(
        X3[trainval_mask], y_class[trainval_mask], selected_alpha
    )
    final_model = fit_histgb(X, y, w, args.seed + 10000 + int(round(selected_alpha * 1000)))
    test_scores = candidate_scores(final_model, X3[test_mask])
    test_metrics = rank_metrics(test_scores, y_class[test_mask])

    model_path = outdir / "selected_mode_balanced_histgb.joblib"
    joblib.dump(final_model, model_path)

    report = {
        "experiment": "true-mode example weighting for HistGradientBoosting candidate ranker",
        "important_method_note": (
            "The earlier ChoCo test split was already inspected during the mode-specific diagnostic. "
            "Therefore this new model is selected strictly on validation, while the reported test comparison "
            "should be treated as a post-hoc evaluation rather than a pristine untouched holdout."
        ),
        "weighting_definition": (
            "Every example's 24 candidate rows share the same true-mode weight. This is multiplied by the "
            "existing 1-positive-vs-23-negative candidate-class weight. alpha=0 means no mode balancing; "
            "alpha=0.5 uses sqrt inverse-frequency; alpha=1 uses full inverse-frequency."
        ),
        "selection_rule": (
            "highest validation balanced_true_mode_key_accuracy, then overall top1_accuracy, "
            "macro_key_recall, then MRR"
        ),
        "alphas": alphas,
        "split_counts": {
            "train": int(np.sum(train_mask)),
            "validation": int(np.sum(val_mask)),
            "test": int(np.sum(test_mask)),
        },
        "validation": validation,
        "weighting": weighting,
        "selected": {
            "tag": selected_tag,
            "alpha": float(selected_alpha),
            "refit_weighting": final_weighting,
            "model_path": str(model_path),
        },
        "posthoc_test_after_refit_train_plus_validation": test_metrics,
        "next_step": (
            "Compare this report with the original HistGB. Do not replace the production model or rerun "
            "Chordonomicon inference yet. If the trade-off is acceptable, calibrate this selected model "
            "separately and repeat the mode-specific labeled audit before deployment."
        ),
    }

    report_path = outdir / "mode_balance_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFINAL SELECTED VARIANT")
    print(json.dumps({
        "selected": report["selected"],
        "validation": validation[selected_tag],
        "posthoc_test": test_metrics,
    }, indent=2))
    print(f"Wrote: {report_path}")
    print(f"Wrote: {model_path}")


if __name__ == "__main__":
    main()
