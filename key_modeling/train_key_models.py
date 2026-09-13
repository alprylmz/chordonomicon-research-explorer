from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit("ML dependencies missing. Install with: python -m pip install -r key_modeling/requirements_ml.txt") from exc

from chord_utils import PC_NAME
from key_modeling.ml_features import candidate_from_index

SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def load_metadata(path: str) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def binary_rows(X3: np.ndarray, y_class: np.ndarray, mask: np.ndarray):
    Xg = X3[mask]
    yg = y_class[mask]
    n = Xg.shape[0]
    X = Xg.reshape(n * 24, Xg.shape[2])
    y = np.zeros(n * 24, dtype=np.uint8)
    y[np.arange(n) * 24 + yg.astype(int)] = 1
    return X, y


def candidate_scores(model, X3: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Xg = X3[mask]
    n = Xg.shape[0]
    flat = Xg.reshape(n * 24, Xg.shape[2])
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(flat)[:, 1]
    else:
        s = model.decision_function(flat)
    return np.asarray(s).reshape(n, 24)


def rank_metrics(scores: np.ndarray, y_true: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1)
    n = len(y_true)
    ranks = np.empty(n, dtype=np.int16)
    pred = order[:, 0]
    for i in range(n):
        ranks[i] = int(np.where(order[i] == int(y_true[i]))[0][0]) + 1

    top = {f"top{k}_accuracy": float(np.mean(ranks <= k)) for k in (1, 2, 3, 5)}
    top["mrr"] = float(np.mean(1.0 / ranks))

    true_tonic = y_true // 2
    true_mode = y_true % 2
    pred_tonic = pred // 2
    pred_mode = pred % 2
    top["tonic_accuracy"] = float(np.mean(pred_tonic == true_tonic))
    top["mode_accuracy"] = float(np.mean(pred_mode == true_mode))

    mode_recalls = {}
    for mode_idx, label in ((0, "major"), (1, "minor")):
        m = true_mode == mode_idx
        mode_recalls[label] = float(np.mean(pred_mode[m] == mode_idx)) if np.any(m) else None
    valid_mode_recalls = [x for x in mode_recalls.values() if x is not None]
    top["mode_recall"] = mode_recalls
    top["balanced_mode_accuracy"] = float(np.mean(valid_mode_recalls)) if valid_mode_recalls else None

    key_recalls = {}
    for cls in range(24):
        m = y_true == cls
        if not np.any(m):
            continue
        tonic, mode = candidate_from_index(cls)
        key_recalls[f"{PC_NAME[tonic]} {mode}"] = float(np.mean(pred[m] == cls))
    top["macro_key_recall"] = float(np.mean(list(key_recalls.values()))) if key_recalls else None
    top["per_key_recall"] = key_recalls
    top["rank_histogram"] = {str(k): int(v) for k, v in sorted(Counter(map(int, ranks)).items())}
    return top


def choose_model(validation_reports: dict[str, dict]) -> str:
    # Primary: top-1 key accuracy. Secondary: MRR. Tertiary: macro key recall.
    return max(
        validation_reports,
        key=lambda name: (
            validation_reports[name]["top1_accuracy"],
            validation_reports[name]["mrr"],
            validation_reports[name]["macro_key_recall"],
        ),
    )


def fit_model(name: str, X, y, seed: int):
    if name == "logistic":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ])
        return model.fit(X, y)
    if name == "histgb":
        # One positive candidate vs 23 negatives per example. Balance the two
        # binary classes while preserving every candidate row.
        pos = max(1, int(np.sum(y == 1)))
        neg = max(1, int(np.sum(y == 0)))
        w = np.where(y == 1, (pos + neg) / (2.0 * pos), (pos + neg) / (2.0 * neg))
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
        return model.fit(X, y, sample_weight=w)
    raise ValueError(name)


def logistic_coefficients(model, feature_names: list[str]) -> list[dict]:
    if not isinstance(model, Pipeline) or "clf" not in model.named_steps:
        return []
    clf = model.named_steps["clf"]
    coef = clf.coef_[0]
    rows = [
        {"feature": f, "coefficient": float(c), "abs_coefficient": float(abs(c))}
        for f, c in zip(feature_names, coef)
    ]
    rows.sort(key=lambda r: r["abs_coefficient"], reverse=True)
    return rows


def error_rows(scores: np.ndarray, y_true: np.ndarray, metadata: list[dict], global_indices: np.ndarray) -> list[dict]:
    order = np.argsort(-scores, axis=1)
    out = []
    for local_i, true_cls in enumerate(y_true.astype(int)):
        pred_cls = int(order[local_i, 0])
        if pred_cls == true_cls:
            continue
        true_tonic, true_mode = candidate_from_index(true_cls)
        p1_tonic, p1_mode = candidate_from_index(pred_cls)
        p2_tonic, p2_mode = candidate_from_index(int(order[local_i, 1]))
        rank = int(np.where(order[local_i] == true_cls)[0][0]) + 1
        meta = metadata[int(global_indices[local_i])]
        out.append({
            "piece_id": meta.get("piece_id", ""),
            "partition": meta.get("partition", ""),
            "title": meta.get("title", ""),
            "artist": meta.get("artist", ""),
            "truth": f"{PC_NAME[true_tonic]} {true_mode}",
            "pred1": f"{PC_NAME[p1_tonic]} {p1_mode}",
            "pred2": f"{PC_NAME[p2_tonic]} {p2_mode}",
            "truth_rank": rank,
            "score_gap": float(scores[local_i, order[local_i, 0]] - scores[local_i, order[local_i, 1]]),
            "n_chords": meta.get("n_chords", ""),
            "chords_simple": meta.get("chords_simple", ""),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train candidate-ranking key-inference models on Step-3 features.")
    ap.add_argument("--features", default="data/key_ml_features.npz")
    ap.add_argument("--metadata", default="data/key_ml_examples.jsonl")
    ap.add_argument("--outdir", default="data/key_models")
    ap.add_argument("--models", default="logistic,histgb", help="comma-separated: logistic,histgb")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--baseline-report", default="data/theory_baseline_report.json")
    args = ap.parse_args()

    d = np.load(args.features)
    X3 = d["X"]
    y_class = d["y"].astype(int)
    split = d["split"].astype(int)
    feature_names = [str(x) for x in d["feature_names"]]
    metadata = load_metadata(args.metadata)
    if len(metadata) != len(X3):
        raise SystemExit(f"Metadata rows ({len(metadata)}) != feature examples ({len(X3)})")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    train_mask = split == SPLIT_CODE["train"]
    val_mask = split == SPLIT_CODE["validation"]
    test_mask = split == SPLIT_CODE["test"]

    X_train, y_train = binary_rows(X3, y_class, train_mask)
    validation_reports: dict[str, dict] = {}
    initial_models = {}

    print(f"Training groups: {int(train_mask.sum()):,}; candidate rows: {len(y_train):,}")
    for name in model_names:
        print(f"\nTraining {name} on train split...")
        model = fit_model(name, X_train, y_train, args.seed)
        initial_models[name] = model
        scores = candidate_scores(model, X3, val_mask)
        report = rank_metrics(scores, y_class[val_mask])
        validation_reports[name] = report
        print(json.dumps({name: report}, indent=2))

    selected = choose_model(validation_reports)
    print(f"\nSelected on validation: {selected}")

    # Refit each candidate model on train+validation for a fair final test report.
    trainval_mask = train_mask | val_mask
    X_trainval, y_trainval = binary_rows(X3, y_class, trainval_mask)
    final_models = {}
    test_reports = {}
    for name in model_names:
        print(f"Refitting {name} on train+validation...")
        model = fit_model(name, X_trainval, y_trainval, args.seed)
        final_models[name] = model
        scores = candidate_scores(model, X3, test_mask)
        test_reports[name] = rank_metrics(scores, y_class[test_mask])
        joblib.dump(model, outdir / f"{name}.joblib")

    selected_model = final_models[selected]
    joblib.dump(selected_model, outdir / "selected_model.joblib")

    test_global_idx = np.flatnonzero(test_mask)
    selected_scores = candidate_scores(selected_model, X3, test_mask)
    selected_errors = error_rows(selected_scores, y_class[test_mask], metadata, test_global_idx)
    errors_path = outdir / "selected_model_test_errors.csv"
    if selected_errors:
        with errors_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(selected_errors[0]))
            w.writeheader(); w.writerows(selected_errors)
    else:
        errors_path.write_text("", encoding="utf-8")

    coef_rows = logistic_coefficients(final_models.get("logistic"), feature_names) if "logistic" in final_models else []
    if coef_rows:
        with (outdir / "logistic_coefficients.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(coef_rows[0]))
            w.writeheader(); w.writerows(coef_rows)

    baseline = None
    p = Path(args.baseline_report)
    if p.exists():
        baseline = json.loads(p.read_text(encoding="utf-8"))

    report = {
        "selection_rule": "highest validation top1_accuracy, then MRR, then macro_key_recall",
        "selected_model": selected,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "split_group_counts": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "validation": validation_reports,
        "test_after_refit_train_plus_validation": test_reports,
        "theory_baseline_test_report": baseline,
        "artifacts": {
            "selected_model": str(outdir / "selected_model.joblib"),
            "selected_test_errors": str(errors_path),
            "logistic_coefficients": str(outdir / "logistic_coefficients.csv") if coef_rows else None,
        },
        "important_note": "Candidate classifier scores are used for ranking only in Step 3. They are not calibrated 24-key probabilities yet; calibration is Step 4.",
    }
    (outdir / "step3_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (outdir / "model_metadata.json").write_text(json.dumps({
        "selected_model": selected,
        "feature_names": feature_names,
        "candidate_order": [f"{PC_NAME[t]} {m}" for t, m in [candidate_from_index(i) for i in range(24)]],
        "not_calibrated": True,
    }, indent=2), encoding="utf-8")

    print("\nFINAL TEST REPORT")
    print(json.dumps({"selected_model": selected, "metrics": test_reports[selected]}, indent=2))
    print(f"Wrote {outdir / 'step3_model_report.json'}")


if __name__ == "__main__":
    main()
