from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
# If installed under key_modeling/, project root is one level higher.
if PROJECT_ROOT.name == "key_modeling":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import joblib
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing dependencies. Activate the project venv and install key_modeling/requirements_ml.txt") from exc

from chord_utils import PC_NAME
from key_modeling.ml_features import candidate_from_index
from key_modeling.calibration_utils import probabilities_from_candidate_scores, probability_metrics

SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def load_metadata(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_examples(model, X3: np.ndarray) -> np.ndarray:
    n = X3.shape[0]
    flat = X3.reshape(n * 24, X3.shape[2])
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(flat)[:, 1]
    else:
        scores = model.decision_function(flat)
    return np.asarray(scores, dtype=float).reshape(n, 24)


def class_name(cls: int) -> str:
    tonic, mode = candidate_from_index(int(cls))
    return f"{PC_NAME[tonic]} {mode}"


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean(np.any(order == y[:, None], axis=1)))


def rank_array(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    ranks = np.empty(len(y), dtype=int)
    for i, truth in enumerate(y):
        ranks[i] = int(np.where(order[i] == int(truth))[0][0]) + 1
    return ranks


def margin(probs: np.ndarray) -> np.ndarray:
    order = np.sort(probs, axis=1)
    return order[:, -1] - order[:, -2]


def subset_metrics(probs: np.ndarray, y: np.ndarray, mode_idx: int, bins: int) -> dict:
    true_mode = y % 2
    mask = true_mode == mode_idx
    p = probs[mask]
    yt = y[mask]
    pred = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    margins = margin(p)
    ranks = rank_array(p, yt)
    pred_mode = pred % 2

    pm = probability_metrics(p, yt, n_bins=bins)
    return {
        "count": int(mask.sum()),
        "top1_accuracy": pm["top1_accuracy"],
        "top2_accuracy": topk_accuracy(p, yt, 2),
        "top3_accuracy": topk_accuracy(p, yt, 3),
        "mrr": float(np.mean(1.0 / ranks)),
        "mode_recall": float(np.mean(pred_mode == mode_idx)),
        "nll": pm["nll"],
        "brier": pm["brier"],
        "ece": pm["ece"],
        "mean_top1_confidence": pm["mean_top1_confidence"],
        "median_top1_confidence": float(np.median(conf)),
        "confidence_accuracy_gap": pm["confidence_accuracy_gap"],
        "mean_true_key_probability": float(np.mean(p[np.arange(len(yt)), yt])),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "correct_mean_confidence": float(np.mean(conf[pred == yt])) if np.any(pred == yt) else None,
        "incorrect_mean_confidence": float(np.mean(conf[pred != yt])) if np.any(pred != yt) else None,
        "predicted_major_share": float(np.mean(pred_mode == 0)),
        "predicted_minor_share": float(np.mean(pred_mode == 1)),
        "rank_histogram": {str(k): int(v) for k, v in sorted(Counter(map(int, ranks)).items())},
        "reliability_bins": pm["reliability_bins"],
    }


def threshold_rows(probs: np.ndarray, y: np.ndarray, thresholds: list[float]) -> list[dict]:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    true_mode = y % 2
    rows = []
    for mode_idx, label in ((0, "major"), (1, "minor")):
        base = true_mode == mode_idx
        denom = int(base.sum())
        for t in thresholds:
            m = base & (conf >= t)
            count = int(m.sum())
            rows.append({
                "true_mode": label,
                "threshold": float(t),
                "count": count,
                "coverage_within_true_mode": float(count / denom) if denom else None,
                "top1_accuracy_at_threshold": float(np.mean(pred[m] == y[m])) if count else None,
                "mode_accuracy_at_threshold": float(np.mean((pred[m] % 2) == mode_idx)) if count else None,
                "mean_confidence": float(np.mean(conf[m])) if count else None,
            })
    return rows


def mode_confusion_rows(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    rows = []
    for truth_idx, truth_label in ((0, "major"), (1, "minor")):
        for pred_idx, pred_label in ((0, "major"), (1, "minor")):
            m = ((y % 2) == truth_idx) & ((pred % 2) == pred_idx)
            rows.append({
                "true_mode": truth_label,
                "predicted_mode": pred_label,
                "count": int(m.sum()),
                "share_within_true_mode": float(m.sum() / np.sum((y % 2) == truth_idx)),
                "mean_top1_confidence": float(np.mean(conf[m])) if np.any(m) else None,
            })
    return rows


def key_confusions(probs: np.ndarray, y: np.ndarray, only_true_mode: int | None = None, top_n: int = 30) -> list[dict]:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    counter: Counter[tuple[int, int]] = Counter()
    conf_sum: dict[tuple[int, int], float] = {}
    for i, (truth, guess) in enumerate(zip(y.astype(int), pred.astype(int))):
        if truth == guess:
            continue
        if only_true_mode is not None and truth % 2 != only_true_mode:
            continue
        key = (truth, guess)
        counter[key] += 1
        conf_sum[key] = conf_sum.get(key, 0.0) + float(conf[i])
    rows = []
    for (truth, guess), count in counter.most_common(top_n):
        rows.append({
            "truth": class_name(truth),
            "predicted": class_name(guess),
            "count": int(count),
            "mean_predicted_confidence": conf_sum[(truth, guess)] / count,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Labeled ChoCo audit of major/minor accuracy and calibration on the untouched test split.")
    ap.add_argument("--features", default="data/key_ml_features.npz")
    ap.add_argument("--metadata", default="data/key_ml_examples.jsonl")
    ap.add_argument("--model", default="data/key_models/selected_model.joblib")
    ap.add_argument("--calibrator", default="data/key_models/temperature_calibrator.json")
    ap.add_argument("--outdir", default="data/key_models/choco_mode_audit")
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    d = np.load(args.features)
    X3 = d["X"]
    y = d["y"].astype(int)
    split = d["split"].astype(int)
    metadata = load_metadata(args.metadata)
    if len(metadata) != len(X3):
        raise SystemExit(f"Metadata rows ({len(metadata)}) != feature examples ({len(X3)})")

    calibrator = json.loads(Path(args.calibrator).read_text(encoding="utf-8"))
    temperature = float(calibrator["temperature"])
    model = joblib.load(args.model)

    test_idx = np.flatnonzero(split == SPLIT_CODE["test"])
    Xtest = X3[test_idx]
    ytest = y[test_idx]
    print(f"Test examples: {len(test_idx):,}")
    print(f"Temperature: {temperature:.6f}")

    raw_scores = score_examples(model, Xtest)
    probs = probabilities_from_candidate_scores(raw_scores, temperature=temperature)

    overall = probability_metrics(probs, ytest, n_bins=args.bins)
    overall.update({
        "top2_accuracy": topk_accuracy(probs, ytest, 2),
        "top3_accuracy": topk_accuracy(probs, ytest, 3),
        "mean_margin": float(np.mean(margin(probs))),
    })

    major = subset_metrics(probs, ytest, 0, args.bins)
    minor = subset_metrics(probs, ytest, 1, args.bins)
    thresholds = threshold_rows(probs, ytest, [0.50, 0.60, 0.70, 0.80, 0.90])
    mode_conf = mode_confusion_rows(probs, ytest)
    all_confusions = key_confusions(probs, ytest, None, 40)
    minor_confusions = key_confusions(probs, ytest, 1, 40)

    report = {
        "dataset": args.features,
        "split": "test",
        "test_examples": int(len(test_idx)),
        "temperature": temperature,
        "overall": overall,
        "by_true_mode": {"major": major, "minor": minor},
        "thresholds_by_true_mode": thresholds,
        "mode_confusion": mode_conf,
        "top_key_confusions": all_confusions,
        "top_true_minor_confusions": minor_confusions,
        "interpretation_note": (
            "This audit uses ChoCo ground-truth labels on the untouched test split. "
            "It is designed to distinguish classifier asymmetry from deployment-domain uncertainty."
        ),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "choco_mode_audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(outdir / "mode_metrics.csv", [
        {"true_mode": "major", **{k: v for k, v in major.items() if not isinstance(v, (dict, list))}},
        {"true_mode": "minor", **{k: v for k, v in minor.items() if not isinstance(v, (dict, list))}},
    ])
    write_csv(outdir / "thresholds_by_true_mode.csv", thresholds)
    write_csv(outdir / "mode_confusion.csv", mode_conf)
    write_csv(outdir / "top_key_confusions.csv", all_confusions)
    write_csv(outdir / "top_true_minor_confusions.csv", minor_confusions)
    write_csv(outdir / "reliability_true_major.csv", major["reliability_bins"])
    write_csv(outdir / "reliability_true_minor.csv", minor["reliability_bins"])

    print("\nCHOCO MODE AUDIT")
    print(json.dumps({
        "overall_top1": overall["top1_accuracy"],
        "major": {k: major[k] for k in ("count", "top1_accuracy", "mode_recall", "ece", "mean_top1_confidence", "confidence_accuracy_gap", "mean_margin")},
        "minor": {k: minor[k] for k in ("count", "top1_accuracy", "mode_recall", "ece", "mean_top1_confidence", "confidence_accuracy_gap", "mean_margin")},
    }, indent=2))
    print(f"\nWrote: {outdir / 'choco_mode_audit_summary.json'}")
    print(f"Wrote: {outdir / 'mode_metrics.csv'}")
    print(f"Wrote: {outdir / 'thresholds_by_true_mode.csv'}")
    print(f"Wrote: {outdir / 'top_true_minor_confusions.csv'}")


if __name__ == "__main__":
    main()
