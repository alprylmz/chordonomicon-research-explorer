from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# When this file is executed as ``python key_modeling/evaluate_theory_baseline.py``,
# Python puts key_modeling/ rather than the project root on sys.path. Add the
# project root explicitly so sibling modules can be imported reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chord_utils import parse_chord
from key_inference import infer_keys


def norm_name(name: str) -> str:
    return name.strip().replace("♭", "b").replace("♯", "#")


def parse_dataset_progression(text: str):
    """Parse the whitespace-delimited ``chords_simple`` field from Step 2.

    Important: do *not* split on commas or hyphens here. ChoCo/Harte chord
    qualities can legitimately contain commas (e.g. interval-set spellings)
    and punctuation inside a chord token. The generic Chordonomicon parser
    splits on commas/hyphens because it is designed for human-entered text;
    that behavior caused a large number of false parse errors on ChoCo.
    """
    out = []
    for raw in (text or "").split():
        token = raw.strip()
        chord = parse_chord(token)
        if chord is None:
            raise ValueError(f"Could not parse chord token: {token}")
        out.append(chord)
    if not out:
        raise ValueError("No chords found")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the Step-1 theory baseline on labeled ChoCo examples.")
    ap.add_argument("--data", default="data/choco_key_segments_ccby.csv")
    ap.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    ap.add_argument("--report", default="data/theory_baseline_report.json")
    ap.add_argument("--errors", default="data/theory_baseline_errors.csv")
    ap.add_argument("--parse-errors", default="data/theory_baseline_parse_errors.csv")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    rows = []
    with Path(args.data).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if args.split != "all" and r.get("split") != args.split:
                continue
            rows.append(r)
            if args.max_rows and len(rows) >= args.max_rows:
                break

    counts = Counter()
    source_counts = Counter()
    mode_conf = Counter()
    errors = []
    parse_errors = []
    rank_hist = Counter()
    truth_key_all = Counter()
    truth_mode_all = Counter()
    truth_key_eval = Counter()
    truth_mode_eval = Counter()

    for i, r in enumerate(rows, 1):
        counts["rows_seen"] += 1
        truth = norm_name(r.get("key_label", ""))
        true_mode = r.get("key_mode", "")
        truth_key_all[truth] += 1
        truth_mode_all[true_mode] += 1

        try:
            chords = parse_dataset_progression(r.get("chords_simple", ""))
        except Exception as exc:
            counts["parse_errors"] += 1
            parse_errors.append({
                "piece_id": r.get("piece_id", ""),
                "partition": r.get("partition", ""),
                "title": r.get("title", ""),
                "artist": r.get("artist", ""),
                "truth": truth,
                "n_chords": r.get("n_chords", ""),
                "error": str(exc),
                "chords_simple": r.get("chords_simple", ""),
            })
            continue

        ranked = infer_keys(chords, top_k=24)
        if not ranked:
            counts["no_prediction"] += 1
            continue

        preds = [norm_name(x.name) for x in ranked]
        counts["evaluated"] += 1
        source_counts[r.get("partition", "unknown")] += 1
        truth_key_eval[truth] += 1
        truth_mode_eval[true_mode] += 1

        try:
            rank = preds.index(truth) + 1
        except ValueError:
            rank = None
        if rank is not None:
            rank_hist[rank] += 1
            for k in (1, 2, 3, 5):
                if rank <= k:
                    counts[f"top{k}_correct"] += 1

        pred_mode = ranked[0].mode
        mode_conf[(true_mode, pred_mode)] += 1
        if pred_mode == true_mode:
            counts["mode_correct"] += 1
        tonic_true = int(r["key_tonic_pc"])
        if ranked[0].tonic == tonic_true:
            counts["tonic_correct"] += 1

        if rank != 1:
            errors.append({
                "piece_id": r.get("piece_id", ""),
                "partition": r.get("partition", ""),
                "title": r.get("title", ""),
                "artist": r.get("artist", ""),
                "truth": truth,
                "pred1": preds[0],
                "pred2": preds[1] if len(preds) > 1 else "",
                "truth_rank": rank if rank is not None else "",
                "score_gap": ranked[0].score - ranked[1].score if len(ranked) > 1 else "",
                "heuristic_share1": ranked[0].heuristic_share,
                "n_chords": r.get("n_chords", ""),
                "chords_simple": r.get("chords_simple", ""),
            })

        if i % 1000 == 0:
            print(f"processed {i:,}/{len(rows):,}")

    n = counts["evaluated"]
    metrics = {}
    if rows:
        metrics["evaluation_coverage"] = n / len(rows)
    if n:
        for k in (1, 2, 3, 5):
            metrics[f"top{k}_accuracy"] = counts[f"top{k}_correct"] / n
        metrics["mode_accuracy"] = counts["mode_correct"] / n
        metrics["tonic_accuracy"] = counts["tonic_correct"] / n
        metrics["mrr"] = sum(c / rank for rank, c in rank_hist.items()) / n

        majority_key, majority_key_n = truth_key_eval.most_common(1)[0]
        majority_mode, majority_mode_n = truth_mode_eval.most_common(1)[0]
        metrics["majority_key_baseline_accuracy"] = majority_key_n / n
        metrics["majority_mode_baseline_accuracy"] = majority_mode_n / n
    else:
        majority_key = majority_mode = None

    report = {
        "dataset": args.data,
        "split": args.split,
        "metrics": metrics,
        "counts": dict(counts),
        "majority_baselines": {
            "key": majority_key,
            "mode": majority_mode,
        },
        "truth_key_counts_evaluated": dict(truth_key_eval.most_common()),
        "truth_mode_counts_evaluated": dict(truth_mode_eval.most_common()),
        "truth_key_counts_all_selected_rows": dict(truth_key_all.most_common()),
        "truth_mode_counts_all_selected_rows": dict(truth_mode_all.most_common()),
        "rank_histogram": {str(k): v for k, v in sorted(rank_hist.items())},
        "mode_confusion": {f"true={a}|pred={b}": n for (a, b), n in sorted(mode_conf.items())},
        "partition_evaluated": dict(source_counts.most_common()),
        "note": (
            "Heuristic shares are not calibrated probabilities. This is a fixed theory baseline, "
            "so no fitting occurs here. Step-2 evaluator uses whitespace tokenization for ChoCo "
            "chords_simple fields so Harte interval-set punctuation is preserved."
        ),
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def write_rows(path: str, rows_to_write: list[dict]):
        p = Path(path)
        if rows_to_write:
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_to_write[0]))
                w.writeheader()
                w.writerows(rows_to_write)
        else:
            p.write_text("", encoding="utf-8")

    write_rows(args.errors, errors)
    write_rows(args.parse_errors, parse_errors)

    print(json.dumps(report, indent=2))
    print(f"Wrote {args.report}")
    print(f"Wrote {args.errors}")
    print(f"Wrote {args.parse_errors}")


if __name__ == "__main__":
    main()
