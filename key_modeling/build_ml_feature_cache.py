from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: python -m pip install -r key_modeling/requirements_ml.txt") from exc

from key_modeling.ml_features import (
    FEATURE_NAMES,
    candidate_index,
    feature_matrix_for_progression,
    parse_dataset_progression,
)

SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a transposition-invariant candidate-key feature cache from the Step-2 ChoCo dataset.")
    ap.add_argument("--data", default="data/choco_key_segments_ccby.csv")
    ap.add_argument("--output", default="data/key_ml_features.npz")
    ap.add_argument("--metadata", default="data/key_ml_examples.jsonl")
    ap.add_argument("--report", default="data/key_ml_feature_report.json")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    rows = []
    with Path(args.data).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("split") not in SPLIT_CODE:
                continue
            rows.append(r)
            if args.max_rows and len(rows) >= args.max_rows:
                break

    n = len(rows)
    fcount = len(FEATURE_NAMES)
    X = np.empty((n, 24, fcount), dtype=np.float32)
    y = np.empty(n, dtype=np.uint8)
    split = np.empty(n, dtype=np.uint8)
    n_chords = np.empty(n, dtype=np.uint16)

    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    parse_failures = []

    with metadata_path.open("w", encoding="utf-8") as meta_f:
        write_i = 0
        for source_i, r in enumerate(rows):
            try:
                chords = parse_dataset_progression(r.get("chords_simple", ""))
            except Exception as exc:
                parse_failures.append({"source_row": source_i, "piece_id": r.get("piece_id", ""), "error": str(exc)})
                continue

            feats = feature_matrix_for_progression(chords)
            X[write_i, :, :] = np.asarray([fr.values for fr in feats], dtype=np.float32)
            y[write_i] = candidate_index(int(r["key_tonic_pc"]), r["key_mode"])
            split[write_i] = SPLIT_CODE[r["split"]]
            n_chords[write_i] = min(len(chords), np.iinfo(np.uint16).max)

            meta = {
                "cache_index": write_i,
                "source_row": source_i,
                "piece_id": r.get("piece_id", ""),
                "partition": r.get("partition", ""),
                "title": r.get("title", ""),
                "artist": r.get("artist", ""),
                "split": r.get("split", ""),
                "truth": r.get("key_label", ""),
                "truth_tonic": int(r["key_tonic_pc"]),
                "truth_mode": r["key_mode"],
                "n_chords": len(chords),
                "chords_simple": r.get("chords_simple", ""),
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            write_i += 1
            if write_i % 1000 == 0:
                print(f"features: {write_i:,}/{n:,}")

    if write_i != n:
        X = X[:write_i]
        y = y[:write_i]
        split = split[:write_i]
        n_chords = n_chords[:write_i]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        y=y,
        split=split,
        n_chords=n_chords,
        feature_names=np.asarray(FEATURE_NAMES, dtype="U64"),
    )

    counts = {name: int((split == code).sum()) for name, code in SPLIT_CODE.items()}
    report = {
        "source": args.data,
        "output": args.output,
        "metadata": args.metadata,
        "examples": int(write_i),
        "candidate_keys_per_example": 24,
        "feature_count": fcount,
        "feature_names": list(FEATURE_NAMES),
        "split_counts": counts,
        "parse_failures": parse_failures,
        "note": "X has shape [examples, 24 candidate keys, features]. Features are candidate-relative and transposition-invariant; absolute tonic pitch is not included as a predictor.",
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
