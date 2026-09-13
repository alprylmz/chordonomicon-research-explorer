from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import duckdb
    import joblib
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Install project + ML dependencies first: python -m pip install -r requirements.txt "
        "&& python -m pip install -r key_modeling/requirements_ml.txt"
    ) from exc

from chord_utils import PC_NAME, parse_sections
from key_modeling.calibration_utils import probabilities_from_candidate_scores
from key_modeling.ml_features import candidate_from_index, feature_matrix_for_progression


PREDICTION_COLUMNS = [
    "source_row",
    "song_id",
    "predicted_key",
    "tonic_pc",
    "mode",
    "calibrated_probability",
    "second_key",
    "second_probability",
    "probability_margin",
    "normalized_entropy",
    "n_chords",
    "n_sections",
]


def _flatten_chart(chord_text: str):
    sections = parse_sections(chord_text or "")
    chords = []
    for _section_type, _section_label, seq in sections:
        chords.extend(seq)
    return sections, chords


def _feature_task(item: tuple[int, str, str]):
    """Worker-safe feature extraction for one raw Chordonomicon row."""
    source_row, song_id, chord_text = item
    sections, chords = _flatten_chart(chord_text)
    if not chords:
        return source_row, song_id, 0, len(sections), None
    rows = feature_matrix_for_progression(chords)
    matrix = np.asarray([r.values for r in rows], dtype=np.float32)
    return source_row, song_id, len(chords), len(sections), matrix


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    return ent / math.log(p.shape[1])


def _summaries_from_probabilities(
    source_rows: Sequence[int],
    song_ids: Sequence[str],
    n_chords: Sequence[int],
    n_sections: Sequence[int],
    probs: np.ndarray,
) -> list[tuple]:
    order = np.argsort(-probs, axis=1)
    entropy = _normalized_entropy(probs)
    out: list[tuple] = []
    for i in range(len(song_ids)):
        first = int(order[i, 0])
        second = int(order[i, 1])
        tonic1, mode1 = candidate_from_index(first)
        tonic2, mode2 = candidate_from_index(second)
        p1 = float(probs[i, first])
        p2 = float(probs[i, second])
        out.append((
            int(source_rows[i]),
            str(song_ids[i]),
            f"{PC_NAME[tonic1]} {mode1}",
            int(tonic1),
            str(mode1),
            p1,
            f"{PC_NAME[tonic2]} {mode2}",
            p2,
            p1 - p2,
            float(entropy[i]),
            int(n_chords[i]),
            int(n_sections[i]),
        ))
    return out


def _ensure_output_schema(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS song_key_predictions (
            source_row BIGINT PRIMARY KEY,
            song_id VARCHAR,
            predicted_key VARCHAR,
            tonic_pc UTINYINT,
            mode VARCHAR,
            calibrated_probability DOUBLE,
            second_key VARCHAR,
            second_probability DOUBLE,
            probability_margin DOUBLE,
            normalized_entropy DOUBLE,
            n_chords INTEGER,
            n_sections INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS inference_state (
            id INTEGER PRIMARY KEY,
            last_source_row BIGINT,
            songs_written BIGINT,
            skipped_no_chords BIGINT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    if con.execute("SELECT COUNT(*) FROM inference_state WHERE id = 1").fetchone()[0] == 0:
        con.execute("INSERT INTO inference_state(id, last_source_row, songs_written, skipped_no_chords) VALUES (1, 0, 0, 0)")


def _write_rows(con, rows: list[tuple], last_source_row: int, skipped_delta: int) -> None:
    if rows:
        placeholders = ",".join(["?"] * len(PREDICTION_COLUMNS))
        con.executemany(
            f"INSERT OR REPLACE INTO song_key_predictions VALUES ({placeholders})",
            rows,
        )
    con.execute("""
        UPDATE inference_state
        SET last_source_row = ?,
            songs_written = (SELECT COUNT(*) FROM song_key_predictions),
            skipped_no_chords = skipped_no_chords + ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = 1
    """, [int(last_source_row), int(skipped_delta)])


def _iter_csv_rows(csv_path: Path, start_after: int, max_songs: int | None):
    yielded = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"id", "chords"}.issubset(fields):
            raise ValueError("Chordonomicon CSV must contain id and chords columns")
        for source_row, row in enumerate(reader, start=1):
            if source_row <= start_after:
                continue
            song_id = (row.get("id") or str(source_row)).strip() or str(source_row)
            yield source_row, song_id, row.get("chords") or ""
            yielded += 1
            if max_songs is not None and yielded >= max_songs:
                return


def _batched(iterator: Iterable, batch_size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_summary(con, output_path: Path, *, temperature: float, model_path: str, source_csv: str, elapsed: float) -> dict:
    total = int(con.execute("SELECT COUNT(*) FROM song_key_predictions").fetchone()[0])
    state = con.execute("SELECT last_source_row, skipped_no_chords FROM inference_state WHERE id = 1").fetchone()
    avg_conf, avg_entropy = con.execute(
        "SELECT AVG(calibrated_probability), AVG(normalized_entropy) FROM song_key_predictions"
    ).fetchone()

    thresholds = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        count = int(con.execute(
            "SELECT COUNT(*) FROM song_key_predictions WHERE calibrated_probability >= ?",
            [threshold],
        ).fetchone()[0])
        thresholds.append({
            "threshold": threshold,
            "count": count,
            "share_of_inferred_songs": (count / total if total else None),
        })

    key_rows = con.execute("""
        SELECT predicted_key, COUNT(*) AS n,
               AVG(calibrated_probability) AS mean_probability
        FROM song_key_predictions
        GROUP BY 1
        ORDER BY n DESC, predicted_key
    """).fetchall()

    summary = {
        "source_csv": source_csv,
        "model_path": model_path,
        "calibration_method": "temperature_scaling",
        "temperature": float(temperature),
        "songs_inferred": total,
        "last_source_row": int(state[0] if state else 0),
        "skipped_no_parseable_chords": int(state[1] if state else 0),
        "mean_top1_probability": float(avg_conf) if avg_conf is not None else None,
        "mean_normalized_entropy": float(avg_entropy) if avg_entropy is not None else None,
        "probability_threshold_counts": thresholds,
        "predicted_key_distribution": [
            {
                "key": str(key),
                "count": int(n),
                "mean_probability": float(mean_p),
            }
            for key, n, mean_p in key_rows
        ],
        "elapsed_seconds_this_run": round(float(elapsed), 3),
        "domain_shift_note": (
            "Probabilities were calibrated on ChoCo held-out data. They are model probabilities when applied to "
            "Chordonomicon, but calibration may shift because Chordonomicon is a different corpus/domain."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch-infer song-level major/minor keys for the full Chordonomicon CSV with the Step-3 model and Step-4 calibrator."
    )
    ap.add_argument("--csv", default="data/chordonomicon_v2.csv")
    ap.add_argument("--model", default="data/key_models/selected_model.joblib")
    ap.add_argument("--calibrator", default="data/key_models/temperature_calibrator.json")
    ap.add_argument("--output-db", default="data/key_models/chordonomicon_key_predictions.duckdb")
    ap.add_argument("--output-parquet", default="data/key_models/chordonomicon_song_keys.parquet")
    ap.add_argument("--summary", default="data/key_models/chordonomicon_song_keys_summary.json")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--max-songs", type=int, default=None, help="Smoke-test on only N new CSV rows.")
    ap.add_argument("--reset", action="store_true", help="Delete prior Step-5 output DB and start from source row 1.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    model_path = Path(args.model)
    calibrator_path = Path(args.calibrator)
    output_db = Path(args.output_db)
    output_parquet = Path(args.output_parquet)
    summary_path = Path(args.summary)

    for required in (csv_path, model_path, calibrator_path):
        if not required.exists():
            raise SystemExit(f"Missing required file: {required}")

    cal = json.loads(calibrator_path.read_text(encoding="utf-8"))
    temperature = float(cal["temperature"])
    model = joblib.load(model_path)

    output_db.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and output_db.exists():
        output_db.unlink()
    con = duckdb.connect(str(output_db))
    _ensure_output_schema(con)
    last_source_row = int(con.execute("SELECT last_source_row FROM inference_state WHERE id = 1").fetchone()[0])

    print(f"Resuming after source row {last_source_row:,}")
    print(f"Workers: {args.workers}; batch size: {args.batch_size}; temperature: {temperature:.6f}")
    start = time.perf_counter()
    total_new = 0
    skipped_new = 0

    row_iter = _iter_csv_rows(csv_path, start_after=last_source_row, max_songs=args.max_songs)

    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for raw_batch in _batched(row_iter, args.batch_size):
            if executor is None:
                extracted = [_feature_task(x) for x in raw_batch]
            else:
                extracted = list(executor.map(_feature_task, raw_batch, chunksize=max(1, len(raw_batch) // max(1, args.workers * 4))))

            valid = [x for x in extracted if x[4] is not None]
            skipped_delta = len(extracted) - len(valid)
            skipped_new += skipped_delta

            rows_to_write: list[tuple] = []
            if valid:
                source_rows = [x[0] for x in valid]
                song_ids = [x[1] for x in valid]
                n_chords = [x[2] for x in valid]
                n_sections = [x[3] for x in valid]
                X = np.concatenate([x[4] for x in valid], axis=0)
                raw_scores = np.asarray(model.predict_proba(X)[:, 1], dtype=float).reshape(len(valid), 24)
                probs = probabilities_from_candidate_scores(raw_scores, temperature=temperature)
                rows_to_write = _summaries_from_probabilities(source_rows, song_ids, n_chords, n_sections, probs)

            batch_last = int(raw_batch[-1][0])
            _write_rows(con, rows_to_write, batch_last, skipped_delta)
            total_new += len(rows_to_write)

            state = con.execute("SELECT songs_written FROM inference_state WHERE id = 1").fetchone()[0]
            elapsed = time.perf_counter() - start
            rate = (total_new + skipped_new) / elapsed if elapsed > 0 else 0.0
            print(
                f"source row {batch_last:,} | inferred total {int(state):,} | "
                f"new {total_new:,} | skipped {skipped_new:,} | {rate:,.1f} rows/s",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    # Export a portable Parquet snapshot from the separate inference DB. This does
    # not alter the main chordonomicon.duckdb explorer database.
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet_sql_path = str(output_parquet).replace("'", "''")
    con.execute(f"COPY (SELECT * FROM song_key_predictions ORDER BY source_row) TO '{parquet_sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    summary = _write_summary(
        con,
        summary_path,
        temperature=temperature,
        model_path=str(model_path),
        source_csv=str(csv_path),
        elapsed=time.perf_counter() - start,
    )
    con.close()

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {output_db}")
    print(f"Wrote: {output_parquet}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
