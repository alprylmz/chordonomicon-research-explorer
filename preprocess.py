from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

from chord_utils import absolute_progression, parse_genres, parse_sections, relative_shape, relative_token


PROG_COLUMNS = [
    "length", "section_type", "main_genre", "rock_genre", "decade",
    "progression_abs", "progression_shape", "occurrences", "example_song_id",
]
TRANS_COLUMNS = [
    "context_len", "section_type", "main_genre", "rock_genre", "decade",
    "context_abs", "context_shape", "next_abs", "next_rel", "occurrences", "example_song_id",
]
SONG_COLUMNS = [
    "song_id", "release_date", "decade", "main_genre", "rock_genre", "genres",
    "artist_id", "spotify_song_id", "spotify_artist_id",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def clean_decade(value: Any) -> Optional[int]:
    s = clean_text(value)
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("PRAGMA threads=4")
    con.execute("CREATE TABLE IF NOT EXISTS songs (song_id VARCHAR, release_date VARCHAR, decade INTEGER, main_genre VARCHAR, rock_genre VARCHAR, genres VARCHAR, artist_id VARCHAR, spotify_song_id VARCHAR, spotify_artist_id VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS progression_partial (length UTINYINT, section_type VARCHAR, main_genre VARCHAR, rock_genre VARCHAR, decade INTEGER, progression_abs VARCHAR, progression_shape VARCHAR, occurrences UBIGINT, example_song_id VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS transition_partial (context_len UTINYINT, section_type VARCHAR, main_genre VARCHAR, rock_genre VARCHAR, decade INTEGER, context_abs VARCHAR, context_shape VARCHAR, next_abs VARCHAR, next_rel VARCHAR, occurrences UBIGINT, example_song_id VARCHAR)")


def append_df(con: duckdb.DuckDBPyConnection, table: str, rows: List[Tuple], columns: Sequence[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    con.register("_batch_df", df)
    con.execute(f"INSERT INTO {table} SELECT * FROM _batch_df")
    con.unregister("_batch_df")


def flush_counters(
    con: duckdb.DuckDBPyConnection,
    prog_counts: Dict[Tuple, List[Any]],
    trans_counts: Dict[Tuple, List[Any]],
) -> None:
    prog_rows = [(*key, value[0], value[1]) for key, value in prog_counts.items()]
    trans_rows = [(*key, value[0], value[1]) for key, value in trans_counts.items()]
    append_df(con, "progression_partial", prog_rows, PROG_COLUMNS)
    append_df(con, "transition_partial", trans_rows, TRANS_COLUMNS)
    prog_counts.clear()
    trans_counts.clear()


def add_count(store: Dict[Tuple, List[Any]], key: Tuple, song_id: str) -> None:
    if key in store:
        store[key][0] += 1
    else:
        store[key] = [1, song_id]


def process_csv(
    csv_path: Path,
    db_path: Path,
    flush_every: int,
    min_len: int,
    max_len: int,
    max_rows: Optional[int],
) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    prog_counts: Dict[Tuple, List[Any]] = {}
    trans_counts: Dict[Tuple, List[Any]] = {}
    song_rows: List[Tuple] = []

    total_rows = 0
    parsed_sections = 0
    parsed_chords = 0
    skipped_short_sections = 0
    start = time.perf_counter()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "chords"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            if max_rows is not None and total_rows > max_rows:
                break

            song_id = clean_text(row.get("id")) or str(total_rows)
            decade = clean_decade(row.get("decade"))
            main_genre = clean_text(row.get("main_genre")).lower()
            rock_genre = clean_text(row.get("rock_genre")).lower()
            genres = parse_genres(clean_text(row.get("genres")))

            song_rows.append((
                song_id,
                clean_text(row.get("release_date")),
                decade,
                main_genre,
                rock_genre,
                json.dumps(genres, ensure_ascii=False),
                clean_text(row.get("artist_id")),
                clean_text(row.get("spotify_song_id")),
                clean_text(row.get("spotify_artist_id")),
            ))

            sections = parse_sections(clean_text(row.get("chords")))
            for sec_type, _sec_label, chords in sections:
                parsed_sections += 1
                parsed_chords += len(chords)
                if len(chords) < min_len:
                    skipped_short_sections += 1
                    continue

                # Sliding windows never cross a section boundary.
                for length in range(min_len, max_len + 1):
                    if len(chords) < length:
                        continue
                    for i in range(0, len(chords) - length + 1):
                        window = chords[i:i + length]
                        abs_prog = absolute_progression(window)
                        shape = relative_shape(window)
                        pkey = (
                            length, sec_type, main_genre, rock_genre, decade,
                            abs_prog, shape,
                        )
                        add_count(prog_counts, pkey, song_id)

                # Context -> next chord tables for interactive exploration.
                max_context = min(max_len - 1, len(chords) - 1)
                for context_len in range(1, max_context + 1):
                    for i in range(0, len(chords) - context_len):
                        context = chords[i:i + context_len]
                        nxt = chords[i + context_len]
                        anchor = context[0].root
                        tkey = (
                            context_len, sec_type, main_genre, rock_genre, decade,
                            absolute_progression(context),
                            relative_shape(context),
                            nxt.absolute_token,
                            relative_token(nxt, anchor),
                        )
                        add_count(trans_counts, tkey, song_id)

            if total_rows % flush_every == 0:
                append_df(con, "songs", song_rows, SONG_COLUMNS)
                song_rows.clear()
                flush_counters(con, prog_counts, trans_counts)
                print(f"processed {total_rows:,} songs", flush=True)

    append_df(con, "songs", song_rows, SONG_COLUMNS)
    flush_counters(con, prog_counts, trans_counts)

    print("aggregating progression counts...", flush=True)
    con.execute("DROP TABLE IF EXISTS progression_counts")
    con.execute("""
        CREATE TABLE progression_counts AS
        SELECT
            length, section_type, main_genre, rock_genre, decade,
            progression_abs, progression_shape,
            SUM(occurrences)::UBIGINT AS occurrences,
            COUNT(DISTINCT example_song_id)::UBIGINT AS sample_song_chunks,
            list_slice(list(DISTINCT example_song_id), 1, 8) AS example_song_ids
        FROM progression_partial
        GROUP BY ALL
    """)

    print("aggregating transition counts...", flush=True)
    con.execute("DROP TABLE IF EXISTS transition_counts")
    con.execute("""
        CREATE TABLE transition_counts AS
        SELECT
            context_len, section_type, main_genre, rock_genre, decade,
            context_abs, context_shape, next_abs, next_rel,
            SUM(occurrences)::UBIGINT AS occurrences,
            list_slice(list(DISTINCT example_song_id), 1, 8) AS example_song_ids
        FROM transition_partial
        GROUP BY ALL
    """)

    con.execute("DROP TABLE progression_partial")
    con.execute("DROP TABLE transition_partial")

    # Small dimension tables make Streamlit filters fast.
    con.execute("CREATE TABLE dimensions AS SELECT DISTINCT section_type, main_genre, rock_genre, decade FROM progression_counts")

    con.execute("CREATE INDEX IF NOT EXISTS idx_prog_shape ON progression_counts(length, progression_shape)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_prog_abs ON progression_counts(length, progression_abs)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trans_shape ON transition_counts(context_len, context_shape)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trans_abs ON transition_counts(context_len, context_abs)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_song_id ON songs(song_id)")

    stats = {
        "source_csv": str(csv_path),
        "songs_processed": total_rows,
        "sections_parsed": parsed_sections,
        "parsed_chords": parsed_chords,
        "skipped_short_sections": skipped_short_sections,
        "min_window": min_len,
        "max_window": max_len,
        "progression_rows": con.execute("SELECT COUNT(*) FROM progression_counts").fetchone()[0],
        "transition_rows": con.execute("SELECT COUNT(*) FROM transition_counts").fetchone()[0],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    con.execute("CREATE TABLE build_metadata AS SELECT ? AS json", [json.dumps(stats)])
    con.close()

    print(json.dumps(stats, indent=2))
    print(f"Wrote {db_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the Chordonomicon Research Explorer DuckDB index.")
    p.add_argument("--csv", default="data/chordonomicon_v2.csv")
    p.add_argument("--db", default="data/chordonomicon.duckdb")
    p.add_argument("--flush-every", type=int, default=2500)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--max-len", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=None, help="Useful for a quick smoke test.")
    args = p.parse_args()

    process_csv(
        csv_path=Path(args.csv),
        db_path=Path(args.db),
        flush_every=args.flush_every,
        min_len=args.min_len,
        max_len=args.max_len,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
