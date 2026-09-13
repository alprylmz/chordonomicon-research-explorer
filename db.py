from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import duckdb
import pandas as pd


PRESENCE_TABLES = ("progression_presence_counts", "transition_presence_counts")


def connect(db_path: str | Path):
    return duckdb.connect(str(db_path), read_only=True)


def table_exists(con, name: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchone()[0]
    )


def has_presence_metrics(con) -> bool:
    return all(table_exists(con, t) for t in PRESENCE_TABLES)


def _filters_sql(
    filters: Dict[str, object],
    params: List[object],
    *,
    presence: bool = False,
) -> str:
    clauses = []
    mapping = {
        "section_type": "section_type",
        "main_genre": "main_genre",
        "rock_genre": "rock_genre",
        "decade": "decade",
    }

    # Presence tables contain an explicit __ALL__ rollup. It is necessary for
    # exact distinct-song counts when a song uses the same pattern in multiple
    # section types.
    if presence:
        section = filters.get("section_type")
        if section in (None, "", "All"):
            clauses.append("section_type = '__ALL__'")
        else:
            clauses.append("section_type = ?")
            params.append(section)

    for key, column in mapping.items():
        if presence and key == "section_type":
            continue
        value = filters.get(key)
        if value not in (None, "", "All"):
            clauses.append(f"{column} = ?")
            params.append(value)
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def dimensions(con) -> Dict[str, List]:
    out = {}
    for col in ("section_type", "main_genre", "rock_genre", "decade"):
        rows = con.execute(
            f"SELECT DISTINCT {col} FROM dimensions "
            f"WHERE {col} IS NOT NULL AND CAST({col} AS VARCHAR) <> '' ORDER BY {col}"
        ).fetchall()
        out[col] = [r[0] for r in rows if r[0] != "__ALL__"]
    return out


def _transition_metric_spec(count_by: str) -> tuple[str, str, bool]:
    if count_by == "Occurrences":
        return "transition_counts", "occurrences", False
    if count_by == "Unique songs":
        return "transition_presence_counts", "unique_songs", True
    if count_by == "Unique song sections":
        return "transition_presence_counts", "unique_sections", True
    raise ValueError(f"Unknown count metric: {count_by}")


def next_chords(
    con,
    context_len: int,
    context_key: str,
    mode: str,
    filters: Dict[str, object],
    count_by: str = "Occurrences",
    limit: int = 20,
) -> pd.DataFrame:
    table, metric_col, presence = _transition_metric_spec(count_by)
    if presence and not has_presence_metrics(con):
        return pd.DataFrame(columns=["next_chord", "count", "probability"])

    params: List[object]
    if presence:
        # Presence tables are pre-deduplicated separately for Absolute and
        # Relative-root representation. This avoids double-counting one song
        # if it contains transposed versions of the same relative pattern.
        params = [mode, context_len, context_key]
        extra = _filters_sql(filters, params, presence=True)
        params.append(limit)
        q = f"""
            WITH grouped AS (
                SELECT next_key AS next_chord,
                       SUM({metric_col})::BIGINT AS count
                FROM {table}
                WHERE mode = ? AND context_len = ? AND context_key = ? {extra}
                GROUP BY 1
            ), scored AS (
                SELECT next_chord,
                       count,
                       count::DOUBLE / NULLIF(SUM(count) OVER (), 0) AS probability
                FROM grouped
            )
            SELECT next_chord, count, probability
            FROM scored
            ORDER BY count DESC, next_chord
            LIMIT ?
        """
    else:
        key_col = "context_abs" if mode == "Absolute" else "context_shape"
        next_col = "next_abs" if mode == "Absolute" else "next_rel"
        params = [context_len, context_key]
        extra = _filters_sql(filters, params, presence=False)
        params.append(limit)
        q = f"""
            WITH grouped AS (
                SELECT {next_col} AS next_chord,
                       SUM(occurrences)::BIGINT AS count
                FROM transition_counts
                WHERE context_len = ? AND {key_col} = ? {extra}
                GROUP BY 1
            ), scored AS (
                SELECT next_chord,
                       count,
                       count::DOUBLE / NULLIF(SUM(count) OVER (), 0) AS probability
                FROM grouped
            )
            SELECT next_chord, count, probability
            FROM scored
            ORDER BY count DESC, next_chord
            LIMIT ?
        """
    return con.execute(q, params).df()


def _progression_metric_spec(count_by: str) -> tuple[str, str, bool]:
    if count_by == "Occurrences":
        return "progression_counts", "occurrences", False
    if count_by == "Unique songs":
        return "progression_presence_counts", "unique_songs", True
    if count_by == "Unique song sections":
        return "progression_presence_counts", "unique_sections", True
    raise ValueError(f"Unknown count metric: {count_by}")


def _progression_examples(
    con,
    length: int,
    progression_key: str,
    mode: str,
    filters: Dict[str, object],
) -> tuple[list, str]:
    key_col = "progression_abs" if mode == "Absolute" else "progression_shape"
    params: List[object] = [length, progression_key]
    extra = _filters_sql(filters, params, presence=False)
    q = f"""
        SELECT
            list_distinct(flatten(list(example_song_ids))) AS example_song_ids,
            any_value(progression_abs) AS sample_absolute
        FROM progression_counts
        WHERE length = ? AND {key_col} = ? {extra}
    """
    row = con.execute(q, params).fetchone()
    if not row:
        return [], ""
    return (row[0] or []), (row[1] or "")


def progression_stats(
    con,
    length: int,
    progression_key: str,
    mode: str,
    filters: Dict[str, object],
    count_by: str = "Occurrences",
) -> pd.DataFrame:
    table, metric_col, presence = _progression_metric_spec(count_by)
    if presence and not has_presence_metrics(con):
        return pd.DataFrame()

    if presence:
        params: List[object] = [mode, length, progression_key]
        extra = _filters_sql(filters, params, presence=True)
        q = f"""
            SELECT
                SUM({metric_col})::BIGINT AS count,
                any_value(sample_absolute) AS sample_absolute
            FROM {table}
            WHERE mode = ? AND length = ? AND progression_key = ? {extra}
        """
    else:
        key_col = "progression_abs" if mode == "Absolute" else "progression_shape"
        params = [length, progression_key]
        extra = _filters_sql(filters, params, presence=False)
        q = f"""
            SELECT
                SUM(occurrences)::BIGINT AS count,
                any_value(progression_abs) AS sample_absolute
            FROM progression_counts
            WHERE length = ? AND {key_col} = ? {extra}
        """

    df = con.execute(q, params).df()
    if df.empty:
        return df

    ids, fallback_abs = _progression_examples(con, length, progression_key, mode, filters)
    df["example_song_ids"] = [ids]
    if not df.iloc[0].get("sample_absolute"):
        df.loc[df.index[0], "sample_absolute"] = fallback_abs
    return df


def top_progressions(
    con,
    length: int,
    mode: str,
    filters: Dict[str, object],
    count_by: str = "Occurrences",
    limit: int = 100,
) -> pd.DataFrame:
    table, metric_col, presence = _progression_metric_spec(count_by)
    if presence and not has_presence_metrics(con):
        return pd.DataFrame()

    if presence:
        params: List[object] = [mode, length]
        extra = _filters_sql(filters, params, presence=True)
        params.append(limit)
        q = f"""
            SELECT
                progression_key AS progression,
                SUM({metric_col})::BIGINT AS count,
                any_value(sample_absolute) AS sample_absolute
            FROM {table}
            WHERE mode = ? AND length = ? {extra}
            GROUP BY 1
            ORDER BY count DESC, progression
            LIMIT ?
        """
    else:
        key_col = "progression_abs" if mode == "Absolute" else "progression_shape"
        params = [length]
        extra = _filters_sql(filters, params, presence=False)
        params.append(limit)
        q = f"""
            SELECT
                {key_col} AS progression,
                SUM(occurrences)::BIGINT AS count,
                any_value(progression_abs) AS sample_absolute
            FROM progression_counts
            WHERE length = ? {extra}
            GROUP BY 1
            ORDER BY count DESC, progression
            LIMIT ?
        """
    return con.execute(q, params).df()


def songs_by_ids(con, song_ids: Sequence[str], limit: int = 12) -> pd.DataFrame:
    ids = [str(x) for x in song_ids if x]
    if not ids:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in ids[:limit])
    q = f"""
        SELECT song_id, release_date, decade, main_genre, rock_genre, genres,
               spotify_song_id, spotify_artist_id
        FROM songs
        WHERE song_id IN ({placeholders})
        LIMIT {int(limit)}
    """
    df = con.execute(q, ids[:limit]).df()
    if "genres" in df.columns:
        df["genres"] = df["genres"].map(lambda x: ", ".join(json.loads(x)) if x else "")
    return df



def song_key_predictions_by_ids(
    con,
    parquet_path: str | Path,
    song_ids: Sequence[str],
    limit: int = 12,
) -> pd.DataFrame:
    """Return precomputed balanced key predictions for selected songs.

    The Parquet file is intentionally kept separate from the main explorer DB so
    Step-7 can be installed without rebuilding the very large aggregate tables.
    DuckDB applies the song_id filter directly while scanning the Parquet file.
    """
    ids = [str(x) for x in song_ids if x][:limit]
    path = Path(parquet_path)
    if not ids or not path.exists():
        return pd.DataFrame()

    escaped_path = str(path).replace("'", "''")
    placeholders = ",".join("?" for _ in ids)
    q = f"""
        SELECT song_id, predicted_key, calibrated_probability,
               second_key, second_probability, probability_margin,
               normalized_entropy
        FROM read_parquet('{escaped_path}')
        WHERE song_id IN ({placeholders})
    """
    return con.execute(q, ids).df()

def build_metadata(con) -> Dict:
    row = con.execute("SELECT json FROM build_metadata LIMIT 1").fetchone()
    return json.loads(row[0]) if row else {}
