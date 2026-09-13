from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chord_utils import (
    absolute_progression,
    parse_progression,
    pretty_relative_shape,
    pretty_relative_token,
    relative_shape,
    relative_to_absolute,
    relative_token,
)
from db import (
    build_metadata,
    connect,
    dimensions,
    has_presence_metrics,
    next_chords,
    progression_stats,
    song_key_predictions_by_ids,
    songs_by_ids,
    top_progressions,
)
from key_inference import roman_numeral
from key_modeling.live_key_inference import KeyPredictor, LiveKeyAnalysis, artifact_stamp

DB_PATH = Path("data/chordonomicon.duckdb")
PRESENCE_STATS_PATH = Path("data/presence_build_stats.json")
KEY_MODEL_PATH = Path("data/key_models/mode_balance_experiment/selected_mode_balanced_histgb.joblib")
KEY_CALIBRATOR_PATH = Path("data/key_models/mode_balance_experiment/mode_balanced_temperature_calibrator.json")
KEY_PREDICTIONS_PATH = Path("data/key_models/chordonomicon_song_keys_mode_balanced.parquet")
KEY_SUMMARY_PATH = Path("data/key_models/chordonomicon_song_keys_mode_balanced_summary.json")

st.set_page_config(
    page_title="Chordonomicon Harmony Explorer",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1420px;
        padding-top: 1.7rem;
        padding-bottom: 4rem;
    }

    div[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.16);
    }

    .ce-eyebrow {
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        opacity: 0.58;
        margin-bottom: 0.35rem;
    }

    .ce-title {
        font-size: clamp(2.15rem, 4vw, 3.6rem);
        line-height: 0.98;
        letter-spacing: -0.045em;
        font-weight: 820;
        margin: 0 0 0.7rem 0;
    }

    .ce-subtitle {
        max-width: 900px;
        font-size: 1.02rem;
        line-height: 1.55;
        opacity: 0.70;
        margin-bottom: 0.9rem;
    }

    .ce-chip-row {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin: 0.25rem 0 1.2rem 0;
    }

    .ce-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.34rem 0.66rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 650;
        background: rgba(128, 128, 128, 0.06);
    }

    .ce-section-title {
        font-size: 1.48rem;
        letter-spacing: -0.025em;
        font-weight: 780;
        margin: 1.65rem 0 0.18rem 0;
    }

    .ce-section-note {
        opacity: 0.63;
        font-size: 0.88rem;
        margin-bottom: 0.85rem;
    }

    .ce-chord-strip {
        display: flex;
        align-items: center;
        gap: 0.48rem;
        overflow-x: auto;
        padding: 0.95rem 0.1rem 1.1rem 0.1rem;
        scrollbar-width: thin;
    }

    .ce-chord-card {
        min-width: 118px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 15px;
        padding: 0.82rem 0.88rem 0.76rem 0.88rem;
        background: rgba(128, 128, 128, 0.055);
        box-shadow: 0 1px 0 rgba(255, 255, 255, 0.03) inset;
    }

    .ce-chord-abs {
        font-size: 1.42rem;
        font-weight: 820;
        letter-spacing: -0.035em;
        line-height: 1.1;
    }

    .ce-chord-rel {
        font-size: 0.79rem;
        font-weight: 700;
        opacity: 0.62;
        margin-top: 0.38rem;
    }

    .ce-chord-roman {
        font-size: 0.92rem;
        font-weight: 820;
        margin-top: 0.22rem;
        letter-spacing: -0.01em;
    }

    .ce-tonal-panel {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 17px;
        padding: 1rem 1.05rem;
        background: rgba(128, 128, 128, 0.045);
        margin: 0.25rem 0 1.15rem 0;
    }

    .ce-tonal-key {
        font-size: 2rem;
        font-weight: 840;
        letter-spacing: -0.045em;
        line-height: 1.05;
    }

    .ce-tonal-prob {
        font-size: 1.25rem;
        font-weight: 820;
        opacity: 0.82;
        margin-top: 0.25rem;
    }

    .ce-tonal-roman {
        font-size: 1.25rem;
        font-weight: 760;
        letter-spacing: -0.025em;
        margin-top: 0.55rem;
        overflow-x: auto;
        white-space: nowrap;
    }

    .ce-arrow {
        font-size: 1.08rem;
        opacity: 0.36;
        flex: 0 0 auto;
    }

    .ce-method-note {
        border-left: 3px solid rgba(128, 128, 128, 0.38);
        padding: 0.12rem 0 0.12rem 0.75rem;
        opacity: 0.67;
        font-size: 0.83rem;
        margin-top: 0.35rem;
    }

    .ce-metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.72rem;
        margin: 0.5rem 0 0.9rem 0;
    }

    .ce-metric-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 14px;
        padding: 0.78rem 0.88rem;
        background: rgba(128, 128, 128, 0.045);
        min-height: 92px;
    }

    .ce-metric-card.selected {
        border-width: 1.5px;
        background: rgba(128, 128, 128, 0.085);
    }

    .ce-metric-label {
        font-size: 0.75rem;
        font-weight: 760;
        letter-spacing: 0.025em;
        opacity: 0.61;
    }

    .ce-metric-value {
        font-size: 1.55rem;
        font-weight: 830;
        letter-spacing: -0.035em;
        line-height: 1.2;
        margin-top: 0.15rem;
    }

    .ce-metric-sub {
        font-size: 0.75rem;
        opacity: 0.57;
        margin-top: 0.15rem;
    }

    .ce-leader {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        background: rgba(128, 128, 128, 0.045);
        margin-top: 0.2rem;
    }

    .ce-leader-chord {
        font-size: 2rem;
        font-weight: 840;
        letter-spacing: -0.045em;
        margin-top: 0.25rem;
        line-height: 1.05;
    }

    .ce-leader-prob {
        font-size: 2.2rem;
        font-weight: 840;
        letter-spacing: -0.05em;
        margin-top: 0.65rem;
    }

    .ce-divider {
        height: 1px;
        background: rgba(128, 128, 128, 0.16);
        margin: 1.25rem 0 0.8rem 0;
    }

    @media (max-width: 900px) {
        .ce-metric-grid { grid-template-columns: 1fr; }
        .ce-chord-card { min-width: 104px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Safe cached DB access. Each cached function opens its own read-only connection.
# This keeps Streamlit reruns from sharing a live DuckDB result object.
# -----------------------------------------------------------------------------


def db_stamp() -> Tuple[int, int]:
    if not DB_PATH.exists():
        return (0, 0)
    stat = DB_PATH.stat()
    return (stat.st_size, stat.st_mtime_ns)


@st.cache_data(show_spinner=False)
def cached_dimensions(_stamp: Tuple[int, int]) -> Dict[str, List]:
    con = connect(DB_PATH)
    try:
        return dimensions(con)
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def cached_metadata(_stamp: Tuple[int, int]) -> Dict:
    con = connect(DB_PATH)
    try:
        return build_metadata(con)
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def cached_presence_ready(_stamp: Tuple[int, int]) -> bool:
    con = connect(DB_PATH)
    try:
        return has_presence_metrics(con)
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def cached_next_chords(
    _stamp: Tuple[int, int],
    context_len: int,
    context_key: str,
    mode: str,
    filter_items: Tuple[Tuple[str, object], ...],
    count_by: str,
    limit: int,
) -> pd.DataFrame:
    con = connect(DB_PATH)
    try:
        return next_chords(
            con,
            context_len,
            context_key,
            mode,
            dict(filter_items),
            count_by=count_by,
            limit=limit,
        )
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def cached_progression_stats(
    _stamp: Tuple[int, int],
    length: int,
    progression_key: str,
    mode: str,
    filter_items: Tuple[Tuple[str, object], ...],
    count_by: str,
) -> pd.DataFrame:
    con = connect(DB_PATH)
    try:
        return progression_stats(
            con,
            length,
            progression_key,
            mode,
            dict(filter_items),
            count_by=count_by,
        )
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def cached_top_progressions(
    _stamp: Tuple[int, int],
    length: int,
    mode: str,
    filter_items: Tuple[Tuple[str, object], ...],
    count_by: str,
    limit: int,
) -> pd.DataFrame:
    con = connect(DB_PATH)
    try:
        return top_progressions(
            con,
            length,
            mode,
            dict(filter_items),
            count_by=count_by,
            limit=limit,
        )
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def cached_song_examples(
    _stamp: Tuple[int, int],
    _key_stamp: Tuple[int, int],
    song_ids: Tuple[str, ...],
) -> pd.DataFrame:
    con = connect(DB_PATH)
    try:
        base = songs_by_ids(con, song_ids)
        if base.empty or not KEY_PREDICTIONS_PATH.exists():
            return base
        keys = song_key_predictions_by_ids(con, KEY_PREDICTIONS_PATH, song_ids)
        if keys.empty:
            return base
        out = base.merge(keys, on="song_id", how="left")
        if "calibrated_probability" in out.columns:
            out = out.rename(columns={
                "predicted_key": "estimated_key",
                "calibrated_probability": "key_confidence",
                "second_key": "second_key_candidate",
                "second_probability": "second_key_confidence",
            })
        return out
    finally:
        con.close()


def simple_file_stamp(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


@st.cache_resource(show_spinner=False)
def cached_key_predictor(_artifact_stamp) -> KeyPredictor:
    return KeyPredictor(KEY_MODEL_PATH, KEY_CALIBRATOR_PATH)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def filter_items(filters: Dict[str, object]) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(filters.items()))


def compact_number(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    n = float(value)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n):,}"


def load_presence_stats() -> Dict:
    if not PRESENCE_STATS_PATH.exists():
        return {}
    try:
        return json.loads(PRESENCE_STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def active_filter_summary(filters: Dict[str, object]) -> List[str]:
    labels = {
        "section_type": "section",
        "main_genre": "genre",
        "rock_genre": "subgenre",
        "decade": "decade",
    }
    out = []
    for key, label in labels.items():
        value = filters.get(key)
        if value not in (None, "", "All"):
            out.append(f"{label}: {value}")
    return out or ["all corpus"]


def filter_controls(dims: Dict[str, List]) -> Dict[str, object]:
    st.sidebar.markdown("### Corpus filters")
    st.sidebar.caption("Narrow the corpus before comparing harmonic patterns.")

    section = st.sidebar.selectbox(
        "Section",
        ["All"] + dims.get("section_type", []),
        key="filter_section",
    )
    main_genre = st.sidebar.selectbox(
        "Main genre",
        ["All"] + dims.get("main_genre", []),
        key="filter_main_genre",
    )
    rock_genre = st.sidebar.selectbox(
        "Rock / subgenre",
        ["All"] + dims.get("rock_genre", []),
        key="filter_rock_genre",
    )
    decade = st.sidebar.selectbox(
        "Decade",
        ["All"] + dims.get("decade", []),
        key="filter_decade",
    )

    if st.sidebar.button("Clear filters", use_container_width=True):
        st.session_state.filter_section = "All"
        st.session_state.filter_main_genre = "All"
        st.session_state.filter_rock_genre = "All"
        st.session_state.filter_decade = "All"
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Relative-root mode only groups transpositions. Estimated key and Roman numerals come from the separate balanced key model."
    )

    return {
        "section_type": section,
        "main_genre": main_genre,
        "rock_genre": rock_genre,
        "decade": decade,
    }


def init_builder() -> None:
    if "builder" not in st.session_state:
        st.session_state.builder = parse_progression("Am F C G")
    if "view" not in st.session_state:
        st.session_state.view = "Explorer"


def load_builder(text: str) -> None:
    st.session_state.builder = parse_progression(text)


def relative_label_for_chord(chord, anchor_root: int) -> str:
    return pretty_relative_token(relative_token(chord, anchor_root))


def render_chord_strip(chords, roman_labels: Tuple[str, ...] | None = None) -> None:
    if not chords:
        return
    anchor = chords[0].root
    pieces = []
    for i, chord in enumerate(chords):
        if i:
            pieces.append('<div class="ce-arrow">→</div>')
        abs_label = html.escape(chord.absolute_token)
        rel_label = html.escape(relative_label_for_chord(chord, anchor))
        roman_html = ""
        if roman_labels and i < len(roman_labels):
            roman_html = f'<div class="ce-chord-roman">{html.escape(roman_labels[i])}</div>'
        pieces.append(
            f'<div class="ce-chord-card">'
            f'<div class="ce-chord-abs">{abs_label}</div>'
            f'<div class="ce-chord-rel">relative-root {rel_label}</div>'
            f'{roman_html}'
            f'</div>'
        )
    st.markdown(
        '<div class="ce-chord-strip">' + "".join(pieces) + "</div>",
        unsafe_allow_html=True,
    )
    if roman_labels:
        st.markdown(
            '<div class="ce-method-note">Relative-root is still first-chord transposition normalization. Bold Roman numerals are a separate tonal reading under the model-estimated key.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ce-method-note">Relative-root labels are interval labels anchored on the first chord, not inferred tonal functions.</div>',
            unsafe_allow_html=True,
        )


def render_tonal_analysis(analysis: LiveKeyAnalysis) -> None:
    top = analysis.top
    second = analysis.second
    second_text = (
        f"Second candidate: {html.escape(second.name)} · {second.probability:.1%}"
        if second is not None
        else "No second candidate"
    )
    tonal_html = (
        f'<div class="ce-tonal-panel">'
        f'<div class="ce-eyebrow">Estimated tonal center · balanced HistGB v1</div>'
        f'<div class="ce-tonal-key">{html.escape(top.name)}</div>'
        f'<div class="ce-tonal-prob">{top.probability:.1%} model confidence</div>'
        f'<div style="opacity:.65;font-size:.82rem;margin-top:.28rem">{second_text} · top-two margin {analysis.probability_margin:.1%} · normalized entropy {analysis.normalized_entropy:.3f}</div>'
        f'<div class="ce-tonal-roman">{html.escape(analysis.roman_progression)}</div>'
        f'</div>'
    )
    st.markdown(tonal_html, unsafe_allow_html=True)
    st.caption(
        "Roman numerals are the harmonic reading under the selected estimated key. "
        "Confidence was calibrated on held-out ChoCo data; it is not a verified correctness probability for arbitrary user input or Chordonomicon. "
        "Slash-bass inversion figures are not encoded yet."
    )


def candidate_roman_label(raw: str, mode: str, chords, analysis: LiveKeyAnalysis | None) -> str:
    if analysis is None or not chords:
        return ""
    try:
        if mode == "Absolute":
            candidate = parse_progression(raw)[0]
        else:
            candidate = relative_to_absolute(raw, chords[0].root)
        return roman_numeral(candidate, analysis.top.tonic, analysis.top.mode)
    except Exception:
        return ""


def candidate_labels(raw: str, mode: str, chords) -> Tuple[str, str, str]:
    """Return absolute label, relative label and a combined display label."""
    anchor = chords[0].root
    if mode == "Relative-root":
        rel = pretty_relative_token(raw)
        abs_label = relative_to_absolute(raw, anchor).absolute_token
    else:
        abs_label = raw
        parsed = parse_progression(raw)
        rel = relative_label_for_chord(parsed[0], anchor) if parsed else ""
    return abs_label, rel, f"{abs_label}  ·  {rel}"


def next_probability_bars(df: pd.DataFrame, mode: str, chords, count_label: str, tonal_analysis: LiveKeyAnalysis | None = None):
    if df.empty:
        return None
    top = df.head(9).copy()
    labels = []
    for raw in top["next_chord"]:
        base = candidate_labels(raw, mode, chords)[2]
        roman = candidate_roman_label(str(raw), mode, chords, tonal_analysis)
        labels.append(f"{base}  ·  {roman}" if roman else base)
    top["display"] = labels
    top = top.iloc[::-1]

    max_prob = float(top["probability"].max()) if not top.empty else 1.0
    xmax = min(1.0, max(0.10, max_prob * 1.14))

    fig = go.Figure(
        go.Bar(
            x=top["probability"],
            y=top["display"],
            orientation="h",
            text=[f"{p:.1%}" for p in top["probability"]],
            textposition="outside",
            cliponaxis=False,
            customdata=top["count"].astype(int),
            hovertemplate=(
                "%{y}<br>Share: %{x:.2%}<br>"
                + count_label
                + ": %{customdata:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(330, 43 * len(top) + 85),
        margin=dict(l=8, r=74, t=10, b=42),
        xaxis=dict(
            title="Share of observed continuations",
            tickformat=".0%",
            range=[0, xmax],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.12)",
            zeroline=False,
        ),
        yaxis=dict(title=None, automargin=True),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.34,
    )
    return fig



def render_metric_cards(cards: Iterable[Tuple[str, str, str, bool]]) -> None:
    'Render metric cards with native Streamlit containers.'
    cards = list(cards)
    cols = st.columns(len(cards), gap="small") if cards else []
    for col, (label, value, sub, selected) in zip(cols, cards):
        with col:
            with st.container(border=True):
                marker = "● " if selected else ""
                st.caption(f"{marker}{label}")
                st.markdown(f"### {value}")
                st.caption(sub)


def probability_card_style(probability: float) -> tuple[str, str, str, float]:
    """Return a high-contrast probability treatment for a continuation card.

    Probability is encoded redundantly in three ways: hue, background intensity,
    and bar length. The piecewise hue scale deliberately exaggerates the visual
    distance between a dominant continuation (for example 70%) and a plausible
    alternative (for example 10%) while keeping the numeric percentage visible.
    """
    p = max(0.0, min(1.0, float(probability)))

    if p >= 0.50:
        # Dominant continuation: vivid emerald.
        rgb = (16, 185, 129)
        bg_alpha = 0.30
        border_alpha = 0.96
    elif p >= 0.20:
        # Strong alternative: cyan.
        rgb = (6, 182, 212)
        bg_alpha = 0.22
        border_alpha = 0.88
    elif p >= 0.05:
        # Plausible but clearly secondary: amber.
        rgb = (245, 158, 11)
        bg_alpha = 0.13
        border_alpha = 0.78
    else:
        # Rare continuation: subdued slate.
        rgb = (100, 116, 139)
        bg_alpha = 0.055
        border_alpha = 0.46

    r, g, b = rgb
    background = f"rgba({r}, {g}, {b}, {bg_alpha:.3f})"
    border = f"rgba({r}, {g}, {b}, {border_alpha:.3f})"
    accent = f"rgb({r}, {g}, {b})"
    return background, border, accent, p


def render_continuation_cards(nxt: pd.DataFrame, mode: str, chords, count_by: str, tonal_analysis: LiveKeyAnalysis | None = None) -> None:
    'Interactive next-chord cards, colored by empirical continuation share.'
    if nxt.empty:
        return

    top = nxt.head(9).reset_index(drop=True)
    st.markdown('<div class="ce-section-title">Continue the progression</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ce-section-note">Hue, color intensity, and bar length all encode continuation probability. Green marks a dominant continuation; amber marks a secondary alternative; muted cards are rare. Click Add to extend the progression.</div>',
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(top), 3):
        cols = st.columns(3, gap="medium")
        for offset, col in enumerate(cols):
            idx = row_start + offset
            if idx >= len(top):
                continue
            row = top.iloc[idx]
            raw = str(row["next_chord"])
            abs_label, rel_label, _ = candidate_labels(raw, mode, chords)
            roman_label = candidate_roman_label(raw, mode, chords, tonal_analysis)
            probability = float(row["probability"])
            support = int(row["count"])
            background, border, accent, absolute_fill = probability_card_style(probability)
            fill_width = max(1.5, absolute_fill * 100.0)

            with col:
                roman_html = (
                    f'<div style="font-size:.88rem;font-weight:820;margin-top:.18rem">{html.escape(roman_label)}</div>'
                    if roman_label else ''
                )
                card_html = (
                    f'<div style="border:1px solid {border};background:{background};border-radius:15px;padding:0.85rem 0.9rem 0.72rem 0.9rem;min-height:128px;margin-bottom:0.38rem;">'
                    f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:.75rem">'
                    f'<div>'
                    f'<div style="font-size:1.42rem;font-weight:820;letter-spacing:-.035em;line-height:1.05">{html.escape(abs_label)}</div>'
                    f'<div style="font-size:.77rem;font-weight:700;opacity:.62;margin-top:.28rem">relative-root {html.escape(rel_label)}</div>'
                    f'{roman_html}'
                    f'</div>'
                    f'<div style="font-size:1.34rem;font-weight:840;letter-spacing:-.035em">{probability:.1%}</div>'
                    f'</div>'
                    f'<div style="height:8px;background:rgba(128,128,128,.14);border-radius:99px;margin-top:.82rem;overflow:hidden">'
                    f'<div style="height:100%;width:{fill_width:.1f}%;background:{accent};border-radius:99px;box-shadow:0 0 10px {border}"></div>'
                    f'</div>'
                    f'<div style="font-size:.72rem;opacity:.56;margin-top:.42rem">{support:,} {html.escape(count_by.lower())}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button(
                    f"Add {abs_label}",
                    key=f"next_card_{mode}_{count_by}_{raw}_{idx}",
                    use_container_width=True,
                ):
                    if mode == "Absolute":
                        next_abs = parse_progression(raw)[0]
                    else:
                        next_abs = relative_to_absolute(raw, chords[0].root)
                    st.session_state.builder = chords + [next_abs]
                    st.rerun()


def find_candidate(df: pd.DataFrame, raw: str) -> Tuple[float | None, int | None]:
    if df.empty:
        return None, None
    match = df[df["next_chord"] == raw]
    if match.empty:
        return None, None
    row = match.iloc[0]
    return float(row["probability"]), int(row["count"])


def top_progression_chart(df: pd.DataFrame, label_col: str, count_col: str):
    top = df.head(15).copy().iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top[count_col],
            y=top[label_col],
            orientation="h",
            text=[compact_number(v) for v in top[count_col]],
            textposition="outside",
            cliponaxis=False,
            customdata=top["sample_absolute"],
            hovertemplate=(
                "%{y}<br>" + count_col + ": %{x:,}<br>Example: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(470, 38 * len(top) + 110),
        margin=dict(l=8, r=85, t=10, b=42),
        xaxis=dict(
            title=count_col,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.12)",
            zeroline=False,
        ),
        yaxis=dict(title=None, automargin=True),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.30,
    )
    return fig


# -----------------------------------------------------------------------------
# App bootstrap
# -----------------------------------------------------------------------------

if not DB_PATH.exists():
    st.error("Database not found. Expected `data/chordonomicon.duckdb`.")
    st.stop()

stamp = db_stamp()
dims = cached_dimensions(stamp)
meta = cached_metadata(stamp)
presence_ready = cached_presence_ready(stamp)
presence_stats = load_presence_stats()

init_builder()
filters = filter_controls(dims)
f_items = filter_items(filters)

songs_n = int(meta.get("songs_processed", 0) or 0)
filter_chips = active_filter_summary(filters)

st.markdown('<div class="ce-eyebrow">Chordonomicon · personal research tool</div>', unsafe_allow_html=True)
st.markdown('<div class="ce-title">Harmony Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ce-subtitle">Explore how chord sequences behave across genres, decades and song sections. Build a progression, estimate its tonal center, read Roman numerals, inspect corpus presence, and branch through empirically observed next chords.</div>',
    unsafe_allow_html=True,
)

chips = [f"{songs_n:,} songs" if songs_n else "Chordonomicon v2"] + filter_chips
if presence_ready:
    chips.append("song-level weighting ready")
key_stamp = artifact_stamp(KEY_MODEL_PATH, KEY_CALIBRATOR_PATH)
if key_stamp is not None:
    chips.append("balanced key model ready")
if KEY_PREDICTIONS_PATH.exists():
    chips.append("679k song keys ready")
st.markdown(
    '<div class="ce-chip-row">'
    + "".join(f'<span class="ce-chip">{html.escape(str(x))}</span>' for x in chips)
    + "</div>",
    unsafe_allow_html=True,
)

# Horizontal navigation. Unlike st.tabs, only the selected view executes its heavy queries.
# Streamlit does not allow changing a widget-backed session-state key after that
# widget has already been instantiated in the same run. Navigation requests from
# controls rendered later in the page are therefore queued under a separate key
# and applied here, before the radio widget is created.
_pending_view = st.session_state.pop("_pending_view", None)
if _pending_view in {"Explorer", "Top progressions", "Dataset"}:
    st.session_state["view"] = _pending_view

view = st.radio(
    "View",
    ["Explorer", "Top progressions", "Dataset"],
    horizontal=True,
    label_visibility="collapsed",
    key="view",
)

st.markdown('<div class="ce-divider"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Explorer
# -----------------------------------------------------------------------------

if view == "Explorer":
    control_left, control_right = st.columns([2.25, 1])

    with control_left:
        with st.form("progression_loader", border=False):
            input_col, load_col = st.columns([5, 1])
            input_text = input_col.text_input(
                "Progression",
                value=absolute_progression(st.session_state.builder),
                placeholder="Am F C G",
                label_visibility="collapsed",
            )
            submitted = load_col.form_submit_button("Load", use_container_width=True)
        if submitted:
            try:
                load_builder(input_text)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        b1, b2 = st.columns([1, 4])
        if b1.button("← Back", use_container_width=True, disabled=len(st.session_state.builder) <= 1):
            st.session_state.builder = st.session_state.builder[:-1]
            st.rerun()
        if b2.button("Reset progression", use_container_width=False):
            st.session_state.builder = parse_progression("Am F C G")
            st.rerun()

    with control_right:
        mode = st.radio(
            "Representation",
            ["Relative-root", "Absolute"],
            horizontal=True,
            help="Relative-root groups transpositions by anchoring the first chord at 0.",
        )

        count_options = ["Occurrences"]
        if presence_ready:
            count_options += ["Unique songs", "Unique song sections"]
        count_by = st.radio(
            "Counting basis",
            count_options,
            horizontal=False,
            help=(
                "Occurrences counts every observed window. Unique songs and unique song sections "
                "deduplicate repeated loops before aggregation."
            ),
        )

    chords = st.session_state.builder
    tonal_analysis: LiveKeyAnalysis | None = None
    if key_stamp is not None and len(chords) >= 4:
        try:
            predictor = cached_key_predictor(key_stamp)
            tonal_analysis = predictor.analyze(chords, top_k=3)
        except Exception as exc:
            st.warning(f"Key inference is temporarily unavailable: {exc}")

    render_chord_strip(chords, tonal_analysis.roman_numerals if tonal_analysis else None)

    if tonal_analysis is not None:
        render_tonal_analysis(tonal_analysis)
    elif key_stamp is not None and len(chords) < 4:
        st.caption("Add at least 4 chords to activate the estimated-key and Roman-numeral layer.")
    elif key_stamp is None:
        st.caption("Balanced key model artifacts are not installed yet, so Roman-numeral analysis is disabled.")

    # Next-chord exploration comes first so the page behaves like a composition tool.
    if len(chords) > 7:
        st.info("Next-chord branching is indexed for contexts up to 7 chords. Use Back to branch from the first 7 chords.")
    else:
        abs_key = absolute_progression(chords)
        shape_key = relative_shape(chords)
        context_key = abs_key if mode == "Absolute" else shape_key

        nxt = cached_next_chords(
            stamp,
            len(chords),
            context_key,
            mode,
            f_items,
            count_by,
            30,
        )

        if nxt.empty:
            st.info("No matching continuations under the current filters and counting basis.")
        else:
            # Put the actionable branching choices first. The chart below then
            # provides the analytical summary of the same distribution.
            render_continuation_cards(nxt, mode, chords, count_by, tonal_analysis)

            st.markdown('<div class="ce-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="ce-section-title">What comes next?</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ce-section-note">Observed continuations after this context, weighted by <b>{html.escape(count_by.lower())}</b>.</div>',
                unsafe_allow_html=True,
            )

            chart_col, insight_col = st.columns([2.2, 1], gap="large")
            with chart_col:
                fig = next_probability_bars(nxt, mode, chords, count_by, tonal_analysis)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            lead_raw = str(nxt.iloc[0]["next_chord"])
            lead_abs, lead_rel, _ = candidate_labels(lead_raw, mode, chords)
            lead_prob = float(nxt.iloc[0]["probability"])
            lead_count = int(nxt.iloc[0]["count"])
            lead_roman = candidate_roman_label(lead_raw, mode, chords, tonal_analysis)

            with insight_col:
                lead_roman_html = (
                    f'<div style="font-size:1rem;font-weight:820;margin-top:.18rem">{html.escape(lead_roman)}</div>'
                    if lead_roman else ''
                )
                leader_html = (
                    f'<div class="ce-leader">'
                    f'<div class="ce-eyebrow">Leading continuation</div>'
                    f'<div class="ce-leader-chord">{html.escape(lead_abs)}</div>'
                    f'<div style="opacity:.62;font-weight:700;margin-top:.25rem">relative-root {html.escape(lead_rel)}</div>'
                    f'{lead_roman_html}'
                    f'<div class="ce-leader-prob">{lead_prob:.1%}</div>'
                    f'<div style="opacity:.60;font-size:.82rem">{lead_count:,} · {html.escape(count_by.lower())}</div>'
                    f'</div>'
                )
                st.markdown(leader_html, unsafe_allow_html=True)

                if presence_ready:
                    st.markdown("**Same continuation, three weighting schemes**")
                    compare_cards = []
                    for metric in ["Occurrences", "Unique songs", "Unique song sections"]:
                        metric_df = (
                            nxt
                            if metric == count_by
                            else cached_next_chords(
                                stamp,
                                len(chords),
                                context_key,
                                mode,
                                f_items,
                                metric,
                                100,
                            )
                        )
                        prob, count = find_candidate(metric_df, lead_raw)
                        prob_text = f"{prob:.1%}" if prob is not None else "—"
                        count_text = f"{count:,} support" if count is not None else "not in top results"
                        compare_cards.append((metric, prob_text, count_text, metric == count_by))
                    render_metric_cards(compare_cards)
                    st.caption(
                        "Unique-song and unique-section percentages are normalized continuation support. "
                        "A song or section can support more than one continuation for the same context."
                    )

            with st.expander("Full continuation table"):
                detail = nxt.copy()
                detail["Chord"] = [candidate_labels(raw, mode, chords)[0] for raw in detail["next_chord"]]
                detail["Relative-root"] = [candidate_labels(raw, mode, chords)[1] for raw in detail["next_chord"]]
                if tonal_analysis is not None:
                    detail["Roman"] = [candidate_roman_label(str(raw), mode, chords, tonal_analysis) for raw in detail["next_chord"]]
                detail = detail.rename(columns={"count": count_by, "probability": "Share"})
                display_cols = ["Chord", "Relative-root"] + (["Roman"] if tonal_analysis is not None else []) + [count_by, "Share"]
                st.dataframe(
                    detail[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Share": st.column_config.ProgressColumn(
                            "Share", min_value=0.0, max_value=1.0, format="%.1%%"
                        )
                    },
                )

    # Corpus presence follows the compositional branching controls.
    if 3 <= len(chords) <= 8:
        st.markdown('<div class="ce-section-title">Corpus presence</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ce-section-note">How often this exact pattern appears under the current corpus filters.</div>',
            unsafe_allow_html=True,
        )
        pkey = absolute_progression(chords) if mode == "Absolute" else relative_shape(chords)
        metric_names = ["Occurrences"] + (["Unique songs", "Unique song sections"] if presence_ready else [])
        pstats: Dict[str, pd.DataFrame] = {}
        presence_cards = []
        for metric in metric_names:
            df = cached_progression_stats(stamp, len(chords), pkey, mode, f_items, metric)
            pstats[metric] = df
            if df.empty or pd.isna(df.iloc[0].get("count")):
                value = "0"
            else:
                value = f"{int(df.iloc[0]['count']):,}"
            sub = {
                "Occurrences": "all matching windows",
                "Unique songs": "songs containing the pattern",
                "Unique song sections": "section instances containing it",
            }[metric]
            presence_cards.append((metric, value, sub, metric == count_by))
        render_metric_cards(presence_cards)

        example_source = pstats.get(count_by)
        if example_source is not None and not example_source.empty:
            row = example_source.iloc[0]
            sample_abs = row.get("sample_absolute") or ""
            ids = row.get("example_song_ids")
            ids = ids if isinstance(ids, (list, tuple)) else []
            with st.expander("Pattern details and example song metadata"):
                if mode == "Relative-root" and sample_abs:
                    st.caption("Example absolute voicing")
                    st.code(sample_abs)
                if ids:
                    examples = cached_song_examples(stamp, simple_file_stamp(KEY_PREDICTIONS_PATH), tuple(str(x) for x in ids if x))
                    if not examples.empty:
                        st.dataframe(examples, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Top progressions
# -----------------------------------------------------------------------------

elif view == "Top progressions":
    st.markdown('<div class="ce-section-title">Top progressions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ce-section-note">Rank the most common 3–8 chord patterns under the current filters, then send one directly to the explorer.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1.2, 1.6])
    length = c1.slider("Progression length", 3, 8, 4)
    top_mode = c2.radio("Grouping", ["Relative-root", "Absolute"], horizontal=True)
    top_count_options = ["Occurrences"] + (["Unique songs", "Unique song sections"] if presence_ready else [])
    top_count = c3.radio("Counting basis", top_count_options, horizontal=True)

    top_df = cached_top_progressions(stamp, length, top_mode, f_items, top_count, 60)
    if top_df.empty:
        st.info("No progressions match the current filters.")
    else:
        display = top_df.copy()
        display["Pattern"] = (
            display["progression"].map(pretty_relative_shape)
            if top_mode == "Relative-root"
            else display["progression"]
        )
        display = display.rename(columns={"count": top_count})

        chart = top_progression_chart(display, "Pattern", top_count)
        st.plotly_chart(chart, use_container_width=True, theme="streamlit")

        st.markdown("**Open a pattern in the explorer**")
        options = [
            f"#{i + 1}  {row['Pattern']}  ·  {compact_number(row[top_count])}"
            for i, (_, row) in enumerate(display.head(40).iterrows())
        ]
        selected_idx = st.selectbox(
            "Pattern",
            range(len(options)),
            format_func=lambda idx: options[idx],
            label_visibility="collapsed",
        )
        if st.button("Open selected progression", type="primary"):
            chosen = display.iloc[int(selected_idx)]
            load_builder(str(chosen["sample_absolute"]))
            # Queue the navigation change. It will be applied at the top of the
            # next rerun, before the radio widget with key="view" is instantiated.
            st.session_state["_pending_view"] = "Explorer"
            st.rerun()

        with st.expander("Ranking table"):
            st.dataframe(
                display[["Pattern", top_count, "sample_absolute"]].head(60),
                use_container_width=True,
                hide_index=True,
            )

# -----------------------------------------------------------------------------
# Dataset / methods
# -----------------------------------------------------------------------------

else:
    st.markdown('<div class="ce-section-title">Dataset & methods</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ce-section-note">Build status, corpus scale and interpretation notes for this local research index.</div>',
        unsafe_allow_html=True,
    )

    section_count = presence_stats.get("sections_scanned")
    cards = [
        ("Songs indexed", f"{songs_n:,}" if songs_n else "—", "Chordonomicon song records", False),
        (
            "Parsed sections",
            f"{int(section_count):,}" if section_count is not None else "—",
            "from the later presence scan" if section_count is not None else "not retained by recovery metadata",
            False,
        ),
        (
            "Progression groups",
            compact_number(meta.get("progression_rows")),
            "aggregated 3–8 chord groups",
            False,
        ),
    ]
    render_metric_cards(cards)

    if presence_ready:
        st.success("Unique-song and unique-song-section weighting tables are installed and ready.")
        if presence_stats:
            p1, p2, p3 = st.columns(3)
            p1.metric("Presence progression rows", f"{int(presence_stats.get('progression_presence_rows', 0)):,}")
            p2.metric("Presence transition rows", f"{int(presence_stats.get('transition_presence_rows', 0)):,}")
            elapsed = float(presence_stats.get("elapsed_seconds", 0) or 0)
            p3.metric("Presence build time", f"{elapsed / 3600:.2f} h" if elapsed else "—")

    st.markdown("#### Key inference v1")
    if key_stamp is not None:
        key_cards = [
            ("Model", "Balanced HistGB", "true-mode inverse-frequency weighting · alpha 1.0", False),
            ("Calibration", "T = 1.3006", "temperature scaling fitted with grouped ChoCo OOF predictions", False),
            ("Song predictions", "ready" if KEY_PREDICTIONS_PATH.exists() else "missing", "balanced full-corpus Parquet snapshot", False),
        ]
        render_metric_cards(key_cards)
        if KEY_SUMMARY_PATH.exists():
            try:
                key_summary = json.loads(KEY_SUMMARY_PATH.read_text(encoding="utf-8"))
                st.caption(
                    f"Full Chordonomicon run: {int(key_summary.get('songs_inferred', 0)):,} songs · "
                    f"mean top-1 confidence {float(key_summary.get('mean_top1_probability', 0)):.1%} · "
                    f"temperature {float(key_summary.get('temperature', 0)):.4f}."
                )
            except Exception:
                pass
    else:
        st.warning("Balanced key model artifacts are missing; the tonal-analysis layer is unavailable.")

    st.markdown("#### Interpretation")
    st.markdown(
        """
        - **Relative-root remains transposition normalization.** The first chord is mapped to 0 and is not assumed to be the tonic.
        - **Roman numerals are now a separate model-based tonal reading.** They are computed only after the balanced key model selects an estimated major/minor key.
        - **Key confidence is ChoCo-calibrated, not Chordonomicon-calibrated.** Treat it as a model confidence under domain shift, not a literal verified probability of correctness for this corpus or arbitrary input.
        - **Occurrences** counts every indexed window or transition, so repeated loops can contribute repeatedly.
        - **Unique songs** gives a song at most one support count for a given pattern or continuation.
        - **Unique song sections** deduplicates within each parsed section instance, reducing repeated-loop dominance while preserving verse/chorus distinctions.
        - Chordonomicon is based on symbolic chord transcriptions, so the explorer studies the corpus representation rather than audio-level harmonic ground truth.
        """
    )

    with st.expander("Raw build metadata"):
        st.json(meta)
        if presence_stats:
            st.json(presence_stats)
