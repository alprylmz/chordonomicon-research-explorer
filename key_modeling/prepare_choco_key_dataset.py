from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# ChoCo v1.0.0: these three source collections are CC BY-NC-SA 4.0.
# Default behavior excludes them so the modeling dataset is built only from
# ChoCo's CC BY 4.0 subsets. --include-noncommercial opts back in for research.
NC_PARTITION_HINTS = {
    "chordify": "Chordify Annotator Subjectivity Dataset",
    "mozart": "Mozart Piano Sonata",
    "jaah": "Jazz Audio-Aligned Harmony",
}

ROOT_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PC_NAME = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)(?::\s*([A-Za-z_-]+))?\s*$")
HARTE_ROOT_RE = re.compile(r"^\s*([A-G])([#b]?)(?::([^/]+))?(?:/([^\s]+))?\s*$")
MEASURE_BEAT_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[:|,]\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class ParsedChord:
    root: int
    suffix: str
    token: str


@dataclass(frozen=True)
class TimedChord:
    chord: ParsedChord
    start: Optional[float]
    duration: Optional[float]


@dataclass(frozen=True)
class KeyObs:
    tonic: int
    mode: str
    label: str
    start: Optional[float]
    duration: Optional[float]


def accidental_offset(acc: str) -> int:
    return 1 if acc == "#" else -1 if acc == "b" else 0


def parse_key(value: Any) -> Optional[tuple[int, str, str]]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.upper() == "N":
        return None
    m = KEY_RE.match(value)
    if not m:
        return None
    letter, acc, raw_mode = m.groups()
    tonic = (ROOT_PC[letter.upper()] + accidental_offset(acc)) % 12
    if not raw_mode:
        return None  # 24-key target needs major/minor labels, not tonic only.
    mode = raw_mode.lower().replace("_", "-")
    if mode == "ionian":
        mode = "major"
    elif mode == "aeolian":
        mode = "minor"
    if mode not in {"major", "minor"}:
        return None
    label = f"{PC_NAME[tonic]} {mode}"
    return tonic, mode, label


def _harte_quality_to_suffix(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return ""
    low = q.lower()
    # Interval-set forms are intentionally broad. They still carry root evidence;
    # quality becomes 'other' in the transparent baseline rather than being guessed.
    if low.startswith("("):
        return low
    aliases = {
        "maj": "",
        "min": "m",
        "dim": "dim",
        "aug": "aug",
        "sus2": "sus2",
        "sus4": "sus4",
        "7": "7",
        "maj7": "maj7",
        "min7": "m7",
        "minmaj7": "mmaj7",
        "dim7": "dim7",
        "hdim7": "m7b5",
        "min6": "m6",
        "maj6": "6",
        "9": "9",
        "maj9": "maj9",
        "min9": "m9",
        "11": "11",
        "13": "13",
    }
    if low in aliases:
        return aliases[low]
    return low.replace("min", "m").replace("maj", "maj")


def parse_harte(value: Any) -> Optional[ParsedChord]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s.upper() in {"N", "NC", "X", "?"}:
        return None
    m = HARTE_ROOT_RE.match(s)
    if not m:
        return None
    letter, acc, quality, _bass = m.groups()
    root = (ROOT_PC[letter] + accidental_offset(acc)) % 12
    suffix = _harte_quality_to_suffix(quality or "")
    token = f"{PC_NAME[root]}{suffix}"
    return ParsedChord(root=root, suffix=suffix, token=token)


def time_scalar(value: Any) -> Optional[float]:
    """Best-effort monotonic scalar for seconds or measure:beat-like values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, str):
        try:
            x = float(value)
            return x if math.isfinite(x) else None
        except ValueError:
            pass
        m = MEASURE_BEAT_RE.match(value)
        if m:
            # Only order matters for segmentation fallback. A large base keeps
            # successive measures separated without assuming a time signature.
            return float(m.group(1)) * 1000.0 + float(m.group(2))
        return None
    if isinstance(value, dict):
        # Common extended-JAMS shapes. Prefer explicit measure/beat fields.
        keys = {str(k).lower(): v for k, v in value.items()}
        if "measure" in keys:
            meas = time_scalar(keys["measure"])
            beat = time_scalar(keys.get("beat", 0)) or 0.0
            if meas is not None:
                return meas * 1000.0 + beat
        if "value" in keys and len(keys) <= 4:
            return time_scalar(keys["value"])
        # Recursively inspect simple scalar components.
        vals = []
        for v in value.values():
            sv = time_scalar(v)
            if sv is not None:
                vals.append(sv)
        if len(vals) == 1:
            return vals[0]
    return None


def _obs_data(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    d = annotation.get("data", [])
    return d if isinstance(d, list) else []


def _annotation_text(annotation: dict[str, Any]) -> str:
    try:
        return json.dumps(annotation, ensure_ascii=False).lower()
    except Exception:
        return str(annotation).lower()


def choose_chord_annotation(annotations: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    candidates = []
    for a in annotations:
        ns = str(a.get("namespace", "")).lower()
        if "chord" not in ns:
            continue
        data = _obs_data(a)
        values = [o.get("value") for o in data[:250] if isinstance(o, dict)]
        parseable = sum(parse_harte(v) is not None for v in values)
        denom = max(1, sum(isinstance(v, str) for v in values))
        ratio = parseable / denom
        txt = _annotation_text({k: a.get(k) for k in ("namespace", "annotation_metadata", "sandbox")})
        harte_bonus = 2.0 if "harte" in txt else 0.0
        converted_bonus = 0.5 if "converted" in txt else 0.0
        candidates.append((ratio + harte_bonus + converted_bonus, len(data), a))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def choose_key_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for a in annotations:
        ns = str(a.get("namespace", "")).lower()
        if ns == "key_mode" or "key" in ns or "tonality" in ns:
            valid = sum(parse_key(o.get("value")) is not None for o in _obs_data(a) if isinstance(o, dict))
            if valid:
                out.append(a)
    return out


def extract_timed_chords(annotation: dict[str, Any]) -> list[TimedChord]:
    out: list[TimedChord] = []
    for o in _obs_data(annotation):
        if not isinstance(o, dict):
            continue
        ch = parse_harte(o.get("value"))
        if ch is None:
            continue
        out.append(TimedChord(ch, time_scalar(o.get("time")), time_scalar(o.get("duration"))))
    return out


def extract_keys(annotations: list[dict[str, Any]]) -> list[KeyObs]:
    found: list[KeyObs] = []
    for a in annotations:
        for o in _obs_data(a):
            if not isinstance(o, dict):
                continue
            parsed = parse_key(o.get("value"))
            if parsed is None:
                continue
            tonic, mode, label = parsed
            found.append(KeyObs(tonic, mode, label, time_scalar(o.get("time")), time_scalar(o.get("duration"))))
    # Remove exact duplicates often introduced by parallel provenance annotations.
    uniq = {}
    for k in found:
        uniq[(k.tonic, k.mode, k.start, k.duration)] = k
    return list(uniq.values())


def _collapse_immediate(tokens: Iterable[ParsedChord]) -> list[ParsedChord]:
    out = []
    for ch in tokens:
        if not out or ch.token != out[-1].token:
            out.append(ch)
    return out


def _segment_by_numeric_interval(chords: list[TimedChord], key: KeyObs) -> list[ParsedChord]:
    if key.start is None:
        return []
    key_end = None if key.duration is None else key.start + max(0.0, key.duration)
    selected = []
    for tc in chords:
        if tc.start is None:
            continue
        if key_end is None:
            if tc.start >= key.start:
                selected.append(tc.chord)
            continue
        tc_end = tc.start if tc.duration is None else tc.start + max(0.0, tc.duration)
        # Any overlap, with point-event fallback.
        if tc.duration is None or tc.duration == 0:
            overlaps = key.start <= tc.start < key_end
        else:
            overlaps = tc_end > key.start and tc.start < key_end
        if overlaps:
            selected.append(tc.chord)
    return _collapse_immediate(selected)


def segment_examples(chords: list[TimedChord], keys: list[KeyObs]) -> tuple[list[tuple[KeyObs, list[ParsedChord]]], str]:
    if not chords or not keys:
        return [], "none"

    # Global key annotation: safest and most common fallback.
    if len(keys) == 1:
        return [(keys[0], _collapse_immediate(tc.chord for tc in chords))], "global"

    if all(tc.start is not None for tc in chords) and all(k.start is not None for k in keys):
        segs = []
        sorted_keys = sorted(keys, key=lambda k: k.start if k.start is not None else -math.inf)
        for i, k in enumerate(sorted_keys):
            effective = k
            # If duration is absent, infer interval end from next key onset.
            if effective.duration is None and i + 1 < len(sorted_keys):
                nxt = sorted_keys[i + 1]
                if nxt.start is not None and effective.start is not None:
                    effective = KeyObs(k.tonic, k.mode, k.label, k.start, max(0.0, nxt.start - k.start))
            selected = _segment_by_numeric_interval(chords, effective)
            if selected:
                segs.append((k, selected))
        if segs:
            return segs, "timed-local"

    return [], "unalignable-local"


def flatten_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(flatten_strings(v))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(flatten_strings(x))
    return out


def detect_partition(jam: dict[str, Any], path: Path) -> str:
    hay = " ".join(flatten_strings({"sandbox": jam.get("sandbox", {}), "file_metadata": jam.get("file_metadata", {})})).lower()
    aliases = {
        "isophonics": ["isophonics"],
        "jaah": ["jaah", "jazz audio-aligned harmony"],
        "schubert-winterreise": ["schubert", "winterreise"],
        "billboard": ["billboard"],
        "chordify": ["chordify"],
        "robbie-williams": ["robbie williams", "robbie_williams"],
        "real-book": ["real book", "realbook", "real-book"],
        "uspop": ["uspop"],
        "rwc-pop": ["rwc", "rwc-pop"],
        "weimar-jazz": ["weimar"],
        "wikifonia": ["wikifonia"],
        "ireal-pro": ["ireal", "ireal pro"],
        "band-in-a-box": ["band-in-a-box", "band in a box", "biab"],
        "when-in-rome": ["when in rome"],
        "rock-corpus": ["rock corpus", "rock-corpus"],
        "mozart-piano-sonata": ["mozart piano sonata", "mozart"],
        "jazz-corpus": ["jazz corpus"],
        "nottingham": ["nottingham"],
    }
    filename = path.stem.lower()
    for canonical, needles in aliases.items():
        if any(n in hay for n in needles) or any(filename.startswith(n.replace(" ", "_")) for n in needles):
            return canonical
    return filename.split("_", 1)[0]


def license_class(partition: str) -> str:
    p = partition.lower()
    if any(h in p for h in NC_PARTITION_HINTS):
        return "CC BY-NC-SA 4.0"
    return "CC BY 4.0"


def stable_split(piece_id: str) -> str:
    x = int(hashlib.sha1(piece_id.encode("utf-8")).hexdigest()[:8], 16) % 10000
    if x < 7000:
        return "train"
    if x < 8500:
        return "validation"
    return "test"


def metadata(jam: dict[str, Any], path: Path) -> dict[str, str]:
    fm = jam.get("file_metadata") or {}
    if not isinstance(fm, dict):
        fm = {}
    title = str(fm.get("title") or "")
    artist = str(fm.get("artist") or "")
    identifiers = fm.get("identifiers") if isinstance(fm.get("identifiers"), dict) else {}
    piece_id = ""
    for k in ("musicbrainz", "mbid", "id", "uri"):
        if identifiers.get(k):
            piece_id = str(identifiers[k])
            break
    if not piece_id:
        piece_id = path.stem
    return {"title": title, "artist": artist, "piece_id": piece_id}


def find_jams_root(path: Path) -> Path:
    if any(path.glob("*.jams")):
        return path
    candidates = []
    for p in path.rglob("jams"):
        if p.is_dir() and any(p.glob("*.jams")):
            candidates.append(p)
    if candidates:
        return max(candidates, key=lambda p: len(list(p.glob("*.jams"))))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a labeled major/minor key dataset from ChoCo JAMS files.")
    ap.add_argument("--choco", default="data/choco/v1.0.0", help="Extracted ChoCo release folder or its jams folder")
    ap.add_argument("--output", default="data/choco_key_segments_ccby.csv")
    ap.add_argument("--audit", default="data/choco_key_segments_audit.json")
    ap.add_argument("--min-chords", type=int, default=4)
    ap.add_argument("--include-noncommercial", action="store_true", help="Include Chordify, Mozart Piano Sonata, and JAAH (CC BY-NC-SA)")
    ap.add_argument("--max-files", type=int, default=None, help="Debug only")
    args = ap.parse_args()

    root = find_jams_root(Path(args.choco))
    files = sorted(root.rglob("*.jams"))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No .jams files found under {root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    partitions = Counter()
    exclusions = Counter()
    alignment = Counter()
    key_counts = Counter()

    fields = [
        "example_id", "piece_id", "source_file", "partition", "license", "split",
        "title", "artist", "key_tonic_pc", "key_mode", "key_label",
        "n_chords", "chords_simple", "chords_harte", "alignment",
    ]

    with output.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for idx, path in enumerate(files, 1):
            stats["files_seen"] += 1
            try:
                jam = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                stats["json_errors"] += 1
                continue
            if not isinstance(jam, dict):
                stats["json_errors"] += 1
                continue

            partition = detect_partition(jam, path)
            partitions[partition] += 1
            lic = license_class(partition)
            if lic != "CC BY 4.0" and not args.include_noncommercial:
                exclusions[partition] += 1
                stats["excluded_noncommercial_files"] += 1
                continue

            annotations = jam.get("annotations") or []
            if not isinstance(annotations, list):
                stats["missing_annotations"] += 1
                continue
            annotations = [a for a in annotations if isinstance(a, dict)]
            chord_ann = choose_chord_annotation(annotations)
            if chord_ann is None:
                stats["no_chord_annotation"] += 1
                continue
            key_anns = choose_key_annotations(annotations)
            if not key_anns:
                stats["no_major_minor_key_annotation"] += 1
                continue

            chords = extract_timed_chords(chord_ann)
            keys = extract_keys(key_anns)
            if not chords:
                stats["no_parseable_chords"] += 1
                continue
            if not keys:
                stats["no_major_minor_key_annotation"] += 1
                continue

            segs, method = segment_examples(chords, keys)
            alignment[method] += 1
            if not segs:
                stats["unalignable_key_segments"] += 1
                continue

            meta = metadata(jam, path)
            piece_id = meta["piece_id"]
            split = stable_split(piece_id)
            local_n = 0
            for seg_idx, (key, seg_chords) in enumerate(segs):
                if len(seg_chords) < args.min_chords:
                    stats["segments_too_short"] += 1
                    continue
                simple = " ".join(c.token for c in seg_chords)
                # Preserve original Harte values insofar as the simple parser retained them.
                # This field is for provenance/debug; model features use chords_simple.
                harte = " ".join(c.token for c in seg_chords)
                example_id = f"{piece_id}::{seg_idx}::{key.label}"
                writer.writerow({
                    "example_id": example_id,
                    "piece_id": piece_id,
                    "source_file": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                    "partition": partition,
                    "license": lic,
                    "split": split,
                    "title": meta["title"],
                    "artist": meta["artist"],
                    "key_tonic_pc": key.tonic,
                    "key_mode": key.mode,
                    "key_label": key.label,
                    "n_chords": len(seg_chords),
                    "chords_simple": simple,
                    "chords_harte": harte,
                    "alignment": method,
                })
                local_n += 1
                stats["examples_written"] += 1
                key_counts[key.label] += 1
            if local_n:
                stats["files_with_examples"] += 1

            if idx % 1000 == 0 or idx == len(files):
                print(f"processed {idx:,}/{len(files):,} JAMS files | examples {stats['examples_written']:,}")

    report = {
        "source_root": str(root),
        "output": str(output),
        "include_noncommercial": bool(args.include_noncommercial),
        "stats": dict(stats),
        "partition_files": dict(partitions.most_common()),
        "excluded_noncommercial_files": dict(exclusions.most_common()),
        "alignment_methods": dict(alignment.most_common()),
        "key_examples": dict(key_counts.most_common()),
        "split_policy": "SHA1(piece_id): 70% train, 15% validation, 15% test; all segments from a piece stay together",
    }
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["stats"], indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
