#!/usr/bin/env python3
"""
Safely organize Chordonomicon development leftovers.

Default: dry-run only.
Apply: python scripts/organize_project.py --apply

Nothing is deleted. Matching development artifacts are moved into _archive/.
Core app/data/model files are left untouched.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import shutil

PROJECT = Path(__file__).resolve().parents[1]

ARCHIVE = PROJECT / "_archive"
PATCH_ARCHIVE = ARCHIVE / "patches"
REPORT_ARCHIVE = ARCHIVE / "experiment_reports"
CACHE_ARCHIVE = ARCHIVE / "old_outputs"

# Explicitly protected names / paths.
PROTECTED_NAMES = {
    "app.py",
    "db.py",
    "chord_utils.py",
    "requirements.txt",
    "README.md",
    "run.command",
    ".gitignore",
}

# Only clearly disposable/development artifacts are moved.
PATCH_PATTERNS = [
    "*patch*.zip",
    "*hotfix*.zip",
    "chordonomicon_step*.zip",
    "chordonomicon_key_inference_step*.zip",
    "chordonomicon_choco_mode_audit.zip",
]

REPORT_PATTERNS = [
    "*audit_summary*.json",
    "*model_report*.json",
    "*calibration_report*.json",
    "*theory_baseline_report*.json",
    "*mode_balance_report*.json",
    "*test_errors*.csv",
    "*coefficients*.csv",
]

OLD_OUTPUT_PATTERNS = [
    "chordonomicon_song_keys_summary.json",
    "chordonomicon_song_keys.parquet",
]

def move_file(src: Path, dest_dir: Path, apply: bool):
    if src.name in PROTECTED_NAMES or not src.is_file():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        stem, suffix = src.stem, src.suffix
        i = 2
        while (dest_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        dest = dest_dir / f"{stem}_{i}{suffix}"
    print(f"{'MOVE' if apply else 'WOULD MOVE'}: {src.relative_to(PROJECT)} -> {dest.relative_to(PROJECT)}")
    if apply:
        shutil.move(str(src), str(dest))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually move files. Without this, only preview.")
    args = ap.parse_args()

    seen = set()
    for pattern in PATCH_PATTERNS:
        for src in PROJECT.glob(pattern):
            if src not in seen:
                seen.add(src)
                move_file(src, PATCH_ARCHIVE, args.apply)

    for pattern in REPORT_PATTERNS:
        for src in PROJECT.glob(pattern):
            if src not in seen:
                seen.add(src)
                move_file(src, REPORT_ARCHIVE, args.apply)

    # Old unbalanced full-corpus outputs are archived only if the balanced outputs exist.
    balanced_parquet = PROJECT / "data/key_models/chordonomicon_song_keys_mode_balanced.parquet"
    balanced_summary = PROJECT / "data/key_models/chordonomicon_song_keys_mode_balanced_summary.json"
    if balanced_parquet.exists() and balanced_summary.exists():
        for pattern in OLD_OUTPUT_PATTERNS:
            for src in (PROJECT / "data/key_models").glob(pattern):
                if src not in seen:
                    seen.add(src)
                    move_file(src, CACHE_ARCHIVE, args.apply)

    print()
    if args.apply:
        print("✅ Düzenleme tamamlandı. Hiçbir dosya silinmedi.")
    else:
        print("ℹ️ Bu sadece önizlemeydi. Uygulamak için:")
        print("   python scripts/organize_project.py --apply")

if __name__ == "__main__":
    main()
