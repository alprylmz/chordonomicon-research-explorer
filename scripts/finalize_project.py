#!/usr/bin/env python3
"""
Final root cleanup for Chordonomicon Research Explorer.

Default mode is preview-only:
    python scripts/finalize_project.py

Apply:
    python scripts/finalize_project.py --apply

Nothing important is deleted. Historical development files are moved under
_archive/. Python cache directories are removed only in --apply mode because
they are generated automatically and contain no project source.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import shutil

PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "_archive"

DOCS_ARCHIVE = ARCHIVE / "legacy_docs"
SCRIPTS_ARCHIVE = ARCHIVE / "legacy_scripts"
UI_ARCHIVE = ARCHIVE / "legacy_ui"
MISC_ARCHIVE = ARCHIVE / "misc"

# These are historical installation / step notes that are not needed in root.
LEGACY_DOC_PATTERNS = [
    "INSTALL*.txt",
    "KEY_INFERENCE*.md",
    "README_FIX.txt",
    "README_STEP*_FIX.md",
]

# These scripts were needed to build/audit earlier stages but are not needed
# to launch the current v1 application.
LEGACY_SCRIPT_NAMES = {
    "chordonomicon_step5_audit.py",
    "build_presence_metrics.py",
    "finish_aggregation.py",
}

# Old extracted patch folders that can safely live in the archive.
LEGACY_DIR_NAMES = {
    "chordonomicon_visual_redesign",
    "chordonomicon_streamlit_hotfix",
    "chordonomicon_ui_presence_patch",
}

# Files/directories that should remain at the project root.
PROTECTED = {
    "app.py",
    "db.py",
    "chord_utils.py",
    "key_inference.py",
    "preprocess.py",
    "download_data.py",
    "requirements.txt",
    "README.md",
    "README_GITHUB.md",
    ".gitignore",
    "run.command",
    "cleanup_downloads.command",
    "data",
    "key_modeling",
    "scripts",
    "tests",
    "docs",
    "_archive",
}

def unique_destination(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    if not dest.exists():
        return dest
    p = Path(name)
    i = 2
    while True:
        candidate = dest_dir / f"{p.stem}_{i}{p.suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def move(src: Path, dest_dir: Path, apply: bool):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_destination(dest_dir, src.name)
    verb = "MOVE" if apply else "WOULD MOVE"
    print(f"{verb}: {src.relative_to(PROJECT)} -> {dest.relative_to(PROJECT)}")
    if apply:
        shutil.move(str(src), str(dest))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    seen = set()

    for pattern in LEGACY_DOC_PATTERNS:
        for src in PROJECT.glob(pattern):
            if src.name in PROTECTED or src in seen:
                continue
            seen.add(src)
            move(src, DOCS_ARCHIVE, args.apply)

    for name in LEGACY_SCRIPT_NAMES:
        src = PROJECT / name
        if src.exists() and src.name not in PROTECTED:
            move(src, SCRIPTS_ARCHIVE, args.apply)

    for name in LEGACY_DIR_NAMES:
        src = PROJECT / name
        if src.exists() and src.is_dir() and src.name not in PROTECTED:
            move(src, UI_ARCHIVE, args.apply)

    # Keep the tiny sample, but organize it as an example fixture.
    sample = PROJECT / "data_sample.csv"
    if sample.exists():
        examples = PROJECT / "examples"
        examples.mkdir(parents=True, exist_ok=True)
        dest = examples / sample.name
        print(f"{'MOVE' if args.apply else 'WOULD MOVE'}: data_sample.csv -> examples/data_sample.csv")
        if args.apply:
            if dest.exists():
                dest.unlink()
            shutil.move(str(sample), str(dest))

    cache = PROJECT / "__pycache__"
    if cache.exists():
        if args.apply:
            print("REMOVE GENERATED CACHE: __pycache__/")
            shutil.rmtree(cache)
        else:
            print("WOULD REMOVE GENERATED CACHE: __pycache__/")

    print()
    if args.apply:
        print("✅ Final cleanup complete.")
        print("Historical files are under _archive/. Core v1 files stayed in place.")
    else:
        print("ℹ️ Preview only. If this looks right, run:")
        print("   python scripts/finalize_project.py --apply")

if __name__ == "__main__":
    main()
