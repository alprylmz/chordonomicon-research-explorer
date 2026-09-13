from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://github.com/smashub/choco/releases/download/v1.0.0/v1.0.0.zip"


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ChoCo v1.0.0 from:\n  {url}")
    print(f"Destination:\n  {out}")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        pct = 100.0 * done / total
        sys.stdout.write(f"\r{pct:6.2f}%  {done / (1024**2):,.1f} / {total / (1024**2):,.1f} MiB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, out, reporthook=progress)
    print("\nDownload complete.")


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {dest} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    print("Extraction complete.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download the official ChoCo v1.0.0 release.")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--archive", default="data/choco/v1.0.0.zip")
    p.add_argument("--dest", default="data/choco/v1.0.0")
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args()

    archive = Path(args.archive)
    dest = Path(args.dest)
    if not archive.exists():
        download(args.url, archive)
    else:
        print(f"Archive already exists: {archive}")
    extract(archive, dest)
    if not args.keep_archive:
        try:
            archive.unlink()
            print("Removed downloaded zip after extraction.")
        except OSError:
            pass

    jams = list(dest.rglob("*.jams"))
    print(f"Found {len(jams):,} JAMS files under {dest}")
    if not jams:
        print("WARNING: no .jams files found. Inspect the extracted folder structure.")


if __name__ == "__main__":
    main()
