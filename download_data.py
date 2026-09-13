from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ailsntua/Chordonomicon"
FILENAME = "chordonomicon_v2.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Download Chordonomicon v2 from Hugging Face.")
    p.add_argument("--output", default="data/chordonomicon_v2.csv")
    args = p.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
    )
    shutil.copy2(cached, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
