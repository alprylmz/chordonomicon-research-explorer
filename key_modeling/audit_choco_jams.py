from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick audit of ChoCo JAMS namespaces before dataset extraction.")
    ap.add_argument("--choco", default="data/choco/v1.0.0")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()
    files = sorted(Path(args.choco).rglob("*.jams"))
    if args.max_files:
        files = files[:args.max_files]
    ns = Counter()
    key_values = Counter()
    files_with_chord = 0
    files_with_key = 0
    for i, p in enumerate(files, 1):
        try:
            jam = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        anns = jam.get("annotations", []) if isinstance(jam, dict) else []
        has_c = has_k = False
        for a in anns:
            if not isinstance(a, dict):
                continue
            name = str(a.get("namespace", ""))
            ns[name] += 1
            low = name.lower()
            if "chord" in low:
                has_c = True
            if low == "key_mode" or "key" in low or "tonality" in low:
                has_k = True
                for o in a.get("data", []) or []:
                    if isinstance(o, dict):
                        key_values[str(o.get("value"))] += 1
        files_with_chord += has_c
        files_with_key += has_k
    print(json.dumps({
        "files": len(files),
        "files_with_chord_namespace": files_with_chord,
        "files_with_key_like_namespace": files_with_key,
        "namespaces": ns.most_common(),
        "sample_key_values": key_values.most_common(40),
    }, indent=2))


if __name__ == "__main__":
    main()
