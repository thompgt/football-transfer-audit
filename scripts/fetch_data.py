#!/usr/bin/env python
"""Verify (or explain how to re-obtain) the bundled third-party data archives.

Both source datasets live behind a Kaggle login that requires accepting each
dataset's terms, so there is no honest anonymous download URL to bake in. What
this script *can* do is pin exactly which bytes the published results were
computed from, and fail loudly if they change underneath the analysis.

See data/SOURCES.md for provenance and terms.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = {
    "data/transfers.zip": {
        "sha256": "47c5ed7476a0b60331bb482198f65c35a332cd6789aa86f0456972f7a6d9c279",
        "bytes": 109432,
        "kaggle": "vardan95ghazaryan/top-250-football-transfers-from-2000-to-2018",
    },
    "data/ratings.zip": {
        "sha256": "01500821468bfc6b90e5a7d2155371886b048ff3273f2d2fcf19a2deb5347272",
        "bytes": 15206638,
        "kaggle": "stefanoleone992/fifa-20-complete-player-dataset",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-hashes",
        action="store_true",
        help="Print a fresh manifest for the files currently on disk.",
    )
    args = parser.parse_args()

    if args.print_hashes:
        for rel in MANIFEST:
            path = REPO_ROOT / rel
            if path.exists():
                print(f'"{rel}": {{"sha256": "{sha256(path)}", "bytes": {path.stat().st_size}}},')
        return 0

    failures = []
    for rel, meta in MANIFEST.items():
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"{rel}: MISSING")
            continue
        digest = sha256(path)
        if digest != meta["sha256"]:
            failures.append(f"{rel}: sha256 {digest} != expected {meta['sha256']}")
        else:
            print(f"OK   {rel}  ({path.stat().st_size:,} bytes)")

    if failures:
        print("\nData verification FAILED:", file=sys.stderr)
        for line in failures:
            print("  " + line, file=sys.stderr)
        print(
            "\nRe-obtain with the Kaggle CLI (requires `pip install kaggle`, an API\n"
            "token in ~/.kaggle/kaggle.json, and accepting each dataset's terms):\n",
            file=sys.stderr,
        )
        for rel, meta in MANIFEST.items():
            print(f"  kaggle datasets download -d {meta['kaggle']} -p data/", file=sys.stderr)
        print("\nSee data/SOURCES.md for provenance and licensing.", file=sys.stderr)
        return 1

    print("\nAll data archives match the recorded manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
