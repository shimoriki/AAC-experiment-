#!/usr/bin/env python
"""Concatenate all raw per-run CSVs into a single tidy file (convenience)."""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "results" / "raw"))
    ap.add_argument("--out", default=str(REPO / "results" / "processed" / "all_trials.csv"))
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in sorted(Path(args.raw_dir).glob("*.csv"))]
    if not frames:
        raise SystemExit(f"no raw CSVs in {args.raw_dir}")
    df = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Collected {len(frames)} runs / {len(df)} trial rows -> {args.out}")


if __name__ == "__main__":
    main()
