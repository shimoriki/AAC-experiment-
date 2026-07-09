#!/usr/bin/env python
"""Produce all report figures from the post-processed CSVs."""
import argparse
from pathlib import Path

from aac_tsp.plotting import make_all

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=str(REPO / "results" / "processed"))
    ap.add_argument("--summaries_dir", default=str(REPO / "results" / "summaries"))
    ap.add_argument("--raw_dir", default=str(REPO / "results" / "raw"))
    ap.add_argument("--fig_dir", default=str(REPO / "figures"))
    args = ap.parse_args()

    produced = make_all(Path(args.processed_dir), Path(args.summaries_dir),
                        Path(args.raw_dir), Path(args.fig_dir))
    print(f"Wrote {len(produced)} figures to {args.fig_dir}:")
    for p in produced:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
