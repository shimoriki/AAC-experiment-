#!/usr/bin/env python
"""Aggregate raw runs into incumbent trajectories + summaries."""
import argparse
from pathlib import Path

from aac_tsp.postprocess import build_outputs

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "results" / "raw"))
    ap.add_argument("--processed_dir", default=str(REPO / "results" / "processed"))
    ap.add_argument("--summaries_dir", default=str(REPO / "results" / "summaries"))
    args = ap.parse_args()

    res = build_outputs(Path(args.raw_dir), Path(args.processed_dir), Path(args.summaries_dir))
    print(f"Processed {res['n_runs']} runs (epsilon={res['eps']}).")
    print(f"  trajectories -> {res['trajectories']}")
    print(f"  run summary  -> {res['summary']}")
    print(f"  conditions   -> {res['condition']}")


if __name__ == "__main__":
    main()
