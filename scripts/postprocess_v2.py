#!/usr/bin/env python
"""CLI wrapper: aggregate v2 raw runs into grid trajectories, summaries, and stats."""
import argparse

from aac_tsp.postprocess_v2 import build_outputs_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="results/v2/raw")
    ap.add_argument("--processed_dir", default="results/v2/processed")
    ap.add_argument("--summaries_dir", default="results/v2/summaries")
    args = ap.parse_args()

    out = build_outputs_v2(args.raw_dir, args.processed_dir, args.summaries_dir)
    print(f"Post-processed {out['n_runs']} runs")
    for k in ("grid", "summary", "condition", "tests", "paired", "parameters",
              "resampling_tests", "interactions", "checkpoint_runs",
              "checkpoint_summary", "contrasts"):
        print(f"  {k}: {out[k]}")


if __name__ == "__main__":
    main()
