#!/usr/bin/env python
"""CLI wrapper: generate all v2 comparison figures."""
import argparse

from aac_tsp.plotting_v2 import make_all_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="results/v2/raw")
    ap.add_argument("--processed_dir", default="results/v2/processed")
    ap.add_argument("--summaries_dir", default="results/v2/summaries")
    ap.add_argument("--fig_dir", default="figures/v2")
    args = ap.parse_args()

    make_all_v2(args.processed_dir, args.summaries_dir, args.raw_dir, args.fig_dir)


if __name__ == "__main__":
    main()
