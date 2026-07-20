#!/usr/bin/env python
"""Compute best-known reference tour lengths (multi-start NN + 2-opt) for all pools."""
import argparse
from pathlib import Path

import pandas as pd

from aac_tsp import FAMILIES
from aac_tsp.best_known import best_known_path, compute_best_known_for_pool
from aac_tsp.instances import load_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cities", type=int, default=50)
    ap.add_argument("--n_starts", type=int, default=20)
    ap.add_argument("--data_dir", default="/local/rohit/datasets/tsp")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    frames = []
    for family in FAMILIES:
        for role in ("train", "test"):
            pool = load_pool(data_dir, family, args.n_cities, role)
            df = compute_best_known_for_pool(pool, n_starts=args.n_starts)
            df["role"] = role
            frames.append(df)
            print(f"  {family}/{role}: {len(df)} instances, mean best-known={df['best_known_length'].mean():.4f}")

    out = pd.concat(frames, ignore_index=True)
    path = best_known_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} best-known references -> {path}")


if __name__ == "__main__":
    main()
