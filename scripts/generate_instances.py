#!/usr/bin/env python
"""Generate train + test TSP instance pools for all families."""
import argparse
from pathlib import Path

from aac_tsp.instances import generate_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cities", type=int, default=50)
    ap.add_argument("--n_train_pool", type=int, default=150)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--data_dir", default="/local/anmol/datasets/tsp")
    args = ap.parse_args()

    manifest = generate_all(Path(args.data_dir), args.n_cities, args.n_train_pool, args.n_test)
    total = sum(len(p["train"]) + len(p["test"]) for p in manifest["pools"].values())
    print(f"Generated {total} instances across {len(manifest['pools'])} families "
          f"(n_cities={args.n_cities}, train_pool={args.n_train_pool}, test={args.n_test})")
    print(f"Data dir: {Path(args.data_dir) / 'instances'}")


if __name__ == "__main__":
    main()
