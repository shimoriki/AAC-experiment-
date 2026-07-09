#!/usr/bin/env python
"""Run one experiment cell (a full BO loop) and write its raw CSV + sidecar JSON.

Thread env vars are pinned to 1 *before* importing numpy/numba so that running many
of these processes in parallel (one per core) does not oversubscribe the machine.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402

from aac_tsp.runner import RunConfig, run_experiment  # noqa: E402
from aac_tsp.solver_sa import DEFAULT_MAX_STEPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["optuna_tpe", "random"], default="optuna_tpe")
    ap.add_argument("--resampling", choices=["holdout", "cv5", "repeated_cv5", "bootstrap_oob"],
                    default="holdout")
    ap.add_argument("--config_space", choices=["full", "fixed_2opt"], default="full")
    ap.add_argument("--train_family", choices=["uniform", "clustered", "mixed"], default="mixed")
    ap.add_argument("--train_size", type=int, default=25)
    ap.add_argument("--bo_budget", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_cities", type=int, default=50)
    ap.add_argument("--solver_repeats", type=int, default=3)
    ap.add_argument("--test_solver_repeats", type=int, default=3)
    ap.add_argument("--test_size", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--cv_repeats", type=int, default=3)
    ap.add_argument("--bootstrap_repeats", type=int, default=10)
    ap.add_argument("--data_dir", default="/local/anmol/datasets/tsp")
    ap.add_argument("--out_dir", default="results/raw")
    args = ap.parse_args()

    cfg = RunConfig(**vars(args))
    t0 = time.perf_counter()
    path = run_experiment(cfg)
    print(f"[{cfg.run_id}] done in {time.perf_counter() - t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()
