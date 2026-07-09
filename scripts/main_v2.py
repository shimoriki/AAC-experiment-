#!/usr/bin/env python
"""Run one v2 experiment cell (one configurator run) and write its outputs.

Thread env vars are pinned to 1 *before* importing numpy/numba so that SLURM array
tasks (one per core) do not oversubscribe the node.

Idempotent: if the run's sidecar JSON reports status=complete, the run is skipped
(use --force to re-run), so a failed/timed-out SLURM array can simply be resubmitted.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from aac_tsp.runner_v2 import OPTIMIZERS_V2, RunConfigV2, run_experiment_v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=list(OPTIMIZERS_V2), required=True)
    ap.add_argument("--train_family", choices=["uniform", "clustered", "mixed"], default="mixed")
    ap.add_argument("--train_size", type=int, default=40)
    ap.add_argument("--n_trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_cities", type=int, default=50)
    ap.add_argument("--solver_repeats", type=int, default=1)
    ap.add_argument("--test_solver_repeats", type=int, default=2)
    ap.add_argument("--test_size", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=30_000)
    ap.add_argument("--tpe_startup_trials", type=int, default=25)
    ap.add_argument("--smac_initial_configs", type=int, default=25)
    ap.add_argument("--smac_max_config_calls", type=int, default=0)
    ap.add_argument("--smac_retrain_after", type=int, default=8)
    ap.add_argument("--data_dir", default="/local/anmol/datasets/tsp")
    ap.add_argument("--out_dir", default="results/v2/raw")
    ap.add_argument("--smac_scratch", default="results/v2/smac_output")
    ap.add_argument("--force", action="store_true", help="re-run even if already complete")
    args = ap.parse_args()

    force = args.force
    del args.force
    cfg = RunConfigV2(**vars(args))

    sidecar = Path(cfg.out_dir) / f"{cfg.run_id}.json"
    if not force and sidecar.exists():
        try:
            if json.loads(sidecar.read_text()).get("status") == "complete":
                print(f"[{cfg.run_id}] already complete -> skipping (use --force to re-run)")
                return
        except (json.JSONDecodeError, OSError):
            pass  # unreadable sidecar: re-run

    t0 = time.perf_counter()
    path = run_experiment_v2(cfg)
    print(f"[{cfg.run_id}] done in {time.perf_counter() - t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()
