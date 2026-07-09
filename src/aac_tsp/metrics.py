"""Overtuning metrics, adapted from Schneider, Bischl & Feurer (AutoML 2025).

Along the validation-incumbent trajectory (theta*_t = argmin_{i<=t} val_i):
  val_t                  validation cost of the incumbent (non-increasing by construction)
  test_t                 test cost of the incumbent (large independent test pool)
  generalization_gap_t   = test_t - val_t
  overtuning_t           = test_t - min_{t'<=t} test_{t'}            (>= 0)
  relative_overtuning_t  = overtuning_t / (test_1 - min_{t'<=t} test_{t'})
                           (NaN when the denominator < EPS; >= 1 means all the
                            test-side progress over the first incumbent was lost)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def incumbent_trajectory(run_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-trial incumbent trajectory and overtuning metrics for one run."""
    df = run_df.sort_values("trial_id").reset_index(drop=True)

    # Index of the incumbent (argmin of validation cost up to and including t).
    val = df["validation_cost"].to_numpy(dtype=float)
    inc_idx = np.empty(len(val), dtype=int)
    best_i, best_v = 0, val[0]
    for t in range(len(val)):
        if val[t] < best_v - 1e-15:
            best_v, best_i = val[t], t
        inc_idx[t] = best_i

    # test_cost is logged only on incumbent changes; the incumbent's own row therefore
    # always carries a non-NaN test_cost. Map each step to its incumbent's test cost.
    test_same = df["test_cost"].to_numpy(dtype=float)
    fam_cols = ["test_cost_uniform", "test_cost_clustered", "test_cost_mixed"]

    out = {
        "run_id": df["run_id"],
        "trial_id": df["trial_id"],
        "optimizer": df["optimizer"],
        "resampling": df["resampling"],
        "train_family": df["train_family"],
        "train_size": df["train_size"],
        "seed": df["seed"],
    }
    val_t = val[inc_idx]
    test_t = test_same[inc_idx]
    out["val_t"] = val_t
    out["test_t"] = test_t
    out["generalization_gap_t"] = test_t - val_t
    for c in fam_cols:
        out[c.replace("test_cost", "test_t")] = df[c].to_numpy(dtype=float)[inc_idx]

    traj = pd.DataFrame(out)

    # overtuning along the incumbent trajectory
    best_test_so_far = np.minimum.accumulate(test_t)
    overtuning = test_t - best_test_so_far
    denom = test_t[0] - best_test_so_far
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.abs(denom) < EPS, np.nan, overtuning / denom)
    traj["overtuning_t"] = overtuning
    traj["relative_overtuning_t"] = rel
    return traj


def summarize_run(traj: pd.DataFrame) -> dict:
    """One-row summary for a run from its incumbent trajectory."""
    last = traj.iloc[-1]
    rel = traj["relative_overtuning_t"].to_numpy(dtype=float)
    rel_valid = rel[~np.isnan(rel)]
    over = traj["overtuning_t"].to_numpy(dtype=float)
    return {
        "run_id": last["run_id"],
        "optimizer": last["optimizer"],
        "resampling": last["resampling"],
        "train_family": last["train_family"],
        "train_size": int(last["train_size"]),
        "seed": int(last["seed"]),
        "final_validation_cost": float(last["val_t"]),
        "final_test_cost": float(last["test_t"]),
        "final_generalization_gap": float(last["generalization_gap_t"]),
        "final_overtuning": float(last["overtuning_t"]),
        "final_relative_overtuning": float(rel[-1]) if not np.isnan(rel[-1]) else np.nan,
        "max_overtuning": float(np.nanmax(over)) if len(over) else np.nan,
        "best_test_seen": float(np.min(traj["test_t"].to_numpy())),
        "proportion_nonzero_overtuning": float(np.mean(over > EPS)),
        "mean_relative_overtuning": float(np.mean(rel_valid)) if len(rel_valid) else np.nan,
        "test_t_uniform": float(last["test_t_uniform"]),
        "test_t_clustered": float(last["test_t_clustered"]),
        "test_t_mixed": float(last["test_t_mixed"]),
    }
