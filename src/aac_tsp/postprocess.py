"""Aggregate raw per-run CSVs into incumbent trajectories, per-run summaries, and
per-condition summaries (including the runtime cost of each resampling method)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import EPS, incumbent_trajectory, summarize_run


def load_raw_runs(raw_dir: Path) -> list[pd.DataFrame]:
    raw_dir = Path(raw_dir)
    runs = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "validation_cost" in df.columns and len(df):
            runs.append(df)
    return runs


def build_outputs(raw_dir: Path, processed_dir: Path, summaries_dir: Path) -> dict:
    processed_dir = Path(processed_dir)
    summaries_dir = Path(summaries_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    runs = load_raw_runs(raw_dir)
    if not runs:
        raise RuntimeError(f"no raw runs found in {raw_dir}")

    trajectories, run_summaries, runtime_rows = [], [], []
    for df in runs:
        traj = incumbent_trajectory(df)
        trajectories.append(traj)
        run_summaries.append(summarize_run(traj))
        runtime_rows.append({
            "run_id": df["run_id"].iloc[0],
            "optimizer": df["optimizer"].iloc[0],
            "resampling": df["resampling"].iloc[0],
            "train_family": df["train_family"].iloc[0],
            "train_size": int(df["train_size"].iloc[0]),
            "seed": int(df["seed"].iloc[0]),
            "total_validation_runtime_sec": float(df["validation_runtime_sec"].sum()),
            "mean_validation_runtime_per_trial": float(df["validation_runtime_sec"].mean()),
            "mean_n_eval_instances": float(df["n_eval_instances"].mean()),
        })

    traj_df = pd.concat(trajectories, ignore_index=True)
    summary_df = pd.DataFrame(run_summaries)
    runtime_df = pd.DataFrame(runtime_rows)
    summary_df = summary_df.merge(
        runtime_df[["run_id", "total_validation_runtime_sec", "mean_validation_runtime_per_trial",
                    "mean_n_eval_instances"]],
        on="run_id", how="left")

    traj_path = processed_dir / "incumbent_trajectories.csv"
    summary_path = summaries_dir / "final_summary.csv"
    traj_df.to_csv(traj_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # per-condition aggregation
    cond = (summary_df
            .groupby(["resampling", "train_size"], as_index=False)
            .agg(mean_final_test_cost=("final_test_cost", "mean"),
                 std_final_test_cost=("final_test_cost", "std"),
                 mean_generalization_gap=("final_generalization_gap", "mean"),
                 mean_relative_overtuning=("final_relative_overtuning", "mean"),
                 proportion_nonzero_overtuning=("proportion_nonzero_overtuning", "mean"),
                 proportion_severe_overtuning=("final_relative_overtuning",
                                               lambda s: (s >= 1).mean()),
                 mean_runtime_sec=("total_validation_runtime_sec", "mean"),
                 n_runs=("run_id", "count")))
    cond_path = summaries_dir / "condition_summary.csv"
    cond.to_csv(cond_path, index=False)

    return {"trajectories": traj_path, "summary": summary_path, "condition": cond_path,
            "n_runs": len(runs), "eps": EPS}
