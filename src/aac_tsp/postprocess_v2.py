"""Post-processing for experiment v2 (configurator comparison).

The four arms allocate their budget differently (fixed-budget arms spend
train_size calls per proposal; SMAC's intensifier spends 1 call per target
evaluation), so nothing here is indexed by "trial". The common axis is
``cum_sa_calls`` -- cumulative target-algorithm (SA) calls -- and every incumbent
trajectory is a right-continuous step function on that axis.

Outputs:
  processed/trajectories_grid.csv   step functions evaluated on a shared log-spaced
                                    budget grid (per train_size), one row per
                                    (run, grid point)
  summaries/final_summary.csv       one row per run (final costs, overtuning, AUC,
                                    overheads, budget-to-target)
  summaries/condition_summary.csv   aggregates per (optimizer, train_size)
  summaries/stats_tests.csv         pairwise Wilcoxon signed-rank tests (paired by
                                    seed), Holm-corrected, plus win rates
  summaries/summary_table.md        human-readable results table
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import EPS

GRID_POINTS = 64
TEST_METRICS = ["final_test_cost", "final_val_cost"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_runs_v2(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (concatenated traj rows, one sidecar row per run)."""
    raw_dir = Path(raw_dir)
    trajs, sides = [], []
    for sc_path in sorted(raw_dir.glob("*.json")):
        try:
            sc = json.loads(sc_path.read_text())
        except json.JSONDecodeError:
            continue
        if sc.get("status") != "complete":
            continue
        traj_path = raw_dir / f"{sc['run_id']}_traj.csv"
        if not traj_path.exists():
            continue
        traj = pd.read_csv(traj_path)
        if not len(traj):
            continue
        trajs.append(traj)
        rc = sc["run_config"]
        sides.append({
            "run_id": sc["run_id"],
            "optimizer": rc["optimizer"],
            "train_family": rc["train_family"],
            "train_size": rc["train_size"],
            "seed": rc["seed"],
            "n_trials": rc["n_trials"],
            "total_budget_calls": sc["total_budget_calls"],
            "total_sa_calls_optimization": sc["total_sa_calls_optimization"],
            "total_sa_calls_analysis": sc["total_sa_calls_analysis"],
            "ask_sec": sc["ask_sec"],
            "eval_sec": sc["eval_sec"],
            "tell_sec": sc["tell_sec"],
            "analysis_runtime_sec": sc["analysis_runtime_sec"],
            "wallclock_sec": sc["wallclock_sec"],
            "n_incumbent_changes": sc["n_incumbent_changes"],
            "n_distinct_configs": sc["n_distinct_configs"],
        })
    if not trajs:
        raise RuntimeError(f"no complete v2 runs found in {raw_dir}")
    return pd.concat(trajs, ignore_index=True), pd.DataFrame(sides)


# --------------------------------------------------------------------------- #
# Common-budget-grid trajectories
# --------------------------------------------------------------------------- #

def _step_at(cum: np.ndarray, values: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Right-continuous step lookup: value of the last event with cum <= budget
    (NaN before the first event)."""
    idx = np.searchsorted(cum, budgets, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], np.nan)
    return out


def build_grid_trajectories(traj_df: pd.DataFrame, side_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value_cols = ["val_cost", "test_cost", "test_cost_uniform",
                  "test_cost_clustered", "test_cost_mixed"]
    for (family, size), side_grp in side_df.groupby(["train_family", "train_size"]):
        total = int(side_grp["total_budget_calls"].max())
        run_ids = set(side_grp["run_id"])
        grp = traj_df[traj_df["run_id"].isin(run_ids)]
        lo = max(1, int(grp.groupby("run_id")["cum_sa_calls"].min().min()))
        budgets = np.unique(np.geomspace(lo, total, GRID_POINTS).round().astype(int))
        for run_id, d in grp.groupby("run_id"):
            d = d.sort_values("cum_sa_calls")
            cum = d["cum_sa_calls"].to_numpy(dtype=float)
            base = {
                "run_id": run_id,
                "optimizer": d["optimizer"].iloc[0],
                "train_family": family,
                "train_size": size,
                "seed": int(d["seed"].iloc[0]),
            }
            interp = {c: _step_at(cum, d[c].to_numpy(dtype=float), budgets)
                      for c in value_cols}
            for i, b in enumerate(budgets):
                row = dict(base)
                row["budget"] = int(b)
                row["budget_frac"] = b / total
                for c in value_cols:
                    row[c] = interp[c][i]
                row["generalization_gap"] = row["test_cost"] - row["val_cost"]
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Per-run summaries
# --------------------------------------------------------------------------- #

def summarize_run_v2(d: pd.DataFrame) -> dict:
    """Final metrics + overtuning along the incumbent-event trajectory of one run."""
    d = d.sort_values("cum_sa_calls").reset_index(drop=True)
    events = d[d["event_type"] == "incumbent"]
    if not len(events):
        events = d
    test_t = events["test_cost"].to_numpy(dtype=float)
    best_so_far = np.minimum.accumulate(test_t)
    overtuning = test_t - best_so_far
    denom = test_t[0] - best_so_far
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.abs(denom) < EPS, np.nan, overtuning / denom)
    last = d.iloc[-1]
    return {
        "run_id": last["run_id"],
        "optimizer": last["optimizer"],
        "train_family": last["train_family"],
        "train_size": int(last["train_size"]),
        "seed": int(last["seed"]),
        "final_val_cost": float(last["val_cost"]),
        "final_test_cost": float(last["test_cost"]),
        "final_generalization_gap": float(last["test_cost"] - last["val_cost"]),
        "best_test_seen": float(np.min(test_t)),
        "final_overtuning": float(overtuning[-1]),
        "final_relative_overtuning": float(rel[-1]) if not np.isnan(rel[-1]) else np.nan,
        "final_test_uniform": float(last["test_cost_uniform"]),
        "final_test_clustered": float(last["test_cost_clustered"]),
        "final_test_mixed": float(last["test_cost_mixed"]),
    }


def _anytime_scores(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Mean test/val cost over the log-spaced budget grid = anytime performance
    (lower is better; equals normalized area under the convergence curve)."""
    return (grid_df.groupby("run_id", as_index=False)
            .agg(auc_test=("test_cost", "mean"),
                 auc_val=("val_cost", "mean")))


def _budget_to_target(traj_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Budget (SA calls) each run needed to first reach the reference target:
    the median-across-seeds *final* validation cost of the random arm in the same
    (train_family, train_size) condition. Speedup = random's own median budget-to-
    target divided by the arm's budget-to-target."""
    rows = []
    for (family, size), grp in summary_df.groupby(["train_family", "train_size"]):
        rs = grp[grp["optimizer"] == "random"]
        if not len(rs):
            continue
        target = float(rs["final_val_cost"].median())
        run_ids = set(grp["run_id"])
        for run_id, d in traj_df[traj_df["run_id"].isin(run_ids)].groupby("run_id"):
            d = d.sort_values("cum_sa_calls")
            hit = d[d["val_cost"] <= target]
            rows.append({
                "run_id": run_id,
                "optimizer": d["optimizer"].iloc[0],
                "train_family": family,
                "train_size": size,
                "seed": int(d["seed"].iloc[0]),
                "target_val_cost": target,
                "budget_to_target": float(hit["cum_sa_calls"].iloc[0]) if len(hit) else np.nan,
            })
    btt = pd.DataFrame(rows)
    if len(btt):
        ref = (btt[btt["optimizer"] == "random"]
               .groupby(["train_family", "train_size"])["budget_to_target"]
               .median().rename("random_median_budget"))
        btt = btt.merge(ref, on=["train_family", "train_size"], how="left")
        btt["speedup_vs_random"] = btt["random_median_budget"] / btt["budget_to_target"]
    return btt


# --------------------------------------------------------------------------- #
# Statistics: pairwise Wilcoxon signed-rank tests, Holm-corrected
# --------------------------------------------------------------------------- #

def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def pairwise_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TEST_METRICS:
        for (family, size), grp in summary_df.groupby(["train_family", "train_size"]):
            opts = sorted(grp["optimizer"].unique())
            pivot = grp.pivot_table(index="seed", columns="optimizer", values=metric)
            pvals, meta = [], []
            for a, b in itertools.combinations(opts, 2):
                paired = pivot[[a, b]].dropna()
                x, y = paired[a].to_numpy(), paired[b].to_numpy()
                if len(x) < 5:
                    continue
                diff = x - y
                if np.allclose(diff, 0.0):
                    p = 1.0
                else:
                    try:
                        p = float(stats.wilcoxon(x, y).pvalue)
                    except ValueError:
                        p = 1.0
                win_a = float(np.mean(x < y) + 0.5 * np.mean(x == y))
                pvals.append(p)
                meta.append({
                    "metric": metric,
                    "train_family": family,
                    "train_size": size,
                    "optimizer_a": a,
                    "optimizer_b": b,
                    "n_pairs": len(x),
                    "median_a": float(np.median(x)),
                    "median_b": float(np.median(y)),
                    "median_diff_a_minus_b": float(np.median(diff)),
                    "win_rate_a_over_b": win_a,
                    "p_value": p,
                })
            adj = _holm(pvals) if pvals else []
            for m_row, p_adj in zip(meta, adj):
                m_row["p_holm"] = p_adj
                m_row["significant_0.05"] = p_adj < 0.05
                rows.append(m_row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_outputs_v2(raw_dir: Path, processed_dir: Path, summaries_dir: Path) -> dict:
    processed_dir = Path(processed_dir)
    summaries_dir = Path(summaries_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    traj_df, side_df = load_runs_v2(raw_dir)

    grid_df = build_grid_trajectories(traj_df, side_df)
    grid_path = processed_dir / "trajectories_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    summary_df = pd.DataFrame([summarize_run_v2(d) for _, d in traj_df.groupby("run_id")])
    summary_df = summary_df.merge(_anytime_scores(grid_df), on="run_id", how="left")
    summary_df = summary_df.merge(
        side_df.drop(columns=["optimizer", "train_family", "train_size", "seed"]),
        on="run_id", how="left")
    summary_df["configurator_overhead_sec"] = summary_df["ask_sec"] + summary_df["tell_sec"]

    btt = _budget_to_target(traj_df, summary_df)
    if len(btt):
        summary_df = summary_df.merge(
            btt[["run_id", "target_val_cost", "budget_to_target", "speedup_vs_random"]],
            on="run_id", how="left")
    summary_path = summaries_dir / "final_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    cond = (summary_df
            .groupby(["optimizer", "train_family", "train_size"], as_index=False)
            .agg(n_runs=("run_id", "count"),
                 mean_final_test=("final_test_cost", "mean"),
                 std_final_test=("final_test_cost", "std"),
                 median_final_test=("final_test_cost", "median"),
                 mean_final_val=("final_val_cost", "mean"),
                 median_final_val=("final_val_cost", "median"),
                 mean_gap=("final_generalization_gap", "mean"),
                 mean_rel_overtuning=("final_relative_overtuning", "mean"),
                 mean_auc_test=("auc_test", "mean"),
                 median_budget_to_target=("budget_to_target", "median"),
                 median_speedup_vs_random=("speedup_vs_random", "median"),
                 mean_distinct_configs=("n_distinct_configs", "mean"),
                 mean_incumbent_changes=("n_incumbent_changes", "mean"),
                 mean_overhead_sec=("configurator_overhead_sec", "mean"),
                 mean_wallclock_sec=("wallclock_sec", "mean")))
    cond_path = summaries_dir / "condition_summary.csv"
    cond.to_csv(cond_path, index=False)

    tests = pairwise_tests(summary_df)
    tests_path = summaries_dir / "stats_tests.csv"
    tests.to_csv(tests_path, index=False)

    _write_markdown_table(cond, tests, summaries_dir / "summary_table.md")

    return {"grid": grid_path, "summary": summary_path, "condition": cond_path,
            "tests": tests_path, "n_runs": len(summary_df)}


def _write_markdown_table(cond: pd.DataFrame, tests: pd.DataFrame, out: Path):
    lines = ["# Experiment v2 — configurator comparison", "",
             "## Per-condition results", ""]
    show = cond.copy()
    float_cols = [c for c in show.columns if show[c].dtype.kind == "f"]
    for c in float_cols:
        show[c] = show[c].map(lambda v: f"{v:.5g}" if pd.notna(v) else "—")
    lines.append("| " + " | ".join(show.columns) + " |")
    lines.append("|" + "---|" * len(show.columns))
    for _, r in show.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    lines += ["", "## Pairwise Wilcoxon signed-rank tests (paired by seed, Holm-corrected)", ""]
    if len(tests):
        tshow = tests.copy()
        for c in ["median_a", "median_b", "median_diff_a_minus_b", "win_rate_a_over_b",
                  "p_value", "p_holm"]:
            tshow[c] = tshow[c].map(lambda v: f"{v:.4g}")
        lines.append("| " + " | ".join(tshow.columns) + " |")
        lines.append("|" + "---|" * len(tshow.columns))
        for _, r in tshow.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
