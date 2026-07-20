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
  summaries/paired_vs_random.csv    matched-seed differences from Random Search
                                    for final test cost and anytime AUC
  summaries/budget_checkpoint_summary.csv
                                    performance/overtuning at 25/50/75/100% of
                                    the common SA-call budget
  summaries/condition_contrasts.csv paired tests of whether method-vs-RS effects
                                    change with size, family, or search space
  summaries/summary_table.md        human-readable results table
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import EPS, RELATIVE_OVERTUNING_MIN_PROGRESS
from .space_v2 import PARAM_COLUMNS

GRID_POINTS = 64
CHECKPOINT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
TEST_METRICS = ["final_test_cost", "final_val_cost", "auc_test",
                "abs_generalization_gap", "final_relative_overtuning"]
BASE_CONDITION_COLUMNS = ["config_space", "train_family", "train_size"]
CONDITION_COLUMNS = [*BASE_CONDITION_COLUMNS, "resampling"]


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
        if "resampling" not in traj.columns:
            traj["resampling"] = rc_resampling = sc.get("run_config", {}).get(
                "resampling", "full_train"
            )
        else:
            rc_resampling = str(traj["resampling"].iloc[0])
        trajs.append(traj)
        rc = sc["run_config"]
        sides.append({
            "run_id": sc["run_id"],
            "optimizer": rc["optimizer"],
            "config_space": rc.get("config_space", "full"),
            "resampling": rc.get("resampling", rc_resampling),
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
            "n_proposals": sc.get("n_proposals", sc["n_distinct_configs"]),
            "resampling_n_subsets": sc.get("resampling_n_subsets", 1),
            "resampling_sa_calls_per_config": sc.get(
                "resampling_sa_calls_per_config",
                int(rc["train_size"]) * int(rc.get("solver_repeats", 1)),
            ),
            "fixed_proposal_budget": sc.get("fixed_proposal_budget", rc["n_trials"]),
            "unused_budget_calls": sc.get(
                "fixed_unused_budget_calls",
                max(0, int(sc["total_budget_calls"])
                    - int(sc["total_sa_calls_optimization"])),
            ),
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
    value_cols = ["val_cost", "test_cost", "best_test_cost", "test_cost_uniform",
                  "test_cost_clustered", "test_cost_mixed"]
    for condition, side_grp in side_df.groupby(CONDITION_COLUMNS):
        condition_values = dict(zip(CONDITION_COLUMNS, condition))
        total = int(side_grp["total_budget_calls"].max())
        run_ids = set(side_grp["run_id"])
        grp = traj_df[traj_df["run_id"].isin(run_ids)]
        # Start where every arm has produced at least one incumbent.  Using the
        # earliest arm's first event would leave leading NaNs for fixed-budget
        # arms and make their normalized AUC cover a shorter interval.
        lo = max(1, int(grp.groupby("run_id")["cum_sa_calls"].min().max()))
        budgets = np.unique(np.geomspace(lo, total, GRID_POINTS).round().astype(int))
        for run_id, d in grp.groupby("run_id"):
            d = d.sort_values("cum_sa_calls").copy()
            # Keep the primary test distribution fixed across train_family. This
            # also makes older raw trajectories (whose generic test_cost followed
            # train_family) safe to re-process because every run logged all three
            # family-specific test costs.
            d["test_cost"] = d["test_cost_mixed"]
            # Post-hoc oracle envelope for the explicitly requested best-so-far
            # test trajectory. It is analysis-only and never selects an incumbent.
            d["best_test_cost"] = d["test_cost"].cummin()
            cum = d["cum_sa_calls"].to_numpy(dtype=float)
            base = {
                "run_id": run_id,
                "optimizer": d["optimizer"].iloc[0],
                **condition_values,
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
                row["overtuning"] = row["test_cost"] - row["best_test_cost"]
                row["test_progress_from_initial"] = (
                    float(d["test_cost"].iloc[0]) - row["best_test_cost"]
                )
                progress = row["test_progress_from_initial"]
                row["relative_overtuning_eligible"] = bool(
                    progress >= RELATIVE_OVERTUNING_MIN_PROGRESS
                )
                row["relative_overtuning_raw"] = (
                    row["overtuning"] / progress if progress >= EPS else np.nan
                )
                row["relative_overtuning"] = (
                    row["relative_overtuning_raw"]
                    if row["relative_overtuning_eligible"] else np.nan
                )
                row["is_overtuned"] = bool(row["overtuning"] > EPS)
                row["is_severe_overtuning"] = bool(
                    row["relative_overtuning_eligible"]
                    and row["relative_overtuning"] >= 1.0
                )
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
    # Use a common unseen-test distribution for every training-family condition.
    test_t = events["test_cost_mixed"].to_numpy(dtype=float)
    best_so_far = np.minimum.accumulate(test_t)
    overtuning = test_t - best_so_far
    progress = test_t[0] - best_so_far
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_raw = np.where(progress < EPS, np.nan, overtuning / progress)
    eligible = progress >= RELATIVE_OVERTUNING_MIN_PROGRESS
    rel = np.where(eligible, rel_raw, np.nan)
    last = d.iloc[-1]
    result = {
        "run_id": last["run_id"],
        "optimizer": last["optimizer"],
        "config_space": last.get("config_space", "full"),
        "resampling": last.get("resampling", "full_train"),
        "train_family": last["train_family"],
        "train_size": int(last["train_size"]),
        "seed": int(last["seed"]),
        "final_val_cost": float(last["val_cost"]),
        "final_test_cost": float(last["test_cost_mixed"]),
        "final_generalization_gap": float(last["test_cost_mixed"] - last["val_cost"]),
        "abs_generalization_gap": float(
            abs(last["test_cost_mixed"] - last["val_cost"])
        ),
        "best_test_seen": float(np.min(test_t)),
        "final_overtuning": float(overtuning[-1]),
        "test_progress_from_initial": float(progress[-1]),
        "relative_overtuning_eligible": bool(eligible[-1]),
        "final_relative_overtuning_raw": (
            float(rel_raw[-1]) if not np.isnan(rel_raw[-1]) else np.nan
        ),
        "final_relative_overtuning": float(rel[-1]) if not np.isnan(rel[-1]) else np.nan,
        "is_final_overtuned": bool(overtuning[-1] > EPS),
        "is_severe_overtuning": bool(
            eligible[-1] and not np.isnan(rel[-1]) and rel[-1] >= 1.0
        ),
        "final_test_uniform": float(last["test_cost_uniform"]),
        "final_test_clustered": float(last["test_cost_clustered"]),
        "final_test_mixed": float(last["test_cost_mixed"]),
    }
    for parameter in PARAM_COLUMNS:
        if parameter in last.index:
            result[parameter] = last[parameter]
    move_total = sum(float(last.get(p, 0.0))
                     for p in ("p_swap", "p_insert", "p_2opt", "p_oropt"))
    result["move_share_2opt"] = (
        float(last.get("p_2opt", 0.0)) / move_total if move_total > 0 else np.nan
    )
    return result


def _anytime_scores(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Normalized AUC on the shared log-budget grid (lower is better).

    ``auc_test`` is the requested best-so-far test envelope.  The incumbent-test
    AUC is kept separately so the post-hoc oracle view cannot be mistaken for the
    validation-selected incumbent trajectory.
    """
    return (grid_df.groupby("run_id", as_index=False)
            .agg(auc_test=("best_test_cost", "mean"),
                 auc_incumbent_test=("test_cost", "mean"),
                 auc_val=("val_cost", "mean")))


def _budget_checkpoints(grid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run and aggregate metrics at common fractions of the SA-call budget.

    These are descriptive repeated measurements from each run, not independent
    experiments at four separately chosen budgets.  They are nevertheless the
    honest way to study the budget trajectory because every arm is sampled on
    the same target-algorithm-call axis.
    """
    rows = []
    for run_id, grp in grid_df.groupby("run_id"):
        grp = grp.sort_values("budget_frac")
        for target in CHECKPOINT_FRACTIONS:
            available = grp[grp["budget_frac"] <= target + 1e-12]
            row = (available.iloc[-1] if len(available) else grp.iloc[0]).to_dict()
            row["checkpoint_frac"] = target
            row["actual_budget_frac"] = float(row["budget_frac"])
            rows.append(row)
    per_run = pd.DataFrame(rows)
    if not len(per_run):
        return per_run, pd.DataFrame()

    aggregate = (per_run
                 .groupby(["optimizer", *CONDITION_COLUMNS, "checkpoint_frac"],
                          as_index=False)
                 .agg(
                     n_runs=("run_id", "count"),
                     mean_actual_budget_frac=("actual_budget_frac", "mean"),
                     mean_val_cost=("val_cost", "mean"),
                     se_val_cost=("val_cost", "sem"),
                     mean_test_cost=("test_cost", "mean"),
                     se_test_cost=("test_cost", "sem"),
                     mean_generalization_gap=("generalization_gap", "mean"),
                     mean_overtuning=("overtuning", "mean"),
                     proportion_overtuned=("is_overtuned", "mean"),
                     n_relative_overtuning_eligible=(
                         "relative_overtuning_eligible", "sum"),
                     mean_relative_overtuning=("relative_overtuning", "mean"),
                     median_relative_overtuning=("relative_overtuning", "median"),
                     n_severe_overtuning=("is_severe_overtuning", "sum"),
                 ))
    aggregate["proportion_relative_overtuning_eligible"] = (
        aggregate["n_relative_overtuning_eligible"] / aggregate["n_runs"]
    )
    aggregate["proportion_severe_among_eligible"] = np.where(
        aggregate["n_relative_overtuning_eligible"] > 0,
        aggregate["n_severe_overtuning"]
        / aggregate["n_relative_overtuning_eligible"],
        np.nan,
    )
    return per_run, aggregate


def _budget_to_target(traj_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Budget (SA calls) each run needed to first reach the reference target:
    the median-across-seeds *final* validation cost of the random arm in the same
    (train_family, train_size) condition. Speedup = random's own median budget-to-
    target divided by the arm's budget-to-target."""
    rows = []
    for condition, grp in summary_df.groupby(CONDITION_COLUMNS):
        condition_values = dict(zip(CONDITION_COLUMNS, condition))
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
                **condition_values,
                "seed": int(d["seed"].iloc[0]),
                "target_val_cost": target,
                "budget_to_target": float(hit["cum_sa_calls"].iloc[0]) if len(hit) else np.nan,
            })
    btt = pd.DataFrame(rows)
    if len(btt):
        ref = (btt[btt["optimizer"] == "random"]
               .groupby(CONDITION_COLUMNS)["budget_to_target"]
               .median().rename("random_median_budget"))
        btt = btt.merge(ref, on=CONDITION_COLUMNS, how="left")
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


def _bootstrap_mean_ci(values, n_resamples: int = 10_000) -> tuple[float, float]:
    """Deterministic percentile CI for a paired mean difference."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(20250717)
    means = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def pairwise_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TEST_METRICS:
        for condition, grp in summary_df.groupby(CONDITION_COLUMNS):
            condition_values = dict(zip(CONDITION_COLUMNS, condition))
            pivot = grp.pivot_table(index="seed", columns="optimizer", values=metric)
            # ``pivot_table`` drops an optimizer whose metric is entirely NaN.
            # This is common in the one-seed smoke profile for denominator-gated
            # relative overtuning. Compare only columns that actually survived.
            opts = sorted(pivot.columns)
            if len(opts) < 2:
                continue
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
                    **condition_values,
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


def paired_vs_random(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Matched-seed method-minus-Random-Search differences (lower is better)."""
    rows = []
    metrics = ("final_test_cost", "auc_test", "abs_generalization_gap",
               "final_relative_overtuning")
    for condition, grp in summary_df.groupby(CONDITION_COLUMNS):
        for metric in metrics:
            pivot = grp.pivot_table(index="seed", columns="optimizer", values=metric)
            if "random" not in pivot.columns:
                continue
            for optimizer in sorted(o for o in pivot.columns if o != "random"):
                paired = pivot[["random", optimizer]].dropna()
                diff = paired[optimizer] - paired["random"]
                if not len(diff):
                    continue
                ci_low, ci_high = _bootstrap_mean_ci(diff.to_numpy(dtype=float))
                rows.append({
                    **dict(zip(CONDITION_COLUMNS, condition)),
                    "metric": metric,
                    "optimizer": optimizer,
                    "reference": "random",
                    "difference": f"{optimizer} minus random",
                    "n_pairs": int(len(diff)),
                    "mean_difference": float(diff.mean()),
                    "median_difference": float(diff.median()),
                    "std_difference": float(diff.std(ddof=1)) if len(diff) > 1 else np.nan,
                    "mean_difference_ci95_low": ci_low,
                    "mean_difference_ci95_high": ci_high,
                    "n_beats_random": int((diff < 0).sum()),
                    "n_ties_random": int((diff == 0).sum()),
                    "win_rate": float((diff < 0).mean()),
                })
    return pd.DataFrame(rows)


def paired_resampling_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Matched-seed pairwise resampling effects within each configurator."""
    rows = []
    for metric in ("final_test_cost", "auc_test", "abs_generalization_gap",
                   "final_relative_overtuning"):
        for base_condition, condition_group in summary_df.groupby(BASE_CONDITION_COLUMNS):
            for optimizer, group in condition_group.groupby("optimizer"):
                pivot = group.pivot_table(
                    index="seed", columns="resampling", values=metric
                )
                pvals, pending = [], []
                for method_a, method_b in itertools.combinations(sorted(pivot.columns), 2):
                    paired = pivot[[method_a, method_b]].dropna()
                    if not len(paired):
                        continue
                    difference = paired[method_a] - paired[method_b]
                    ci_low, ci_high = _bootstrap_mean_ci(
                        difference.to_numpy(dtype=float)
                    )
                    if len(difference) < 5 or np.allclose(difference, 0.0):
                        p_value = 1.0 if np.allclose(difference, 0.0) else np.nan
                    else:
                        try:
                            p_value = float(stats.wilcoxon(difference).pvalue)
                        except ValueError:
                            p_value = 1.0
                    row = {
                        **dict(zip(BASE_CONDITION_COLUMNS, base_condition)),
                        "metric": metric,
                        "optimizer": optimizer,
                        "resampling_a": method_a,
                        "resampling_b": method_b,
                        "difference": f"{method_a} minus {method_b}",
                        "n_pairs": int(len(difference)),
                        "mean_difference": float(difference.mean()),
                        "median_difference": float(difference.median()),
                        "std_difference": (
                            float(difference.std(ddof=1))
                            if len(difference) > 1 else np.nan
                        ),
                        "mean_difference_ci95_low": ci_low,
                        "mean_difference_ci95_high": ci_high,
                        "win_rate_a_over_b": float((difference < 0).mean()),
                        "p_value": p_value,
                    }
                    pending.append(row)
                    pvals.append(p_value)
                valid_indices = [i for i, p in enumerate(pvals) if pd.notna(p)]
                adjusted = _holm([pvals[i] for i in valid_indices]) if valid_indices else []
                for i, p_adj in zip(valid_indices, adjusted):
                    pending[i]["p_holm"] = p_adj
                    pending[i]["significant_0.05"] = p_adj < 0.05
                for row in pending:
                    row.setdefault("p_holm", np.nan)
                    row.setdefault("significant_0.05", False)
                    rows.append(row)
    return pd.DataFrame(rows)


def configurator_resampling_interactions(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Matched-seed difference-in-differences for the two factorial axes.

    For configurator ``c`` and resampling methods A/B, the estimand is
    ``(c - random)_A - (c - random)_B``. Negative values mean configurator ``c``
    has a stronger advantage over Random Search under A than under B.
    """
    rows = []
    metrics = ("final_test_cost", "auc_test", "abs_generalization_gap",
               "final_relative_overtuning")
    for metric in metrics:
        for base_condition, group in summary_df.groupby(BASE_CONDITION_COLUMNS):
            pivot = group.pivot_table(
                index="seed", columns=["optimizer", "resampling"], values=metric
            )
            methods = sorted(group["resampling"].dropna().unique())
            optimizers = sorted(
                optimizer for optimizer in group["optimizer"].unique()
                if optimizer != "random"
            )
            pending, p_values = [], []
            for optimizer in optimizers:
                for method_a, method_b in itertools.combinations(methods, 2):
                    required = [
                        ("random", method_a), (optimizer, method_a),
                        ("random", method_b), (optimizer, method_b),
                    ]
                    if any(column not in pivot.columns for column in required):
                        continue
                    paired = pivot[required].dropna()
                    if not len(paired):
                        continue
                    effect_a = (
                        paired[(optimizer, method_a)] - paired[("random", method_a)]
                    )
                    effect_b = (
                        paired[(optimizer, method_b)] - paired[("random", method_b)]
                    )
                    interaction = effect_a - effect_b
                    ci_low, ci_high = _bootstrap_mean_ci(
                        interaction.to_numpy(dtype=float)
                    )
                    if len(interaction) < 5 or np.allclose(interaction, 0.0):
                        p_value = 1.0 if np.allclose(interaction, 0.0) else np.nan
                    else:
                        try:
                            p_value = float(stats.wilcoxon(interaction).pvalue)
                        except ValueError:
                            p_value = 1.0
                    pending.append({
                        **dict(zip(BASE_CONDITION_COLUMNS, base_condition)),
                        "metric": metric,
                        "optimizer": optimizer,
                        "reference_optimizer": "random",
                        "resampling_a": method_a,
                        "resampling_b": method_b,
                        "difference_in_differences": (
                            f"({optimizer} - random) under {method_a} minus "
                            f"({optimizer} - random) under {method_b}"
                        ),
                        "n_pairs": int(len(interaction)),
                        "mean_effect_a": float(effect_a.mean()),
                        "mean_effect_b": float(effect_b.mean()),
                        "mean_interaction": float(interaction.mean()),
                        "median_interaction": float(interaction.median()),
                        "interaction_ci95_low": ci_low,
                        "interaction_ci95_high": ci_high,
                        "p_value": p_value,
                    })
                    p_values.append(p_value)

            valid = [i for i, value in enumerate(p_values) if pd.notna(value)]
            adjusted = _holm([p_values[i] for i in valid]) if valid else []
            for i, p_adj in zip(valid, adjusted):
                pending[i]["p_holm"] = p_adj
                pending[i]["significant_0.05"] = p_adj < 0.05
            for row in pending:
                row.setdefault("p_holm", np.nan)
                row.setdefault("significant_0.05", False)
                rows.append(row)
    return pd.DataFrame(rows)


def _method_minus_random_effect(summary_df: pd.DataFrame, metric: str,
                                condition: tuple,
                                optimizer: str) -> pd.Series:
    grp = summary_df[_condition_mask_frame(summary_df, condition)]
    pivot = grp.pivot_table(index="seed", columns="optimizer", values=metric)
    if "random" not in pivot.columns or optimizer not in pivot.columns:
        return pd.Series(dtype=float)
    paired = pivot[["random", optimizer]].dropna()
    return paired[optimizer] - paired["random"]


def _condition_mask_frame(df: pd.DataFrame, condition: tuple):
    mask = pd.Series(True, index=df.index)
    for column, value in zip(CONDITION_COLUMNS, condition):
        mask &= df[column] == value
    return mask


def condition_contrasts(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Does a configurator's paired advantage over RS change by condition?

    The estimand is a difference-in-paired-differences:
      (method - RS at level B) - (method - RS at level A).
    Negative values mean the method's advantage (lower cost) becomes stronger
    from A to B.  The targeted condition grid is intentionally not treated as a
    full factorial design.
    """
    columns = [
        "metric", "optimizer", "resampling", "factor", "level_a", "level_b",
        "condition_a", "condition_b", "n_pairs", "mean_effect_a",
        "mean_effect_b", "mean_contrast_b_minus_a", "median_contrast_b_minus_a",
        "contrast_ci95_low", "contrast_ci95_high", "p_value", "p_holm",
        "significant_0.05",
    ]
    resampling_values = sorted(summary_df["resampling"].dropna().unique())
    if len(resampling_values) > 1:
        pieces = [
            condition_contrasts(summary_df[summary_df["resampling"] == value])
            for value in resampling_values
        ]
        pieces = [piece for piece in pieces if len(piece)]
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    resampling_value = str(resampling_values[0]) if resampling_values else "full_train"
    available = {
        (str(r.config_space), str(r.train_family), int(r.train_size))
        for r in summary_df[CONDITION_COLUMNS].drop_duplicates().itertuples(index=False)
    }
    specs = []
    fixed_spaces = [s for s in ("large_fixed_2opt", "fixed_2opt")
                    if any(c[0] == s for c in available)]
    for fixed_space in fixed_spaces:
        sizes = sorted(c[2] for c in available
                       if c[0] == fixed_space and c[1] == "mixed")
        for a, b in itertools.combinations(sizes, 2):
            specs.append(("train_size", str(a), str(b),
                          (fixed_space, "mixed", a),
                          (fixed_space, "mixed", b)))

        families = sorted(c[1] for c in available
                          if c[0] == fixed_space and c[2] == 25)
        for a, b in itertools.combinations(families, 2):
            specs.append(("train_family", a, b,
                          (fixed_space, a, 25),
                          (fixed_space, b, 25)))

        fixed_c = (fixed_space, "mixed", 25)
        full_c = ("full", "mixed", 25)
        if fixed_c in available and full_c in available:
            specs.append(("config_space", fixed_space, "full", fixed_c, full_c))

    optimizers = sorted(o for o in summary_df["optimizer"].unique() if o != "random")
    rows = []
    for metric in ("final_test_cost", "auc_test", "abs_generalization_gap",
                   "final_relative_overtuning"):
        for optimizer in optimizers:
            for factor, level_a, level_b, condition_a, condition_b in specs:
                effect_a = _method_minus_random_effect(
                    summary_df, metric, condition_a, optimizer).rename("effect_a")
                effect_b = _method_minus_random_effect(
                    summary_df, metric, condition_b, optimizer).rename("effect_b")
                paired = pd.concat([effect_a, effect_b], axis=1).dropna()
                if not len(paired):
                    continue
                contrast = paired["effect_b"] - paired["effect_a"]
                ci_low, ci_high = _bootstrap_mean_ci(contrast.to_numpy(dtype=float))
                if len(contrast) < 5 or np.allclose(contrast, 0.0):
                    p_value = 1.0 if np.allclose(contrast, 0.0) else np.nan
                else:
                    try:
                        p_value = float(stats.wilcoxon(contrast).pvalue)
                    except ValueError:
                        p_value = 1.0
                rows.append({
                    "metric": metric,
                    "optimizer": optimizer,
                    "resampling": resampling_value,
                    "factor": factor,
                    "level_a": level_a,
                    "level_b": level_b,
                    "condition_a": "|".join(map(str, condition_a)),
                    "condition_b": "|".join(map(str, condition_b)),
                    "n_pairs": int(len(contrast)),
                    "mean_effect_a": float(paired["effect_a"].mean()),
                    "mean_effect_b": float(paired["effect_b"].mean()),
                    "mean_contrast_b_minus_a": float(contrast.mean()),
                    "median_contrast_b_minus_a": float(contrast.median()),
                    "contrast_ci95_low": ci_low,
                    "contrast_ci95_high": ci_high,
                    "p_value": p_value,
                })

    out = pd.DataFrame(rows)
    if not len(out):
        return pd.DataFrame(columns=columns)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["metric", "factor"]).groups.items():
        valid = [i for i in idx if pd.notna(out.at[i, "p_value"])]
        if valid:
            adjusted = _holm([float(out.at[i, "p_value"]) for i in valid])
            for i, p_adj in zip(valid, adjusted):
                out.at[i, "p_holm"] = p_adj
    out["significant_0.05"] = out["p_holm"] < 0.05
    return out[columns]


def summarize_parameters(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Final-incumbent parameter distributions per optimizer and condition."""
    rows = []
    parameters = [*PARAM_COLUMNS, "move_share_2opt"]
    group_columns = ["optimizer", *CONDITION_COLUMNS]
    for group, grp in summary_df.groupby(group_columns):
        base = dict(zip(group_columns, group))
        for parameter in parameters:
            if parameter not in grp:
                continue
            values = grp[parameter].dropna()
            if not len(values):
                continue
            row = {**base, "parameter": parameter, "n": int(len(values))}
            if pd.api.types.is_numeric_dtype(values):
                numeric = values.astype(float)
                row.update({
                    "kind": "numeric",
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std(ddof=1)) if len(numeric) > 1 else np.nan,
                    "q25": float(numeric.quantile(0.25)),
                    "median": float(numeric.median()),
                    "q75": float(numeric.quantile(0.75)),
                    "top_value": "",
                    "top_fraction": np.nan,
                })
            else:
                counts = values.astype(str).value_counts()
                row.update({
                    "kind": "categorical",
                    "mean": np.nan,
                    "std": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "top_value": counts.index[0],
                    "top_fraction": float(counts.iloc[0] / len(values)),
                })
            rows.append(row)
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

    checkpoint_runs, checkpoint_summary = _budget_checkpoints(grid_df)
    checkpoint_runs_path = processed_dir / "budget_checkpoints.csv"
    checkpoint_summary_path = summaries_dir / "budget_checkpoint_summary.csv"
    checkpoint_runs.to_csv(checkpoint_runs_path, index=False)
    checkpoint_summary.to_csv(checkpoint_summary_path, index=False)

    summary_df = pd.DataFrame([summarize_run_v2(d) for _, d in traj_df.groupby("run_id")])
    summary_df = summary_df.merge(_anytime_scores(grid_df), on="run_id", how="left")
    summary_df = summary_df.merge(
        side_df.drop(columns=["optimizer", "config_space", "resampling", "train_family",
                              "train_size", "seed"]),
        on="run_id", how="left")
    summary_df["configurator_overhead_sec"] = summary_df["ask_sec"] + summary_df["tell_sec"]

    btt = _budget_to_target(traj_df, summary_df)
    if len(btt):
        summary_df = summary_df.merge(
            btt[["run_id", "target_val_cost", "budget_to_target", "speedup_vs_random"]],
            on="run_id", how="left")
    else:
        summary_df["target_val_cost"] = np.nan
        summary_df["budget_to_target"] = np.nan
        summary_df["speedup_vs_random"] = np.nan
    summary_path = summaries_dir / "final_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    cond = (summary_df
            .groupby(["optimizer", *CONDITION_COLUMNS], as_index=False)
            .agg(n_runs=("run_id", "count"),
                 mean_final_test=("final_test_cost", "mean"),
                 se_final_test=("final_test_cost", "sem"),
                 std_final_test=("final_test_cost", "std"),
                 median_final_test=("final_test_cost", "median"),
                 mean_final_val=("final_val_cost", "mean"),
                 median_final_val=("final_val_cost", "median"),
                 mean_gap=("final_generalization_gap", "mean"),
                 mean_abs_gap=("abs_generalization_gap", "mean"),
                 se_abs_gap=("abs_generalization_gap", "sem"),
                 mean_final_overtuning=("final_overtuning", "mean"),
                 mean_test_progress_from_initial=("test_progress_from_initial", "mean"),
                 n_relative_overtuning_eligible=("relative_overtuning_eligible", "sum"),
                 proportion_relative_overtuning_eligible=(
                     "relative_overtuning_eligible", "mean"),
                 mean_rel_overtuning=("final_relative_overtuning", "mean"),
                 proportion_final_overtuned=("is_final_overtuned", "mean"),
                 n_severe_overtuning=("is_severe_overtuning", "sum"),
                 mean_auc_test=("auc_test", "mean"),
                 se_auc_test=("auc_test", "sem"),
                 median_budget_to_target=("budget_to_target", "median"),
                 median_speedup_vs_random=("speedup_vs_random", "median"),
                 mean_distinct_configs=("n_distinct_configs", "mean"),
                 mean_proposals=("n_proposals", "mean"),
                 mean_resampling_sa_calls_per_config=(
                     "resampling_sa_calls_per_config", "mean"),
                 mean_unused_budget_calls=("unused_budget_calls", "mean"),
                 mean_optimization_sa_calls=("total_sa_calls_optimization", "mean"),
                 mean_incumbent_changes=("n_incumbent_changes", "mean"),
                 mean_overhead_sec=("configurator_overhead_sec", "mean"),
                 mean_wallclock_sec=("wallclock_sec", "mean")))
    cond["proportion_severe_among_eligible"] = np.where(
        cond["n_relative_overtuning_eligible"] > 0,
        cond["n_severe_overtuning"] / cond["n_relative_overtuning_eligible"],
        np.nan,
    )
    cond_path = summaries_dir / "condition_summary.csv"
    cond.to_csv(cond_path, index=False)

    tests = pairwise_tests(summary_df)
    tests_path = summaries_dir / "stats_tests.csv"
    tests.to_csv(tests_path, index=False)

    paired = paired_vs_random(summary_df)
    paired_path = summaries_dir / "paired_vs_random.csv"
    paired.to_csv(paired_path, index=False)

    resampling_tests = paired_resampling_tests(summary_df)
    resampling_tests_path = summaries_dir / "paired_resampling_tests.csv"
    resampling_tests.to_csv(resampling_tests_path, index=False)

    interactions = configurator_resampling_interactions(summary_df)
    interactions_path = summaries_dir / "configurator_resampling_interactions.csv"
    interactions.to_csv(interactions_path, index=False)

    parameters = summarize_parameters(summary_df)
    parameters_path = summaries_dir / "parameter_summary.csv"
    parameters.to_csv(parameters_path, index=False)

    contrasts = condition_contrasts(summary_df)
    contrasts_path = summaries_dir / "condition_contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False)

    _write_markdown_table(cond, paired, resampling_tests, interactions, tests, contrasts,
                          summaries_dir / "summary_table.md")

    return {"grid": grid_path, "summary": summary_path, "condition": cond_path,
            "tests": tests_path, "paired": paired_path,
            "resampling_tests": resampling_tests_path,
            "interactions": interactions_path,
            "parameters": parameters_path,
            "checkpoint_runs": checkpoint_runs_path,
            "checkpoint_summary": checkpoint_summary_path,
            "contrasts": contrasts_path,
            "n_runs": len(summary_df)}


def _write_markdown_table(cond: pd.DataFrame, paired: pd.DataFrame,
                          resampling_tests: pd.DataFrame,
                          interactions: pd.DataFrame,
                          tests: pd.DataFrame, contrasts: pd.DataFrame, out: Path):
    lines = [
        "# Experiment v2 — configurator × instance-resampling comparison",
        "",
        "All comparisons use the common cumulative SA-call cap. "
        "Unseen-test values are analysis-only and never reach a configurator.",
        "",
        f"Relative overtuning is reported only when test improvement from the "
        f"initial incumbent is at least {RELATIVE_OVERTUNING_MIN_PROGRESS:g}; "
        "this follows Schneider et al. (2025, Section 5).",
        "",
        "## Per-condition results",
        "",
    ]
    show = cond.copy()
    float_cols = [c for c in show.columns if show[c].dtype.kind == "f"]
    for c in float_cols:
        show[c] = show[c].map(lambda v: f"{v:.5g}" if pd.notna(v) else "—")
    lines.append("| " + " | ".join(show.columns) + " |")
    lines.append("|" + "---|" * len(show.columns))
    for _, r in show.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")

    lines += [
        "",
        "## Paired differences from Random Search",
        "",
        "Differences are method minus Random Search on matched seeds; negative is better. "
        "Confidence intervals are paired percentile-bootstrap intervals.",
        "",
    ]
    if len(paired):
        pshow = paired.copy()
        for c in ["mean_difference", "median_difference", "std_difference",
                  "mean_difference_ci95_low", "mean_difference_ci95_high", "win_rate"]:
            pshow[c] = pshow[c].map(
                lambda v: f"{v:.5g}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(pshow.columns) + " |")
        lines.append("|" + "---|" * len(pshow.columns))
        for _, r in pshow.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r) + " |")

    lines += [
        "",
        "## Paired instance-resampling comparisons within each configurator",
        "",
        "Differences are resampling A minus resampling B on matched seeds; "
        "negative is better. Holm correction is applied within each metric, "
        "condition, and configurator.",
        "",
    ]
    if len(resampling_tests):
        rshow = resampling_tests.copy()
        for c in ["mean_difference", "median_difference", "std_difference",
                  "mean_difference_ci95_low", "mean_difference_ci95_high",
                  "win_rate_a_over_b", "p_value", "p_holm"]:
            rshow[c] = rshow[c].map(
                lambda v: f"{v:.5g}" if pd.notna(v) else "—"
            )
        lines.append("| " + " | ".join(rshow.columns) + " |")
        lines.append("|" + "---|" * len(rshow.columns))
        for _, r in rshow.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r) + " |")

    lines += [
        "",
        "## Configurator × instance-resampling interaction tests",
        "",
        "Each interaction is `(configurator - Random Search under A) - "
        "(configurator - Random Search under B)` on matched seeds. Negative "
        "means the configurator has a stronger advantage under A. Holm correction "
        "is applied within each metric and base condition.",
        "",
    ]
    if len(interactions):
        ishow = interactions.copy()
        for c in ["mean_effect_a", "mean_effect_b", "mean_interaction",
                  "median_interaction", "interaction_ci95_low",
                  "interaction_ci95_high", "p_value", "p_holm"]:
            ishow[c] = ishow[c].map(
                lambda v: f"{v:.5g}" if pd.notna(v) else "—"
            )
        lines.append("| " + " | ".join(ishow.columns) + " |")
        lines.append("|" + "---|" * len(ishow.columns))
        for _, r in ishow.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r) + " |")

    lines += [
        "",
        "Final-incumbent parameter distributions are in `parameter_summary.csv`; "
        "repeated budget summaries are in `budget_checkpoint_summary.csv`.",
        "",
        "## Cross-condition contrasts of the paired method-vs-RS effect",
        "",
        "Each contrast is `(method - RS at B) - (method - RS at A)` on matched "
        "seeds. Negative values mean the method's advantage improves at B. "
        "Holm correction is applied within each metric/factor family.",
        "",
    ]
    if len(contrasts):
        cshow = contrasts.copy()
        for c in ["mean_effect_a", "mean_effect_b", "mean_contrast_b_minus_a",
                  "median_contrast_b_minus_a", "contrast_ci95_low",
                  "contrast_ci95_high", "p_value", "p_holm"]:
            cshow[c] = cshow[c].map(
                lambda v: f"{v:.5g}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(cshow.columns) + " |")
        lines.append("|" + "---|" * len(cshow.columns))
        for _, r in cshow.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r) + " |")

    lines += [
        "",
        "## Pairwise Wilcoxon signed-rank tests (paired by seed, Holm-corrected)",
        "",
    ]
    if len(tests):
        tshow = tests.copy()
        for c in ["median_a", "median_b", "median_diff_a_minus_b", "win_rate_a_over_b",
                  "p_value", "p_holm"]:
            tshow[c] = tshow[c].map(
                lambda v: f"{v:.4g}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(tshow.columns) + " |")
        lines.append("|" + "---|" * len(tshow.columns))
        for _, r in tshow.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
