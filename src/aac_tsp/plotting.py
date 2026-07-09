"""Figures for the report. All read from the post-processed CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
RESAMPLING_ORDER = ["holdout", "cv5", "repeated_cv5", "bootstrap_oob"]
# Readable names for the report/figures. Keys stay as-is in code/CSVs; these are
# display-only. "cv5" is 5-fold *instance* resampling, not model cross-validation.
RESAMPLING_LABELS = {
    "holdout": "holdout",
    "cv5": "5-fold instance resampling",
    "repeated_cv5": "repeated 5-fold instance resampling",
    "bootstrap_oob": "bootstrap OOB",
}


def _present(values, order):
    return [v for v in order if v in set(values)]


def plot_trajectory(traj: pd.DataFrame, out: Path):
    """Validation vs test incumbent for one representative run (shows overtuning)."""
    runs = traj["run_id"].unique()
    # pick the run with the largest final overtuning to make the effect visible
    end = traj.sort_values("trial_id").groupby("run_id").tail(1)
    pick = end.loc[end["overtuning_t"].idxmax(), "run_id"] if len(end) else runs[0]
    d = traj[traj["run_id"] == pick].sort_values("trial_id")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(d["trial_id"], d["val_t"], label="validation incumbent", lw=2.5)
    ax.plot(d["trial_id"], d["test_t"], label="test (same-distribution)", lw=2.5)
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("normalized cost (gap)")
    ax.set_title(f"Validation vs test incumbent\n{pick}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ecdf_relative_overtuning(traj: pd.DataFrame, out: Path):
    """ECDF of relative overtuning by resampling (mirrors the paper, Fig. 2).

    Two panels: left = full x-range (shows the heavy tail); right = zoomed to the
    top 12% of the y-axis (starts at 0.88) so inter-resampling separation is visible.
    Mirrors the presentation style in Schneider et al. (AutoML 2025), who start the
    y-axis at 0.3 for the same reason.
    """
    end = traj.sort_values("trial_id").groupby("run_id").tail(1).copy()
    end = end.dropna(subset=["relative_overtuning_t"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, zoomed in zip(axes, [False, True]):
        for r in _present(end["resampling"], RESAMPLING_ORDER):
            vals = np.sort(end.loc[end["resampling"] == r, "relative_overtuning_t"].to_numpy())
            if len(vals) == 0:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", label=RESAMPLING_LABELS.get(r, r), lw=2.5)
        ax.axvline(1.0, color="k", ls="--", lw=1.5, label="all progress lost (=1)")
        ax.set_xlabel("relative overtuning")
        ax.set_ylabel("proportion of runs")
        if zoomed:
            ax.set_ylim(0.88, 1.005)
            ax.set_title("ECDF — zoomed (y ≥ 0.88)\nshows resampling separation")
        else:
            ax.set_title("ECDF — full scale\nshows heavy tail of holdout")
        ax.legend(fontsize=11)

    fig.suptitle("ECDF of relative overtuning by resampling", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_final_test_by_resampling(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    sns.boxplot(data=summary, x="resampling", y="final_test_cost", order=order, ax=ax)
    sns.stripplot(data=summary, x="resampling", y="final_test_cost", order=order,
                  color="0.25", size=4, alpha=0.5, ax=ax)
    ax.set_xlabel("resampling")
    ax.set_ylabel("final test cost (gap)")
    ax.set_title("Final test performance by resampling")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gap_by_train_size(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    sns.pointplot(data=summary, x="train_size", y="final_generalization_gap",
                  hue="resampling", hue_order=order, dodge=0.3, ax=ax, errorbar="sd")
    ax.set_xlabel("training-pool size")
    ax.set_ylabel("generalization gap (test - validation)")
    ax.set_title("Generalization gap vs training size")
    ax.legend(title="resampling", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_runtime_by_resampling(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    sns.barplot(data=summary, x="resampling", y="total_validation_runtime_sec",
                order=order, errorbar="sd", ax=ax)
    ax.set_xlabel("resampling")
    ax.set_ylabel("total validation runtime per run (s)")
    ax.set_title("Computational cost by resampling")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_param_stability(raw_dir: Path, out: Path):
    """Distribution of selected SA parameters of the final incumbent, across seeds."""
    rows = []
    for csv_path in sorted(Path(raw_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        if not len(df):
            continue
        inc = df.loc[df["validation_cost"].idxmin()]
        rows.append(inc)
    if not rows:
        return
    sel = pd.DataFrame(rows)
    order = _present(sel["resampling"], RESAMPLING_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.boxplot(data=sel, x="resampling", y="cooling_rate", order=order, ax=axes[0])
    axes[0].set_title("cooling_rate")
    sns.boxplot(data=sel, x="resampling", y="initial_temperature", order=order, ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_title("initial_temperature")
    sns.countplot(data=sel, x="move_type", hue="resampling", hue_order=order, ax=axes[2])
    axes[2].set_title("move_type")
    for a in axes:
        a.set_xlabel("")
        a.tick_params(axis="x", rotation=30)
    fig.suptitle("Selected-configuration stability across seeds")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_composition_heatmap(summary: pd.DataFrame, out: Path):
    """Mean final test cost for each train-family × test-family pair (distribution shift).

    Each cell is the mean normalised tour gap m(θ,π) = (tour − best_known)/best_known
    of the final incumbent, averaged across all seeds, resamplings, and train sizes.
    Note: this is *test cost*, NOT the generalisation gap (test − validation).
    The dominant axis is test-family difficulty (columns), not the train↔test match (rows).
    """
    fam = ["uniform", "clustered", "mixed"]
    mat = np.full((len(fam), len(fam)), np.nan)
    for i, tf in enumerate(fam):
        sub = summary[summary["train_family"] == tf]
        if not len(sub):
            continue
        for j, ef in enumerate(fam):
            col = f"test_t_{ef}"
            if col in sub.columns:
                mat[i, j] = sub[col].mean()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(mat, annot=True, fmt=".4f", xticklabels=fam, yticklabels=fam,
                cmap="viridis", cbar_kws={"label": "mean final test cost"}, ax=ax)
    ax.set_xlabel("test family")
    ax.set_ylabel("train family")
    ax.set_title("Composition / distribution shift\n(mean final test cost, not gap)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_optimizer_comparison(traj: pd.DataFrame, summary: pd.DataFrame, out: Path):
    """TPE vs Random Search: convergence curves (full + zoomed) + final test cost.

    Three panels:
      Left:   full-scale convergence — shows the iteration-0 spike of RS and how quickly
              both optimizers drop to their operating range.
      Centre: zoomed convergence (iter ≥ 2) — shows the actual TPE vs RS separation once
              the initial random samples are past.
      Right:  boxplot of final test cost by optimizer × resampling.
    """
    OPTIMIZER_LABELS = {"optuna_tpe": "Bayesian Opt. (TPE)", "random": "Random Search"}
    optimizers = ["optuna_tpe", "random"]
    colors = {"optuna_tpe": "#1f77b4", "random": "#ff7f0e"}

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Build mean convergence curves once
    curves = {}
    for opt in optimizers:
        sub = traj[traj["optimizer"] == opt].copy()
        sub["test_t"] = sub.groupby("run_id")["test_t"].ffill()
        curves[opt] = sub.groupby("trial_id")["test_t"].mean()

    # Left: full scale (shows iter-0 spike)
    ax0 = axes[0]
    for opt in optimizers:
        c = curves[opt]
        ax0.plot(c.index, c.values, label=OPTIMIZER_LABELS[opt], lw=2.5, color=colors[opt])
    ax0.set_xlabel("BO iteration")
    ax0.set_ylabel("mean test cost (gap)")
    ax0.set_title("Convergence — full scale\n(shows initial random spike)")
    ax0.legend(fontsize=11)

    # Centre: zoomed (iter >= 2, removes the spike)
    ax1 = axes[1]
    for opt in optimizers:
        c = curves[opt]
        c_zoom = c[c.index >= 2]
        ax1.plot(c_zoom.index, c_zoom.values, label=OPTIMIZER_LABELS[opt], lw=2.5,
                 color=colors[opt])
    ax1.set_xlabel("BO iteration")
    ax1.set_ylabel("mean test cost (gap)")
    ax1.set_title("Convergence — zoomed (iter ≥ 2)\nshows TPE vs RS separation")
    ax1.legend(fontsize=11)

    # Right: final test cost by resampling, split by optimizer
    ax2 = axes[2]
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    summary_plot = summary.copy()
    summary_plot["resampling_label"] = summary_plot["resampling"].map(
        lambda x: RESAMPLING_LABELS.get(x, x))
    summary_plot["optimizer_label"] = summary_plot["optimizer"].map(
        lambda x: OPTIMIZER_LABELS.get(x, x))
    label_order = [RESAMPLING_LABELS.get(k, k) for k in order]
    sns.boxplot(data=summary_plot, x="resampling_label", y="final_test_cost",
                hue="optimizer_label", order=label_order,
                hue_order=[OPTIMIZER_LABELS["optuna_tpe"], OPTIMIZER_LABELS["random"]],
                palette={"Bayesian Opt. (TPE)": colors["optuna_tpe"],
                         "Random Search": colors["random"]},
                ax=ax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("final test cost (gap)")
    ax2.set_title("Final test cost: TPE vs Random Search")
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(title="optimizer", fontsize=11)

    fig.suptitle("Bayesian Optimization (TPE) vs Random Search baseline", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def make_all(processed_dir: Path, summaries_dir: Path, raw_dir: Path, fig_dir: Path) -> list[Path]:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(Path(processed_dir) / "incumbent_trajectories.csv")
    summary = pd.read_csv(Path(summaries_dir) / "final_summary.csv")
    produced = []

    def _do(fn, name, *args):
        path = fig_dir / name
        fn(*args, path)
        produced.append(path)

    _do(plot_trajectory, "trajectory_validation_vs_test.png", traj)
    _do(plot_ecdf_relative_overtuning, "ecdf_relative_overtuning.png", traj)
    _do(plot_final_test_by_resampling, "final_test_by_resampling.png", summary)
    _do(plot_gap_by_train_size, "generalization_gap_by_train_size.png", summary)
    _do(plot_runtime_by_resampling, "runtime_by_resampling.png", summary)
    _do(plot_param_stability, "selected_params_stability.png", raw_dir)
    _do(plot_composition_heatmap, "composition_heatmap.png", summary)
    _do(plot_optimizer_comparison, "optimizer_comparison.png", traj, summary)
    return produced
