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
    "holdout": "Holdout instance sampling",
    "cv5": "5-fold instance resampling",
    "repeated_cv5": "Repeated 5-fold instance resampling",
    "bootstrap_oob": "Bootstrap OOB instance resampling",
}
RESAMPLING_COLORS = {
    "holdout": "#d55e00",
    "cv5": "#0072b2",
    "repeated_cv5": "#009e73",
    "bootstrap_oob": "#cc79a7",
}


def _present(values, order):
    return [v for v in order if v in set(values)]


def _label_resampling(df: pd.DataFrame) -> pd.DataFrame:
    """Add report-facing method names without changing stored CSV keys."""
    labelled = df.copy()
    labelled["resampling_method"] = labelled["resampling"].map(
        lambda value: RESAMPLING_LABELS.get(value, value)
    )
    return labelled


def _label_order(keys: list[str]) -> list[str]:
    return [RESAMPLING_LABELS.get(key, key) for key in keys]


def _label_palette(keys: list[str]) -> dict[str, str]:
    return {
        RESAMPLING_LABELS.get(key, key): RESAMPLING_COLORS.get(key, "#777777")
        for key in keys
    }


def _rotate_method_ticks(ax, angle: int = 22):
    ax.tick_params(axis="x", rotation=angle)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")


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
    ax.set_xlabel("configurator iteration")
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
            ax.step(
                vals,
                y,
                where="post",
                label=RESAMPLING_LABELS.get(r, r),
                color=RESAMPLING_COLORS.get(r),
                lw=2.5,
            )
        ax.axvline(
            1.0,
            color="k",
            ls="--",
            lw=1.5,
            label="100% of tuning progress lost (= 1)",
        )
        ax.set_xlabel("relative overtuning")
        ax.set_ylabel("proportion of runs")
        if zoomed:
            ax.set_ylim(0.88, 1.005)
            ax.set_title("ECDF — zoomed (y ≥ 0.88)\nshows method separation")
        else:
            ax.set_title("ECDF — full scale\nshows the complete overtuning tail")
        ax.legend(fontsize=9)

    fig.suptitle("ECDF of relative overtuning by instance-resampling method", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_final_test_by_resampling(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    labelled = _label_resampling(summary)
    label_order = _label_order(order)
    sns.boxplot(
        data=labelled,
        x="resampling_method",
        y="final_test_cost",
        order=label_order,
        palette=_label_palette(order),
        hue="resampling_method",
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=labelled,
        x="resampling_method",
        y="final_test_cost",
        order=label_order,
        color="0.20",
        size=3,
        alpha=0.35,
        ax=ax,
    )
    ax.set_xlabel("instance-resampling method")
    ax.set_ylabel("final test cost (gap)")
    ax.set_title("Final test performance by instance-resampling method")
    _rotate_method_ticks(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gap_by_train_size(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    labelled = _label_resampling(summary)
    sns.pointplot(
        data=labelled,
        x="train_size",
        y="final_generalization_gap",
        hue="resampling_method",
        hue_order=_label_order(order),
        palette=_label_palette(order),
        dodge=0.3,
        ax=ax,
        errorbar="se",
    )
    ax.set_xlabel("training-pool size")
    ax.set_ylabel("generalization gap (test - validation)")
    ax.set_title("Generalization gap vs training size")
    ax.legend(title="instance-resampling method", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_runtime_by_resampling(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    labelled = _label_resampling(summary)
    sns.barplot(
        data=labelled,
        x="resampling_method",
        y="total_validation_runtime_sec",
        order=_label_order(order),
        palette=_label_palette(order),
        hue="resampling_method",
        legend=False,
        errorbar="se",
        ax=ax,
    )
    ax.set_xlabel("instance-resampling method")
    ax.set_ylabel("total validation runtime per run (s)")
    ax.set_title("Computational cost by instance-resampling method")
    _rotate_method_ticks(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_trajectories_by_resampling(traj: pd.DataFrame, out: Path):
    """Mean incumbent trajectories for all four instance-resampling methods."""
    order = _present(traj["resampling"], RESAMPLING_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharex=True)
    for ax, metric, ylabel, title in [
        (axes[0], "val_t", "mean incumbent validation cost", "Validation trajectory"),
        (axes[1], "test_t", "mean incumbent test cost", "Unseen-test trajectory"),
    ]:
        for method in order:
            curve = (
                traj.loc[traj["resampling"] == method]
                .groupby("trial_id")[metric]
                .agg(["mean", "sem"])
                .sort_index()
            )
            x = curve.index.to_numpy(dtype=float)
            mean = curve["mean"].to_numpy(dtype=float)
            sem = curve["sem"].fillna(0.0).to_numpy(dtype=float)
            color = RESAMPLING_COLORS.get(method)
            ax.plot(
                x,
                mean,
                color=color,
                lw=2.5,
                label=RESAMPLING_LABELS.get(method, method),
            )
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.14)
        ax.set_xlabel("configurator iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nmean ± standard error across runs")
        ax.legend(title="instance-resampling method", fontsize=9)
    fig.suptitle("Incumbent trajectories by instance-resampling method", fontsize=16)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_relative_overtuning_by_train_size(summary: pd.DataFrame, out: Path):
    """Eligible final relative overtuning as the training-instance pool grows."""
    data = summary.dropna(subset=["final_relative_overtuning"]).copy()
    if data.empty:
        return
    order = _present(data["resampling"], RESAMPLING_ORDER)
    data = _label_resampling(data)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.pointplot(
        data=data,
        x="train_size",
        y="final_relative_overtuning",
        hue="resampling_method",
        hue_order=_label_order(order),
        palette=_label_palette(order),
        dodge=0.3,
        errorbar="se",
        ax=ax,
    )
    ax.axhline(1.0, color="black", ls="--", lw=1.3, label="all progress lost (= 1)")
    ax.set_xlabel("training-pool size")
    ax.set_ylabel("final relative overtuning")
    ax.set_title(
        "Relative overtuning vs training size\n"
        "reported only for runs with sufficient test-side progress"
    )
    ax.legend(title="instance-resampling method", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_overtuning_frequency(summary: pd.DataFrame, out: Path):
    """Compare overtuning-event rates and relative-metric eligibility."""
    metric_labels = {
        "is_final_overtuned": "Final incumbent overtuned",
        "is_severe_overtuning": "All progress lost (relative ≥ 1)",
        "relative_overtuning_eligible": "Relative metric eligible",
    }
    columns = [column for column in metric_labels if column in summary.columns]
    if not columns:
        return
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    data = _label_resampling(summary)
    long = data.melt(
        id_vars=["resampling_method"],
        value_vars=columns,
        var_name="metric",
        value_name="indicator",
    )
    long["indicator"] = long["indicator"].astype(float)
    long["outcome"] = long["metric"].map(metric_labels)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        data=long,
        x="resampling_method",
        y="indicator",
        hue="outcome",
        order=_label_order(order),
        errorbar="se",
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("instance-resampling method")
    ax.set_ylabel("proportion of runs")
    ax.set_title("Overtuning outcomes by instance-resampling method")
    ax.legend(title="run-level outcome", fontsize=10)
    _rotate_method_ticks(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_resampling_tradeoff(summary: pd.DataFrame, out: Path):
    """Show validation-evaluation cost against two generalization outcomes."""
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    outcomes = [
        ("final_generalization_gap", "mean generalization gap", "Runtime vs generalization gap"),
        ("final_relative_overtuning", "mean final relative overtuning", "Runtime vs relative overtuning"),
    ]
    for ax, (outcome, ylabel, title) in zip(axes, outcomes):
        plotted = False
        for method in order:
            sub = summary.loc[summary["resampling"] == method, [
                "total_validation_runtime_sec", outcome
            ]].dropna()
            if sub.empty:
                continue
            x = float(sub["total_validation_runtime_sec"].mean())
            y = float(sub[outcome].mean())
            xerr = float(sub["total_validation_runtime_sec"].sem()) if len(sub) > 1 else 0.0
            yerr = float(sub[outcome].sem()) if len(sub) > 1 else 0.0
            color = RESAMPLING_COLORS.get(method)
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                ms=10,
                capsize=4,
                color=color,
                label=RESAMPLING_LABELS.get(method, method),
            )
            plotted = True
        if plotted:
            ax.set_xscale("log")
            ax.set_xlabel("mean validation runtime per run (s, log scale)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}\npoints show mean ± standard error")
            ax.legend(title="instance-resampling method", fontsize=8)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No eligible data for {ylabel}", ha="center", va="center")
    fig.suptitle("Quality–cost trade-off of instance-resampling methods", fontsize=16)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_param_stability(raw_dir: Path, out: Path):
    """Compare every selected v1 SA parameter across instance-resampling methods."""
    rows = []
    for csv_path in sorted(Path(raw_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        if not len(df) or "validation_cost" not in df.columns:
            continue
        inc = df.loc[df["validation_cost"].idxmin()]
        rows.append(inc)
    if not rows:
        return
    # Each selected Series is named by its original trial index. Resetting avoids
    # duplicate row labels when several runs select the same trial number.
    sel = _label_resampling(pd.DataFrame(rows).reset_index(drop=True))
    order = _present(sel["resampling"], RESAMPLING_ORDER)
    label_order = _label_order(order)
    palette = _label_palette(order)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    numeric = [
        ("initial_temperature", "Initial temperature", True),
        ("cooling_rate", "Cooling rate", False),
        ("iterations_per_temp", "Iterations per temperature", False),
        ("restarts", "Restarts", False),
    ]
    for ax, (column, title, log_scale) in zip(axes.flat[:4], numeric):
        sns.boxplot(
            data=sel,
            x="resampling_method",
            y=column,
            order=label_order,
            palette=palette,
            hue="resampling_method",
            legend=False,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=sel,
            x="resampling_method",
            y=column,
            order=label_order,
            color="0.15",
            alpha=0.25,
            size=2.5,
            jitter=0.22,
            ax=ax,
        )
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("")
        _rotate_method_ticks(ax, angle=20)

    move_ax = axes.flat[4]
    move_order = [m for m in ["swap", "insert", "2opt"] if m in set(sel["move_type"])]
    shares = pd.crosstab(
        sel["resampling_method"], sel["move_type"], normalize="index"
    ).reindex(index=label_order, columns=move_order, fill_value=0.0)
    shares.plot(
        kind="bar",
        stacked=True,
        color={"swap": "#e69f00", "insert": "#56b4e9", "2opt": "#009e73"},
        width=0.78,
        ax=move_ax,
    )
    move_ax.set_ylim(0, 1)
    move_ax.set_ylabel("proportion of selected incumbents")
    move_ax.set_xlabel("")
    move_ax.set_title("Move-type selection")
    move_ax.legend(title="move type", fontsize=10)
    _rotate_method_ticks(move_ax, angle=20)

    effect_ax = axes.flat[5]
    if sel["move_type"].nunique() > 1 and sel["test_cost"].notna().any():
        sns.boxplot(
            data=sel,
            x="move_type",
            y="test_cost",
            hue="resampling_method",
            hue_order=label_order,
            palette=palette,
            showfliers=False,
            ax=effect_ax,
        )
        effect_ax.set_xlabel("selected move type")
        effect_ax.set_ylabel("final test cost (gap)")
        effect_ax.set_title("Move type and final test performance")
        effect_ax.legend(title="instance-resampling method", fontsize=8)
    else:
        effect_ax.axis("off")
        effect_ax.text(
            0.5,
            0.55,
            "Move type is fixed to 2-opt\nin this configuration-space variant.",
            ha="center",
            va="center",
            fontsize=16,
        )

    fig.suptitle(
        "Selected SA parameter comparisons across instance-resampling methods",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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


def plot_composition_by_resampling(summary: pd.DataFrame, out: Path):
    """Distribution-shift heatmaps separated by instance-resampling method."""
    families = ["uniform", "clustered", "mixed"]
    order = _present(summary["resampling"], RESAMPLING_ORDER)
    matrices: dict[str, np.ndarray] = {}
    for method in order:
        matrix = np.full((len(families), len(families)), np.nan)
        method_data = summary[summary["resampling"] == method]
        for i, train_family in enumerate(families):
            cell_rows = method_data[method_data["train_family"] == train_family]
            for j, test_family in enumerate(families):
                column = f"test_t_{test_family}"
                if column in cell_rows and not cell_rows.empty:
                    matrix[i, j] = cell_rows[column].mean()
        matrices[method] = matrix
    if not matrices:
        return

    finite = np.concatenate([m[np.isfinite(m)] for m in matrices.values()])
    vmin = float(finite.min()) if len(finite) else None
    vmax = float(finite.max()) if len(finite) else None
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), sharex=True, sharey=True)
    for index, (ax, method) in enumerate(zip(axes.flat, order)):
        sns.heatmap(
            matrices[method],
            annot=True,
            fmt=".4f",
            xticklabels=families,
            yticklabels=families,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=index == len(order) - 1,
            cbar_kws={"label": "mean final test cost"},
            ax=ax,
        )
        ax.set_title(RESAMPLING_LABELS.get(method, method))
        ax.set_xlabel("test family")
        ax.set_ylabel("train family")
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    fig.suptitle(
        "Composition / distribution shift by instance-resampling method\n"
        "cells show mean final test cost (not generalization gap)",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    ax0.set_xlabel("configurator iteration")
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
    ax1.set_xlabel("configurator iteration")
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
    ax2.set_xlabel("instance-resampling method")
    ax2.set_ylabel("final test cost (gap)")
    ax2.set_title("Final test cost: TPE vs Random Search")
    _rotate_method_ticks(ax2, angle=22)
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
    _do(plot_trajectories_by_resampling, "trajectories_by_resampling.png", traj)
    _do(plot_ecdf_relative_overtuning, "ecdf_relative_overtuning.png", traj)
    _do(plot_relative_overtuning_by_train_size,
        "relative_overtuning_by_train_size.png", summary)
    _do(plot_overtuning_frequency, "overtuning_frequency_by_resampling.png", summary)
    _do(plot_final_test_by_resampling, "final_test_by_resampling.png", summary)
    _do(plot_gap_by_train_size, "generalization_gap_by_train_size.png", summary)
    _do(plot_runtime_by_resampling, "runtime_by_resampling.png", summary)
    _do(plot_resampling_tradeoff, "resampling_quality_runtime_tradeoff.png", summary)
    _do(plot_param_stability, "selected_params_stability.png", raw_dir)
    _do(plot_composition_heatmap, "composition_heatmap.png", summary)
    _do(plot_composition_by_resampling, "composition_by_resampling.png", summary)
    _do(plot_optimizer_comparison, "optimizer_comparison.png", traj, summary)
    return produced
