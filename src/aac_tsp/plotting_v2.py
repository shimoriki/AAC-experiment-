"""Figures for experiment v2: configurator × instance-resampling comparison.

All figures read the post-processed CSVs (plus, for the exploration/intensification
figures, the raw per-proposal trials CSVs). Every plotting function tolerates a
subset of arms being present, so partial results (e.g. a local smoke test without
SMAC) still plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import RELATIVE_OVERTUNING_MIN_PROGRESS
from .plotting import (
    plot_ecdf_relative_overtuning as plot_v1_ecdf_relative_overtuning,
    plot_final_test_by_resampling as plot_v1_final_test_by_resampling,
    plot_gap_by_train_size as plot_v1_gap_by_train_size,
    plot_overtuning_frequency as plot_v1_overtuning_frequency,
    plot_relative_overtuning_by_train_size as plot_v1_relative_overtuning_by_train_size,
    plot_resampling_tradeoff as plot_v1_resampling_tradeoff,
    plot_runtime_by_resampling as plot_v1_runtime_by_resampling,
)

sns.set_theme(style="whitegrid", context="talk")

OPTIMIZER_ORDER = ["random", "optuna_tpe", "smac_bo", "smac_aac"]
OPTIMIZER_LABELS = {
    "random": "Random Search",
    "optuna_tpe": "Optuna TPE",
    "smac_bo": "SMAC (fixed budget)",
    "smac_aac": "SMAC (intensification)",
}
PALETTE = {
    "random": "#7f7f7f",
    "optuna_tpe": "#0173b2",
    "smac_bo": "#de8f05",
    "smac_aac": "#029e73",
}
OPTIMIZER_MARKERS = {
    "random": "o",
    "optuna_tpe": "s",
    "smac_bo": "^",
    "smac_aac": "D",
}
SPACE_LABELS = {
    "large_fixed_2opt": "fixed 2-opt (11p)",
    "fixed_2opt": "fixed 2-opt (11p)",
    "full": "full space (15p)",
}
RESAMPLING_ORDER = ["holdout", "cv5", "repeated_cv5", "bootstrap_oob", "full_train"]
RESAMPLING_LABELS = {
    "holdout": "Holdout instance sampling",
    "cv5": "5-fold instance resampling",
    "repeated_cv5": "Repeated 5-fold instance resampling",
    "bootstrap_oob": "Bootstrap OOB instance resampling",
    "full_train": "Full training-set evaluation",
}
RESAMPLING_PALETTE = {
    "holdout": "#d55e00",
    "cv5": "#0072b2",
    "repeated_cv5": "#009e73",
    "bootstrap_oob": "#cc79a7",
    "full_train": "#7f7f7f",
}


def _present(values) -> list[str]:
    have = set(values)
    return [o for o in OPTIMIZER_ORDER if o in have]


def _save(fig, out: Path, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _sizes(df) -> list[int]:
    return sorted(df["train_size"].unique())


def _condition_tuples(df: pd.DataFrame) -> list[tuple[str, str, int, str]]:
    work = df.copy()
    if "resampling" not in work:
        work["resampling"] = "full_train"
    conditions = {
        (str(r.config_space), str(r.train_family), int(r.train_size), str(r.resampling))
        for r in work[["config_space", "train_family", "train_size", "resampling"]]
        .drop_duplicates().itertuples(index=False)
    }

    def key(condition):
        space, family, size, resampling = condition
        resampling_rank = (RESAMPLING_ORDER.index(resampling)
                           if resampling in RESAMPLING_ORDER else len(RESAMPLING_ORDER))
        if space in {"large_fixed_2opt", "fixed_2opt"} and family == "mixed":
            return (0, size, resampling_rank)
        if space == "full" and family == "mixed":
            return (1, size, resampling_rank)
        return (2 + {"uniform": 0, "clustered": 1}.get(family, 2), size,
                resampling_rank)

    return sorted(conditions, key=key)


def _condition_label(condition: tuple[str, str, int, str]) -> str:
    space, family, size, resampling = condition
    return (f"{SPACE_LABELS.get(space, space)} | {family} | n={size} | "
            f"{RESAMPLING_LABELS.get(resampling, resampling)}")


def _condition_mask(df: pd.DataFrame, condition: tuple[str, str, int, str]):
    space, family, size, resampling = condition
    resampling_series = (df["resampling"] if "resampling" in df
                         else pd.Series("full_train", index=df.index))
    return ((df["config_space"] == space) &
            (df["train_family"] == family) &
            (df["train_size"] == size) &
            (resampling_series == resampling))


# --------------------------------------------------------------------------- #
# Cross-condition headline comparisons
# --------------------------------------------------------------------------- #

def plot_bo_vs_random_conditions(paired: pd.DataFrame, out: Path):
    """Paired method-minus-RS effects with paired-bootstrap 95% CIs."""
    panels = [
        ("final_test_cost", "final unseen-test cost"),
        ("auc_test", "anytime test-trajectory AUC"),
        ("abs_generalization_gap", "absolute generalization gap"),
        ("final_relative_overtuning", "relative overtuning"),
    ]
    conditions = _condition_tuples(paired)
    labels = [_condition_label(c) for c in conditions]
    methods = [o for o in OPTIMIZER_ORDER if o != "random" and
               o in set(paired["optimizer"])]
    fig, axes = plt.subplots(2, 2, figsize=(20, max(11, 1.15 * len(conditions))),
                             squeeze=False)
    offsets = np.linspace(-0.22, 0.22, max(1, len(methods)))
    for ax, (metric, title) in zip(axes.flat, panels):
        data = paired[paired["metric"] == metric]
        for offset, method in zip(offsets, methods):
            xs, ys, err_low, err_high = [], [], [], []
            for y, condition in enumerate(conditions):
                row = data[_condition_mask(data, condition) &
                           (data["optimizer"] == method)]
                if not len(row):
                    continue
                r = row.iloc[0]
                xs.append(float(r["mean_difference"]))
                ys.append(y + offset)
                mean = float(r["mean_difference"])
                if ("mean_difference_ci95_low" in r.index
                        and pd.notna(r["mean_difference_ci95_low"])
                        and pd.notna(r["mean_difference_ci95_high"])):
                    err_low.append(max(0.0, mean - float(r["mean_difference_ci95_low"])))
                    err_high.append(max(0.0, float(r["mean_difference_ci95_high"]) - mean))
                else:
                    n = max(1, int(r["n_pairs"]))
                    normal = (1.96 * float(r["std_difference"]) / np.sqrt(n)
                              if pd.notna(r["std_difference"]) else 0.0)
                    err_low.append(normal)
                    err_high.append(normal)
            ax.errorbar(xs, ys, xerr=np.asarray([err_low, err_high]),
                        fmt="o", ms=7, capsize=3,
                        color=PALETTE[method], label=OPTIMIZER_LABELS[method])
        ax.axvline(0.0, color="black", ls="--", lw=1.2)
        show_labels = ax is axes[0, 0] or ax is axes[1, 0]
        ax.set_yticks(range(len(labels)), labels if show_labels else [])
        ax.invert_yaxis()
        ax.set_xlabel("method minus Random Search (negative is better)")
        ax.set_title(title)
    axes[0, 1].legend(fontsize=11, loc="best")
    fig.suptitle("BO-style configurators versus Random Search across conditions\n"
                 "(paired seeds; points are mean differences, bars are bootstrap 95% CIs)",
                 y=1.01)
    _save(fig, out)


def plot_bo_vs_random_winrates(paired: pd.DataFrame, out: Path):
    """Condition x method heatmaps for the two primary performance metrics."""
    metrics = [("final_test_cost", "final test cost"),
               ("auc_test", "anytime AUC")]
    conditions = _condition_tuples(paired)
    labels = [_condition_label(c) for c in conditions]
    methods = [o for o in OPTIMIZER_ORDER if o != "random" and
               o in set(paired["optimizer"])]
    fig, axes = plt.subplots(1, 2, figsize=(17, max(7, 0.9 * len(conditions))),
                             squeeze=False)
    for ax, (metric, title) in zip(axes[0], metrics):
        values = np.full((len(conditions), len(methods)), np.nan)
        annotations = np.full(values.shape, "", dtype=object)
        data = paired[paired["metric"] == metric]
        for i, condition in enumerate(conditions):
            for j, method in enumerate(methods):
                row = data[_condition_mask(data, condition) &
                           (data["optimizer"] == method)]
                if not len(row):
                    continue
                r = row.iloc[0]
                values[i, j] = float(r["win_rate"])
                annotations[i, j] = f"{values[i, j]:.2f}\n{int(r['n_beats_random'])}/{int(r['n_pairs'])}"
        sns.heatmap(values, vmin=0, vmax=1, center=0.5, cmap="RdYlGn",
                    annot=annotations, fmt="", linewidths=0.5,
                    xticklabels=[OPTIMIZER_LABELS[m] for m in methods],
                    yticklabels=labels, cbar_kws={"label": "P(method beats RS)"}, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.suptitle("Paired win rates against Random Search across conditions\n"
                 "(annotation: win rate and seeds won/paired)", y=1.01)
    _save(fig, out)


def plot_final_parameter_shifts(summary: pd.DataFrame, out: Path):
    """Robustly standardized final-incumbent parameter shifts versus RS."""
    work = summary.copy()
    work["log10_initial_temperature"] = np.log10(work["initial_temperature"])
    work["log10_iterations_per_temp"] = np.log10(work["iterations_per_temp"])
    parameters = [
        ("log10_initial_temperature", "log10 initial T"),
        ("cooling_rate", "cooling rate"),
        ("log10_iterations_per_temp", "log10 iterations/T"),
        ("restarts", "restarts"),
        ("move_share_2opt", "2-opt move share"),
    ]
    conditions = _condition_tuples(work)
    methods = [o for o in OPTIMIZER_ORDER if o != "random" and
               o in set(work["optimizer"])]
    rows, labels = [], []
    for condition in conditions:
        data = work[_condition_mask(work, condition)]
        reference = data[data["optimizer"] == "random"]
        for method in methods:
            selected = data[data["optimizer"] == method]
            values = []
            for column, _ in parameters:
                pooled = data[column].dropna().astype(float)
                rs = reference[column].dropna().astype(float)
                candidate = selected[column].dropna().astype(float)
                scale = float(pooled.quantile(0.75) - pooled.quantile(0.25))
                if not len(rs) or not len(candidate) or scale <= 1e-12:
                    values.append(np.nan)
                else:
                    values.append(float((candidate.median() - rs.median()) / scale))
            rows.append(values)
            labels.append(f"{_condition_label(condition)} | {OPTIMIZER_LABELS[method]}")
    matrix = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(13, max(9, 0.48 * len(labels))))
    sns.heatmap(matrix, center=0, vmin=-2, vmax=2, cmap="vlag", annot=True,
                fmt=".1f", xticklabels=[label for _, label in parameters],
                yticklabels=labels, cbar_kws={"label": "median shift vs RS / pooled IQR"},
                ax=ax)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    ax.set_title("How BO-selected final parameters differ from Random Search\n"
                 "(descriptive selection shift; sign does not imply better performance)")
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 1-2. Convergence on the common budget axis (median + IQR band across seeds)
# --------------------------------------------------------------------------- #

def plot_convergence(grid: pd.DataFrame, out: Path, metric: str, title: str):
    sizes = _sizes(grid)
    fig, axes = plt.subplots(1, len(sizes), figsize=(8 * len(sizes), 6),
                             sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = grid[grid["train_size"] == size]
        for opt in _present(d["optimizer"]):
            g = (d[d["optimizer"] == opt].groupby("budget")[metric]
                 .quantile([0.25, 0.5, 0.75]).unstack())
            g = g.dropna()
            if not len(g):
                continue
            ax.plot(g.index, g[0.5], lw=2.5, color=PALETTE[opt],
                    label=OPTIMIZER_LABELS[opt])
            ax.fill_between(g.index, g[0.25], g[0.75], color=PALETTE[opt], alpha=0.18)
        ax.set_xscale("log")
        ax.set_xlabel("budget (SA solver calls)")
        ax.set_title(f"train pool n={size}")
    axes[0][0].set_ylabel(f"incumbent {title}\n(median, IQR band)")
    axes[0][-1].legend(fontsize=12)
    fig.suptitle(f"Anytime performance — incumbent {title} vs budget", y=1.02)
    _save(fig, out)


def plot_best_test_trajectory(grid: pd.DataFrame, out: Path):
    """Post-hoc best test cost seen among validation incumbents (never feedback)."""
    sizes = _sizes(grid)
    fig, axes = plt.subplots(1, len(sizes), figsize=(8 * len(sizes), 6),
                             sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = grid[grid["train_size"] == size]
        for opt in _present(d["optimizer"]):
            g = (d[d["optimizer"] == opt].groupby("budget")["best_test_cost"]
                 .quantile([0.25, 0.5, 0.75]).unstack().dropna())
            if not len(g):
                continue
            ax.plot(g.index, g[0.5], lw=2.5, color=PALETTE[opt],
                    label=OPTIMIZER_LABELS[opt])
            ax.fill_between(g.index, g[0.25], g[0.75],
                            color=PALETTE[opt], alpha=0.18)
        ax.set_xscale("log")
        ax.set_xlabel("budget (SA solver calls)")
        ax.set_title(f"train pool n={size}")
    axes[0][0].set_ylabel("best test cost seen\n(median, IQR band)")
    axes[0][-1].legend(fontsize=12)
    fig.suptitle("Best-so-far test trajectory (post-hoc analysis only; no test feedback)",
                 y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 3. Final cost distributions
# --------------------------------------------------------------------------- #

def plot_final_boxplots(summary: pd.DataFrame, out: Path):
    sizes = _sizes(summary)
    metrics = [("final_val_cost", "final validation cost"),
               ("final_test_cost", "final test cost")]
    fig, axes = plt.subplots(len(metrics), len(sizes),
                             figsize=(7 * len(sizes), 5.5 * len(metrics)),
                             squeeze=False)
    for i, (metric, label) in enumerate(metrics):
        for j, size in enumerate(sizes):
            ax = axes[i][j]
            d = summary[summary["train_size"] == size]
            order = _present(d["optimizer"])
            sns.boxplot(data=d, x="optimizer", y=metric, order=order, hue="optimizer",
                        palette=PALETTE, legend=False, showfliers=False, width=0.6, ax=ax)
            sns.stripplot(data=d, x="optimizer", y=metric, order=order, color="0.2",
                          size=4, alpha=0.6, jitter=0.18, ax=ax)
            ax.set_xticks(range(len(order)),
                          [OPTIMIZER_LABELS[o] for o in order],
                          rotation=20, ha="right", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel(label if j == 0 else "")
            ax.set_title(f"n={size}")
    fig.suptitle("Final incumbent quality after the full budget (each dot = one seed)",
                 y=1.0)
    _save(fig, out)


def plot_anytime_auc(summary: pd.DataFrame, out: Path):
    """Normalized AUC of the post-hoc best-so-far test trajectory."""
    sizes = _sizes(summary)
    fig, axes = plt.subplots(1, len(sizes), figsize=(7 * len(sizes), 6), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = summary[summary["train_size"] == size]
        order = _present(d["optimizer"])
        sns.boxplot(data=d, x="optimizer", y="auc_test", order=order,
                    hue="optimizer", palette=PALETTE, legend=False,
                    showfliers=False, width=0.6, ax=ax)
        sns.stripplot(data=d, x="optimizer", y="auc_test", order=order,
                      color="0.2", size=4, alpha=0.65, jitter=0.18, ax=ax)
        rs = d.loc[d["optimizer"] == "random", "auc_test"].median()
        if pd.notna(rs):
            ax.axhline(rs, color=PALETTE["random"], ls="--", lw=1.4,
                       label="Random Search median")
            ax.legend(fontsize=10)
        ax.set_xticks(range(len(order)), [OPTIMIZER_LABELS[o] for o in order],
                      rotation=20, ha="right", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("normalized best-test AUC (lower is better)")
        ax.set_title(f"n={size}")
    fig.suptitle("Anytime performance across matched seeds (post-hoc test envelope)",
                 y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 4. ECDF of final test cost
# --------------------------------------------------------------------------- #

def plot_ecdf_final_test(summary: pd.DataFrame, out: Path):
    sizes = _sizes(summary)
    fig, axes = plt.subplots(1, len(sizes), figsize=(8 * len(sizes), 6),
                             sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = summary[summary["train_size"] == size]
        for opt in _present(d["optimizer"]):
            vals = np.sort(d.loc[d["optimizer"] == opt, "final_test_cost"].to_numpy())
            if not len(vals):
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", lw=2.5, color=PALETTE[opt],
                    label=OPTIMIZER_LABELS[opt])
        ax.set_xlabel("final test cost")
        ax.set_title(f"n={size}")
    axes[0][0].set_ylabel("proportion of seeds")
    axes[0][-1].legend(fontsize=12)
    fig.suptitle("ECDF of final test cost (further left = better)", y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 5. Mean rank over budget (paired across seeds; complete cases only)
# --------------------------------------------------------------------------- #

def plot_rank_over_budget(grid: pd.DataFrame, out: Path):
    sizes = _sizes(grid)
    fig, axes = plt.subplots(1, len(sizes), figsize=(8 * len(sizes), 6),
                             sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = grid[grid["train_size"] == size]
        opts = _present(d["optimizer"])
        pivot = d.pivot_table(index=["seed", "budget"], columns="optimizer",
                              values="test_cost")
        pivot = pivot.dropna(subset=[o for o in opts if o in pivot.columns])
        ranks = pivot.rank(axis=1, method="average")
        mean_rank = ranks.groupby(level="budget").mean()
        for opt in opts:
            if opt not in mean_rank.columns:
                continue
            ax.plot(mean_rank.index, mean_rank[opt], lw=2.5, color=PALETTE[opt],
                    label=OPTIMIZER_LABELS[opt])
        ax.axhline((len(opts) + 1) / 2, color="k", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("budget (SA solver calls)")
        ax.set_title(f"n={size}")
        ax.invert_yaxis()  # rank 1 (best) on top
    axes[0][0].set_ylabel("mean rank on test cost\n(1 = best)")
    axes[0][-1].legend(fontsize=12)
    fig.suptitle("Average rank across seeds over the budget", y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 6. Pairwise win-rate heatmap with Holm-corrected Wilcoxon p-values
# --------------------------------------------------------------------------- #

def plot_pairwise_winrate(tests: pd.DataFrame, out: Path,
                          metric: str = "final_test_cost"):
    t = tests[tests["metric"] == metric]
    if not len(t):
        return
    conds = sorted(t.groupby(["train_family", "train_size"]).groups)
    fig, axes = plt.subplots(1, len(conds), figsize=(8 * len(conds), 7), squeeze=False)
    for ax, (family, size) in zip(axes[0], conds):
        d = t[(t["train_family"] == family) & (t["train_size"] == size)]
        opts = _present(pd.concat([d["optimizer_a"], d["optimizer_b"]]))
        n = len(opts)
        win = np.full((n, n), np.nan)
        ann = [["" for _ in range(n)] for _ in range(n)]
        for _, r in d.iterrows():
            i, j = opts.index(r["optimizer_a"]), opts.index(r["optimizer_b"])
            win[i, j] = r["win_rate_a_over_b"]
            win[j, i] = 1.0 - r["win_rate_a_over_b"]
            star = "*" if r["p_holm"] < 0.05 else ""
            ann[i][j] = f"{win[i, j]:.2f}{star}\np={r['p_holm']:.3g}"
            ann[j][i] = f"{win[j, i]:.2f}{star}\np={r['p_holm']:.3g}"
        im = ax.imshow(win, vmin=0, vmax=1, cmap="RdYlGn")
        for i in range(n):
            for j in range(n):
                if i != j and ann[i][j]:
                    ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=10)
        ax.set_xticks(range(n), [OPTIMIZER_LABELS[o] for o in opts],
                      rotation=25, ha="right", fontsize=11)
        ax.set_yticks(range(n), [OPTIMIZER_LABELS[o] for o in opts], fontsize=11)
        ax.set_title(f"{family}, n={size}")
        fig.colorbar(im, ax=ax, shrink=0.8, label="P(row better than column)")
    fig.suptitle("Pairwise win rate on final test cost\n"
                 "(paired by seed; * = significant, Wilcoxon + Holm at 0.05)", y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 7. Budget to reach Random Search's final quality
# --------------------------------------------------------------------------- #

def plot_budget_to_target(summary: pd.DataFrame, out: Path):
    if "budget_to_target" not in summary.columns:
        return
    sizes = _sizes(summary)
    fig, axes = plt.subplots(1, len(sizes), figsize=(8 * len(sizes), 6), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = summary[summary["train_size"] == size].copy()
        order = _present(d["optimizer"])
        reached = d.dropna(subset=["budget_to_target"])
        sns.boxplot(data=reached, x="optimizer", y="budget_to_target", order=order,
                    hue="optimizer", palette=PALETTE, legend=False,
                    showfliers=False, width=0.6, ax=ax)
        sns.stripplot(data=reached, x="optimizer", y="budget_to_target", order=order,
                      color="0.2", size=4, alpha=0.6, jitter=0.18, ax=ax)
        ax.set_yscale("log")
        rs_med = reached.loc[reached["optimizer"] == "random", "budget_to_target"].median()
        if pd.notna(rs_med):
            ax.axhline(rs_med, color=PALETTE["random"], ls="--", lw=1.5)
        labels = []
        for o in order:
            sub = d[d["optimizer"] == o]
            miss = int(sub["budget_to_target"].isna().sum())
            sp = sub["speedup_vs_random"].median()
            lab = OPTIMIZER_LABELS[o]
            if pd.notna(sp):
                lab += f"\n{sp:.1f}x speedup"
            if miss:
                lab += f"\n({miss} not reached)"
            labels.append(lab)
        ax.set_xticks(range(len(order)), labels, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("SA calls to reach target" if size == sizes[0] else "")
        ax.set_title(f"n={size}")
    fig.suptitle("Budget needed to reach Random Search's median final validation cost",
                 y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 8. Generalization gap over budget + 9. overtuning over budget/ECDF
# --------------------------------------------------------------------------- #

def plot_generalization_gap(grid: pd.DataFrame, out: Path):
    sizes = _sizes(grid)
    fig, axes = plt.subplots(1, len(sizes), figsize=(8 * len(sizes), 6),
                             sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = grid[grid["train_size"] == size]
        for opt in _present(d["optimizer"]):
            g = (d[d["optimizer"] == opt].groupby("budget")["generalization_gap"]
                 .median().dropna())
            if not len(g):
                continue
            ax.plot(g.index, g.values, lw=2.5, color=PALETTE[opt],
                    label=OPTIMIZER_LABELS[opt])
        ax.axhline(0.0, color="k", ls="--", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("budget (SA solver calls)")
        ax.set_title(f"n={size}")
    axes[0][0].set_ylabel("generalization gap\n(test − validation, median)")
    axes[0][-1].legend(fontsize=12)
    fig.suptitle("Does more aggressive optimization overfit the training instances?",
                 y=1.02)
    _save(fig, out)


def plot_overtuning_over_budget(grid: pd.DataFrame, out: Path):
    """Frequency and magnitude of overtuning on the common SA-call axis."""
    required = {"is_overtuned", "relative_overtuning",
                "relative_overtuning_eligible"}
    if not required.issubset(grid.columns):
        return
    sizes = _sizes(grid)
    fig, axes = plt.subplots(len(sizes), 2, figsize=(16, 5.8 * len(sizes)),
                             squeeze=False)
    for row_idx, size in enumerate(sizes):
        d = grid[grid["train_size"] == size]
        ax_prob, ax_rel = axes[row_idx]
        for opt in _present(d["optimizer"]):
            ds = d[d["optimizer"] == opt]
            probability = ds.groupby("budget")["is_overtuned"].mean().dropna()
            magnitude = ds.groupby("budget")["relative_overtuning"].median().dropna()
            final_budget = ds["budget"].max()
            final = ds[ds["budget"] == final_budget]
            eligible_n = int(final["relative_overtuning_eligible"].sum())
            total_n = int(final["run_id"].nunique())
            ax_prob.plot(probability.index, probability.values, lw=2.4,
                         color=PALETTE[opt], label=OPTIMIZER_LABELS[opt])
            ax_rel.plot(magnitude.index, magnitude.values, lw=2.4,
                        color=PALETTE[opt],
                        label=f"{OPTIMIZER_LABELS[opt]} ({eligible_n}/{total_n} eligible)")
        for ax in (ax_prob, ax_rel):
            ax.set_xscale("log")
            ax.set_xlabel("budget (SA solver calls)")
        ax_prob.set_ylim(-0.02, 1.02)
        ax_prob.set_ylabel("proportion of runs overtuned")
        ax_prob.set_title(f"Probability of nonzero overtuning (n={size})")
        ax_rel.axhline(1.0, color="black", ls="--", lw=1.2,
                       label="all progress lost (=1)")
        ax_rel.set_ylabel("median relative overtuning\namong eligible runs")
        ax_rel.set_title(f"Magnitude of relative overtuning (n={size})")
        ax_prob.legend(fontsize=10)
        ax_rel.legend(fontsize=9)
    fig.suptitle(
        "Overtuning across the equal SA-call budget\n"
        f"(relative metric requires at least {RELATIVE_OVERTUNING_MIN_PROGRESS:g} "
        "test improvement from the initial incumbent)",
        y=1.01,
    )
    _save(fig, out)


def _plot_overtuning_factorial_over_budget(
    grid: pd.DataFrame, out: Path, metric: str
):
    """Training-size x resampling small multiples on the SA-call axis."""
    required = {
        "budget", "train_size", "resampling", "optimizer", "run_id",
        "is_overtuned", "relative_overtuning", "relative_overtuning_eligible",
    }
    if not required.issubset(grid.columns):
        return
    data = grid[grid["budget"] > 0].copy()
    sizes = _sizes(data)
    methods = _present_resamplings(data["resampling"])
    optimizers = _present(data["optimizer"])
    if len(sizes) < 2 or len(methods) < 2 or not optimizers:
        return

    metric_specs = {
        "frequency": {
            "column": "is_overtuned",
            "ylabel": "proportion of runs overtuned",
            "title": "Overtuning frequency throughout the equal solver-call budget",
            "ylim": (-0.02, 1.02),
        },
        "relative": {
            "column": "relative_overtuning",
            "ylabel": "median relative overtuning\namong eligible runs",
            "title": "Relative overtuning throughout the equal solver-call budget",
            "ylim": None,
        },
        "eligibility": {
            "column": "relative_overtuning_eligible",
            "ylabel": "proportion eligible for\nrelative overtuning",
            "title": "When relative overtuning becomes identifiable",
            "ylim": (-0.02, 1.02),
        },
    }
    if metric not in metric_specs:
        raise ValueError(f"unknown overtuning budget metric: {metric}")
    spec = metric_specs[metric]

    fig, axes = plt.subplots(
        len(sizes), len(methods),
        figsize=(5.8 * len(methods), 4.5 * len(sizes)),
        sharex=True, sharey=True, squeeze=False,
    )
    for row_idx, size in enumerate(sizes):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            cell = data[(data["train_size"] == size)
                        & (data["resampling"] == method)]
            for optimizer in optimizers:
                arm = cell[cell["optimizer"] == optimizer]
                if not len(arm):
                    continue
                grouped = arm.groupby("budget")[spec["column"]]
                if metric == "relative":
                    centre = grouped.median().dropna()
                    low = grouped.quantile(0.25).reindex(centre.index)
                    high = grouped.quantile(0.75).reindex(centre.index)
                else:
                    centre = grouped.mean().dropna()
                    low = high = None
                if not len(centre):
                    continue
                x = centre.index.to_numpy(dtype=float)
                ax.plot(
                    x, centre.to_numpy(dtype=float),
                    color=PALETTE[optimizer], lw=2.0,
                    marker=OPTIMIZER_MARKERS[optimizer], markersize=4,
                    markevery=max(1, len(x) // 7),
                    label=OPTIMIZER_LABELS[optimizer],
                )
                if metric == "relative":
                    ax.fill_between(
                        x, low.to_numpy(dtype=float), high.to_numpy(dtype=float),
                        color=PALETTE[optimizer], alpha=0.10,
                    )

            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", alpha=0.35)
            if spec["ylim"] is not None:
                ax.set_ylim(*spec["ylim"])
            if metric == "relative":
                ax.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.75)
            if row_idx == 0:
                ax.set_title(RESAMPLING_LABELS.get(method, method), fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"n={size}\n{spec['ylabel']}", fontsize=11)
            if row_idx == len(sizes) - 1:
                ax.set_xlabel("cumulative SA solver calls", fontsize=11)

    handles = [
        plt.Line2D([0], [0], color=PALETTE[o], marker=OPTIMIZER_MARKERS[o],
                   lw=2.2, markersize=6)
        for o in optimizers
    ]
    fig.legend(
        handles, [OPTIMIZER_LABELS[o] for o in optimizers],
        loc="upper center", ncol=len(optimizers), frameon=False,
        bbox_to_anchor=(0.5, 0.945), fontsize=11,
    )
    subtitle = (
        f"relative metric requires at least {RELATIVE_OVERTUNING_MIN_PROGRESS:g} "
        "test improvement from the initial incumbent"
        if metric in {"relative", "eligibility"}
        else "all panels use matched seeds and the same cumulative SA-call axis"
    )
    fig.suptitle(f"{spec['title']}\n{subtitle}", y=0.995, fontsize=18)
    _save(fig, out, rect=(0, 0, 1, 0.89))


def plot_overtuning_frequency_factorial_over_budget(grid: pd.DataFrame, out: Path):
    _plot_overtuning_factorial_over_budget(grid, out, "frequency")


def plot_relative_overtuning_factorial_over_budget(grid: pd.DataFrame, out: Path):
    _plot_overtuning_factorial_over_budget(grid, out, "relative")


def plot_relative_overtuning_eligibility_factorial_over_budget(
    grid: pd.DataFrame, out: Path
):
    _plot_overtuning_factorial_over_budget(grid, out, "eligibility")


def plot_overtuning_ecdf(summary: pd.DataFrame, out: Path):
    d = summary.dropna(subset=["final_relative_overtuning"])
    if not len(d):
        return
    sizes = _sizes(summary)
    fig, axes = plt.subplots(1, len(sizes), figsize=(8 * len(sizes), 6),
                             sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        ds = d[d["train_size"] == size]
        for opt in _present(ds["optimizer"]):
            vals = np.sort(ds.loc[ds["optimizer"] == opt,
                                  "final_relative_overtuning"].to_numpy())
            if not len(vals):
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", lw=2.5, color=PALETTE[opt],
                    label=f"{OPTIMIZER_LABELS[opt]} (eligible n={len(vals)})")
        ax.axvline(1.0, color="k", ls="--", lw=1.2, label="all progress lost (=1)")
        ax.set_xlabel("final relative overtuning")
        ax.set_title(f"n={size}")
    axes[0][0].set_ylabel("proportion of runs")
    axes[0][-1].legend(fontsize=11)
    fig.suptitle("ECDF of relative overtuning by configurator\n"
                 f"(Schneider et al. metric; minimum test progress = "
                 f"{RELATIVE_OVERTUNING_MIN_PROGRESS:g})",
                 y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 10. Proposal quality over time (exploration -> exploitation)
# --------------------------------------------------------------------------- #

def plot_proposal_quality(trials: pd.DataFrame, out: Path):
    sizes = _sizes(trials)
    fig, axes = plt.subplots(1, len(sizes), figsize=(8 * len(sizes), 6),
                             sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = trials[trials["train_size"] == size]
        for opt in _present(d["optimizer"]):
            ds = d[d["optimizer"] == opt]
            if opt == "smac_aac":
                # one point per *distinct configuration*, in proposal order, with the
                # mean observed cost over all its intensification calls
                per_cfg = (ds.groupby(["run_id", "config_id"])
                           .agg(cost=("observed_cost", "mean"),
                                first=("trial_id", "min")).reset_index())
                per_cfg["prop_idx"] = per_cfg.groupby("run_id")["first"].rank(
                    method="first").astype(int) - 1
                med = per_cfg.groupby("prop_idx")["cost"].median()
                n_runs = per_cfg["run_id"].nunique()
                counts = per_cfg.groupby("prop_idx").size()
                med = med[counts >= max(1, n_runs // 4)]
            else:
                med = ds.groupby("trial_id")["observed_cost"].median()
            if len(med) > 10:
                med = med.rolling(9, center=True, min_periods=1).median()
            ax.plot(med.index, med.values, lw=2.5, color=PALETTE[opt],
                    label=OPTIMIZER_LABELS[opt])
        ax.set_yscale("log")
        ax.set_xlabel("proposal index (distinct configuration)")
        ax.set_title(f"n={size}")
    axes[0][0].set_ylabel("cost of *proposed* config\n(median across seeds, smoothed)")
    axes[0][-1].legend(fontsize=12)
    fig.suptitle("Quality of proposals over time — RS stays flat, model-based "
                 "configurators improve", y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 11. Where the search concentrates: early vs late parameter distributions
# --------------------------------------------------------------------------- #

def plot_param_concentration(trials: pd.DataFrame, out: Path):
    tot = (trials[["p_swap", "p_insert", "p_2opt", "p_oropt"]].sum(axis=1)
           .replace(0.0, np.nan))
    d = trials.copy()
    d["log10_T0"] = np.log10(d["initial_temperature"])
    d["share_2opt"] = d["p_2opt"] / tot
    d["log10_iters_per_temp"] = np.log10(d["iterations_per_temp"])
    third = (("share_2opt", "2-opt share of move mixture")
             if d["share_2opt"].nunique(dropna=True) > 1
             else ("log10_iters_per_temp", "log10 iterations per temperature"))
    params = [("log10_T0", "log10 initial temperature"),
              ("cooling_rate", "cooling rate"),
              third]
    # proposal order: distinct configs per run
    d = d.sort_values("trial_id")
    d = d.drop_duplicates(subset=["run_id", "config_id"], keep="first")
    d["prop_idx"] = d.groupby("run_id").cumcount()
    d["n_props"] = d.groupby("run_id")["prop_idx"].transform("max") + 1
    d["phase"] = np.where(d["prop_idx"] < d["n_props"] / 3, "early",
                          np.where(d["prop_idx"] >= 2 * d["n_props"] / 3, "late", "mid"))
    d = d[d["phase"] != "mid"]

    opts = _present(d["optimizer"])
    fig, axes = plt.subplots(len(params), len(opts),
                             figsize=(5 * len(opts), 4 * len(params)),
                             squeeze=False)
    for i, (col, label) in enumerate(params):
        for j, opt in enumerate(opts):
            ax = axes[i][j]
            ds = d[d["optimizer"] == opt]
            for phase, color in (("early", "0.55"), ("late", PALETTE[opt])):
                vals = ds.loc[ds["phase"] == phase, col].dropna()
                if len(vals) < 2:
                    continue
                ax.hist(vals, bins=24, density=True, alpha=0.55, color=color,
                        label=f"{phase} third")
            if i == 0:
                ax.set_title(OPTIMIZER_LABELS[opt], fontsize=13)
            if j == 0:
                ax.set_ylabel(label, fontsize=12)
            ax.set_yticks([])
            if i == 0 and j == len(opts) - 1:
                ax.legend(fontsize=10)
    fig.suptitle("Sampled parameter distributions: early vs late proposals\n"
                 "(model-based configurators concentrate; Random Search cannot)",
                 y=1.0)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 12. Configurator overhead
# --------------------------------------------------------------------------- #

def plot_overhead(summary: pd.DataFrame, out: Path):
    d = (summary.groupby("optimizer", as_index=False)
         .agg(sa=("eval_sec", "mean"), overhead=("configurator_overhead_sec", "mean")))
    order = _present(d["optimizer"])
    d = d.set_index("optimizer").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(d))
    ax.bar(x, d["sa"], width=0.6, label="target algorithm (SA) time",
           color=[PALETTE[o] for o in d["optimizer"]], alpha=0.9)
    ax.bar(x, d["overhead"], width=0.6, bottom=d["sa"],
           label="configurator overhead (ask/tell, model)", color="0.25")
    for xi, (_, r) in zip(x, d.iterrows()):
        frac = r["overhead"] / max(r["sa"] + r["overhead"], 1e-9)
        ax.text(xi, r["sa"] + r["overhead"], f"{frac * 100:.0f}% overhead",
                ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, [OPTIMIZER_LABELS[o] for o in d["optimizer"]],
                  rotation=15, ha="right")
    ax.set_ylabel("mean wallclock per run (s)")
    ax.set_title("Where the time goes: solver evaluations vs configurator machinery")
    ax.legend(fontsize=12)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 13. SMAC intensification behaviour
# --------------------------------------------------------------------------- #

def plot_smac_intensification(trials: pd.DataFrame, summary: pd.DataFrame, out: Path):
    aac = trials[trials["optimizer"] == "smac_aac"]
    if not len(aac):
        return
    sizes = _sizes(aac)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for size in sizes:
        d = aac[aac["train_size"] == size]
        calls = d.groupby(["run_id", "config_id"]).size().to_numpy()
        vals = np.sort(calls)
        y = np.arange(1, len(vals) + 1) / len(vals)
        line, = ax.step(vals, y, where="post", lw=2.5, label=f"smac_aac, n={size}")
        fixed = trials[(trials["optimizer"] != "smac_aac")
                       & (trials["train_size"] == size)]
        fixed_calls = (float(fixed["n_sa_calls"].median()) if len(fixed)
                       else float(size))
        ax.axvline(fixed_calls, color=line.get_color(), ls="--", lw=1.5,
                   label=f"fixed-estimator arms spend {fixed_calls:g}")
    ax.set_xscale("log")
    ax.set_xlabel("SA calls spent on a configuration")
    ax.set_ylabel("proportion of configurations")
    ax.set_title("Racing allocation: most configs killed early,\nincumbents intensified")
    ax.legend(fontsize=10)

    ax = axes[1]
    d = (summary.groupby(["optimizer", "train_size"], as_index=False)
         .agg(n=("n_distinct_configs", "mean")))
    order = _present(d["optimizer"])
    sns.barplot(data=d, x="optimizer", y="n", hue="train_size", order=order, ax=ax)
    ax.set_xticks(range(len(order)), [OPTIMIZER_LABELS[o] for o in order],
                  rotation=15, ha="right", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("distinct configurations evaluated")
    ax.set_title("Same budget, very different coverage\nof the configuration space")
    fig.suptitle("SMAC native intensification: how the equal budget is allocated", y=1.03)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 14. Distribution shift: final test cost per test family
# --------------------------------------------------------------------------- #

def plot_test_family_heatmap(summary: pd.DataFrame, out: Path):
    fams = ["uniform", "clustered", "mixed"]
    sizes = _sizes(summary)
    fig, axes = plt.subplots(1, len(sizes), figsize=(7.5 * len(sizes), 6), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = summary[summary["train_size"] == size]
        opts = _present(d["optimizer"])
        mat = np.array([[d.loc[d["optimizer"] == o, f"final_test_{f}"].mean()
                         for f in fams] for o in opts])
        im = ax.imshow(mat, cmap="viridis_r")
        for i in range(len(opts)):
            for j in range(len(fams)):
                ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center",
                        color="w", fontsize=11)
        ax.set_xticks(range(len(fams)), fams)
        ax.set_yticks(range(len(opts)), [OPTIMIZER_LABELS[o] for o in opts], fontsize=11)
        ax.set_title(f"n={size}")
        fig.colorbar(im, ax=ax, shrink=0.8, label="mean final test cost")
    fig.suptitle("Out-of-distribution robustness: mean final test cost by test family",
                 y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# Factorial configurator x instance-resampling comparisons
# --------------------------------------------------------------------------- #

def _present_resamplings(values) -> list[str]:
    have = set(values)
    return [method for method in RESAMPLING_ORDER if method in have]


def _label_factorial_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "resampling" not in data:
        data["resampling"] = "full_train"
    data["resampling_label"] = data["resampling"].map(
        lambda value: RESAMPLING_LABELS.get(value, value)
    )
    data["optimizer_label"] = data["optimizer"].map(
        lambda value: OPTIMIZER_LABELS.get(value, value)
    )
    return data


def plot_factorial_final_performance(summary: pd.DataFrame, out: Path):
    """Final outcomes for every configurator x resampling combination."""
    data = _label_factorial_data(summary)
    methods = _present_resamplings(data["resampling"])
    optimizers = _present(data["optimizer"])
    panels = [
        ("final_test_cost", "Final unseen-test cost"),
        ("auc_test", "Anytime test-trajectory AUC"),
        ("abs_generalization_gap", "Absolute generalization gap"),
        ("final_relative_overtuning", "Final relative overtuning"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(22, 15), squeeze=False)
    for ax, (metric, title) in zip(axes.flat, panels):
        panel = data.dropna(subset=[metric])
        sns.boxplot(
            data=panel,
            x="resampling_label",
            y=metric,
            hue="optimizer_label",
            order=[RESAMPLING_LABELS[m] for m in methods],
            hue_order=[OPTIMIZER_LABELS[o] for o in optimizers],
            palette={OPTIMIZER_LABELS[o]: PALETTE[o] for o in optimizers},
            showfliers=False,
            ax=ax,
        )
        ax.set_xlabel("instance-resampling method")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        if ax is not axes[0, 1]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    axes[0, 1].legend(title="configurator", fontsize=10)
    fig.suptitle(
        "Configurator × instance-resampling comparison at equal SA-call cap",
        y=1.01,
    )
    _save(fig, out)


def plot_factorial_test_trajectories(grid: pd.DataFrame, out: Path):
    """One equal-budget test trajectory panel per resampling estimator."""
    methods = _present_resamplings(grid["resampling"])
    ncols = 2
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, method in zip(axes.flat, methods):
        data = grid[grid["resampling"] == method]
        for optimizer in _present(data["optimizer"]):
            curve = (data[data["optimizer"] == optimizer]
                     .groupby("budget")["test_cost"].agg(["mean", "sem"])
                     .dropna(subset=["mean"]))
            x = curve.index.to_numpy(dtype=float)
            mean = curve["mean"].to_numpy(dtype=float)
            sem = curve["sem"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, mean, lw=2.5, color=PALETTE[optimizer],
                    label=OPTIMIZER_LABELS[optimizer])
            ax.fill_between(x, mean - sem, mean + sem,
                            color=PALETTE[optimizer], alpha=0.14)
        ax.set_xscale("log")
        ax.set_xlabel("cumulative SA solver calls")
        ax.set_ylabel("mean incumbent unseen-test cost")
        ax.set_title(RESAMPLING_LABELS.get(method, method))
        ax.legend(title="configurator", fontsize=9)
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    fig.suptitle(
        "Anytime configurator performance within each instance-resampling method\n"
        "bands show standard error across matched seeds",
        y=1.01,
    )
    _save(fig, out)


def plot_all_configurators_all_resamplings(
    grid: pd.DataFrame, summary: pd.DataFrame, out: Path
):
    """One reference-style matrix containing every factorial cell.

    Each row fixes one instance-resampling method.  The two trajectory columns
    and the final-cost column then compare Random Search, TPE, fixed-budget
    SMAC-BO, and native SMAC-AAC together.  Keeping the resampling methods in
    separate aligned rows avoids averaging four different estimators into one
    convergence curve while still presenting the complete 4 x 4 comparison in
    a single figure.
    """
    methods = _present_resamplings(summary["resampling"])
    preferred_order = ["optuna_tpe", "random", "smac_bo", "smac_aac"]
    optimizers = [optimizer for optimizer in preferred_order
                  if optimizer in set(summary["optimizer"])]
    if not methods or not optimizers:
        return

    fig, axes = plt.subplots(
        len(methods), 3, figsize=(24, 5.2 * len(methods)), squeeze=False
    )
    for row, method in enumerate(methods):
        method_grid = grid[grid["resampling"] == method]
        method_summary = summary[summary["resampling"] == method].copy()
        curves = {
            optimizer: _aligned_step_curve(
                method_grid[method_grid["optimizer"] == optimizer], "test_cost"
            )
            for optimizer in optimizers
        }

        for column, zoomed in enumerate((False, True)):
            ax = axes[row, column]
            for optimizer in optimizers:
                budgets, mean, sem = curves[optimizer]
                if not len(budgets):
                    continue
                mask = np.ones(len(budgets), dtype=bool)
                if zoomed:
                    cutoff = max(
                        float(budgets.min()), 0.05 * float(budgets.max())
                    )
                    mask = budgets >= cutoff
                ax.plot(
                    budgets[mask], mean[mask], color=PALETTE[optimizer], lw=2.4,
                    label=OPTIMIZER_LABELS[optimizer],
                )
                ax.fill_between(
                    budgets[mask], mean[mask] - sem[mask], mean[mask] + sem[mask],
                    color=PALETTE[optimizer], alpha=0.13,
                )
            ax.set_xlabel("cumulative SA solver calls")
            ax.set_ylabel("mean incumbent unseen-test cost")
            column_title = (
                "Convergence - full SA-call range"
                if not zoomed
                else "Convergence - after first 5% of SA-call cap"
            )
            ax.set_title(
                f"{RESAMPLING_LABELS.get(method, method)}\n{column_title}"
            )

        right = axes[row, 2]
        method_summary["optimizer_label"] = method_summary["optimizer"].map(
            OPTIMIZER_LABELS
        )
        order = [OPTIMIZER_LABELS[optimizer] for optimizer in optimizers]
        palette = {OPTIMIZER_LABELS[optimizer]: PALETTE[optimizer]
                   for optimizer in optimizers}
        sns.boxplot(
            data=method_summary,
            x="optimizer_label",
            y="final_test_cost",
            order=order,
            hue="optimizer_label",
            hue_order=order,
            palette=palette,
            legend=False,
            showfliers=False,
            ax=right,
        )
        sns.stripplot(
            data=method_summary,
            x="optimizer_label",
            y="final_test_cost",
            order=order,
            color="0.2",
            alpha=0.48,
            size=3.5,
            ax=right,
        )
        right.set_xlabel("")
        right.set_ylabel("final unseen-test cost")
        right.set_title(
            f"{RESAMPLING_LABELS.get(method, method)}\n"
            "Final unseen-test cost"
        )
        right.tick_params(axis="x", rotation=18)
        for tick in right.get_xticklabels():
            tick.set_ha("right")

    handles = [plt.Line2D([0], [0], color=PALETTE[optimizer], lw=7)
               for optimizer in optimizers]
    fig.legend(
        handles,
        [OPTIMIZER_LABELS[optimizer] for optimizer in optimizers],
        title="configurator",
        loc="upper center",
        ncol=len(optimizers),
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        "All four configurators across all four instance-resampling methods\n"
        "trajectories use the common cumulative-SA-call axis; bands show standard error",
        y=0.998,
    )
    _save(fig, out, rect=(0, 0, 1, 0.89))


def plot_factorial_overtuning_ecdf_by_resampling(
    summary: pd.DataFrame, out: Path, y_min: float = 0.0
):
    """Relative-overtuning ECDFs faceted by resampling, with all configurators."""
    data = summary.dropna(subset=["final_relative_overtuning"])
    methods = _present_resamplings(summary["resampling"])
    preferred_order = ["optuna_tpe", "random", "smac_bo", "smac_aac"]
    optimizers = [optimizer for optimizer in preferred_order
                  if optimizer in set(summary["optimizer"])]
    if not methods or not optimizers:
        return

    ncols = 2
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 6.5 * nrows), sharex=True, sharey=True,
        squeeze=False,
    )
    for ax, method in zip(axes.flat, methods):
        panel = data[data["resampling"] == method]
        for optimizer in optimizers:
            values = np.sort(
                panel.loc[panel["optimizer"] == optimizer,
                          "final_relative_overtuning"].to_numpy()
            )
            if not len(values):
                continue
            y = np.arange(1, len(values) + 1) / len(values)
            ax.step(
                values,
                y,
                where="post",
                lw=2.4,
                color=PALETTE[optimizer],
                label=f"{OPTIMIZER_LABELS[optimizer]} (n={len(values)})",
            )
        ax.axvline(1.0, color="black", ls="--", lw=1.2)
        ax.set_xlabel("final relative overtuning")
        ax.set_ylabel("proportion of eligible runs")
        ax.set_ylim(y_min, 1.005)
        ax.set_title(RESAMPLING_LABELS.get(method, method))

    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=PALETTE[optimizer], lw=5)
               for optimizer in optimizers]
    labels = [OPTIMIZER_LABELS[optimizer] for optimizer in optimizers]
    handles.append(plt.Line2D([0], [0], color="black", ls="--", lw=1.2))
    labels.append("all tuning progress lost (= 1)")
    fig.legend(
        handles,
        labels,
        title="configurator",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.955),
    )
    scale_note = " - zoomed" if y_min > 0 else " - full scale"
    fig.suptitle(
        "Relative-overtuning ECDF: all configurators within every "
        f"instance-resampling method{scale_note}",
        y=0.995,
    )
    _save(fig, out, rect=(0, 0, 1, 0.88))


def plot_factorial_overtuning_ecdf_matrix(
    summary: pd.DataFrame, out: Path, zoom_y_min: float = 0.75
):
    """Paper-style full/zoomed ECDF pair for every resampling method.

    This is the factorial extension of the original experiment's two-panel
    ECDF: rows are instance-resampling methods, columns are the full and zoomed
    scales, and every panel compares all four configurators.
    """
    data = summary.dropna(subset=["final_relative_overtuning"])
    methods = _present_resamplings(summary["resampling"])
    preferred_order = ["optuna_tpe", "random", "smac_bo", "smac_aac"]
    optimizers = [optimizer for optimizer in preferred_order
                  if optimizer in set(summary["optimizer"])]
    if not methods or not optimizers:
        return

    fig, axes = plt.subplots(
        len(methods), 2, figsize=(19, 5.2 * len(methods)), sharex=True,
        squeeze=False,
    )
    for row, method in enumerate(methods):
        panel = data[data["resampling"] == method]
        for column, y_min in enumerate((0.0, zoom_y_min)):
            ax = axes[row, column]
            for optimizer in optimizers:
                values = np.sort(
                    panel.loc[panel["optimizer"] == optimizer,
                              "final_relative_overtuning"].to_numpy()
                )
                if not len(values):
                    continue
                y = np.arange(1, len(values) + 1) / len(values)
                ax.step(
                    values,
                    y,
                    where="post",
                    lw=2.4,
                    color=PALETTE[optimizer],
                    label=f"{OPTIMIZER_LABELS[optimizer]} (n={len(values)})",
                )
            ax.axvline(1.0, color="black", ls="--", lw=1.2)
            ax.set_xlabel("final relative overtuning")
            ax.set_ylabel("proportion of eligible runs")
            ax.set_ylim(y_min, 1.005)
            scale_label = (
                "Full scale" if column == 0
                else f"Zoomed (y >= {zoom_y_min:.2f})"
            )
            ax.set_title(
                f"{RESAMPLING_LABELS.get(method, method)}\n{scale_label}"
            )

    handles = [plt.Line2D([0], [0], color=PALETTE[optimizer], lw=5)
               for optimizer in optimizers]
    labels = [OPTIMIZER_LABELS[optimizer] for optimizer in optimizers]
    handles.append(plt.Line2D([0], [0], color="black", ls="--", lw=1.2))
    labels.append("all tuning progress lost (= 1)")
    fig.legend(
        handles,
        labels,
        title="configurator",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        "Relative overtuning across the complete configurator x "
        "instance-resampling factorial",
        y=0.997,
    )
    _save(fig, out, rect=(0, 0, 1, 0.89))


def plot_factorial_overtuning_ecdf(
    summary: pd.DataFrame, out: Path, y_min: float = 0.0
):
    """Paper-style relative-overtuning ECDF, faceted by configurator."""
    data = summary.dropna(subset=["final_relative_overtuning"])
    optimizers = _present(summary["optimizer"])
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), sharex=True, sharey=True,
                             squeeze=False)
    for ax, optimizer in zip(axes.flat, optimizers):
        panel = data[data["optimizer"] == optimizer]
        for method in _present_resamplings(summary["resampling"]):
            values = np.sort(panel.loc[panel["resampling"] == method,
                                       "final_relative_overtuning"].to_numpy())
            if not len(values):
                continue
            y = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, y, where="post", lw=2.4,
                    color=RESAMPLING_PALETTE[method],
                    label=f"{RESAMPLING_LABELS[method]} (n={len(values)})")
        ax.axvline(1.0, color="black", ls="--", lw=1.2,
                   label="all tuning progress lost (= 1)")
        ax.set_xlabel("final relative overtuning")
        ax.set_ylabel("proportion of eligible runs")
        ax.set_ylim(y_min, 1.005)
        ax.set_title(OPTIMIZER_LABELS[optimizer])
        ax.legend(fontsize=8)
    for ax in axes.flat[len(optimizers):]:
        ax.axis("off")
    scale_note = " — zoomed" if y_min > 0 else " — full scale"
    fig.suptitle(
        "ECDF of relative overtuning by configurator and instance resampling"
        + scale_note,
        y=1.01,
    )
    _save(fig, out)


def plot_factorial_heatmaps(summary: pd.DataFrame, out: Path):
    """Compact matrix view of the two experimental factors."""
    methods = _present_resamplings(summary["resampling"])
    optimizers = _present(summary["optimizer"])
    panels = [
        ("final_test_cost", "Mean final unseen-test cost"),
        ("auc_test", "Mean anytime AUC"),
        ("abs_generalization_gap", "Mean absolute generalization gap"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), squeeze=False)
    for ax, (metric, title) in zip(axes[0], panels):
        matrix = (summary.pivot_table(index="resampling", columns="optimizer",
                                      values=metric, aggfunc="mean")
                  .reindex(index=methods, columns=optimizers))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".4f",
            cmap="viridis_r",
            xticklabels=[OPTIMIZER_LABELS[o] for o in optimizers],
            yticklabels=[RESAMPLING_LABELS[m] for m in methods],
            cbar_kws={"label": metric.replace("_", " ")},
            ax=ax,
        )
        ax.set_xlabel("configurator")
        ax.set_ylabel("instance-resampling method")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.suptitle("Factorial mean outcomes: configurator × instance resampling", y=1.02)
    _save(fig, out)


def plot_factorial_efficiency(summary: pd.DataFrame, out: Path):
    """Compute use and configurator overhead for every factorial cell."""
    data = _label_factorial_data(summary)
    methods = _present_resamplings(data["resampling"])
    optimizers = _present(data["optimizer"])
    panels = [
        ("wallclock_sec", "Total run wall-clock time", "seconds"),
        ("eval_sec", "Optimization solver time", "seconds"),
        ("analysis_runtime_sec", "Analysis-only evaluation time", "seconds"),
        ("configurator_overhead_sec", "Configurator ask/tell overhead", "seconds"),
        ("total_sa_calls_optimization", "Optimization SA calls used", "SA calls"),
        ("n_distinct_configs", "Distinct configurations evaluated", "configurations"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), squeeze=False)
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        panel = data.dropna(subset=[metric])
        sns.boxplot(
            data=panel,
            x="resampling_label",
            y=metric,
            hue="optimizer_label",
            order=[RESAMPLING_LABELS[m] for m in methods],
            hue_order=[OPTIMIZER_LABELS[o] for o in optimizers],
            palette={OPTIMIZER_LABELS[o]: PALETTE[o] for o in optimizers},
            showfliers=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=22)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    handles = [plt.Line2D([0], [0], color=PALETTE[o], lw=6)
               for o in optimizers]
    fig.legend(handles, [OPTIMIZER_LABELS[o] for o in optimizers],
               title="configurator", loc="upper center", ncol=len(optimizers),
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Runtime, budget use, and search breadth by factorial cell", y=0.995
    )
    _save(fig, out, rect=(0, 0, 1, 0.90))


def plot_factorial_selected_parameters(summary: pd.DataFrame, out: Path):
    """Final selected numeric SA parameters for every factorial cell."""
    data = _label_factorial_data(summary)
    methods = _present_resamplings(data["resampling"])
    optimizers = _present(data["optimizer"])
    parameters = [
        ("initial_temperature", "Initial temperature", "log"),
        ("cooling_rate", "Cooling rate", None),
        ("min_temperature", "Minimum temperature", "log"),
        ("iterations_per_temp", "Iterations per temperature", "log"),
        ("restarts", "Restarts", None),
        ("perturbation_kicks", "Perturbation kicks", None),
        ("reheat_interval", "Reheat interval", None),
        ("reheat_factor", "Reheat factor", "log"),
        ("move_share_2opt", "2-opt move share", None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(24, 20), squeeze=False)
    for ax, (column, title, scale) in zip(axes.flat, parameters):
        panel = data.dropna(subset=[column])
        sns.boxplot(
            data=panel,
            x="resampling_label",
            y=column,
            hue="optimizer_label",
            order=[RESAMPLING_LABELS[m] for m in methods],
            hue_order=[OPTIMIZER_LABELS[o] for o in optimizers],
            palette={OPTIMIZER_LABELS[o]: PALETTE[o] for o in optimizers},
            showfliers=False,
            ax=ax,
        )
        if scale:
            ax.set_yscale(scale)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(column.replace("_", " "))
        ax.tick_params(axis="x", rotation=22)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    handles = [plt.Line2D([0], [0], color=PALETTE[o], lw=6)
               for o in optimizers]
    fig.legend(handles, [OPTIMIZER_LABELS[o] for o in optimizers],
               title="configurator", loc="upper center", ncol=len(optimizers),
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Final selected SA parameters by configurator and instance resampling",
        y=0.995,
    )
    _save(fig, out, rect=(0, 0, 1, 0.90))


# --------------------------------------------------------------------------- #
# V1-style report suite, computed exclusively from new v2 factorial results
# --------------------------------------------------------------------------- #

def _aligned_step_curve(data: pd.DataFrame, metric: str,
                        n_points: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align run-level step trajectories on one honest cumulative-SA-call axis."""
    runs = [d.sort_values("budget") for _, d in data.groupby("run_id")]
    runs = [d for d in runs if len(d) and d[metric].notna().any()]
    if not runs:
        return np.array([]), np.array([]), np.array([])
    lower = max(float(d["budget"].min()) for d in runs)
    upper = min(float(d["budget"].max()) for d in runs)
    if upper < lower:
        return np.array([]), np.array([]), np.array([])
    budgets = np.linspace(lower, upper, n_points)
    aligned = []
    for run in runs:
        x = run["budget"].to_numpy(dtype=float)
        y = run[metric].to_numpy(dtype=float)
        indices = np.searchsorted(x, budgets, side="right") - 1
        indices = np.clip(indices, 0, len(y) - 1)
        aligned.append(y[indices])
    matrix = np.asarray(aligned, dtype=float)
    counts = np.sum(np.isfinite(matrix), axis=0)
    means = np.nanmean(matrix, axis=0)
    sem = np.zeros_like(means)
    if matrix.shape[0] > 1:
        sem = np.divide(
            np.nanstd(matrix, axis=0, ddof=1),
            np.sqrt(counts),
            out=np.zeros_like(means),
            where=counts > 1,
        )
    return budgets, means, sem


def plot_v1_style_representative_trajectory(
    grid: pd.DataFrame, summary: pd.DataFrame, out: Path
):
    """V1 validation-vs-test diagnostic using a v2 incumbent trajectory."""
    eligible = summary.dropna(subset=["final_overtuning"])
    run_id = (eligible.loc[eligible["final_overtuning"].idxmax(), "run_id"]
              if len(eligible) else summary["run_id"].iloc[0])
    data = grid[grid["run_id"] == run_id].sort_values("budget")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.step(data["budget"], data["val_cost"], where="post", lw=2.5,
            label="canonical full-training validation")
    ax.step(data["budget"], data["test_cost"], where="post", lw=2.5,
            label="unseen-test cost")
    ax.set_xlabel("cumulative SA solver calls")
    ax.set_ylabel("normalized cost (gap)")
    ax.set_title(f"Validation vs unseen-test incumbent\n{run_id}")
    ax.legend()
    _save(fig, out)


def plot_v1_style_trajectories_by_resampling(grid: pd.DataFrame, out: Path):
    """V1 two-panel trajectory view on the equal-compute v2 budget axis."""
    methods = _present_resamplings(grid["resampling"])
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), sharex=True)
    panels = [
        ("val_cost", "canonical validation cost", "Validation trajectory"),
        ("test_cost", "unseen-test cost", "Unseen-test trajectory"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for method in methods:
            budgets, mean, sem = _aligned_step_curve(
                grid[grid["resampling"] == method], metric
            )
            if not len(budgets):
                continue
            color = RESAMPLING_PALETTE[method]
            ax.plot(budgets, mean, color=color, lw=2.5,
                    label=RESAMPLING_LABELS[method])
            ax.fill_between(budgets, mean - sem, mean + sem,
                            color=color, alpha=0.14)
        ax.set_xlabel("cumulative SA solver calls")
        ax.set_ylabel(f"mean incumbent {ylabel}")
        ax.set_title(f"{title}\nmean ± standard error across new v2 runs")
        ax.legend(title="instance-resampling method", fontsize=8)
    fig.suptitle("Incumbent trajectories by instance-resampling method", y=1.02)
    _save(fig, out)


def plot_v1_style_composition(summary: pd.DataFrame, out: Path):
    """Adaptive v1 train-family × test-family heatmap for available v2 cells."""
    test_families = ["uniform", "clustered", "mixed"]
    train_families = [f for f in test_families
                      if f in set(summary["train_family"])]
    matrix = np.full((len(train_families), len(test_families)), np.nan)
    for row, family in enumerate(train_families):
        data = summary[summary["train_family"] == family]
        for column, test_family in enumerate(test_families):
            key = f"final_test_{test_family}"
            if key in data:
                matrix[row, column] = data[key].mean()
    fig, ax = plt.subplots(figsize=(9, max(4.5, 1.4 * len(train_families) + 3)))
    sns.heatmap(matrix, annot=True, fmt=".4f", cmap="viridis",
                xticklabels=test_families, yticklabels=train_families,
                cbar_kws={"label": "mean final unseen-test cost"}, ax=ax)
    ax.set_xlabel("test family")
    ax.set_ylabel("training family")
    ax.set_title("Composition / distribution shift\nnew v2 factorial results only")
    _save(fig, out)


def plot_v1_style_composition_by_resampling(summary: pd.DataFrame, out: Path):
    """Adaptive composition heatmaps, one panel per v2 resampling method."""
    methods = _present_resamplings(summary["resampling"])
    test_families = ["uniform", "clustered", "mixed"]
    train_families = [f for f in test_families
                      if f in set(summary["train_family"])]
    matrices = {}
    for method in methods:
        matrix = np.full((len(train_families), len(test_families)), np.nan)
        method_data = summary[summary["resampling"] == method]
        for row, family in enumerate(train_families):
            data = method_data[method_data["train_family"] == family]
            for column, test_family in enumerate(test_families):
                key = f"final_test_{test_family}"
                if key in data and len(data):
                    matrix[row, column] = data[key].mean()
        matrices[method] = matrix
    finite = np.concatenate([matrix[np.isfinite(matrix)]
                             for matrix in matrices.values()])
    vmin = float(finite.min()) if len(finite) else None
    vmax = float(finite.max()) if len(finite) else None
    ncols = 2
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6 * nrows), squeeze=False)
    for index, (ax, method) in enumerate(zip(axes.flat, methods)):
        sns.heatmap(matrices[method], annot=True, fmt=".4f", cmap="viridis",
                    vmin=vmin, vmax=vmax, xticklabels=test_families,
                    yticklabels=train_families, cbar=index == len(methods) - 1,
                    cbar_kws={"label": "mean final unseen-test cost"}, ax=ax)
        ax.set_xlabel("test family")
        ax.set_ylabel("training family")
        ax.set_title(RESAMPLING_LABELS[method])
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    fig.suptitle("Composition / distribution shift by instance resampling\n"
                 "new v2 factorial results only", y=1.01)
    _save(fig, out)


def plot_optimizer_set_v2(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
    optimizers: list[str],
    comparison_title: str,
    resampling: str | None = None,
):
    """Image-style configurator comparison on the honest SA-call axis."""
    labels = {
        "optuna_tpe": "Bayesian Opt. (TPE)",
        "random": "Random Search",
        "smac_bo": "SMAC (fixed-budget BO)",
        "smac_aac": "SMAC (native intensification)",
    }
    colors = {
        "optuna_tpe": "#1f77b4",
        "random": "#ff7f0e",
        "smac_bo": "#9467bd",
        "smac_aac": "#2ca02c",
    }
    optimizers = [optimizer for optimizer in optimizers
                  if optimizer in set(summary["optimizer"])]
    grid_data = grid[grid["optimizer"].isin(optimizers)]
    summary_data = summary[summary["optimizer"].isin(optimizers)].copy()
    scope = "all instance-resampling methods"
    if resampling is not None:
        grid_data = grid_data[grid_data["resampling"] == resampling]
        summary_data = summary_data[summary_data["resampling"] == resampling]
        scope = RESAMPLING_LABELS.get(resampling, resampling)
    if grid_data.empty or summary_data.empty:
        return

    curves = {
        optimizer: _aligned_step_curve(
            grid_data[grid_data["optimizer"] == optimizer], "test_cost"
        )
        for optimizer in optimizers
    }
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    for ax, zoomed in zip(axes[:2], (False, True)):
        for optimizer in optimizers:
            budgets, mean, sem = curves[optimizer]
            if not len(budgets):
                continue
            mask = np.ones(len(budgets), dtype=bool)
            if zoomed:
                mask = budgets >= max(float(budgets.min()), 0.05 * float(budgets.max()))
            ax.plot(budgets[mask], mean[mask], color=colors[optimizer], lw=2.5,
                    label=labels[optimizer])
            ax.fill_between(budgets[mask], mean[mask] - sem[mask],
                            mean[mask] + sem[mask], color=colors[optimizer], alpha=0.14)
        ax.set_xlabel("cumulative SA solver calls")
        ax.set_ylabel("mean incumbent unseen-test cost")
        ax.set_title(
            "Convergence — full SA-call range"
            if not zoomed else
            "Convergence — after first 5% of SA-call cap\n"
            + ("shows TPE vs RS separation" if len(optimizers) == 2
               else "shows configurator separation")
        )

    right = axes[2]
    summary_data["optimizer_label"] = summary_data["optimizer"].map(labels)
    if resampling is None:
        methods = _present_resamplings(summary_data["resampling"])
        summary_data["resampling_label"] = summary_data["resampling"].map(
            lambda value: RESAMPLING_LABELS.get(value, value)
        )
        sns.boxplot(
            data=summary_data, x="resampling_label", y="final_test_cost",
            hue="optimizer_label", order=[RESAMPLING_LABELS[m] for m in methods],
            hue_order=[labels[o] for o in optimizers],
            palette={labels[o]: colors[o] for o in optimizers},
            showfliers=False, ax=right,
        )
        right.set_xlabel("instance-resampling method")
        right.tick_params(axis="x", rotation=22)
        for tick in right.get_xticklabels():
            tick.set_ha("right")
        legend = right.get_legend()
        if legend is not None:
            legend.remove()
    else:
        sns.boxplot(
            data=summary_data, x="optimizer_label", y="final_test_cost",
            order=[labels[o] for o in optimizers],
            hue="optimizer_label", legend=False,
            palette={labels[o]: colors[o] for o in optimizers},
            showfliers=False, ax=right,
        )
        sns.stripplot(
            data=summary_data, x="optimizer_label", y="final_test_cost",
            order=[labels[o] for o in optimizers], color="0.2", alpha=0.5,
            size=4, ax=right,
        )
        right.set_xlabel("")
        if len(optimizers) > 2:
            right.tick_params(axis="x", rotation=18)
            for tick in right.get_xticklabels():
                tick.set_ha("right")
    right.set_ylabel("final unseen-test cost")
    right.set_title(
        "Final test cost: TPE vs Random Search"
        if len(optimizers) == 2 else "Final unseen-test cost by configurator"
    )

    handles = [plt.Line2D([0], [0], color=colors[o], lw=8) for o in optimizers]
    fig.legend(handles, [labels[o] for o in optimizers], loc="upper center",
               ncol=len(optimizers), bbox_to_anchor=(0.5, 1.045))
    fig.suptitle(f"{comparison_title} — {scope}", y=0.94)
    _save(fig, out, rect=(0, 0, 1, 0.84))


def plot_tpe_vs_random_v2(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
    resampling: str | None = None,
):
    return plot_optimizer_set_v2(
        grid,
        summary,
        out,
        ["optuna_tpe", "random"],
        "Bayesian Optimization (TPE) vs Random Search",
        resampling,
    )


def plot_all_configurators_v2(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
    resampling: str | None = None,
):
    return plot_optimizer_set_v2(
        grid,
        summary,
        out,
        ["optuna_tpe", "random", "smac_bo", "smac_aac"],
        "TPE vs Random Search vs SMAC fixed-budget vs SMAC native",
        resampling,
    )


def make_v1_style_suite_v2(grid: pd.DataFrame, summary: pd.DataFrame, out_dir: Path):
    """Render every v1 report figure from v2 tables, never from v1 CSVs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_summary = summary.copy()
    legacy_summary["total_validation_runtime_sec"] = legacy_summary["eval_sec"]
    for family in ("uniform", "clustered", "mixed"):
        legacy_summary[f"test_t_{family}"] = legacy_summary[f"final_test_{family}"]
    legacy_end = summary[["run_id", "resampling", "final_relative_overtuning"]].copy()
    legacy_end["trial_id"] = 1
    legacy_end["relative_overtuning_t"] = legacy_end["final_relative_overtuning"]

    _run_jobs([
        (plot_v1_style_representative_trajectory,
         (grid, summary, out_dir / "trajectory_validation_vs_test.png")),
        (plot_v1_style_trajectories_by_resampling,
         (grid, out_dir / "trajectories_by_resampling.png")),
        (plot_v1_ecdf_relative_overtuning,
         (legacy_end, out_dir / "ecdf_relative_overtuning.png")),
        (plot_v1_relative_overtuning_by_train_size,
         (legacy_summary, out_dir / "relative_overtuning_by_train_size.png")),
        (plot_v1_overtuning_frequency,
         (legacy_summary, out_dir / "overtuning_frequency_by_resampling.png")),
        (plot_v1_final_test_by_resampling,
         (legacy_summary, out_dir / "final_test_by_resampling.png")),
        (plot_v1_gap_by_train_size,
         (legacy_summary, out_dir / "generalization_gap_by_train_size.png")),
        (plot_v1_runtime_by_resampling,
         (legacy_summary, out_dir / "runtime_by_resampling.png")),
        (plot_v1_resampling_tradeoff,
         (legacy_summary, out_dir / "resampling_quality_runtime_tradeoff.png")),
        (plot_factorial_selected_parameters,
         (summary, out_dir / "selected_params_stability.png")),
        (plot_v1_style_composition,
         (summary, out_dir / "composition_heatmap.png")),
        (plot_v1_style_composition_by_resampling,
         (summary, out_dir / "composition_by_resampling.png")),
        (plot_tpe_vs_random_v2,
         (grid, summary, out_dir / "optimizer_comparison.png")),
        (plot_all_configurators_v2,
         (grid, summary, out_dir / "configurator_comparison.png")),
    ])

    by_resampling = out_dir / "optimizer_comparison_by_resampling"
    by_resampling.mkdir(parents=True, exist_ok=True)
    for method in _present_resamplings(summary["resampling"]):
        plot_tpe_vs_random_v2(
            grid, summary, by_resampling / f"{method}.png", resampling=method
        )

    all_by_resampling = out_dir / "configurator_comparison_by_resampling"
    all_by_resampling.mkdir(parents=True, exist_ok=True)
    for method in _present_resamplings(summary["resampling"]):
        plot_all_configurators_v2(
            grid,
            summary,
            all_by_resampling / f"{method}.png",
            resampling=method,
        )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

TRIALS_USECOLS = ["run_id", "optimizer", "train_family", "train_size", "seed",
                  "config_space", "resampling",
                  "trial_id", "config_id", "observed_cost", "n_sa_calls",
                  "cum_sa_calls", "is_incumbent", "initial_temperature",
                  "cooling_rate", "min_temperature", "iterations_per_temp", "restarts",
                  "p_swap", "p_insert", "p_2opt", "p_oropt"]


def load_trials_v2(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(raw_dir).glob("*_trials.csv")):
        try:
            frames.append(pd.read_csv(path, usecols=TRIALS_USECOLS))
        except ValueError:
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError(f"no trials CSVs in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def _standard_jobs(grid: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame,
                   trials: pd.DataFrame, fig_dir: Path):
    """Create the original figure set for data from one exact condition."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    return [
        (plot_convergence, (grid, fig_dir / "convergence_validation.png",
                            "val_cost", "validation cost")),
        (plot_convergence, (grid, fig_dir / "convergence_test.png",
                            "test_cost", "test cost")),
        (plot_best_test_trajectory, (grid, fig_dir / "best_so_far_test_trajectory.png")),
        (plot_final_boxplots, (summary, fig_dir / "final_cost_boxplots.png")),
        (plot_anytime_auc, (summary, fig_dir / "anytime_auc.png")),
        (plot_ecdf_final_test, (summary, fig_dir / "ecdf_final_test.png")),
        (plot_rank_over_budget, (grid, fig_dir / "rank_over_budget.png")),
        (plot_pairwise_winrate, (tests, fig_dir / "pairwise_winrate.png")),
        (plot_budget_to_target, (summary, fig_dir / "budget_to_target.png")),
        (plot_generalization_gap, (grid, fig_dir / "generalization_gap_over_budget.png")),
        (plot_overtuning_over_budget, (grid, fig_dir / "overtuning_over_budget.png")),
        (plot_overtuning_ecdf, (summary, fig_dir / "overtuning_ecdf.png")),
        (plot_proposal_quality, (trials, fig_dir / "proposal_quality.png")),
        (plot_param_concentration, (trials, fig_dir / "param_concentration.png")),
        (plot_overhead, (summary, fig_dir / "configurator_overhead.png")),
        (plot_smac_intensification, (trials, summary, fig_dir / "smac_intensification.png")),
        (plot_test_family_heatmap, (summary, fig_dir / "test_family_heatmap.png")),
    ]


def _run_jobs(jobs):
    for fn, args in jobs:
        try:
            fn(*args)
        except Exception as e:  # keep going: partial results should still plot
            print(f"  WARNING: {fn.__name__} failed: {e!r}")


def make_all_v2(processed_dir: Path, summaries_dir: Path, raw_dir: Path, fig_dir: Path):
    processed_dir, summaries_dir = Path(processed_dir), Path(summaries_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(processed_dir / "trajectories_grid.csv")
    summary = pd.read_csv(summaries_dir / "final_summary.csv")
    tests_path = summaries_dir / "stats_tests.csv"
    paired_path = summaries_dir / "paired_vs_random.csv"
    try:
        tests = pd.read_csv(tests_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        tests = pd.DataFrame(columns=["metric", "config_space",
                                      "train_family", "train_size", "resampling"])
    try:
        paired = pd.read_csv(paired_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        paired = pd.DataFrame(columns=["metric", "optimizer", "config_space",
                                       "train_family", "train_size", "resampling"])
    trials = load_trials_v2(raw_dir)

    # Preserve the complete v1 report figure vocabulary, but rebuild every
    # figure exclusively from this profile's v2 factorial tables.
    _run_jobs([(make_v1_style_suite_v2, (grid, summary, fig_dir))])

    # Complete training-size factorial: every row is a training-pool size,
    # every column is an instance-resampling method, and all configurators are
    # overlaid on the common cumulative-SA-call axis.  Only combine sizes when
    # their declared caps are genuinely equal.
    size_scopes = list(
        summary[["config_space", "train_family"]]
        .drop_duplicates().itertuples(index=False, name=None)
    )
    for space, family in size_scopes:
        summary_mask = ((summary["config_space"] == space)
                        & (summary["train_family"] == family))
        scope_summary = summary[summary_mask]
        if (scope_summary["train_size"].nunique() < 2
                or scope_summary["resampling"].nunique() < 2
                or "total_budget_calls" not in scope_summary
                or scope_summary["total_budget_calls"].nunique() != 1):
            continue
        grid_mask = ((grid["config_space"] == space)
                     & (grid["train_family"] == family))
        scope_grid = grid[grid_mask]
        target_dir = (fig_dir if len(size_scopes) == 1 else
                      fig_dir / "training_size_factorial" /
                      f"{space}__{family}".replace("-", "_"))
        target_dir.mkdir(parents=True, exist_ok=True)
        _run_jobs([
            (plot_overtuning_frequency_factorial_over_budget,
             (scope_grid,
              target_dir / "overtuning_frequency_over_budget_by_train_size.png")),
            (plot_relative_overtuning_factorial_over_budget,
             (scope_grid,
              target_dir / "relative_overtuning_over_budget_by_train_size.png")),
            (plot_relative_overtuning_eligibility_factorial_over_budget,
             (scope_grid,
              target_dir / "relative_overtuning_eligibility_over_budget_by_train_size.png")),
        ])

    # Direct factorial comparisons.  Existing per-condition plots below remain
    # useful, but these root-level figures answer the interaction question.
    base_columns = ["config_space", "train_family", "train_size"]
    base_conditions = list(
        summary[base_columns].drop_duplicates().itertuples(index=False, name=None)
    )
    for space, family, size in base_conditions:
        summary_mask = ((summary["config_space"] == space)
                        & (summary["train_family"] == family)
                        & (summary["train_size"] == size))
        if "resampling" not in summary or summary.loc[summary_mask, "resampling"].nunique() < 2:
            continue
        grid_mask = ((grid["config_space"] == space)
                     & (grid["train_family"] == family)
                     & (grid["train_size"] == size))
        target_dir = (fig_dir if len(base_conditions) == 1 else
                      fig_dir / "factorial" /
                      f"{space}__{family}__n{size}".replace("-", "_"))
        target_dir.mkdir(parents=True, exist_ok=True)
        factorial_summary = summary[summary_mask]
        factorial_grid = grid[grid_mask]
        _run_jobs([
            (plot_all_configurators_all_resamplings,
             (factorial_grid, factorial_summary,
              target_dir / "all_configurators_all_resamplings.png")),
            (plot_factorial_overtuning_ecdf_by_resampling,
             (factorial_summary,
              target_dir / "all_configurators_all_resamplings_ecdf.png")),
            (plot_factorial_overtuning_ecdf_by_resampling,
             (factorial_summary,
              target_dir / "all_configurators_all_resamplings_ecdf_zoomed.png",
              0.75)),
            (plot_factorial_overtuning_ecdf_matrix,
             (factorial_summary,
              target_dir / "all_configurators_all_resamplings_ecdf_matrix.png",
              0.75)),
            (plot_factorial_final_performance,
             (factorial_summary, target_dir / "configurator_by_resampling_final.png")),
            (plot_factorial_test_trajectories,
             (factorial_grid, target_dir / "configurator_by_resampling_trajectories.png")),
            (plot_factorial_overtuning_ecdf,
             (factorial_summary, target_dir / "resampling_ecdf_by_configurator.png")),
            (plot_factorial_overtuning_ecdf,
             (factorial_summary,
              target_dir / "resampling_ecdf_by_configurator_zoomed.png", 0.75)),
            (plot_factorial_heatmaps,
             (factorial_summary, target_dir / "configurator_resampling_heatmaps.png")),
            (plot_factorial_efficiency,
             (factorial_summary, target_dir / "configurator_resampling_efficiency.png")),
            (plot_factorial_selected_parameters,
             (factorial_summary, target_dir / "selected_parameters_by_factorial_cell.png")),
        ])

    if len(paired):
        _run_jobs([
            (plot_bo_vs_random_conditions,
             (paired, fig_dir / "bo_vs_random_by_condition.png")),
            (plot_bo_vs_random_winrates,
             (paired, fig_dir / "bo_vs_random_winrates.png")),
            (plot_final_parameter_shifts,
             (summary, fig_dir / "final_parameter_shifts_vs_random.png")),
        ])

    conditions = _condition_tuples(summary)
    if len(conditions) == 1:
        _run_jobs(_standard_jobs(grid, summary, tests, trials, fig_dir))
        return

    for condition in conditions:
        space, family, size, resampling = condition
        slug = f"{space}__{family}__n{size}__{resampling}".replace("-", "_")
        condition_dir = fig_dir / "conditions" / slug
        condition_grid = grid[_condition_mask(grid, condition)]
        condition_summary = summary[_condition_mask(summary, condition)]
        condition_tests = tests[_condition_mask(tests, condition)] if len(tests) else tests
        condition_trials = trials[_condition_mask(trials, condition)]
        _run_jobs(_standard_jobs(condition_grid, condition_summary, condition_tests,
                                 condition_trials, condition_dir))
