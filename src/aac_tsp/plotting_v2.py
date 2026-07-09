"""Figures for experiment v2: configurator comparison (BO vs RS vs SMAC).

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
import numpy as np
import pandas as pd
import seaborn as sns

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


def _present(values) -> list[str]:
    have = set(values)
    return [o for o in OPTIMIZER_ORDER if o in have]


def _save(fig, out: Path):
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _sizes(df) -> list[int]:
    return sorted(df["train_size"].unique())


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
# 8. Generalization gap over budget + 9. overtuning ECDF
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
                    label=OPTIMIZER_LABELS[opt])
        ax.axvline(1.0, color="k", ls="--", lw=1.2, label="all progress lost (=1)")
        ax.set_xlabel("final relative overtuning")
        ax.set_title(f"n={size}")
    axes[0][0].set_ylabel("proportion of runs")
    axes[0][-1].legend(fontsize=11)
    fig.suptitle("ECDF of relative overtuning by configurator (Schneider et al. metric)",
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
    params = [("log10_T0", "log10 initial temperature"),
              ("cooling_rate", "cooling rate"),
              ("share_2opt", "2-opt share of move mixture")]
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
        ax.axvline(size, color=line.get_color(), ls="--", lw=1.5,
                   label=f"fixed-budget arms spend exactly {size}")
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
# Driver
# --------------------------------------------------------------------------- #

TRIALS_USECOLS = ["run_id", "optimizer", "train_family", "train_size", "seed",
                  "trial_id", "config_id", "observed_cost", "n_sa_calls",
                  "cum_sa_calls", "is_incumbent", "initial_temperature",
                  "cooling_rate", "p_swap", "p_insert", "p_2opt", "p_oropt"]


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


def make_all_v2(processed_dir: Path, summaries_dir: Path, raw_dir: Path, fig_dir: Path):
    processed_dir, summaries_dir = Path(processed_dir), Path(summaries_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(processed_dir / "trajectories_grid.csv")
    summary = pd.read_csv(summaries_dir / "final_summary.csv")
    tests_path = summaries_dir / "stats_tests.csv"
    try:
        tests = pd.read_csv(tests_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        tests = pd.DataFrame(columns=["metric"])  # too few seeds for tests
    trials = load_trials_v2(raw_dir)

    jobs = [
        (plot_convergence, (grid, fig_dir / "convergence_validation.png",
                            "val_cost", "validation cost")),
        (plot_convergence, (grid, fig_dir / "convergence_test.png",
                            "test_cost", "test cost")),
        (plot_final_boxplots, (summary, fig_dir / "final_cost_boxplots.png")),
        (plot_ecdf_final_test, (summary, fig_dir / "ecdf_final_test.png")),
        (plot_rank_over_budget, (grid, fig_dir / "rank_over_budget.png")),
        (plot_pairwise_winrate, (tests, fig_dir / "pairwise_winrate.png")),
        (plot_budget_to_target, (summary, fig_dir / "budget_to_target.png")),
        (plot_generalization_gap, (grid, fig_dir / "generalization_gap_over_budget.png")),
        (plot_overtuning_ecdf, (summary, fig_dir / "overtuning_ecdf.png")),
        (plot_proposal_quality, (trials, fig_dir / "proposal_quality.png")),
        (plot_param_concentration, (trials, fig_dir / "param_concentration.png")),
        (plot_overhead, (summary, fig_dir / "configurator_overhead.png")),
        (plot_smac_intensification, (trials, summary, fig_dir / "smac_intensification.png")),
        (plot_test_family_heatmap, (summary, fig_dir / "test_family_heatmap.png")),
    ]
    for fn, args in jobs:
        try:
            fn(*args)
        except Exception as e:  # keep going: partial results should still plot
            print(f"  WARNING: {fn.__name__} failed: {e!r}")
