# AGENTS.md — Orientation guide for Claude (and other AI assistants)

This file is for AI assistants continuing work on this project. Read it before touching any code.

---

## What this project is

A university seminar experiment on **configuration generalization in AAC** (Automated Algorithm Configuration). The experiment measures how much the choice of resampling strategy affects whether a Bayesian Optimizer overfits its training instances when configuring Simulated Annealing for TSP.

**Research question:** Does better resampling reduce configuration overfitting in BO-based AAC?

The full experiment has already been run and results are committed. The typical next tasks are: extending the analysis, improving figures for the report/presentation, or re-running with modified parameters.

---

## Repository state (as of the last run)

- **720-run final experiment: COMPLETE** (`results/FINAL_COMPLETE` exists).
- Raw CSVs: `results/raw/full/` (360 files) and `results/raw/fixed_2opt/` (360 files).
- Post-processed trajectories and summaries: `results/processed/` and `results/summaries/`.
- Figures: `figures/full/` and `figures/fixed_2opt/` (8 figures each). Pilot-run figures archived in `figures/pilot/`.
- Pilot artifacts (54 runs): `results/raw/*.csv` (top-level, not subdirectory).

### Figure changes made for the preliminary result presentation (post-experiment)

The three figure functions in `src/aac_tsp/plotting.py` were updated (no re-computation of the 720 runs):

1. **`plot_ecdf_relative_overtuning`** — changed from a single panel to two panels:
   - Left: full x-range (shows holdout's heavy tail extending to ~6.5).
   - Right: zoomed y-axis (y ≥ 0.88), mirroring Schneider et al. Fig. 2, so inter-resampling separation is visible.

2. **`plot_composition_heatmap`** — relabelled from "gap" to "test cost" (the values are the normalised tour gap `m(θ,π)`, not the test−validation gap). The title and colorbar now read "mean final test cost, not gap".

3. **`plot_optimizer_comparison`** — changed from 2 panels to 3:
   - Left: full-scale convergence (shows the iter-0 random spike of RS).
   - Centre: zoomed convergence (iter ≥ 2), showing actual TPE vs RS separation.
   - Right: final test cost boxplot (unchanged).

The README "Composition / distribution shift" result and guideline #5 were also corrected: the prior claim ("mixed training is best, row mean 0.033 vs 0.038") was stale pilot data. The actual finding is that test-family difficulty dominates; train-family choice is negligible (row variation < 0.0005).

---

## Critical design decisions — do not change without understanding the reason

**1. The optimizer never sees test cost.**
In `src/aac_tsp/runner.py`, `test_cost` is evaluated only when the validation incumbent changes and is written to the CSV but never returned to `study.tell()`. Violating this invalidates the experiment.

**2. Two config-space variants exist for a reason.**
- `full`: `move_type` ∈ {swap, insert, 2-opt} is part of the search space. This is the realistic AAC setting. Overtuning shows up in the generalisation gap and overtuning-event proportion.
- `fixed_2opt`: `move_type` is fixed to 2-opt. This removes a large structural performance cliff (2-opt solutions are ~15× better than swap/insert) that dominates the relative-overtuning denominator and makes the ECDF uninformative. Use `fixed_2opt` for the paper-style ECDF figure.

Both variants are in `configs/final.yaml` (`config_spaces: [full, fixed_2opt]`). Do not merge them into a single CSV without adding a `config_space` grouping column.

**3. "CV" here means instance resampling, not model cross-validation.**
There is no model. A `cv5` fold means: split the training *instance pool* into 5 groups, evaluate the SA configuration on each group, average. This is an estimator of `Ĉ(θ) = E_{π}[m(θ,π)]`. In the report and presentation always write "5-fold instance resampling", not "5-fold CV".

**4. Instances are synthetic, generated with fixed seeds.**
Do not replace them with real TSPLIB/DIMACS instances without re-generating best-known references. The normalisation `m(θ,π) = (tour - best_known) / best_known` depends on the `data/best_known/best_known.csv` baseline, which was computed by multi-start NN + 2-opt (20 starts). The absolute values of this baseline do not matter — only consistency across all configurations.

**5. GNU parallel is not installed on the VM. Use xargs -P.**
```bash
tail -n +3 scripts/run_experiments.sh | xargs -P 32 -I{} bash -c '{}'
```
The `run_final.sh` driver already does this with `-P 48`.

---

## Key files to read before editing

| File | Why |
|---|---|
| `src/aac_tsp/runner.py` | The BO loop; `RunConfig` dataclass; test-cost isolation |
| `src/aac_tsp/resampling.py` | All four resampling strategy classes |
| `src/aac_tsp/optimizers.py` | `suggest_config(trial, config_space)` — note the `config_space` param |
| `src/aac_tsp/metrics.py` | Definitions of overtuning, relative overtuning, generalisation gap |
| `src/aac_tsp/plotting.py` | All 7 figures; `RESAMPLING_LABELS` maps code keys to display names |
| `scripts/create_experiments.py` | Expands a profile YAML into `run_experiments.sh`; handles `config_spaces` |
| `configs/final.yaml` | The full experiment grid |

---

## Common tasks and how to do them

### Re-run the full experiment from scratch

```bash
source .venv/bin/activate
# Regenerate instances if needed (only if /local/anmol/datasets/tsp/ is missing)
python scripts/generate_instances.py --n_cities 50 --n_train_pool 150 --n_test 100
python scripts/compute_best_known.py --n_cities 50 --n_starts 20

# Run (this clears old raw results and re-runs everything)
bash scripts/run_final.sh
```

### Re-run postprocessing and re-plot (no recompute)

```bash
source .venv/bin/activate
python scripts/postprocess.py --raw_dir results/raw/full \
    --processed_dir results/processed/full --summaries_dir results/summaries/full
python scripts/make_plots.py --raw_dir results/raw/full \
    --processed_dir results/processed/full --summaries_dir results/summaries/full \
    --fig_dir figures/full

python scripts/postprocess.py --raw_dir results/raw/fixed_2opt \
    --processed_dir results/processed/fixed_2opt --summaries_dir results/summaries/fixed_2opt
python scripts/make_plots.py --raw_dir results/raw/fixed_2opt \
    --processed_dir results/processed/fixed_2opt --summaries_dir results/summaries/fixed_2opt \
    --fig_dir figures/fixed_2opt
```

### Add a new resampling strategy

1. Add a class to `src/aac_tsp/resampling.py` that inherits `ResamplingStrategy` and implements `make_plan(train_ids, seed) -> list[list[str]]`.
2. Register it in the `make_resampling(name, ...)` factory at the bottom of the same file.
3. Add it to `choices` in `scripts/main.py` (`--resampling` arg).
4. Add it to the profile YAML under `resamplings:`.

### Add a new plot

Add a function to `src/aac_tsp/plotting.py` and call it inside `make_all()`. The `RESAMPLING_LABELS` dict at the top of that file controls display names for resampling methods; `OPTIMIZER_LABELS` (inside `plot_optimizer_comparison`) does the same for optimizers.

### Smoke-test a code change before re-running the full experiment

```bash
source .venv/bin/activate
python scripts/create_experiments.py --profile smoke
bash scripts/run_experiments.sh
python scripts/postprocess.py
python scripts/make_plots.py
```
The smoke profile runs 2 cells in under a minute.

### Run a single cell interactively

```bash
source .venv/bin/activate
python scripts/main.py \
    --optimizer optuna_tpe \
    --resampling holdout \
    --config_space fixed_2opt \
    --train_family mixed \
    --train_size 10 \
    --bo_budget 15 \
    --seed 0
```

---

## VM environment

- **Path:** `/local/anmol/aac_tsp_overtuning`
- **Python venv:** `.venv/` (activate with `source .venv/bin/activate`)
- **Instance data:** `/local/anmol/datasets/tsp/` (not in this repo)
- **CPU:** 64 vCPUs (Intel Xeon Platinum 8462Y+), 125 GB RAM, no GPU
- **Disk:** ~600 GB free on `/local`; write large outputs there, not to `/tmp` or `/`
- **Internet:** available; PyPI reachable
- **GNU parallel:** not installed — use `xargs -P N`
- **screen:** available (`screen -dmS name cmd` to detach)

---

## What the results show (summary for context)

**Full config space:**
Holdout has the largest generalisation gap (0.0066 vs 0.0006 for repeated CV / bootstrap OOB) and the highest overtuning frequency (16 % vs 6–8 %). Stronger resampling nearly eliminates the gap at 14–17× runtime cost. Training-pool size matters: the gap at n=10 is 2–3× larger than at n=50 for all resamplings.

**Fixed_2opt config space (ECDF headline):**
Removing the move-type cliff reveals relative overtuning. Holdout: mean 13.7 % of tuning progress lost. Repeated 5-fold instance resampling: 3.5 %. The ECDF of relative overtuning in `figures/fixed_2opt/ecdf_relative_overtuning.png` is the paper-style headline figure.

**Composition / distribution shift:**
Training on `mixed` instances generalises best across all test families. The dominant axis is test-family difficulty (clustered is easier), not the train↔test family match.

---

## Experiment v2 — configurator comparison (added later; separate code path)

v2 compares **configurators** (random, optuna_tpe, smac_bo, smac_aac) at equal
total SA-call budget on a 15-parameter config space. It is fully documented in
`EXPERIMENT_V2.md` and deliberately does **not** touch any v1 module: new files
are `solver_sa_ext.py`, `space_v2.py`, `evaluation_v2.py`, `runner_v2.py`,
`postprocess_v2.py`, `plotting_v2.py`, `scripts/*_v2.py`, `configs/bovsrs*.yaml`,
and `slurm/`. Key invariants:

1. **Budget unit = one SA solver call.** The smac_aac arm gets
   `n_trials * train_size * solver_repeats` single-instance target calls; the
   fixed-budget arms spend the same total. Never compare arms "per iteration".
2. **Trajectories live on the `cum_sa_calls` axis** (`*_traj.csv`, one row per
   incumbent change + a closing `final` row). Post-processing interpolates step
   functions onto a shared log grid; nothing joins on `trial_id` across arms.
3. **Analysis evaluations (test cost; canonical validation re-evaluation for
   smac_aac) never reach a configurator and don't count toward budget.**
4. The v2 space is defined once in `space_v2.py` and exposed to Optuna
   (define-by-run) and SMAC (ConfigSpace with `EqualsCondition`s). Keep the three
   views in sync if you change ranges.
5. SMAC only runs on Linux (pyrfr). Local Windows smoke tests cover the Optuna
   arms; always run `PROFILE=bovsrs_smoke bash slurm/submit_all.sh` on the VM
   before the full 200-run submission.

---

## Future work / TODO (for final written report)

These are **not** implemented. No existing results need to be recomputed; the 720 runs are final. These are extensions for the final report.

### Statistical confidence
- **More seeds (5 → 20):** update `configs/final.yaml` seeds list, re-run `run_final.sh`. No code changes. Stabilises the relative-overtuning vs. train-size curves and smooths the zoomed ECDF staircase.
- **Significance testing:** add Wilcoxon signed-rank tests (or a small LMM mirroring the paper's Table 1) comparing resampling methods pairwise on final relative overtuning and generalisation gap. Would go in a new `scripts/statistics.py` script.

### New experimental levers (paper-aligned)
- **Reshuffling resampling splits:** at each BO trial, use a freshly drawn random split instead of fixed folds. Requires a new `ResamplingStrategy` subclass in `resampling.py` + a YAML key. Most direct extension of the paper's Section 6.
- **Incumbent-selection mitigation:** post-hoc analysis on existing trajectory CSVs — select the incumbent by posterior mean or apply early stopping (no new SA evaluations needed).
- **Larger BO budget (50 → 200):** tests whether the inverted-U budget effect from the paper appears. Update `bo_budget` in the YAML profile.

### Figure improvements (cosmetic)
- Add bootstrap CIs to the `generalization_gap_by_train_size.png` pointplot.
- Add `iterations_per_temp` and `restarts` panels to `selected_params_stability.png`.

---

## Reference

Lennart Schneider, Bernd Bischl, and Matthias Feurer. **Overtuning in Hyperparameter Optimization.** AutoML 2025. https://proceedings.mlr.press/v293/schneider25a.html
