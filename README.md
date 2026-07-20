# AAC Configuration Generalization — Overtuning Study

**Seminar project: Topic 5 — Configuration Generalization**
*Department of Computer Science, University Seminar on Automated Algorithm Configuration*

**Authors:** Anmol Mathur, Rohit Mamgain

---

## Research question

> Does better resampling reduce configuration overfitting in Bayesian-Optimization-based Automated Algorithm Configuration (AAC)?

| Component | This experiment |
|---|---|
| Target problem | Traveling Salesperson Problem (TSP) |
| Target algorithm | Simulated Annealing (SA) |
| Configurator | Bayesian Optimization — Optuna TPE (+ Random Search baseline) |
| Comparison axis | Resampling strategy used to estimate configuration quality |
| Conceptual anchor | Schneider, Bischl & Feurer, *Overtuning in HPO*, AutoML 2025 |

---

> **Experiment v2 — configurator × instance-resampling comparison.**
> The primary new experiment crosses 4 configurators (Random Search, Optuna TPE,
> SMAC-BO, SMAC-AAC) with all 4 instance-resampling methods at equal SA-call
> caps: 192 runs with 12 matched seeds. An optional 512-run profile adds
> targeted training-size, family, and 11-vs-15-parameter-space conditions.
> The complete training-size extension is
> `bovsrs_resampling_train_sizes`: 576 runs crossing the same 4 configurators
> and 4 instance-resampling methods at n=10, 25, and 50 with 12 matched seeds.
> Every size receives the same 5,000 optimization-SA-call cap.
> Its plotting stage regenerates every v1-style report graph from the new v2
> results only, plus overall and per-resampling comparisons for TPE-vs-Random
> and for all four configurators together. The headline
> `all_configurators_all_resamplings*.png` figures show every one of the 16
> configurator-resampling cells simultaneously.
> See **[EXPERIMENT_V2.md](EXPERIMENT_V2.md)** and
> **[EXPERIMENT_COVERAGE.md](EXPERIMENT_COVERAGE.md)**;
> smoke-test with `PROFILE=bovsrs_resampling_smoke bash slurm/run_local.sh`, then
> run `PROFILE=bovsrs_resampling_train_sizes_smoke bash slurm/run_local.sh`
> before the full `PROFILE=bovsrs_resampling_train_sizes bash slurm/run_local.sh`
> job on the VM.

---

## Background

Algorithm configurators are trained on a fixed set of instances. Configurations that perform well on training instances do not always generalise — this is the AAC analogue of overfitting in machine learning.

The key quantity is the **generalisation gap**:

```
gap = test_cost(θ*) − validation_cost(θ*)
```

where `θ*` is the configuration selected by the configurator. A positive gap means the selected configuration is worse on unseen instances than the training signal suggested.

**What "resampling" means in AAC (important distinction)**

In AAC there is no model trained on folds — a configuration `θ` is only *evaluated* on instances. A resampling strategy defines the **estimator of configuration quality** `Ĉ(θ)` that the configurator minimises. Higher-variance estimators (e.g. holdout over a tiny subset) cause the configurator to chase noise, leading to overtuning.

```
Ĉ(θ) = (1/|Π_train|) Σ_{π ∈ Π_train} m(θ, π)
```

The four estimators compared, in order of decreasing variance:

| Code key | Report name | Description |
|---|---|---|
| `holdout` | Holdout instance sampling | Fixed 20 % validation subset |
| `cv5` | 5-fold instance resampling | 5-fold split over training instances |
| `repeated_cv5` | Repeated 5-fold instance resampling | 3 × 5-fold with different splits |
| `bootstrap_oob` | Bootstrap OOB instance resampling | 10 bootstrap resamples; OOB instances are validation |

The optimizer sees **only** `validation_cost`. `test_cost` is logged for analysis and never fed back to the optimizer.

---

## Dataset

Synthetic TSP instances, generated with fixed seeds — full control over size, diversity, and composition. Three families mirror the standard DIMACS TSP Challenge generators:

| Family | Generator analogue | Description |
|---|---|---|
| `uniform` | portgen / RUE | Cities uniform in the unit square |
| `clustered` | portcgen | 3–5 Gaussian clusters, centres uniform in the square |
| `mixed` | — | Blend of uniform and clustered points |

**Scale:** 50 cities, 150 training instances + 100 test instances per family.
Stored under `/local/rohit/datasets/tsp/` (not committed to this repo due to size; regenerate with the script below).
Both v1 and v2 load these exact pool files and the same `best_known.csv`; the
v2 launcher only generates them when the shared dataset is missing or incomplete.

Performance metric — normalised tour gap:
```
m(θ, π) = (tour_length(θ, π) − best_known(π)) / best_known(π)
```
`best_known` is approximated by multi-start nearest-neighbour + 2-opt (20 starts); consistent across all configurations.

---

## SA configuration space

| Parameter | Type | Range |
|---|---|---|
| `initial_temperature` | float (log) | 1 – 1000 |
| `cooling_rate` | float | 0.80 – 0.999 |
| `iterations_per_temp` | int | 10 – 300 |
| `move_type` | categorical | swap, insert, 2-opt |
| `restarts` | int | 0 – 10 |

Two config-space variants are run:

- **`full`** — `move_type` is part of the search space (realistic AAC; overtuning shows in the generalisation gap).
- **`fixed_2opt`** — `move_type` fixed to 2-opt; tunes only continuous/integer params (removes the structural 2-opt performance cliff and makes the ECDF of relative overtuning cleanly discriminating, matching the paper's headline figure).

---

## Setup

Requires Python ≥ 3.10. No GPU needed.

```bash
git clone <this-repo>
cd /local/rohit/projects/aacnewtest

python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade && pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Regenerate instances and best-known references

```bash
# Creates /local/rohit/datasets/tsp/ (or change --data_dir)
python scripts/generate_instances.py --n_cities 50 --n_train_pool 150 --n_test 100
python scripts/compute_best_known.py --n_cities 50 --n_starts 20
```

---

## Running experiments

### Option A — smoke test (< 1 min, confirms the pipeline works)

```bash
python scripts/create_experiments.py --profile smoke
bash scripts/run_experiments.sh
python scripts/postprocess.py
python scripts/make_plots.py
```

### Option B — pilot (54 runs, ~30 s on 32 cores, first real signal)

```bash
python scripts/create_experiments.py --profile pilot
# GNU parallel (if available):
cat scripts/run_experiments.sh | parallel -j 32
# or xargs:
tail -n +3 scripts/run_experiments.sh | xargs -P 32 -I{} bash -c '{}'
# or serial:
bash scripts/run_experiments.sh

python scripts/postprocess.py
python scripts/make_plots.py
```

### Option C — full final experiment (720 runs, ~8 min on 48 cores)

The `run_final.sh` driver generates runs, executes them in parallel, and post-processes + plots both config-space variants automatically.

```bash
# Optionally in a detached screen session:
screen -dmS aac bash scripts/run_final.sh
# Monitor:
tail -f results/final_run.log
# Done when this file appears:
ls results/FINAL_COMPLETE
```

Results land in `results/raw/full/`, `results/raw/fixed_2opt/`, and the corresponding `figures/` subdirectories.

### Running a single cell

```bash
python scripts/main.py \
    --optimizer optuna_tpe \
    --resampling cv5 \
    --config_space fixed_2opt \
    --train_family mixed \
    --train_size 25 \
    --bo_budget 50 \
    --seed 0
```

---

## Experiment profiles

| Profile | YAML | Runs | Purpose |
|---|---|---|---|
| smoke | `configs/smoke.yaml` | 2 | Pipeline check (minutes) |
| pilot | `configs/pilot.yaml` | 54 | First real signal (full space) |
| pilot_fixed2opt | `configs/pilot_fixed2opt.yaml` | 54 | Verify ECDF separation (fixed_2opt) |
| final | `configs/final.yaml` | 720 | Full result (both config spaces) |

Cost knobs in any YAML: `solver_repeats`, `cv_repeats`, `bootstrap_repeats`, `test_size`, `bo_budget`, `max_steps`.

---

## Output structure

```
results/
  raw/
    full/          one CSV + JSON sidecar per run (full config space)
    fixed_2opt/    one CSV + JSON sidecar per run (fixed_2opt config space)
  processed/
    full/incumbent_trajectories.csv
    fixed_2opt/incumbent_trajectories.csv
  summaries/
    full/{final_summary,condition_summary}.csv
    fixed_2opt/{final_summary,condition_summary}.csv
  final_run.log
  FINAL_COMPLETE   (sentinel; appears when run_final.sh finishes)

figures/
  full/            13 figures — full config space (generalization gap + overtuning story)
  fixed_2opt/      13 figures — fixed_2opt (ECDF + optimizer comparison headline figures)
  pilot/           7 figures from the initial 54-run pilot (reference only)
```

The 13 figures produced per variant use the report-facing names “Holdout instance
sampling”, “5-fold instance resampling”, “Repeated 5-fold instance resampling”,
and “Bootstrap OOB instance resampling”. The CSV/code keys remain unchanged for
backward compatibility.

| Filename | What it shows |
|---|---|
| `trajectory_validation_vs_test.png` | Validation vs test incumbent for one illustrative run |
| `trajectories_by_resampling.png` | Mean validation and unseen-test trajectories for all four instance-resampling methods, with standard-error ribbons |
| `ecdf_relative_overtuning.png` | ECDF of relative overtuning — full scale plus paper-style zoom at y ≥ 0.88 |
| `relative_overtuning_by_train_size.png` | Eligible final relative overtuning vs training-pool size |
| `overtuning_frequency_by_resampling.png` | Final overtuning, severe overtuning, and relative-metric eligibility rates |
| `final_test_by_resampling.png` | Boxplot of final test cost by instance-resampling method |
| `generalization_gap_by_train_size.png` | Gap vs training-pool size by instance-resampling method |
| `runtime_by_resampling.png` | Computational cost per instance-resampling method |
| `resampling_quality_runtime_tradeoff.png` | Runtime against generalization gap and relative overtuning |
| `selected_params_stability.png` | All five selected SA parameters: four numeric distributions, move-type proportions, and move-type/test-cost relation |
| `composition_heatmap.png` | Train-family × test-family mean final **test cost** (not gap); shows test-family difficulty dominates, not train↔test match |
| `composition_by_resampling.png` | The same train-family × test-family analysis split into one panel per instance-resampling method |
| `optimizer_comparison.png` | TPE vs Random Search — three panels: full-scale convergence, zoomed convergence (iter ≥ 2), final test cost boxplot |

**Per-run CSV columns:** `trial_id`, `optimizer`, `config_space`, `resampling`, `train_family`, `train_size`, `seed`, SA parameters, `validation_cost`, `test_cost`, `test_cost_{uniform,clustered,mixed}`, `is_incumbent`, runtimes.

**Trajectory CSV columns:** `val_t` (non-increasing incumbent validation cost),
`test_t`, `generalization_gap_t`, `overtuning_t`, test progress from the initial
incumbent, raw relative overtuning, its eligibility flag, and the reported
`relative_overtuning_t`. Following Schneider et al. Section 5, the reported
relative metric is `NaN` when test progress is below 0.001; the raw value remains
available for auditing.

---

## Results summary

Results from the full final experiment (720 runs: 2 config spaces × 2 optimizers × 4 resamplings × 3 families × 3 train sizes × 5 seeds).

### Full config space — generalisation gap by instance-resampling method

| Resampling | Mean gap | Proportion of runs with overtuning | Mean runtime / run (s) |
|---|---|---|---|
| Holdout instance sampling | **0.0066** | **16.2 %** | 2.2 |
| 5-fold instance resampling | 0.0018 | 12.3 % | 10.3 |
| Repeated 5-fold instance resampling | 0.0006 | 7.7 % | 30.7 |
| Bootstrap OOB instance resampling | 0.0006 | 5.8 % | 37.5 |

Holdout instance sampling has the largest generalisation gap and the most overtuning events. Stronger estimators (repeated 5-fold instance resampling and Bootstrap OOB instance resampling) nearly eliminate the gap, at a runtime cost of ~14–17×.

### Fixed_2opt config space — relative overtuning by instance-resampling method (ECDF headline)

| Resampling | Mean relative overtuning | Proportion with overtuning |
|---|---|---|
| Holdout instance sampling | **0.137** | **25.2 %** |
| 5-fold instance resampling | 0.039 | 17.8 % |
| Bootstrap OOB instance resampling | 0.055 | 17.2 % |
| Repeated 5-fold instance resampling | **0.035** | **16.2 %** |

Fixing the move type reveals the relative-overtuning signal: holdout overshoots the optimal configuration by 13.7 % on average; repeated 5-fold instance resampling reduces this to 3.5 %.

### Composition / distribution shift

The dominant axis is **test-family difficulty**, not the train↔test family match. All three training families achieve nearly identical mean final test cost on any given test family (row variation < 0.0005), while the test-family column varies strongly: uniform is hardest (≈ 0.033), clustered is easiest (≈ 0.012), and mixed is intermediate (≈ 0.024).

| Train \ Test | uniform | clustered | mixed |
|---|---|---|---|
| uniform  | 0.0331 | 0.0125 | 0.0237 |
| clustered | 0.0334 | 0.0114 | 0.0238 |
| mixed    | 0.0331 | 0.0118 | 0.0237 |

Note: these are **mean final test cost** values (normalised tour gap), **not** the generalisation gap (test − validation). The figure `composition_heatmap.png` shows these numbers.

Practical implication: in this TSP+SA setup the choice of training family had negligible impact on out-of-distribution performance. A mixed pool is a safe default (no worse than any other), but not demonstrably superior.

---

## Practical guidelines for training/test set design

These guidelines are derived directly from the experimental results.

### 1. Avoid holdout with small training pools

With only 10 training instances and holdout, a single validation subset may contain as few as 2 instances. The configurator treats noise in those 2 instances as signal, and the selected configuration generalises poorly. In our experiment, **holdout at n=10 produced a generalization gap of 0.010 — roughly 5× larger than 5-fold instance resampling (0.002) under the same conditions**. If holdout is your only option (time budget), use at least 50 training instances.

### 2. Use at least 5-fold instance resampling when your training pool is small

5-fold instance resampling evaluates each candidate configuration on the full training pool (in 5 grouped folds) rather than a 20% subset. This single change reduces the generalization gap by 3–5× at only ~5× the runtime cost per BO trial. For a training pool of ≥ 25 instances, 5-fold instance resampling is a practical default.

### 3. Repeated resampling pays off — but only meaningfully at n ≥ 25

Repeated 5-fold instance resampling (3 repetitions) reduces the mean relative overtuning from **13.7% (holdout) to 3.5%**, the best of all methods tested. However, the gain over a single 5-fold instance-resampling pass is modest unless the training pool is large enough that different folds actually see different instance structure. At n=10, repeated 5-fold instance resampling and Bootstrap OOB instance resampling perform similarly to one 5-fold pass.

### 4. Bootstrap OOB is not worth the cost at moderate training sizes

Bootstrap OOB instance resampling costs ~14–17× as much as holdout per BO run, similar to repeated 5-fold instance resampling, but it does not outperform the repeated-fold method in our experiments. Use it only if you have a strong prior that out-of-bag coverage is more representative than fold coverage for your instance distribution.

### 5. A mixed training pool is a safe default, but train-family choice barely mattered here

In this TSP+SA experiment, the choice of training instance family had negligible impact on out-of-distribution performance: all three training families achieved essentially the same mean final test cost on every test family (row variation < 0.0005). The dominant axis is **test-family difficulty**, not the train↔test match. A mixed pool is therefore a safe (and non-inferior) default when the deployment distribution is unknown, but it is not demonstrably superior to a uniform or clustered-only pool in this setting. Note: this finding might differ in setups with stronger distribution shift (e.g., larger instances, clustered vs. random-Euclidean, or real-world TSPLIB data).

### 6. More training instances always help — but with diminishing returns

Across all resampling methods, the generalization gap at n=50 was roughly half the gap at n=10. The sharpest drop is from n=10 to n=25; the marginal benefit of going from n=25 to n=50 is smaller. For practical configuration budgets, **n=25 with 5-fold instance resampling** is a reasonable operating point: it approaches the quality of the same estimator at n=50 at a fraction of the evaluation cost.

### 7. Use Bayesian Optimization — Random Search does not scale

In our experiment, Optuna TPE (BO) achieved 23.5% improvement in test cost over 50 iterations, versus 17.6% for Random Search. The gap is modest here because the SA search space has one dominant categorical variable (move type). On continuous, higher-dimensional configuration spaces the advantage of BO over random search is expected to be larger. Random Search is a useful sanity-check baseline but should not be used as the primary configurator.

---

## Source layout

```
src/aac_tsp/
  instances.py     TSP instance generation and IO (three families)
  solver_sa.py     Simulated Annealing; inner loop numba-JIT compiled
  resampling.py    Four resampling strategies (holdout, cv5, repeated_cv5, bootstrap_oob)
  objective.py     Maps a configuration to a validation cost via a resampling plan
  optimizers.py    Optuna TPE and Random Search wrappers; suggest_config()
  runner.py        Full BO loop for one experiment cell; RunConfig dataclass
  best_known.py    Multi-start NN + 2-opt reference lengths for normalisation
  metrics.py       Overtuning, relative overtuning, generalisation gap (Schneider et al.)
  postprocess.py   Aggregate raw CSVs → trajectories + summaries
  plotting.py      7 report figures

scripts/
  generate_instances.py   Create instance .npz files
  compute_best_known.py   Compute and save best_known.csv
  create_experiments.py   Expand a profile YAML → run_experiments.sh
  main.py                 Single-cell entry point (CLI → RunConfig → run_experiment)
  postprocess.py          CLI wrapper for postprocess.build_outputs
  make_plots.py           CLI wrapper for plotting.make_all
  run_final.sh            Full final driver: generate → run → postprocess → plot
  collect_results.py      (helper) Spot-check raw CSV completeness

configs/
  smoke.yaml          2 runs — pipeline check
  pilot.yaml          54 runs — first real signal
  pilot_fixed2opt.yaml 54 runs — verify ECDF separation
  final.yaml          720 runs — complete experiment
```

---

## Future work / TODO

These items are out of scope for the preliminary result presentation but are candidates for the final written report.

### Strengthen statistical confidence (no new experiments needed)
- **More seeds:** The current 5 seeds / 30 runs per condition produces overlapping error bars and non-monotonic relative-overtuning vs. train-size curves. Increasing to ≥ 20 seeds (still on the same 64-vCPU VM in minutes) would stabilise these. No code changes required — just update `configs/final.yaml` (`seeds: [0,1,2,...,19]`) and re-run.
- **Significance testing:** Add a Wilcoxon signed-rank test or a small linear mixed-effects model (LMM mirroring the paper's Table 1) comparing resampling methods pairwise on final relative overtuning / generalisation gap. This would let the report say "holdout is significantly worse than cv5 (p < 0.05)" rather than "on average".

### Cover more levers from the paper (new experiments)
- **Reshuffling resampling splits** (Nagler et al., 2024 — cited by the paper): the most paper-aligned extension. At each BO trial, use a freshly drawn random split instead of the same fixed fold assignment. Expected to reduce overtuning further, especially for holdout. Requires a small change to `resampling.py` + a new key in `configs/final.yaml`.
- **Incumbent-selection mitigation:** instead of always picking the validation-optimal incumbent, select by posterior mean (BO) or apply an early-stopping rule (Makarova et al., 2022). Purely a post-hoc analysis on existing trajectory CSVs — no new SA evaluations.
- **Larger BO budget (50 → 200 iterations):** the paper finds an inverted-U budget effect (overtuning rises then plateaus). At budget = 50 we may still be on the rising limb. Extending would test whether the plateau appears.

### Figure improvements (cosmetic / report-quality)
- The relative-overtuning ECDF still has a noisy staircase in the zoomed panel (too few runs). More seeds (above) would smooth it significantly.
- Add confidence ribbons (bootstrap CI) to the `generalization_gap_by_train_size.png` pointplot.

---

## Reference

Lennart Schneider, Bernd Bischl, and Matthias Feurer. **Overtuning in Hyperparameter Optimization.** In *Proceedings of the 4th International Conference on Automated Machine Learning (AutoML 2025)*, PMLR Vol. 293, pp. 17/1–43.
https://proceedings.mlr.press/v293/schneider25a.html
