# Experiment v2 — Configurator × instance-resampling comparison

Experiment v2 now answers the joint question: **how do Random Search, Optuna
TPE, SMAC-BO, and SMAC-AAC behave under holdout, 5-fold instance resampling,
repeated 5-fold instance resampling, and bootstrap OOB instance resampling?**
This is a full configurator × resampling factorial comparison, not a
configurator-only analysis and not a reuse of v1 results.

## Recommended factorial benchmark

The primary profile is `configs/bovsrs_resampling_signal.yaml`: 192 new runs
(4 configurators × 4 resampling methods × 12 paired seeds) in the central
mixed-instance, n=25, fixed-2-opt condition. Every cell receives the same 5,000
SA-call cap and the same unseen test pool. Because the estimators cost
different numbers of calls per proposed configuration, the number of proposals
is intentionally different; comparisons are always made on `cum_sa_calls`.

| Resampling method | Calls per proposal at n=25 | Approx. proposals in 5,000 calls |
|---|---:|---:|
| Holdout instance sampling | 5 | 1,000 |
| 5-fold instance resampling | 25 | 200 |
| Repeated 5-fold instance resampling | 75 | 66 |
| Bootstrap OOB instance resampling | seed-dependent, about 90 | about 55 |

For SMAC-AAC, the resampling plan defines weighted scenario-evaluation units;
the native intensifier still chooses which configuration/unit pair to evaluate.
SMAC-AAC spends the 5,000-call cap exactly. Fixed-estimator arms spend the
largest whole number of estimator evaluations that fits; the recorded unused
remainder is always smaller than one evaluation. Canonical full-training
validation and unseen-test evaluation are analysis-only and never reach a
configurator.

Run the 16-cell smoke test first, then the primary factorial experiment:

```bash
cd /local/rohit/projects/aacnewtest
source .venv/bin/activate
python scripts/preflight_v2.py
PROFILE=bovsrs_resampling_smoke JOBS=2 bash slurm/run_local.sh   # 16 runs
PROFILE=bovsrs_resampling_signal JOBS=2 bash slurm/run_local.sh  # 192 runs
```

## Complete training-size factorial

Use `configs/bovsrs_resampling_train_sizes.yaml` for the requested comparison
over n=10, 25, and 50. It crosses all four configurators with all four
instance-resampling methods and 12 matched seeds at every size: 576 runs total.
The proposal reference count is deliberately size-specific so the optimization
budget remains exactly 5,000 SA solver calls in every cell.

| Training size | Reference proposals | Optimization cap | Runs |
|---:|---:|---:|---:|
| 10 | 500 | 5,000 SA calls | 192 |
| 25 | 200 | 5,000 SA calls | 192 |
| 50 | 100 | 5,000 SA calls | 192 |

This is not a different dataset. It loads the same synthetic 50-city pools and
the same `best_known.csv` used by v1 from `/local/rohit/datasets/tsp`: 150 train
and 100 disjoint test instances per family, generated with fixed seeds. The new
profile focuses on the mixed training pool and the 11-parameter fixed-2-opt
space because that removes the move-type performance cliff and makes relative
overtuning interpretable. V1 additionally crossed three training families and
the full move-type space, so the designs are related but not identical.

Run the full-layout smoke first, then the 576-run experiment:

```bash
PROFILE=bovsrs_resampling_train_sizes_smoke JOBS=2 bash slurm/run_local.sh  # 48 runs
PROFILE=bovsrs_resampling_train_sizes JOBS=2 bash slurm/run_local.sh        # 576 runs
```

In addition to every existing v1-style and factorial comparison, this profile
writes three 3-by-4 trajectory matrices. Rows are n=10/25/50, columns are the
four instance-resampling methods, and each panel compares all four
configurators on cumulative SA calls:

```text
figures/v2_bovsrs_resampling_train_sizes/overtuning_frequency_over_budget_by_train_size.png
figures/v2_bovsrs_resampling_train_sizes/relative_overtuning_over_budget_by_train_size.png
figures/v2_bovsrs_resampling_train_sizes/relative_overtuning_eligibility_over_budget_by_train_size.png
```

The eligibility matrix is essential: relative overtuning is undefined until a
run improves its initial unseen-test incumbent by at least 0.001. The relative
overtuning matrix therefore reports the median only among eligible runs; it
never silently turns an undefined value into zero.

The optional `configs/bovsrs_resampling_conditions.yaml` profile adds four
targeted conditions (training size, family, and 11-vs-15-parameter space):
4 configurators × 4 resampling methods × 4 conditions × 8 seeds = 512 runs.
Run it only after the primary profile completes:

```bash
PROFILE=bovsrs_resampling_conditions JOBS=2 bash slurm/run_local.sh
```

Primary outputs include:

```text
results/v2_bovsrs_resampling_signal/summaries/final_summary.csv
results/v2_bovsrs_resampling_signal/summaries/paired_vs_random.csv
results/v2_bovsrs_resampling_signal/summaries/paired_resampling_tests.csv
results/v2_bovsrs_resampling_signal/summaries/configurator_resampling_interactions.csv
results/v2_bovsrs_resampling_signal/summaries/parameter_summary.csv
results/v2_bovsrs_resampling_signal/summaries/summary_table.md
figures/v2_bovsrs_resampling_signal/configurator_by_resampling_final.png
figures/v2_bovsrs_resampling_signal/configurator_by_resampling_trajectories.png
figures/v2_bovsrs_resampling_signal/resampling_ecdf_by_configurator.png
figures/v2_bovsrs_resampling_signal/resampling_ecdf_by_configurator_zoomed.png
figures/v2_bovsrs_resampling_signal/configurator_resampling_heatmaps.png
figures/v2_bovsrs_resampling_signal/configurator_resampling_efficiency.png
figures/v2_bovsrs_resampling_signal/selected_parameters_by_factorial_cell.png
figures/v2_bovsrs_resampling_signal/optimizer_comparison.png
figures/v2_bovsrs_resampling_signal/optimizer_comparison_by_resampling/<method>.png
figures/v2_bovsrs_resampling_signal/configurator_comparison.png
figures/v2_bovsrs_resampling_signal/configurator_comparison_by_resampling/<method>.png
figures/v2_bovsrs_resampling_signal/all_configurators_all_resamplings.png
figures/v2_bovsrs_resampling_signal/all_configurators_all_resamplings_ecdf.png
figures/v2_bovsrs_resampling_signal/all_configurators_all_resamplings_ecdf_zoomed.png
figures/v2_bovsrs_resampling_signal/all_configurators_all_resamplings_ecdf_matrix.png
```

The plotting step also recreates the complete v1 figure vocabulary directly
from the new v2 tables: `trajectory_validation_vs_test.png`,
`trajectories_by_resampling.png`, `ecdf_relative_overtuning.png`,
`relative_overtuning_by_train_size.png`,
`overtuning_frequency_by_resampling.png`, `final_test_by_resampling.png`,
`generalization_gap_by_train_size.png`, `runtime_by_resampling.png`,
`resampling_quality_runtime_tradeoff.png`, `selected_params_stability.png`,
`composition_heatmap.png`, and `composition_by_resampling.png`. No v1 result
CSV is read while producing these figures.

### Recommended result story

The paper and lecture imply a specific reporting order. Relative overtuning is
a within-trajectory lost-progress measure, so it must not be used alone to rank
configurators by generalization. Present the new v2 evidence in this order:

1. **Design and fairness:** state the 4 configurators x 4 instance-resampling
   methods x 12 matched seeds design, the 5,000-SA-call cap, and test isolation.
2. **Anytime and final performance:** use
   `all_configurators_all_resamplings.png`. It mirrors the original
   full/zoomed convergence comparison but gives each resampling estimator its
   own aligned row, avoiding an average over unlike estimators.
3. **Overtuning:** use
   `all_configurators_all_resamplings_ecdf_matrix.png`. It extends the paper's
   and v1's full/zoomed ECDF format to every configurator-resampling cell.
4. **Absolute generalization and robustness:** use
   `configurator_by_resampling_final.png` and
   `bo_vs_random_by_condition.png` for final test cost, anytime AUC,
   generalization gap, paired effects, and confidence intervals.
5. **Cost of robustness:** use `configurator_resampling_efficiency.png` so a
   lower overtuning result is interpreted together with runtime, SA calls, and
   configurator overhead.
6. **Mechanism and stability:** use
   `selected_parameters_by_factorial_cell.png` only after the outcome plots to
   discuss whether methods repeatedly select different SA regions.

The primary 192-run profile fixes training size at 25, training family at
`mixed`, and the 11-parameter fixed-2-opt space. Therefore its one-point
training-size and one-row training-family figures are not evidence about size
or composition effects. Those claims belong to the optional conditions profile
or to the completed v1 experiment, and must be clearly labelled as such.

Completion checks:

```bash
find results/v2_bovsrs_resampling_signal/raw -maxdepth 1 -name "*.json" | wc -l  # 192
find results/v2_bovsrs_resampling_signal -name FAILED.json -print -exec cat {} \;
ls results/v2_bovsrs_resampling_signal/V2_COMPLETE
```

The old `bovsrs_signal` and `bovsrs_conditions` outputs remain valid only for
the legacy full-training configurator comparison. They do not answer the
factorial question and are never mixed into the new summaries.

Relative overtuning follows Schneider et al.'s denominator-stability rule: it
is reported only when the best test improvement over the initial incumbent is
at least 0.001. The raw ratio, progress value, and eligibility flag remain in
`final_summary.csv`, so excluded runs are auditable.

## Completed single-condition signal baseline

The completed baseline run is `configs/bovsrs_signal.yaml`. It contains 48
matched runs for one central condition:

| Axis | Values |
|---|---|
| configurator | Random Search, Optuna TPE, SMAC-BO, SMAC-AAC |
| train family / size | mixed / 25 |
| seeds | 0 … 11 (paired across all four arms) |
| search space | `large_fixed_2opt` (11 tunable parameters; pure 2-opt moves) |
| optimizer budget | 120 proposals × 25 instances = 3,000 SA calls per run |
| SMAC-AAC intensification cap | 8 calls per configuration |

Fixing the neighbourhood to 2-opt removes the large discrete "find 2-opt" cliff
that can make all optimizers look similar once Random Search reaches it. The
remaining temperature, schedule, construction, restart, and reheating space is
continuous/integer-heavy, so model-based search has useful structure to learn.
Twelve matched seeds are enough for trajectory and paired-difference evidence
without paying for the full 200-run grid. This is a design intended to reveal a
signal; it is not a claim that any method wins. Results must come from the run.

Run it locally on the VM:

```bash
cd /local/rohit/projects/aacnewtest
bash slurm/setup_env.sh        # first clone only; creates .venv and installs PyYAML/SMAC
source .venv/bin/activate
python scripts/preflight_v2.py
PROFILE=bovsrs_signal JOBS=2 bash slurm/run_local.sh
```

Both `preflight_v2.py` and `run_local.sh` invoke `setup_env.sh` automatically
when `.venv/bin/python` or required packages are absent; the launcher then runs
preflight itself. The explicit setup line above makes the one-time dependency
installation visible and ensures the following `source` command can succeed.

The optional train-size check is also 48 runs, but only compares the two fast
arms over sizes 8, 25, and 50 with seeds 0 … 7:

```bash
PROFILE=bovsrs_generalization_check JOBS=2 bash slurm/run_local.sh
```

Outputs and checks:

```bash
# Main profile: exactly 48 successful run sidecars
find results/v2_bovsrs_signal/raw -name "*.json" | wc -l
find results/v2_bovsrs_signal -name FAILED.json -print -exec cat {} \;
ls results/v2_bovsrs_signal/V2_COMPLETE
cat results/v2_bovsrs_signal/summaries/summary_table.md

# Optional profile: also exactly 48
find results/v2_bovsrs_generalization_check/raw -name "*.json" | wc -l
```

Each run mirrors stdout/stderr to `results/v2_<profile>/logs/<run_id>.log` and
prints identity, start/end timestamps, progress, and its JSON output path. A
failed run writes `results/v2_<profile>/failures/<run_id>/FAILED.json`.
`V2_COMPLETE` is created only after every manifest run has a valid complete
sidecar. Re-running the command skips complete cells and retries missing ones.

The analysis keeps the validation-selected incumbent trajectory and adds the
requested post-hoc best-so-far test envelope. The latter and its normalized
log-budget AUC are explicitly labelled analysis-only: test values never reach a
configurator. `paired_vs_random.csv` and `summary_table.md` report, for final
test cost and AUC, method-minus-Random-Search mean, median, standard deviation,
and the number of matched seeds won.

**Research questions for the legacy full grid**

1. Does model-based configuration (TPE, SMAC) beat Random Search on a large,
   realistic algorithm-configuration space — and by how much (final quality,
   anytime performance, budget-to-target speedup)?
2. Given the *same total target-algorithm budget*, does SMAC's native
   intensification (racing configurations across instances and seeds) beat the
   standard fixed-budget protocol (every configuration evaluated on the full
   training set)?
3. Do aggressive configurators overtune more at a 250-proposal budget
   (connecting back to Schneider et al., AutoML 2025)?

---

## Legacy 200-run design (`configs/bovsrs.yaml`)

### Arms (4 configurators)

| Arm | Configurator | Evaluation protocol |
|---|---|---|
| `random` | Optuna `RandomSampler` | fixed budget: every proposal evaluated on the full training set |
| `optuna_tpe` | Optuna `TPESampler` (multivariate, grouped, 25 startup trials) | same fixed-budget protocol |
| `smac_bo` | SMAC3 `HyperparameterOptimizationFacade` (random-forest BO, 25-config Sobol initial design) | same fixed-budget protocol |
| `smac_aac` | SMAC3 `AlgorithmConfigurationFacade` + native `Intensifier` | one target call = one SA run on **one instance with one SMAC-chosen seed**; the intensifier races configs and decides where budget goes |

### Budget fairness — the core design decision

The budget unit is **one SA solver call** (one configuration × one instance × one
seed). Per run:

```
fixed-budget arms:  n_trials × train_size × solver_repeats   calls  (250 × 40 × 1 = 10,000 at n=40)
smac_aac:           exactly the same number of calls, allocated by the intensifier
```

Test-set evaluations (and, for `smac_aac`, the canonical re-evaluation of each new
incumbent on the full training set) are **analysis-only**: they are never returned
to any configurator and do not count toward the budget — identically across arms.

All trajectories are therefore step functions over `cum_sa_calls`, the honest
common axis; nothing is compared "per iteration".

### Grid (200 runs)

| Axis | Values |
|---|---|
| configurator | random, optuna_tpe, smac_bo, smac_aac |
| train family | mixed (test pools: uniform, clustered, mixed — logged per family) |
| train size | 20, 40 |
| seeds | 0 … 24 (25 seeds; paired across arms) |
| budget | 250 proposals ⇒ 5,000 / 10,000 SA calls per run |

The seed fixes: the training-instance subsample, the sampler seed, the CRN solver
seed stream, and the SMAC scenario seed — so arm comparisons are **paired by seed**
(Wilcoxon signed-rank in the analysis).

### The enlarged configuration space (15 parameters)

The v1 space had 5 parameters. The v2 solver (`solver_sa_ext.py`) generalizes it
to 15 (12 always active + 3 conditional) — mixed types, log scales, a probability
simplex, and hierarchical structure. See `space_v2.py` for the exact table.
Highlights:

- the single `move_type` categorical becomes a tunable **move mixture**
  (`p_swap, p_insert, p_2opt, p_oropt`, normalized in the solver; a new or-opt
  segment-relocation move is added);
- construction heuristic (`init_method`: random vs nearest-neighbour);
- temperature floor + optional **periodic reheating** (2 conditional params);
- restart strategy: fresh restarts vs **ILS-style double-bridge perturbation** of
  the best tour (`perturbation_kicks`, conditional).

The space deliberately contains large bad regions (quenching cooling rates,
random-walk temperature regimes, swap-only neighbourhoods: gaps 0.25–0.8) next to
a narrow high-quality regime (gap ≈ 0.00–0.02). Verified spread on 5 instances:
tuned configs ≈ 0.00, pure-swap ≈ 0.14, hot-and-slow-cooling ≈ 0.26, partially
improved random tours ≈ 0.78. Random Search keeps paying for the bad regions for
all 250 proposals; a model-based configurator can learn to avoid them — that is
precisely the hypothesis under test.

### Legacy estimator, instances, metric

- The legacy profiles use the mean normalized tour gap over the **full training
  subset**. The new factorial profiles replace that single estimator with the
  four explicit instance-resampling methods documented above.
- Same instance pools, best-known references, and metric
  `m(θ, π) = (tour(θ, π) − best_known(π)) / best_known(π)` as v1
  (50 cities; 150 train + 100 test per family). **No dataset regeneration needed
  on the VM.** Note the extended solver can slightly beat the NN+2-opt reference,
  so small negative gaps are legitimate (normalization only requires consistency).
- `smac_aac` extras: per-instance features (distance moments, NN-distance stats)
  are passed to SMAC's surrogate; `deterministic=False` so SMAC races seeds;
  intensification is capped at `3 × train_size` calls per configuration.

---

## How to run (on the VM, with SLURM)

```bash
cd /local/rohit/projects/aacnewtest
git pull

# 0) one-time: rootless environment setup (no sudo required for Python/SWIG)
bash slurm/setup_env.sh

# 1) all four configurators × all four resampling methods (16 tiny runs)
PROFILE=bovsrs_resampling_smoke bash slurm/submit_all.sh

# 2) primary factorial experiment (192 runs)
PROFILE=bovsrs_resampling_signal bash slurm/submit_all.sh

# 3) optional factorial condition extension (512 runs)
PROFILE=bovsrs_resampling_conditions bash slurm/submit_all.sh

# 4) complete n=10/25/50 factorial (576 runs; smoke profile available)
PROFILE=bovsrs_resampling_train_sizes bash slurm/submit_all.sh

# Legacy full experiment only when explicitly wanted
PROFILE=bovsrs bash slurm/submit_all.sh
```

`submit_all.sh` expands the selected profile into `slurm/experiments_v2.txt`,
submits `01_gen_instances.sbatch` (no-op if the v1 dataset exists),
`02_run_array.sbatch` as a single-threaded array (SMAC-AAC runs
are scheduled first because they are the longest), and `03_postprocess.sbatch`
with an `afterany` dependency.

Useful properties:

- **Idempotent.** `main_v2.py` skips any run whose sidecar says
  `status: complete`. If tasks fail or time out, repeat the same
  `PROFILE=... bash slurm/submit_all.sh` command.
- **Partial results can be inspected manually.** Run the two postprocess/plot CLI
  scripts directly. The automated completion job intentionally refuses to write
  `V2_COMPLETE` until the manifest is complete.
- **No SLURM?** Use `slurm/run_local.sh` — identical behaviour (per-profile
  output roots, idempotent, postprocess at the end) via plain `xargs` fan-out:
  `PROFILE=bovsrs_resampling_smoke bash slurm/run_local.sh`, then
  `PROFILE=bovsrs_resampling_train_sizes_smoke bash slurm/run_local.sh`, then
  `PROFILE=bovsrs_resampling_train_sizes screen -dmS aacv2-sizes bash slurm/run_local.sh`.
- **No sudo?** `setup_env.sh` installs SWIG inside `.venv` and bypasses its Python
  launcher if needed. A system `g++`/`c++` is still required; if it is missing,
  use `conda install -c conda-forge gxx_linux-64` or ask the VM administrator.

Expected cost (64 vCPUs): fixed-budget runs ≈ 2–5 min each; `smac_aac` runs are
dominated by SMAC's surrogate/intensifier machinery over 5–10k tells, expect
~0.5–2 h each. Total wallclock ≈ 1.5–3 h. The 24 h array limit is generous
headroom, not a forecast.

### Outputs

```
results/v2_<profile>/raw/       <run_id>_trials.csv, <run_id>_traj.csv, <run_id>.json
results/v2_<profile>/logs/      one stdout/stderr log per run
results/v2_<profile>/failures/  <run_id>/FAILED.json for failed cells
results/v2_<profile>/processed/ trajectories_grid.csv, budget_checkpoints.csv
results/v2_<profile>/summaries/ final_summary.csv, condition_summary.csv,
                                 stats_tests.csv, paired_vs_random.csv,
                                 paired_resampling_tests.csv,
                                 configurator_resampling_interactions.csv,
                                 budget_checkpoint_summary.csv,
                                 condition_contrasts.csv, parameter_summary.csv,
                                 summary_table.md
figures/v2_<profile>/            comparison figures
results/v2_<profile>/V2_COMPLETE exact-completion sentinel
```

---

## Figure guide (`figures/v2_<profile>/`)

| Figure | What it shows / what to look for |
|---|---|
| `configurator_by_resampling_final.png` | Final test cost, anytime AUC, generalization gap, and relative overtuning for every configurator × resampling cell. |
| `configurator_by_resampling_trajectories.png` | Equal-SA-call test trajectories, faceted by instance-resampling method. |
| `resampling_ecdf_by_configurator.png` | Relative-overtuning ECDFs for all four resampling methods, faceted by configurator. |
| `resampling_ecdf_by_configurator_zoomed.png` | Zoomed ECDF view for separation near the top of the distribution. |
| `configurator_resampling_heatmaps.png` | Matrix comparison of final test cost, AUC, and absolute generalization gap. |
| `configurator_resampling_efficiency.png` | Runtime, solver/analysis cost, budget use, overhead, and number of configurations explored for every factorial cell. |
| `selected_parameters_by_factorial_cell.png` | Selected SA parameter distributions for every factorial cell. |
| `optimizer_comparison.png` | Image-style TPE vs Random Search comparison: full convergence, post-initial-budget zoom, and final test cost across all four resampling methods. |
| `optimizer_comparison_by_resampling/<method>.png` | The same TPE-vs-Random three-panel comparison separately for every instance-resampling method. |
| `configurator_comparison.png` | The same three-panel view with Random Search, TPE, SMAC fixed-budget BO, and SMAC native intensification together. |
| `configurator_comparison_by_resampling/<method>.png` | Four-configurator comparison separately for each of the four instance-resampling methods. |
| `all_configurators_all_resamplings.png` | One 4-row comparison matrix: each row is an instance-resampling method and simultaneously shows full/zoomed equal-SA-call trajectories plus final test distributions for all four configurators. |
| `all_configurators_all_resamplings_ecdf.png` | One 2x2 ECDF figure: each panel is an instance-resampling method and compares relative overtuning for all four configurators. |
| `all_configurators_all_resamplings_ecdf_zoomed.png` | Zoomed ECDF companion for separation near the top of the distribution. |
| `all_configurators_all_resamplings_ecdf_matrix.png` | Paper-style 4x2 ECDF matrix: one row per instance-resampling method and paired full/zoomed columns, with all four configurators in every panel. |
| `convergence_validation.png` | Incumbent validation cost vs budget (median + IQR over seeds, log x). |
| `best_so_far_test_trajectory.png` | Post-hoc lower envelope of test cost among validation incumbents; analysis-only and never fed back. |
| `convergence_test.png` | Same on *test* cost — the honest generalization version. |
| `final_cost_boxplots.png` | Final validation & test cost per arm (each dot = one seed). |
| `anytime_auc.png` | Normalized log-budget AUC of the best-so-far test envelope; lower is better. |
| `ecdf_final_test.png` | ECDF of final test cost; stochastic dominance is visible as a curve entirely left of another. |
| `rank_over_budget.png` | Mean rank (1–4) across seeds over the budget; shows *when* model-based search overtakes RS. |
| `pairwise_winrate.png` | 4×4 win-rate matrix on final test cost, paired by seed; Holm-corrected Wilcoxon p-values annotated. This is the "is it significant" figure. |
| `budget_to_target.png` | SA calls each arm needs to reach Random Search's median final validation cost, with median speedup factors ("TPE reaches RS quality with X× less budget"). |
| `generalization_gap_over_budget.png` | Median (test − validation) gap vs budget per arm: do aggressive optimizers overfit the training instances more? |
| `overtuning_over_budget.png` | Probability of nonzero overtuning and median eligible relative overtuning over the common SA-call budget. |
| `overtuning_ecdf.png` | Schneider-et-al relative overtuning ECDF by configurator; curves report eligible sample counts after the 0.001 filter. |
| `proposal_quality.png` | Cost of *proposed* (not incumbent) configs over time: RS stays flat by construction; TPE/SMAC proposals improve — the mechanism behind the headline result. |
| `param_concentration.png` | Early-vs-late sampled distributions of key parameters; model-based arms visibly concentrate on the good regime. |
| `configurator_overhead.png` | Solver time vs configurator machinery time per run — the practical price of SMAC's model and intensifier. |
| `smac_intensification.png` | ECDF of SA calls spent per configuration for `smac_aac` (most configs killed after a few instances, incumbents intensified) + distinct-configs-per-budget bar chart. Same budget, ~10× more configurations explored. |
| `test_family_heatmap.png` | Mean final test cost per test family — does any configurator find configs that transfer worse out-of-distribution? |

`summaries/summary_table.md` has the per-condition table (means, standard errors,
medians, AUC, gaps, speedups, overheads), matched-seed Random Search differences,
targeted cross-condition contrasts, and pairwise tests in copy-pasteable Markdown.
See [EXPERIMENT_COVERAGE.md](EXPERIMENT_COVERAGE.md) for the mapping from the
paper/lecture/presentation requirements to v1 and v2, plus the explicit limits.

---

## Design decisions worth defending in the report

1. **Budget in SA calls, not iterations.** Comparing "250 iterations" of a
   fixed-budget arm against SMAC's intensifier would be meaningless — the
   intensifier's whole point is spending *less* than a full training-set pass on
   bad configs. Equal total target-algorithm calls is the standard fair unit in
   the AAC literature.
2. **`solver_repeats = 1` for validation.** Averaging over 20–40 instances already
   stabilizes the estimator; CRN seeds make it a paired comparison across
   configurations. This triples throughput vs v1's 3 repeats with no change in
   ranking behaviour.
3. **`smac_aac` keeps its native defaults** (default-config initial design,
   RF surrogate with instance features, seed racing) because "SMAC as its authors
   intend it" is the object of study; `smac_bo` gets the same 25-config initial
   design as TPE's 25 startup trials so the fixed-budget arms differ only in the
   model.
4. **Incumbent trajectories are re-evaluated canonically** (same CRN estimator)
   for `smac_aac`, analysis-only. Otherwise its validation trajectory would be an
   average over *whichever* instance subset SMAC happened to race, which is not
   comparable across arms.
5. **The test set never touches any configurator** — unchanged invariant from v1.

## Troubleshooting

- `pyrfr` build error `command 'swig' failed` → re-run `bash slurm/setup_env.sh`
  (it installs SWIG from the PyPI wheel without sudo).
- `pyrfr` build error about a missing compiler → the VM has no `g++`; ask the
  admin for `build-essential` (or use conda: `conda install gxx swig`).
- Array tasks failing instantly → check `slurm/logs/aac_v2_*_<idx>.out`; the
  manifest line index is the array task id.
- Timed-out `smac_aac` runs → resubmit (`bash slurm/submit_all.sh`); finished runs
  are skipped, unfinished ones restart cleanly (`overwrite=True`).
- Want intermediate figures while the array runs → `sbatch slurm/03_postprocess.sbatch`
  at any time.
