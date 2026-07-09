# Experiment v2 — Configurator comparison at scale: BO vs Random Search vs SMAC

Scaled-up follow-up to the v1 overtuning study. Where v1 varied the *resampling
estimator* under a fixed configurator, v2 fixes the estimator and varies the
**configurator**, on a much larger configuration space and budget, so that the
BO-vs-Random-Search separation (and the value of SMAC's native intensification)
becomes clearly measurable.

**Research questions**

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

## Design

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

### Estimator, instances, metric

- Canonical quality estimator `Ĉ(θ)`: mean normalized tour gap over the **full
  training subset**, with common random numbers (the solver seed for
  (instance, repeat) is a deterministic function of the run seed). This is the
  lowest-variance fixed estimator, isolating the configurator comparison from
  the resampling axis studied in v1.
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
cd /local/anmol/aac_tsp_overtuning          # or wherever the repo lives
git pull

# 0) one-time: environment (installs SMAC; needs swig)
sudo apt-get install -y swig build-essential   # if not present
bash slurm/setup_env.sh

# 1) smoke test all four arms first (8 tiny runs, minutes)
PROFILE=bovsrs_smoke bash slurm/submit_all.sh
#    ... wait, then check figures/v2 + slurm/logs before the real thing

# 2) full experiment (200 runs; dataset check -> array -> postprocess chain)
bash slurm/submit_all.sh
```

`submit_all.sh` expands `configs/bovsrs.yaml` into `slurm/experiments_v2.txt`,
submits `01_gen_instances.sbatch` (no-op if the v1 dataset exists),
`02_run_array.sbatch` as `--array=0-199%64` (single-threaded tasks; smac_aac runs
are scheduled first because they are the longest), and `03_postprocess.sbatch`
with an `afterany` dependency.

Useful properties:

- **Idempotent.** `main_v2.py` skips any run whose sidecar says
  `status: complete`. If tasks fail or time out, just run
  `bash slurm/submit_all.sh` again.
- **Partial results plot fine.** `03_postprocess.sbatch` (or the two CLI scripts)
  can be run at any time on whatever has finished.
- **No SLURM?** Use `slurm/run_local.sh` — identical behaviour (per-profile
  output roots, idempotent, postprocess at the end) via plain `xargs` fan-out:
  `PROFILE=bovsrs_smoke bash slurm/run_local.sh`, then
  `screen -dmS aacv2 bash slurm/run_local.sh` for the full experiment.
- **No sudo?** `setup_env.sh` installs swig from its prebuilt PyPI wheel into the
  venv automatically; only a C++ compiler (`g++`) must already be on the system.

Expected cost (64 vCPUs): fixed-budget runs ≈ 2–5 min each; `smac_aac` runs are
dominated by SMAC's surrogate/intensifier machinery over 5–10k tells, expect
~0.5–2 h each. Total wallclock ≈ 1.5–3 h. The 24 h array limit is generous
headroom, not a forecast.

### Outputs

```
results/v2/raw/            <run_id>_trials.csv, <run_id>_traj.csv, <run_id>.json per run
results/v2/processed/      trajectories_grid.csv (step functions on a shared log budget grid)
results/v2/summaries/      final_summary.csv, condition_summary.csv,
                           stats_tests.csv (Wilcoxon + Holm), summary_table.md
figures/v2/                14 comparison figures (below)
results/v2/V2_COMPLETE     sentinel written by postprocess
```

---

## Figure guide (figures/v2/)

| Figure | What it shows / what to look for |
|---|---|
| `convergence_validation.png` | Incumbent validation cost vs budget (median + IQR over 25 seeds, log x). The headline anytime plot: BO curves should drop below RS and stay there. |
| `convergence_test.png` | Same on *test* cost — the honest generalization version. |
| `final_cost_boxplots.png` | Final validation & test cost per arm (each dot = one seed). |
| `ecdf_final_test.png` | ECDF of final test cost; stochastic dominance is visible as a curve entirely left of another. |
| `rank_over_budget.png` | Mean rank (1–4) across seeds over the budget; shows *when* model-based search overtakes RS. |
| `pairwise_winrate.png` | 4×4 win-rate matrix on final test cost, paired by seed; Holm-corrected Wilcoxon p-values annotated. This is the "is it significant" figure. |
| `budget_to_target.png` | SA calls each arm needs to reach Random Search's median final validation cost, with median speedup factors ("TPE reaches RS quality with X× less budget"). |
| `generalization_gap_over_budget.png` | Median (test − validation) gap vs budget per arm: do aggressive optimizers overfit the training instances more? |
| `overtuning_ecdf.png` | Schneider-et-al relative overtuning ECDF, by configurator instead of by resampling. |
| `proposal_quality.png` | Cost of *proposed* (not incumbent) configs over time: RS stays flat by construction; TPE/SMAC proposals improve — the mechanism behind the headline result. |
| `param_concentration.png` | Early-vs-late sampled distributions of key parameters; model-based arms visibly concentrate on the good regime. |
| `configurator_overhead.png` | Solver time vs configurator machinery time per run — the practical price of SMAC's model and intensifier. |
| `smac_intensification.png` | ECDF of SA calls spent per configuration for `smac_aac` (most configs killed after a few instances, incumbents intensified) + distinct-configs-per-budget bar chart. Same budget, ~10× more configurations explored. |
| `test_family_heatmap.png` | Mean final test cost per test family — does any configurator find configs that transfer worse out-of-distribution? |

`summaries/summary_table.md` has the per-condition table (means, medians, AUC,
speedups, overheads) and all pairwise tests in copy-pasteable Markdown.

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
  (it now installs swig from the PyPI wheel, no sudo needed). With admin rights,
  `sudo apt-get install -y swig build-essential` also works.
- `pyrfr` build error about a missing compiler → the VM has no `g++`; ask the
  admin for `build-essential` (or use conda: `conda install gxx swig`).
- Array tasks failing instantly → check `slurm/logs/aac_v2_*_<idx>.out`; the
  manifest line index is the array task id.
- Timed-out `smac_aac` runs → resubmit (`bash slurm/submit_all.sh`); finished runs
  are skipped, unfinished ones restart cleanly (`overwrite=True`).
- Want intermediate figures while the array runs → `sbatch slurm/03_postprocess.sbatch`
  at any time.
