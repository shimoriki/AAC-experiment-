# Experimental coverage of the reference material

This repository uses the Schneider, Bischl, and Feurer (2025) paper, the AAC
lecture, and the preliminary-results presentation as methodological references.
It does not try to reproduce every tabular-HPO experiment from the paper. The
repository contains the completed historical v1 study and the current v2
factorial study:

- **v1 (complete, 720 runs):** asks how holdout, 5-fold instance resampling,
  repeated 5-fold instance resampling, and bootstrap OOB change configuration
  generalization for BO and Random Search.
- **v2 primary factorial benchmark (192 new runs):** jointly compares Random
  Search, Optuna TPE, SMAC-BO, and SMAC-AAC under all four instance-resampling
  methods at equal SA-call budget. An optional 512-run profile adds targeted
  training-size, family, and configuration-space conditions.
- **v2 complete training-size factorial (576 new runs):** crosses all four
  configurators and all four instance-resampling methods at n=10, 25, and 50
  with 12 matched seeds and the same 5,000-SA-call cap at every size. It adds
  overtime overtuning-frequency, eligible relative-overtuning, and eligibility
  matrices over the full 4 x 4 x 3 design.

## Requirement mapping

| Reference requirement | Where it is covered |
|---|---|
| Finite training instances and unseen test instances | v1 and v2 use disjoint deterministic instance pools and solver-seed streams. |
| Test performance never reaches the configurator | Enforced in both runners; v2 canonical test evaluation is analysis-only and counted separately. |
| Compare resampling strength and runtime | v2 crosses holdout, 5-fold, repeated 5-fold, and bootstrap OOB with every configurator and reports quality/runtime at equal SA-call budgets. |
| Compare BO with Random Search | v2 primary uses 12 matched seeds in every configurator × resampling cell; the optional condition extension uses 8. |
| Compare additional AAC configurators | v2 includes fixed-budget SMAC-BO and native SMAC-AAC intensification in the same resampling factorial. |
| Fair budget across racing and fixed evaluation | v2 uses cumulative SA solver calls, never configurator iterations, as the common axis. |
| Final test performance, variance, and runtime | v2 reports seed distributions, standard errors, wall-clock time, solver time, and configurator overhead. |
| Anytime/trajectory behavior | v2 reports validation-selected incumbent trajectories, an explicitly post-hoc test envelope, AUC, rank, and budget-to-target. |
| Overtuning probability and magnitude | v2 reports absolute overtuning frequency and relative-overtuning magnitude over budget and at the final budget. |
| Stable relative-overtuning denominator | Runs with less than 0.001 test improvement from the initial incumbent are excluded from the reported relative ratio, following Schneider et al. Section 5. Raw ratios and eligibility remain in the CSV. |
| Dataset/training-set size | The v2 primary factorial fixes n=25; the optional v2 condition profile includes n=8 and n=25. |
| Evaluation/data distribution | v2 uses disjoint deterministic uniform, clustered, and mixed TSP pools. The primary score uses the same mixed unseen-test pool; family-specific scores are retained for transfer analysis. |
| Optimizer and parameter-space effects | v2 compares four configurators in fixed-2-opt (11 parameters) and full (15 parameters) spaces and summarizes final parameter shifts. |
| Tuning-budget effect | v2 analyzes 25/50/75/100% checkpoints and continuous trajectories on the common SA-call axis. These are repeated measurements from a run, not four independent budget experiments. |
| Paired statistical evidence | v2 uses matched-seed Wilcoxon tests with Holm correction, paired-bootstrap confidence intervals, within-configurator resampling tests, and configurator × resampling difference-in-differences. |

## Important interpretation boundaries

The best-so-far test curve is a **post-hoc oracle envelope among validation
incumbents**, because test performance is intentionally not computed for every
rejected configuration. It is useful for anytime diagnosis but never selects a
configuration and must not be described as configurator input.

The following paper-inspired extensions remain outside the main run:

- reshuffling an instance split at every optimizer step;
- posterior-mean/conservative incumbent selection;
- early stopping of BO;
- a formal mixed-effects model over a full factorial design;
- other target algorithms, real TSPLIB data, or non-TSP benchmarks.

The four fixed resampling estimators are deliberately included in the v2
factorial. Reshuffling at every optimizer step and alternative incumbent rules
would answer different questions and should remain separate follow-ups.

## Operational requirements retained from the VM bring-up

- Repository: `/local/rohit/projects/aacnewtest`.
- Dataset: `/local/rohit/datasets/tsp`; launchers generate all deterministic
  train/test pools and best-known references when they are missing.
- Environment setup is rootless for Python, virtualenv, and SWIG. A C++ compiler
  must already exist or be supplied through a user-owned conda environment.
- `.venv`, datasets, results, caches, and logs are excluded from source transfer.
- `bovsrs_resampling_smoke` is 16 runs, the primary
  `bovsrs_resampling_signal` profile is 192 runs, and the optional
  `bovsrs_resampling_conditions` profile is 512 runs.
- `bovsrs_resampling_train_sizes_smoke` is 48 runs and the complete
  `bovsrs_resampling_train_sizes` profile is 576 runs. Both reuse the same
  synthetic TSP pools and best-known references as v1.
- The old configurator-only profiles remain available but are not mixed into
  the factorial results.
- Every run writes an individual log, start/end identity, periodic progress, a
  complete sidecar, or a structured `FAILED.json`.
- Re-running a launcher skips complete cells. `V2_COMPLETE` is written only
  after the manifest checker confirms that every expected run completed.
