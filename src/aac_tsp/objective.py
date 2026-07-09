"""Objective evaluation: maps a configuration to a (normalized) cost.

The validation cost is what the configurator optimizes; the test cost is logged
for *analysis only* and must never be returned to the optimizer.

Solver seeds are derived deterministically from (subset_seed, instance, repeat) so
that runs are reproducible. Crucially, validation and test evaluations use disjoint
solver-seed streams (test uses a large offset), so a configuration that got lucky on
the validation seeds does not automatically look good on test -- which is what makes
overtuning observable.
"""

from __future__ import annotations

import time

import numpy as np

from .solver_sa import SAConfig, run_simulated_annealing

TEST_SEED_OFFSET = 5_000_000


def _derive_seed(subset_seed: int, instance_index: int, repeat: int) -> int:
    return (int(subset_seed) * 100_003 + int(instance_index) * 7919 + int(repeat) * 31 + 1) & 0x7FFFFFFF


def _instance_gap(config, dist, best_known, base_seed, instance_index, solver_repeats, max_steps):
    gaps = 0.0
    for r in range(solver_repeats):
        sa_seed = _derive_seed(base_seed, instance_index, r)
        res = run_simulated_annealing(dist, config, sa_seed, max_steps=max_steps)
        gaps += (res["tour_length"] - best_known) / best_known
    return gaps / solver_repeats


def evaluate_config_on_resampling(config: SAConfig, plan, dist_by_id, best_known_by_id,
                                  id_index, solver_repeats: int, max_steps: int) -> dict:
    """Validation cost = mean over subsets of the mean instance gap within each subset."""
    t0 = time.perf_counter()
    subset_means = []
    n_eval = 0
    for subset_ids, subset_seed in plan:
        gaps = []
        for iid in subset_ids:
            gap = _instance_gap(config, dist_by_id[iid], best_known_by_id[iid],
                                subset_seed, id_index[iid], solver_repeats, max_steps)
            gaps.append(gap)
            n_eval += 1
        subset_means.append(float(np.mean(gaps)))
    return {
        "validation_cost": float(np.mean(subset_means)),
        "validation_runtime_sec": time.perf_counter() - t0,
        "n_eval_instances": n_eval,
        "subsets_evaluated": len(plan),
    }


def evaluate_config_on_test(config: SAConfig, test_ids, dist_by_id, best_known_by_id,
                            id_index, test_seed: int, solver_repeats: int, max_steps: int) -> dict:
    """Mean test gap over an independent test pool with an independent solver-seed stream."""
    t0 = time.perf_counter()
    gaps = []
    for iid in test_ids:
        gap = _instance_gap(config, dist_by_id[iid], best_known_by_id[iid],
                            TEST_SEED_OFFSET + test_seed, id_index[iid], solver_repeats, max_steps)
        gaps.append(gap)
    return {"test_cost": float(np.mean(gaps)), "test_runtime_sec": time.perf_counter() - t0}
