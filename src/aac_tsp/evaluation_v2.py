"""Objective evaluation for the v2 configurator-comparison experiment.

Same seed discipline as ``objective.py``:
  - Validation evaluations use common random numbers: the solver seed for
    (instance, repeat) is a deterministic function of the run seed, so every
    configuration inside a run is judged on the *same* seed stream (paired
    comparison, low-variance estimator).
  - Test evaluations use a disjoint seed stream (large offset), so a configuration
    that got lucky on validation seeds does not automatically look good on test.
  - SMAC's native intensification arm additionally performs *its own* stochastic
    (instance, seed) evaluations via ``evaluate_single_call`` -- there the seed comes
    from SMAC, which is exactly the "native" behaviour under study.

The budget unit for the whole experiment is one SA solver call
(= one configuration x one instance x one seed).
"""

from __future__ import annotations

import time

import numpy as np

from .objective import TEST_SEED_OFFSET, _derive_seed
from .resampling_v2 import ResamplingPlanV2
from .solver_sa_ext import SAExtConfig, run_sa_ext


def evaluate_on_ids(config: SAExtConfig, ids, dist_by_id, best_known_by_id, id_index,
                    base_seed: int, repeats: int, max_steps: int) -> dict:
    """Canonical estimator: mean normalized gap over an instance set, CRN seed stream."""
    t0 = time.perf_counter()
    gaps = []
    n_calls = 0
    for iid in ids:
        bk = best_known_by_id[iid]
        g = 0.0
        for r in range(repeats):
            sa_seed = _derive_seed(base_seed, id_index[iid], r)
            res = run_sa_ext(dist_by_id[iid], config, sa_seed, max_steps=max_steps)
            g += (res["tour_length"] - bk) / bk
            n_calls += 1
        gaps.append(g / repeats)
    return {
        "cost": float(np.mean(gaps)),
        "n_sa_calls": n_calls,
        "runtime_sec": time.perf_counter() - t0,
    }


def evaluate_on_resampling_plan(
    config: SAExtConfig,
    plan: ResamplingPlanV2,
    dist_by_id,
    best_known_by_id,
    id_index,
    repeats: int,
    max_steps: int,
) -> dict:
    """Mean of subset means for the optimizer-facing validation estimator."""
    t0 = time.perf_counter()
    subset_means = []
    n_calls = 0
    for ids, subset_seed in plan.subsets:
        instance_gaps = []
        for iid in ids:
            bk = best_known_by_id[iid]
            gap = 0.0
            for repeat in range(repeats):
                sa_seed = _derive_seed(subset_seed, id_index[iid], repeat)
                result = run_sa_ext(
                    dist_by_id[iid], config, sa_seed, max_steps=max_steps
                )
                gap += (result["tour_length"] - bk) / bk
                n_calls += 1
            instance_gaps.append(gap / repeats)
        subset_means.append(float(np.mean(instance_gaps)))
    return {
        "cost": float(np.mean(subset_means)),
        "n_sa_calls": n_calls,
        "runtime_sec": time.perf_counter() - t0,
        "n_subsets": plan.n_subsets,
    }


def evaluate_single_call(config: SAExtConfig, instance_id: str, sa_seed: int,
                         dist_by_id, best_known_by_id, max_steps: int) -> dict:
    """One SA call on one instance with one seed (SMAC-intensification target)."""
    bk = best_known_by_id[instance_id]
    res = run_sa_ext(dist_by_id[instance_id], config, int(sa_seed), max_steps=max_steps)
    return {
        "cost": float((res["tour_length"] - bk) / bk),
        "n_sa_calls": 1,
        "runtime_sec": res["runtime_sec"],
    }


def evaluate_on_test(config: SAExtConfig, test_ids, dist_by_id, best_known_by_id,
                     id_index, run_seed: int, repeats: int, max_steps: int) -> dict:
    """Mean test gap over an independent pool with the disjoint test seed stream."""
    return evaluate_on_ids(config, test_ids, dist_by_id, best_known_by_id, id_index,
                           TEST_SEED_OFFSET + run_seed, repeats, max_steps)


def compute_instance_features(dist: np.ndarray) -> list[float]:
    """Cheap descriptive features of a TSP instance for SMAC's surrogate model
    (used only by the native-intensification arm): distance-distribution moments
    and nearest-neighbour statistics."""
    n = dist.shape[0]
    off_diag = dist[~np.eye(n, dtype=bool)]
    nn = np.where(np.eye(n, dtype=bool), np.inf, dist).min(axis=1)
    return [
        float(off_diag.mean()),
        float(off_diag.std()),
        float(nn.mean()),
        float(nn.std()),
    ]
