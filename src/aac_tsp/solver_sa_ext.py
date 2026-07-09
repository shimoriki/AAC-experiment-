"""Extended Simulated Annealing solver for the v2 configurator-comparison experiment.

This is a strict superset of the SA in ``solver_sa.py``: the original 5-parameter
space is enlarged to 15 parameters (12 always active + 3 conditional) so that the
configuration space is genuinely high-dimensional, mixed-type, and hierarchical --
the regime where model-based configurators (TPE, SMAC) are expected to separate
clearly from Random Search, and where SMAC's native handling of conditionals matters.

New mechanisms over the original solver:
  - Mixed move neighbourhood: each step draws swap / insert / 2-opt / or-opt with
    tunable probabilities (the original single ``move_type`` categorical becomes a
    4-dim simplex; pure strategies are corner cases of the mixture).
  - Or-opt move: relocate a segment of 2-3 consecutive cities.
  - Construction heuristic: random permutation or nearest-neighbour start.
  - Temperature floor (``min_temperature``) and optional periodic reheating.
  - Restart strategy: fresh restart vs ILS-style double-bridge perturbation of the
    best tour found so far (``perturbation_kicks`` controls kick strength).

As before, a fixed total ``max_steps`` move budget is split across the
``restarts + 1`` runs so wallclock stays comparable across configurations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from .solver_sa import DEFAULT_MAX_STEPS, _shuffle, _swap_edges_sum, _tour_length

INIT_METHOD_CODE = {"random": 0, "nearest_neighbor": 1}
RESTART_STRATEGY_CODE = {"fresh": 0, "perturb_best": 1}


@dataclass
class SAExtConfig:
    """One point in the v2 configuration space (see ``space_v2.py`` for ranges)."""

    initial_temperature: float
    cooling_rate: float
    min_temperature: float
    iterations_per_temp: int
    p_swap: float
    p_insert: float
    p_2opt: float
    p_oropt: float
    init_method: str          # random | nearest_neighbor
    restarts: int
    restart_strategy: str     # fresh | perturb_best   (irrelevant when restarts == 0)
    perturbation_kicks: int   # active only for perturb_best
    use_reheat: str           # no | yes
    reheat_interval: int      # coolings between reheats; active only when use_reheat == yes
    reheat_factor: float      # T_reheat = initial_temperature * reheat_factor

    def as_dict(self) -> dict:
        return {
            "initial_temperature": self.initial_temperature,
            "cooling_rate": self.cooling_rate,
            "min_temperature": self.min_temperature,
            "iterations_per_temp": self.iterations_per_temp,
            "p_swap": self.p_swap,
            "p_insert": self.p_insert,
            "p_2opt": self.p_2opt,
            "p_oropt": self.p_oropt,
            "init_method": self.init_method,
            "restarts": self.restarts,
            "restart_strategy": self.restart_strategy,
            "perturbation_kicks": self.perturbation_kicks,
            "use_reheat": self.use_reheat,
            "reheat_interval": self.reheat_interval,
            "reheat_factor": self.reheat_factor,
        }


@njit(cache=True, fastmath=True)
def _nn_tour(dist, t, n, start):
    visited = np.zeros(n, dtype=np.bool_)
    t[0] = start
    visited[start] = True
    cur = start
    for k in range(1, n):
        best_j = -1
        best_d = 1e18
        for j in range(n):
            if not visited[j] and dist[cur, j] < best_d:
                best_d = dist[cur, j]
                best_j = j
        t[k] = best_j
        visited[best_j] = True
        cur = best_j


@njit(cache=True, fastmath=True)
def _double_bridge(t, work, n):
    # Classic 4-opt double bridge A|B|C|D -> A|C|B|D with random cut points
    # 1 <= a < b < c < n. Requires n >= 4.
    a = 1 + np.random.randint(0, n - 3)
    b = a + 1 + np.random.randint(0, n - a - 2)
    c = b + 1 + np.random.randint(0, n - b - 1)
    idx = 0
    for k in range(0, a):
        work[idx] = t[k]
        idx += 1
    for k in range(b, c):
        work[idx] = t[k]
        idx += 1
    for k in range(a, b):
        work[idx] = t[k]
        idx += 1
    for k in range(c, n):
        work[idx] = t[k]
        idx += 1
    for k in range(n):
        t[k] = work[k]


@njit(cache=True, fastmath=True)
def _sa_core_ext(dist, init_temp, cooling_rate, min_temp, iters_per_temp,
                 c_swap, c_insert, c_2opt,
                 init_code, restarts, restart_code, kicks,
                 reheat_interval, reheat_temp, max_steps, seed):
    """Extended SA core.

    c_swap/c_insert/c_2opt are cumulative move-probability thresholds in [0, 1]
    (or-opt takes the remainder). reheat_interval <= 0 disables reheating;
    otherwise every ``reheat_interval`` coolings the temperature is raised to
    ``reheat_temp`` if that exceeds the current temperature.
    """
    np.random.seed(seed)
    n = dist.shape[0]
    n_runs = restarts + 1
    steps_per_run = max_steps // n_runs
    if steps_per_run < 1:
        steps_per_run = 1

    t = np.empty(n, dtype=np.int64)
    work = np.empty(n, dtype=np.int64)
    work2 = np.empty(n, dtype=np.int64)
    best_tour = np.empty(n, dtype=np.int64)

    best_overall = 1e18
    have_best = False
    total_steps = 0

    if min_temp < 1e-12:
        min_temp = 1e-12

    for run in range(n_runs):
        # ---- construct the starting tour of this run
        if run > 0 and restart_code == 1 and have_best and n >= 5:
            for k in range(n):
                t[k] = best_tour[k]
            for _kick in range(kicks):
                _double_bridge(t, work, n)
        elif init_code == 1:
            _nn_tour(dist, t, n, np.random.randint(0, n))
        else:
            for k in range(n):
                t[k] = k
            _shuffle(t, n)

        cur_len = _tour_length(dist, t)
        if cur_len < best_overall:
            best_overall = cur_len
            for k in range(n):
                best_tour[k] = t[k]
            have_best = True

        T = init_temp
        if T < min_temp:
            T = min_temp
        since_cool = 0
        coolings = 0

        for _step in range(steps_per_run):
            total_steps += 1
            u = np.random.random()
            applied = False

            if u < c_swap:  # ---- swap two positions (O(1) delta)
                p = np.random.randint(0, n)
                q = np.random.randint(0, n)
                if p != q:
                    s0 = (p - 1) % n
                    s2 = (q - 1) % n
                    old = _swap_edges_sum(dist, t, n, s0, p, s2, q)
                    tmp = t[p]
                    t[p] = t[q]
                    t[q] = tmp
                    new = _swap_edges_sum(dist, t, n, s0, p, s2, q)
                    delta = new - old
                    if delta <= 0.0 or np.random.random() < math.exp(-delta / T):
                        cur_len += delta
                        applied = True
                    else:
                        tmp = t[p]
                        t[p] = t[q]
                        t[q] = tmp

            elif u < c_insert:  # ---- insert / relocate one city (O(n) re-eval)
                p = np.random.randint(0, n)
                q = np.random.randint(0, n)
                if p != q:
                    for k in range(n):
                        work[k] = t[k]
                    city = work[p]
                    if p < q:
                        for k in range(p, q):
                            work[k] = work[k + 1]
                        work[q] = city
                    else:
                        for k in range(p, q, -1):
                            work[k] = work[k - 1]
                        work[q] = city
                    new_len = _tour_length(dist, work)
                    delta = new_len - cur_len
                    if delta <= 0.0 or np.random.random() < math.exp(-delta / T):
                        for k in range(n):
                            t[k] = work[k]
                        cur_len = new_len
                        applied = True

            elif u < c_2opt:  # ---- 2-opt: reverse segment [i, j] (O(1) delta)
                a = np.random.randint(0, n)
                b = np.random.randint(0, n)
                i = a if a < b else b
                j = b if a < b else a
                if not (i == j or (i == 0 and j == n - 1)):
                    li = (i - 1) % n
                    rj = (j + 1) % n
                    old = dist[t[li], t[i]] + dist[t[j], t[rj]]
                    new = dist[t[li], t[j]] + dist[t[i], t[rj]]
                    delta = new - old
                    if delta <= 0.0 or np.random.random() < math.exp(-delta / T):
                        lo = i
                        hi = j
                        while lo < hi:
                            tmp = t[lo]
                            t[lo] = t[hi]
                            t[hi] = tmp
                            lo += 1
                            hi -= 1
                        cur_len += delta
                        applied = True

            else:  # ---- or-opt: relocate a segment of 2-3 cities (O(n) re-eval)
                seg_len = 2 + np.random.randint(0, 2)
                if seg_len < n:
                    p = np.random.randint(0, n - seg_len + 1)
                    q = np.random.randint(0, n - seg_len + 1)
                    if p != q:
                        # work = tour without the segment (length n - seg_len)
                        idx = 0
                        for k in range(n):
                            if k < p or k >= p + seg_len:
                                work[idx] = t[k]
                                idx += 1
                        # work2 = work with the segment re-inserted at position q
                        idx = 0
                        for k in range(q):
                            work2[idx] = work[k]
                            idx += 1
                        for k in range(seg_len):
                            work2[idx] = t[p + k]
                            idx += 1
                        for k in range(q, n - seg_len):
                            work2[idx] = work[k]
                            idx += 1
                        new_len = _tour_length(dist, work2)
                        delta = new_len - cur_len
                        if delta <= 0.0 or np.random.random() < math.exp(-delta / T):
                            for k in range(n):
                                t[k] = work2[k]
                            cur_len = new_len
                            applied = True

            if applied and cur_len < best_overall - 1e-12:
                best_overall = cur_len
                for k in range(n):
                    best_tour[k] = t[k]
                have_best = True

            since_cool += 1
            if since_cool >= iters_per_temp:
                T *= cooling_rate
                coolings += 1
                since_cool = 0
                if reheat_interval > 0 and coolings % reheat_interval == 0 and reheat_temp > T:
                    T = reheat_temp
                if T < min_temp:
                    T = min_temp

    return best_overall, total_steps


def run_sa_ext(dist: np.ndarray, config: SAExtConfig, seed: int,
               max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Run the extended SA on a precomputed distance matrix; returns the best tour length."""
    total = config.p_swap + config.p_insert + config.p_2opt + config.p_oropt
    if total < 1e-9:
        # Degenerate all-zero mixture: deterministic fallback to pure 2-opt.
        c_swap, c_insert, c_2opt = 0.0, 0.0, 1.0
    else:
        c_swap = config.p_swap / total
        c_insert = c_swap + config.p_insert / total
        c_2opt = c_insert + config.p_2opt / total

    reheat_interval = int(config.reheat_interval) if config.use_reheat == "yes" else 0
    reheat_temp = float(config.initial_temperature) * float(config.reheat_factor)

    t0 = time.perf_counter()
    best_len, n_steps = _sa_core_ext(
        np.ascontiguousarray(dist, dtype=np.float64),
        float(config.initial_temperature),
        float(config.cooling_rate),
        float(config.min_temperature),
        int(config.iterations_per_temp),
        float(c_swap), float(c_insert), float(c_2opt),
        int(INIT_METHOD_CODE[config.init_method]),
        int(config.restarts),
        int(RESTART_STRATEGY_CODE[config.restart_strategy]),
        int(config.perturbation_kicks),
        int(reheat_interval),
        float(reheat_temp),
        int(max_steps),
        int(seed) & 0x7FFFFFFF,
    )
    return {"tour_length": float(best_len), "runtime_sec": time.perf_counter() - t0,
            "n_steps": int(n_steps)}
