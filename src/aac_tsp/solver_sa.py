"""Simulated Annealing solver for symmetric Euclidean TSP.

The target algorithm being configured. The inner loop is numba-JIT compiled and
uses O(1) delta evaluation for the swap and 2-opt neighbourhoods (insert uses an
O(n) re-evaluation, acceptable for the small instances studied here).

A configuration theta = (initial_temperature, cooling_rate, iterations_per_temp,
move_type, restarts). A fixed total ``max_steps`` move budget is split across the
``restarts + 1`` independent runs so wallclock is comparable across configurations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from . import MOVE_CODE

DEFAULT_MAX_STEPS = 30_000


@dataclass
class SAConfig:
    initial_temperature: float
    cooling_rate: float
    iterations_per_temp: int
    move_type: str
    restarts: int

    def as_dict(self) -> dict:
        return {
            "initial_temperature": self.initial_temperature,
            "cooling_rate": self.cooling_rate,
            "iterations_per_temp": self.iterations_per_temp,
            "move_type": self.move_type,
            "restarts": self.restarts,
        }


@njit(cache=True, fastmath=True)
def _tour_length(dist, t):
    n = t.shape[0]
    total = 0.0
    for k in range(n):
        total += dist[t[k], t[(k + 1) % n]]
    return total


@njit(cache=True, fastmath=True)
def _shuffle(t, n):
    # Fisher-Yates using numba-supported np.random.randint
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = t[i]
        t[i] = t[j]
        t[j] = tmp


@njit(cache=True, fastmath=True)
def _swap_edges_sum(dist, t, n, s0, s1, s2, s3):
    # Sum of distinct cyclic edges starting at the given positions (deduped),
    # robust to adjacency / wrap-around.
    starts = (s0, s1, s2, s3)
    total = 0.0
    for a in range(4):
        sa = starts[a]
        dup = False
        for b in range(a):
            if starts[b] == sa:
                dup = True
                break
        if not dup:
            total += dist[t[sa], t[(sa + 1) % n]]
    return total


@njit(cache=True, fastmath=True)
def _sa_core(dist, init_temp, cooling_rate, iters_per_temp, move_code, restarts, max_steps, seed):
    np.random.seed(seed)
    n = dist.shape[0]
    n_runs = restarts + 1
    steps_per_run = max_steps // n_runs
    if steps_per_run < 1:
        steps_per_run = 1

    best_overall = 1e18
    total_steps = 0

    t = np.arange(n)
    work = np.empty(n, dtype=np.int64)

    for _run in range(n_runs):
        # fresh random start
        for k in range(n):
            t[k] = k
        _shuffle(t, n)
        cur_len = _tour_length(dist, t)
        best_len = cur_len
        T = init_temp
        if T < 1e-12:
            T = 1e-12
        since_cool = 0

        for _step in range(steps_per_run):
            total_steps += 1
            delta = 0.0
            applied = False

            if move_code == 0:  # swap two positions
                p = np.random.randint(0, n)
                q = np.random.randint(0, n)
                if p == q:
                    pass
                else:
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
                        # revert
                        tmp = t[p]
                        t[p] = t[q]
                        t[q] = tmp

            elif move_code == 2:  # 2-opt: reverse segment [i, j]
                a = np.random.randint(0, n)
                b = np.random.randint(0, n)
                i = a if a < b else b
                j = b if a < b else a
                if i == j or (i == 0 and j == n - 1):
                    pass
                else:
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

            else:  # move_code == 1: insert (relocate city) -> O(n) re-evaluation
                p = np.random.randint(0, n)
                q = np.random.randint(0, n)
                if p == q:
                    pass
                else:
                    for k in range(n):
                        work[k] = t[k]
                    c = work[p]
                    if p < q:
                        for k in range(p, q):
                            work[k] = work[k + 1]
                        work[q] = c
                    else:
                        for k in range(p, q, -1):
                            work[k] = work[k - 1]
                        work[q] = c
                    new_len = _tour_length(dist, work)
                    delta = new_len - cur_len
                    if delta <= 0.0 or np.random.random() < math.exp(-delta / T):
                        for k in range(n):
                            t[k] = work[k]
                        cur_len = new_len
                        applied = True

            if applied and cur_len < best_len:
                best_len = cur_len

            since_cool += 1
            if since_cool >= iters_per_temp:
                T *= cooling_rate
                if T < 1e-12:
                    T = 1e-12
                since_cool = 0

        if best_len < best_overall:
            best_overall = best_len

    return best_overall, total_steps


def run_simulated_annealing(dist: np.ndarray, config: SAConfig, seed: int,
                            max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Run SA on a precomputed distance matrix. Returns best tour length found."""
    move_code = MOVE_CODE[config.move_type]
    t0 = time.perf_counter()
    best_len, n_steps = _sa_core(
        np.ascontiguousarray(dist, dtype=np.float64),
        float(config.initial_temperature),
        float(config.cooling_rate),
        int(config.iterations_per_temp),
        int(move_code),
        int(config.restarts),
        int(max_steps),
        int(seed) & 0x7FFFFFFF,
    )
    return {"tour_length": float(best_len), "runtime_sec": time.perf_counter() - t0, "n_steps": int(n_steps)}
