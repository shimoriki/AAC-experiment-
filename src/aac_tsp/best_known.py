"""Best-known reference tour lengths for normalization.

Random TSP instances have no published optimum, so we use a strong, *consistent*
deterministic baseline: multi-start nearest-neighbour + full 2-opt local search.
This need not be globally optimal -- only consistent across all configurations --
because every configuration's performance is normalized by the same reference.

Metric used everywhere else:  gap = (tour_length - best_known) / best_known.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .instances import TSPInstance


@njit(cache=True, fastmath=True)
def _tour_length(dist, t):
    n = t.shape[0]
    total = 0.0
    for k in range(n):
        total += dist[t[k], t[(k + 1) % n]]
    return total


@njit(cache=True, fastmath=True)
def _nearest_neighbor(dist, start):
    n = dist.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    tour = np.empty(n, dtype=np.int64)
    tour[0] = start
    visited[start] = True
    for k in range(1, n):
        last = tour[k - 1]
        best_j = -1
        best_d = 1e18
        for j in range(n):
            if not visited[j] and dist[last, j] < best_d:
                best_d = dist[last, j]
                best_j = j
        tour[k] = best_j
        visited[best_j] = True
    return tour


@njit(cache=True, fastmath=True)
def _two_opt(dist, tour):
    # First-improvement 2-opt until no improving move remains.
    n = tour.shape[0]
    t = tour.copy()
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 1):
            li = (i - 1) % n
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                rj = (j + 1) % n
                old = dist[t[li], t[i]] + dist[t[j], t[rj]]
                new = dist[t[li], t[j]] + dist[t[i], t[rj]]
                if new + 1e-12 < old:
                    lo = i
                    hi = j
                    while lo < hi:
                        tmp = t[lo]
                        t[lo] = t[hi]
                        t[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
        # loop again if any improvement was made in the full pass
    return t


@njit(cache=True, fastmath=True)
def _multistart_2opt(dist, n_starts, seed):
    np.random.seed(seed)
    n = dist.shape[0]
    best_len = 1e18
    for s in range(n_starts):
        start = np.random.randint(0, n)
        tour = _nearest_neighbor(dist, start)
        tour = _two_opt(dist, tour)
        length = _tour_length(dist, tour)
        if length < best_len:
            best_len = length
    return best_len


def compute_best_known(instance: TSPInstance, n_starts: int = 20, seed: int = 12345) -> float:
    dist = np.ascontiguousarray(instance.distance_matrix(), dtype=np.float64)
    n_starts = min(n_starts, instance.n_cities)
    return float(_multistart_2opt(dist, int(n_starts), int(seed)))


def compute_best_known_for_pool(instances: list[TSPInstance], n_starts: int = 20) -> pd.DataFrame:
    rows = []
    for inst in instances:
        bk = compute_best_known(inst, n_starts=n_starts)
        rows.append({"instance_id": inst.instance_id, "family": inst.family,
                     "n_cities": inst.n_cities, "best_known_length": bk})
    return pd.DataFrame(rows)


def best_known_path(data_dir: Path) -> Path:
    return Path(data_dir) / "best_known" / "best_known.csv"


def load_best_known(data_dir: Path) -> dict:
    df = pd.read_csv(best_known_path(data_dir))
    return dict(zip(df["instance_id"], df["best_known_length"]))
