"""TSP instance generation and IO.

Three families mirror the standard DIMACS TSP Challenge generators:
  - uniform   : cities uniform in the unit square (portgen / RUE).
  - clustered : 3-5 Gaussian clusters whose centers are uniform in the square (portcgen).
  - mixed     : a blend of uniform and clustered points in one instance.

Coordinates are in [0, 1]^2. We store one compressed ``.npz`` per (family, role)
pool (coords + ids + seeds) rather than one JSON per instance, which is faster to
load for the thousands of solver evaluations the experiment performs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import FAMILIES

# Seed-range convention keeps train and test instances disjoint *by construction*.
TRAIN_SEED_BASE = 0
TEST_SEED_BASE = 1_000_000


@dataclass
class TSPInstance:
    instance_id: str
    family: str
    n_cities: int
    seed: int
    coords: np.ndarray  # (n_cities, 2) float64

    def distance_matrix(self) -> np.ndarray:
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.sqrt((diff * diff).sum(axis=-1)).astype(np.float64)


def generate_uniform_instance(n_cities: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2))


def generate_clustered_instance(n_cities: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_clusters = int(rng.integers(3, 6))  # 3..5 clusters
    centers = rng.random((n_clusters, 2))
    spread = 0.06  # cluster radius relative to unit square
    assignments = rng.integers(0, n_clusters, size=n_cities)
    pts = centers[assignments] + rng.normal(0.0, spread, size=(n_cities, 2))
    return np.clip(pts, 0.0, 1.0)


def generate_mixed_instance(n_cities: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_uniform = n_cities // 2
    n_clustered = n_cities - n_uniform
    uni = rng.random((n_uniform, 2))
    # clustered part
    n_clusters = int(rng.integers(3, 6))
    centers = rng.random((n_clusters, 2))
    assignments = rng.integers(0, n_clusters, size=n_clustered)
    clu = np.clip(centers[assignments] + rng.normal(0.0, 0.06, size=(n_clustered, 2)), 0.0, 1.0)
    pts = np.vstack([uni, clu])
    rng.shuffle(pts)
    return pts


_GENERATORS = {
    "uniform": generate_uniform_instance,
    "clustered": generate_clustered_instance,
    "mixed": generate_mixed_instance,
}


def generate_instance_pool(family: str, n_cities: int, count: int, role: str, seed_base: int) -> list[TSPInstance]:
    """Generate ``count`` instances of a family. ``role`` is 'train' or 'test'."""
    if family not in _GENERATORS:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    gen = _GENERATORS[family]
    instances = []
    for idx in range(count):
        seed = seed_base + idx
        coords = gen(n_cities, seed)
        iid = f"{family}_{n_cities}_{role}_{idx:04d}"
        instances.append(TSPInstance(iid, family, n_cities, seed, coords.astype(np.float64)))
    return instances


def _pool_path(data_dir: Path, family: str, n_cities: int, role: str) -> Path:
    return Path(data_dir) / "instances" / f"{family}_{n_cities}_{role}.npz"


def save_pool(data_dir: Path, family: str, n_cities: int, role: str, instances: list[TSPInstance]) -> Path:
    path = _pool_path(data_dir, family, n_cities, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.stack([inst.coords for inst in instances])  # (count, n, 2)
    ids = np.array([inst.instance_id for inst in instances])
    seeds = np.array([inst.seed for inst in instances])
    np.savez_compressed(path, coords=coords, ids=ids, seeds=seeds,
                        family=family, n_cities=n_cities, role=role)
    return path


def load_pool(data_dir: Path, family: str, n_cities: int, role: str) -> list[TSPInstance]:
    path = _pool_path(data_dir, family, n_cities, role)
    if not path.exists():
        raise FileNotFoundError(f"instance pool not found: {path}; run generate_instances.py first")
    data = np.load(path, allow_pickle=False)
    coords, ids, seeds = data["coords"], data["ids"], data["seeds"]
    out = []
    for i in range(len(ids)):
        out.append(TSPInstance(str(ids[i]), family, int(n_cities), int(seeds[i]), coords[i].astype(np.float64)))
    return out


def generate_all(data_dir: Path, n_cities: int, n_train: int, n_test: int) -> dict:
    """Generate train + test pools for every family; returns a manifest dict."""
    manifest = {"n_cities": n_cities, "n_train": n_train, "n_test": n_test, "families": list(FAMILIES), "pools": {}}
    for family in FAMILIES:
        train = generate_instance_pool(family, n_cities, n_train, "train", TRAIN_SEED_BASE)
        test = generate_instance_pool(family, n_cities, n_test, "test", TEST_SEED_BASE)
        save_pool(data_dir, family, n_cities, "train", train)
        save_pool(data_dir, family, n_cities, "test", test)
        manifest["pools"][family] = {"train": [i.instance_id for i in train], "test": [i.instance_id for i in test]}
    manifest_path = Path(data_dir) / "instances" / f"manifest_n{n_cities}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
