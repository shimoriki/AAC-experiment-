"""One experiment run: a full BO loop for a single (optimizer, resampling, family,
train_size, seed) cell, with per-trial logging.

Key logging rule: the optimizer is told only ``validation_cost``. Test costs are
evaluated only when the validation incumbent changes (cheap) and logged for analysis;
post-processing forward-fills them along the incumbent trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import FAMILIES
from .best_known import load_best_known
from .instances import load_pool
from .objective import evaluate_config_on_resampling, evaluate_config_on_test
from .optimizers import create_study, suggest_config
from .resampling import make_resampling
from .solver_sa import DEFAULT_MAX_STEPS


@dataclass
class RunConfig:
    optimizer: str
    resampling: str
    train_family: str
    train_size: int
    bo_budget: int
    seed: int
    config_space: str = "full"
    n_cities: int = 50
    solver_repeats: int = 3
    test_solver_repeats: int = 3
    test_size: int = 100
    max_steps: int = DEFAULT_MAX_STEPS
    cv_repeats: int = 3
    bootstrap_repeats: int = 10
    data_dir: str = "/local/rohit/datasets/tsp"
    out_dir: str = "results/raw"
    extra: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.optimizer}_{self.resampling}_{self.train_family}_n{self.train_size}_seed{self.seed}"


def _load_all_instances(data_dir, n_cities, train_family, train_size, test_size, seed):
    """Returns dist/best-known maps, the chosen train ids, and per-family test ids."""
    bk = load_best_known(data_dir)
    dist_by_id, id_index = {}, {}
    idx = 0

    def register(instances):
        nonlocal idx
        for inst in instances:
            if inst.instance_id in dist_by_id:
                continue
            dist_by_id[inst.instance_id] = np.ascontiguousarray(inst.distance_matrix(), dtype=np.float64)
            id_index[inst.instance_id] = idx
            idx += 1

    # training pool of the requested family, subsample to train_size with a per-seed RNG
    train_pool = load_pool(data_dir, train_family, n_cities, "train")
    register(train_pool)
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice([i.instance_id for i in train_pool], size=train_size, replace=False))

    # test pools for ALL families (composition / distribution-shift axis)
    test_ids = {}
    for fam in FAMILIES:
        pool = load_pool(data_dir, fam, n_cities, "test")
        register(pool)
        ids = [i.instance_id for i in pool]
        test_ids[fam] = ids[:test_size]

    return bk, dist_by_id, id_index, chosen, test_ids


def run_experiment(cfg: RunConfig) -> Path:
    bk, dist_by_id, id_index, train_ids, test_ids = _load_all_instances(
        cfg.data_dir, cfg.n_cities, cfg.train_family, cfg.train_size, cfg.test_size, cfg.seed)

    strategy = make_resampling(cfg.resampling, cv_repeats=cfg.cv_repeats,
                               bootstrap_repeats=cfg.bootstrap_repeats)
    resampling_seed = cfg.seed
    plan = strategy.make_plan(train_ids, resampling_seed)

    study = create_study(cfg.optimizer, cfg.seed)
    incumbent_val = np.inf
    rows = []

    for trial_id in range(cfg.bo_budget):
        trial = study.ask()
        config = suggest_config(trial, cfg.config_space)
        vres = evaluate_config_on_resampling(
            config, plan, dist_by_id, bk, id_index, cfg.solver_repeats, cfg.max_steps)
        study.tell(trial, vres["validation_cost"])

        is_incumbent = vres["validation_cost"] < incumbent_val - 1e-15
        if is_incumbent:
            incumbent_val = vres["validation_cost"]

        # Test evaluation only on incumbent change (and the very first trial).
        test_costs = {fam: np.nan for fam in FAMILIES}
        test_runtime = np.nan
        if is_incumbent or trial_id == 0:
            test_runtime = 0.0
            for fam in FAMILIES:
                tres = evaluate_config_on_test(
                    config, test_ids[fam], dist_by_id, bk, id_index,
                    cfg.seed, cfg.test_solver_repeats, cfg.max_steps)
                test_costs[fam] = tres["test_cost"]
                test_runtime += tres["test_runtime_sec"]

        row = {
            "run_id": cfg.run_id,
            "trial_id": trial_id,
            "optimizer": cfg.optimizer,
            "config_space": cfg.config_space,
            "resampling": cfg.resampling,
            "train_family": cfg.train_family,
            "train_size": cfg.train_size,
            "seed": cfg.seed,
            "bo_budget": cfg.bo_budget,
            "is_incumbent": bool(is_incumbent or trial_id == 0),
            **config.as_dict(),
            "validation_cost": vres["validation_cost"],
            "validation_runtime_sec": vres["validation_runtime_sec"],
            "n_eval_instances": vres["n_eval_instances"],
            "subsets_evaluated": vres["subsets_evaluated"],
            "test_cost": test_costs[cfg.train_family],  # same-distribution test (primary)
            "test_cost_uniform": test_costs["uniform"],
            "test_cost_clustered": test_costs["clustered"],
            "test_cost_mixed": test_costs["mixed"],
            "test_runtime_sec": test_runtime,
        }
        rows.append(row)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{cfg.run_id}.csv"
    df.to_csv(csv_path, index=False)

    sidecar = {
        "run_id": cfg.run_id,
        "config_space": cfg.config_space,
        "train_instance_ids": train_ids,
        "test_instance_ids": {fam: test_ids[fam] for fam in FAMILIES},
        "resampling_seed": resampling_seed,
        "n_subsets": len(plan),
        "solver_repeats": cfg.solver_repeats,
        "test_solver_repeats": cfg.test_solver_repeats,
        "cv_repeats": cfg.cv_repeats,
        "bootstrap_repeats": cfg.bootstrap_repeats,
        "max_steps": cfg.max_steps,
        "n_cities": cfg.n_cities,
    }
    (out_dir / f"{cfg.run_id}.json").write_text(json.dumps(sidecar, indent=2))
    return csv_path
