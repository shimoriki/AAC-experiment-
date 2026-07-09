"""Experiment v2: configurator comparison at scale (BO vs Random Search vs SMAC).

Four arms, all optimizing the same extended 15-parameter SA space on the same
training instances with the same total target-algorithm budget:

  random      Optuna RandomSampler; fixed budget of ``n_trials`` proposals, each
              evaluated on the full training set (CRN seed stream).
  optuna_tpe  Optuna TPESampler (multivariate, grouped); same protocol as random.
  smac_bo     SMAC3 HyperparameterOptimizationFacade (RF-based BO); same fixed-budget
              protocol as the Optuna arms (each proposal = full training-set eval).
  smac_aac    SMAC3 AlgorithmConfigurationFacade with *native intensification*:
              SMAC decides per configuration how many (instance, seed) pairs to
              spend; each target call is a single SA run on a single instance.

Budget fairness -- the unit is one SA solver call (config x instance x seed):
  fixed-budget arms: n_trials * train_size * solver_repeats calls
  smac_aac:          exactly the same number of calls, but allocated by the
                     intensifier instead of uniformly.

Two invariants carried over from the v1 experiment:
  1. The configurator never sees test cost. Test (and, for smac_aac, the canonical
     re-evaluated validation cost) are analysis-only and do not count toward budget.
  2. Every run writes the same two CSVs so post-processing is arm-agnostic:
       <run_id>_trials.csv  one row per configurator proposal / target call
       <run_id>_traj.csv    one row per incumbent change on the common budget axis
                            (cum_sa_calls), plus a closing row at budget exhaustion
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import FAMILIES
from .evaluation_v2 import (compute_instance_features, evaluate_on_ids,
                            evaluate_on_test, evaluate_single_call)
from .runner import _load_all_instances
from .space_v2 import (PARAM_COLUMNS, config_from_mapping, config_key,
                       optuna_suggest_v2)

OPTIMIZERS_V2 = ("random", "optuna_tpe", "smac_bo", "smac_aac")
INCUMBENT_TOL = 1e-15


@dataclass
class RunConfigV2:
    optimizer: str
    train_family: str
    train_size: int
    n_trials: int              # fixed-budget proposals; defines the SA-call budget
    seed: int
    n_cities: int = 50
    solver_repeats: int = 1    # SA seeds per instance in the canonical estimator
    test_solver_repeats: int = 2
    test_size: int = 100
    max_steps: int = 30_000
    tpe_startup_trials: int = 25
    smac_initial_configs: int = 25   # initial design size of the smac_bo arm
    smac_max_config_calls: int = 0   # cap per config for smac_aac; 0 -> 3 * train_size
    smac_retrain_after: int = 8      # surrogate refit interval for smac_aac
    data_dir: str = "/local/anmol/datasets/tsp"
    out_dir: str = "results/v2/raw"
    smac_scratch: str = "results/v2/smac_output"

    @property
    def run_id(self) -> str:
        return f"{self.optimizer}_{self.train_family}_n{self.train_size}_seed{self.seed}"

    @property
    def total_budget_calls(self) -> int:
        return self.n_trials * self.train_size * self.solver_repeats


class _Context:
    """Instances, distance matrices, and best-known references for one run."""

    def __init__(self, cfg: RunConfigV2):
        (self.bk, self.dist_by_id, self.id_index,
         self.train_ids, self.test_ids) = _load_all_instances(
            cfg.data_dir, cfg.n_cities, cfg.train_family,
            cfg.train_size, cfg.test_size, cfg.seed)


class _CanonicalEvaluator:
    """Analysis-only evaluations (never fed back to any configurator), cached per
    configuration: canonical validation cost (CRN full-train estimator) and test
    cost on every family's test pool."""

    def __init__(self, cfg: RunConfigV2, ctx: _Context):
        self.cfg, self.ctx = cfg, ctx
        self.cache: dict[tuple, dict] = {}
        self.analysis_sa_calls = 0
        self.analysis_runtime_sec = 0.0

    def evaluate(self, config, known_val: float | None = None) -> dict:
        """Full analysis entry for a config. Pass ``known_val`` when the canonical
        validation cost was already computed in the optimization loop (fixed-budget
        arms) to avoid recomputing the identical CRN estimate."""
        key = config_key(config)
        if key in self.cache:
            return self.cache[key]
        cfg, ctx = self.cfg, self.ctx
        n_calls, runtime = 0, 0.0
        if known_val is None:
            vres = evaluate_on_ids(config, ctx.train_ids, ctx.dist_by_id, ctx.bk,
                                   ctx.id_index, cfg.seed, cfg.solver_repeats,
                                   cfg.max_steps)
            entry = {"val_cost": vres["cost"]}
            n_calls, runtime = vres["n_sa_calls"], vres["runtime_sec"]
        else:
            entry = {"val_cost": known_val}
        for fam in FAMILIES:
            tres = evaluate_on_test(config, ctx.test_ids[fam], ctx.dist_by_id, ctx.bk,
                                    ctx.id_index, cfg.seed, cfg.test_solver_repeats,
                                    cfg.max_steps)
            entry[f"test_cost_{fam}"] = tres["cost"]
            n_calls += tres["n_sa_calls"]
            runtime += tres["runtime_sec"]
        self.analysis_sa_calls += n_calls
        self.analysis_runtime_sec += runtime
        self.cache[key] = entry
        return entry


def _traj_row(cfg: RunConfigV2, event: int, event_type: str, trial_id: int,
              cum_sa_calls: int, config, entry: dict, t_start: float) -> dict:
    row = {
        "run_id": cfg.run_id,
        "optimizer": cfg.optimizer,
        "train_family": cfg.train_family,
        "train_size": cfg.train_size,
        "seed": cfg.seed,
        "n_trials": cfg.n_trials,
        "event": event,
        "event_type": event_type,   # incumbent | final
        "trial_id": trial_id,
        "cum_sa_calls": cum_sa_calls,
        "val_cost": entry["val_cost"],
        "test_cost": entry[f"test_cost_{cfg.train_family}"],
        "wallclock_sec": time.perf_counter() - t_start,
    }
    for fam in FAMILIES:
        row[f"test_cost_{fam}"] = entry[f"test_cost_{fam}"]
    row.update(config.as_dict())
    return row


def _trial_row(cfg: RunConfigV2, trial_id: int, config, observed_cost: float,
               n_sa_calls: int, cum_sa_calls: int, is_incumbent: bool,
               ask_sec: float, eval_sec: float, tell_sec: float,
               instance_id: str = "", smac_seed: int = -1) -> dict:
    row = {
        "run_id": cfg.run_id,
        "optimizer": cfg.optimizer,
        "train_family": cfg.train_family,
        "train_size": cfg.train_size,
        "seed": cfg.seed,
        "trial_id": trial_id,
        "config_id": hash(config_key(config)) & 0xFFFFFFFF,
        "observed_cost": observed_cost,
        "n_sa_calls": n_sa_calls,
        "cum_sa_calls": cum_sa_calls,
        "is_incumbent": bool(is_incumbent),
        "instance_id": instance_id,
        "smac_seed": smac_seed,
        "ask_sec": ask_sec,
        "eval_sec": eval_sec,
        "tell_sec": tell_sec,
    }
    row.update(config.as_dict())
    return row


# --------------------------------------------------------------------------- #
# Fixed-budget arms (random, optuna_tpe, smac_bo): identical protocol, only the
# proposal mechanism differs.
# --------------------------------------------------------------------------- #

def _run_fixed_budget(cfg: RunConfigV2, ctx: _Context, evaluator: _CanonicalEvaluator,
                      ask_fn, tell_fn) -> tuple[list, list, dict]:
    """Shared fixed-budget loop. ask_fn() -> (handle, SAExtConfig); tell_fn(handle, cost)."""
    t_start = time.perf_counter()
    trials, traj = [], []
    incumbent_val = np.inf
    cum = 0
    n_inc = 0
    totals = {"ask_sec": 0.0, "eval_sec": 0.0, "tell_sec": 0.0}

    for trial_id in range(cfg.n_trials):
        t0 = time.perf_counter()
        try:
            handle, config = ask_fn()
        except StopIteration:  # SMAC can refuse the ask at exact budget exhaustion
            break
        t1 = time.perf_counter()
        vres = evaluate_on_ids(config, ctx.train_ids, ctx.dist_by_id, ctx.bk,
                               ctx.id_index, cfg.seed, cfg.solver_repeats, cfg.max_steps)
        val = vres["cost"]
        t2 = time.perf_counter()
        tell_fn(handle, val)
        t3 = time.perf_counter()

        cum += vres["n_sa_calls"]
        totals["ask_sec"] += t1 - t0
        totals["eval_sec"] += t2 - t1
        totals["tell_sec"] += t3 - t2

        is_inc = val < incumbent_val - INCUMBENT_TOL
        if is_inc or trial_id == 0:
            incumbent_val = min(incumbent_val, val)
            entry = evaluator.evaluate(config, known_val=val)
            traj.append(_traj_row(cfg, n_inc, "incumbent", trial_id, cum, config,
                                  entry, t_start))
            n_inc += 1

        trials.append(_trial_row(cfg, trial_id, config, val, vres["n_sa_calls"], cum,
                                 is_inc or trial_id == 0,
                                 t1 - t0, t2 - t1, t3 - t2))

    # closing row: extend the incumbent step function to the exact budget end
    if traj:
        last = dict(traj[-1])
        last.update({"event": n_inc, "event_type": "final", "cum_sa_calls": cum,
                     "wallclock_sec": time.perf_counter() - t_start})
        traj.append(last)

    totals.update({"n_incumbent_changes": n_inc, "total_sa_calls_optimization": cum,
                   "wallclock_sec": time.perf_counter() - t_start})
    return trials, traj, totals


def _run_optuna(cfg: RunConfigV2, ctx: _Context, evaluator: _CanonicalEvaluator):
    import warnings

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    if cfg.optimizer == "optuna_tpe":
        sampler = optuna.samplers.TPESampler(seed=cfg.seed,
                                             n_startup_trials=cfg.tpe_startup_trials,
                                             multivariate=True, group=True)
    else:
        sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def ask_fn():
        trial = study.ask()
        return trial, optuna_suggest_v2(trial)

    def tell_fn(trial, cost):
        study.tell(trial, cost)

    return _run_fixed_budget(cfg, ctx, evaluator, ask_fn, tell_fn)


def _require_smac():
    try:
        import smac  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SMAC3 is required for the smac_bo / smac_aac arms. Install it on the "
            "Linux VM with: pip install -r requirements_v2.txt "
            "(needs `apt install swig build-essential` for pyrfr). "
            "SMAC does not build on Windows."
        ) from e


def _run_smac_bo(cfg: RunConfigV2, ctx: _Context, evaluator: _CanonicalEvaluator):
    _require_smac()
    from smac import HyperparameterOptimizationFacade as HPOFacade
    from smac import Scenario
    from smac.runhistory.dataclasses import TrialValue

    from .space_v2 import build_configspace

    cs = build_configspace(cfg.seed)
    scenario = Scenario(
        cs,
        name=cfg.run_id,
        output_directory=Path(cfg.smac_scratch),
        deterministic=True,   # the CRN estimator is deterministic given the config
        n_trials=cfg.n_trials,
        seed=cfg.seed,
    )

    def _target(config, seed: int = 0) -> float:  # required by the facade; loop uses ask/tell
        vres = evaluate_on_ids(config_from_mapping(config), ctx.train_ids, ctx.dist_by_id,
                               ctx.bk, ctx.id_index, cfg.seed, cfg.solver_repeats,
                               cfg.max_steps)
        return vres["cost"]

    smac = HPOFacade(
        scenario,
        _target,
        initial_design=HPOFacade.get_initial_design(scenario,
                                                    n_configs=cfg.smac_initial_configs),
        overwrite=True,
        logging_level=30,
    )

    def ask_fn():
        info = smac.ask()
        return info, config_from_mapping(info.config)

    def tell_fn(info, cost):
        # save=False: we persist everything ourselves; SMAC's default re-writes the
        # full runhistory JSON on every tell.
        smac.tell(info, TrialValue(cost=cost), save=False)

    return _run_fixed_budget(cfg, ctx, evaluator, ask_fn, tell_fn)


# --------------------------------------------------------------------------- #
# smac_aac: native intensification. One target call = one SA run on one instance
# with one SMAC-chosen seed; the intensifier races configurations across
# (instance, seed) pairs and promotes incumbents.
# --------------------------------------------------------------------------- #

def _run_smac_aac(cfg: RunConfigV2, ctx: _Context, evaluator: _CanonicalEvaluator):
    _require_smac()
    from smac import AlgorithmConfigurationFacade as ACFacade
    from smac import Scenario
    from smac.runhistory.dataclasses import TrialValue

    from .space_v2 import build_configspace

    cs = build_configspace(cfg.seed)
    features = {iid: compute_instance_features(ctx.dist_by_id[iid])
                for iid in ctx.train_ids}
    budget = cfg.total_budget_calls
    scenario = Scenario(
        cs,
        name=cfg.run_id,
        output_directory=Path(cfg.smac_scratch),
        deterministic=False,          # SA is stochastic; SMAC races over seeds too
        n_trials=budget,
        instances=list(ctx.train_ids),
        instance_features=features,
        seed=cfg.seed,
    )

    def _target(config, instance: str, seed: int = 0) -> float:
        return evaluate_single_call(config_from_mapping(config), instance, seed,
                                    ctx.dist_by_id, ctx.bk, cfg.max_steps)["cost"]

    max_config_calls = cfg.smac_max_config_calls or 3 * cfg.train_size
    smac = ACFacade(
        scenario,
        _target,
        intensifier=ACFacade.get_intensifier(scenario, max_config_calls=max_config_calls),
        config_selector=ACFacade.get_config_selector(scenario,
                                                     retrain_after=cfg.smac_retrain_after),
        overwrite=True,
        logging_level=30,
    )

    t_start = time.perf_counter()
    trials, traj = [], []
    totals = {"ask_sec": 0.0, "eval_sec": 0.0, "tell_sec": 0.0}
    prev_inc_key = None
    n_inc = 0
    cum = 0
    last_entry, last_config, last_trial = None, None, -1

    for i in range(budget):
        t0 = time.perf_counter()
        try:
            info = smac.ask()
        except StopIteration:
            break
        config = config_from_mapping(info.config)
        t1 = time.perf_counter()
        res = evaluate_single_call(config, info.instance, info.seed,
                                   ctx.dist_by_id, ctx.bk, cfg.max_steps)
        t2 = time.perf_counter()
        # save=False: re-writing the runhistory JSON on each of the ~10k tells would
        # be quadratic disk I/O; all logging is done by this runner instead.
        smac.tell(info, TrialValue(cost=res["cost"], time=res["runtime_sec"]), save=False)
        t3 = time.perf_counter()

        cum += res["n_sa_calls"]
        totals["ask_sec"] += t1 - t0
        totals["eval_sec"] += t2 - t1
        totals["tell_sec"] += t3 - t2

        inc = smac.intensifier.get_incumbent()
        is_inc_config = False
        if inc is not None:
            inc_config = config_from_mapping(inc)
            inc_key = config_key(inc_config)
            is_inc_config = inc_key == config_key(config)
            if inc_key != prev_inc_key:
                prev_inc_key = inc_key
                entry = evaluator.evaluate(inc_config)   # analysis-only, cached
                traj.append(_traj_row(cfg, n_inc, "incumbent", i, cum, inc_config,
                                      entry, t_start))
                n_inc += 1
                last_entry, last_config, last_trial = entry, inc_config, i

        trials.append(_trial_row(cfg, i, config, res["cost"], res["n_sa_calls"], cum,
                                 is_inc_config, t1 - t0, t2 - t1, t3 - t2,
                                 instance_id=str(info.instance),
                                 smac_seed=int(info.seed if info.seed is not None else -1)))

    if last_entry is not None:
        traj.append(_traj_row(cfg, n_inc, "final", last_trial, cum, last_config,
                              last_entry, t_start))

    totals.update({"n_incumbent_changes": n_inc, "total_sa_calls_optimization": cum,
                   "wallclock_sec": time.perf_counter() - t_start})
    return trials, traj, totals


# --------------------------------------------------------------------------- #

def run_experiment_v2(cfg: RunConfigV2) -> Path:
    if cfg.optimizer not in OPTIMIZERS_V2:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}; expected one of {OPTIMIZERS_V2}")

    ctx = _Context(cfg)
    evaluator = _CanonicalEvaluator(cfg, ctx)

    if cfg.optimizer in ("random", "optuna_tpe"):
        trials, traj, totals = _run_optuna(cfg, ctx, evaluator)
    elif cfg.optimizer == "smac_bo":
        trials, traj, totals = _run_smac_bo(cfg, ctx, evaluator)
    else:
        trials, traj, totals = _run_smac_aac(cfg, ctx, evaluator)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / f"{cfg.run_id}_trials.csv"
    traj_path = out_dir / f"{cfg.run_id}_traj.csv"
    pd.DataFrame(trials).to_csv(trials_path, index=False)
    pd.DataFrame(traj).to_csv(traj_path, index=False)

    n_distinct = len({r["config_id"] for r in trials})
    sidecar = {
        "status": "complete",
        "run_config": asdict(cfg),
        "run_id": cfg.run_id,
        "total_budget_calls": cfg.total_budget_calls,
        "total_sa_calls_optimization": totals["total_sa_calls_optimization"],
        "total_sa_calls_analysis": evaluator.analysis_sa_calls,
        "analysis_runtime_sec": evaluator.analysis_runtime_sec,
        "ask_sec": totals["ask_sec"],
        "eval_sec": totals["eval_sec"],
        "tell_sec": totals["tell_sec"],
        "wallclock_sec": totals["wallclock_sec"],
        "n_incumbent_changes": totals["n_incumbent_changes"],
        "n_distinct_configs": n_distinct,
        "param_columns": PARAM_COLUMNS,
        "train_instance_ids": list(ctx.train_ids),
        "versions": _versions(),
    }
    (out_dir / f"{cfg.run_id}.json").write_text(json.dumps(sidecar, indent=2))
    return traj_path


def _versions() -> dict:
    import numba
    import optuna
    v = {"numpy": np.__version__, "pandas": pd.__version__,
         "numba": numba.__version__, "optuna": optuna.__version__}
    try:
        import smac
        v["smac"] = smac.__version__
    except ImportError:
        pass
    return v
