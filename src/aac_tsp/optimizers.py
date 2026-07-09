"""Configurators (Bayesian Optimization) over the SA search space.

Both configurators are driven through Optuna's ask/tell API so the runner code is
identical; only the sampler differs:
  - optuna_tpe : TPESampler (a density-estimation Bayesian-optimization method).
  - random     : RandomSampler (baseline that ignores past observations).

The search space is the SA configuration space from the plan.
"""

from __future__ import annotations

import optuna

from .solver_sa import SAConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_sampler(name: str, seed: int):
    name = name.lower()
    if name == "optuna_tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, multivariate=True)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unknown optimizer {name!r}; expected optuna_tpe or random")


def suggest_config(trial: optuna.Trial, config_space: str = "full") -> SAConfig:
    """Sample an SA configuration.

    config_space:
      - "full"       : the move_type {swap, insert, 2opt} is part of the search space.
      - "fixed_2opt" : move_type is fixed to "2opt" (removes the large move-type
        performance cliff so overtuning among the continuous SA params is measured
        cleanly, in the regime closest to the HPO overtuning paper).

    The suggestion order is preserved so that "full" runs are identical to earlier runs.
    """
    initial_temperature = trial.suggest_float("initial_temperature", 1.0, 1000.0, log=True)
    cooling_rate = trial.suggest_float("cooling_rate", 0.80, 0.999)
    iterations_per_temp = trial.suggest_int("iterations_per_temp", 10, 300)
    if config_space == "fixed_2opt":
        move_type = "2opt"
    elif config_space == "full":
        move_type = trial.suggest_categorical("move_type", ["swap", "insert", "2opt"])
    else:
        raise ValueError(f"unknown config_space {config_space!r}; expected full or fixed_2opt")
    restarts = trial.suggest_int("restarts", 0, 10)
    return SAConfig(
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        iterations_per_temp=iterations_per_temp,
        move_type=move_type,
        restarts=restarts,
    )


def create_study(optimizer: str, seed: int) -> optuna.Study:
    return optuna.create_study(direction="minimize", sampler=get_sampler(optimizer, seed))
