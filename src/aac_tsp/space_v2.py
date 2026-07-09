"""The v2 configuration space (single source of truth for all configurators).

15 hyperparameters: 12 always active, 3 conditional. Mixed types (log-floats,
linear floats, log-ints, categoricals) and a hierarchical structure -- exactly the
regime motivating model-based algorithm configuration.

| name                 | type        | range               | condition                      |
|----------------------|-------------|---------------------|--------------------------------|
| initial_temperature  | float, log  | [0.01, 5000]        |                                |
| cooling_rate         | float       | [0.5, 0.9999]       |                                |
| min_temperature      | float, log  | [1e-6, 1.0]         |                                |
| iterations_per_temp  | int, log    | [1, 500]            |                                |
| p_swap               | float       | [0, 1]              |                                |
| p_insert             | float       | [0, 1]              |                                |
| p_2opt               | float       | [0, 1]              |                                |
| p_oropt              | float       | [0, 1]              |                                |
| init_method          | categorical | random, nearest_neighbor |                           |
| restarts             | int         | [0, 16]             |                                |
| restart_strategy     | categorical | fresh, perturb_best |                                |
| perturbation_kicks   | int         | [1, 8]              | restart_strategy == perturb_best |
| use_reheat           | categorical | no, yes             |                                |
| reheat_interval      | int         | [5, 100]            | use_reheat == yes              |
| reheat_factor        | float, log  | [0.001, 1.0]        | use_reheat == yes              |

The move probabilities are normalized inside the solver, so the 4 p_* parameters
parameterize a 3-simplex (pure move strategies are its corners). The space contains
large genuinely-bad regions (quenching cooling rates, random-walk temperatures,
swap/insert-dominated neighbourhoods), so Random Search keeps paying for them while
a model-based configurator can learn to avoid them.

The same definition is exposed three ways:
  - ``optuna_suggest_v2(trial)``   define-by-run for Optuna TPE / RandomSampler
  - ``build_configspace(seed)``    ConfigSpace object for SMAC (with conditionals)
  - ``config_from_mapping(m)``     any dict-like (incl. SMAC Configuration) -> SAExtConfig
"""

from __future__ import annotations

from .solver_sa_ext import SAExtConfig

# Fill values for parameters that are inactive under the sampled conditionals.
# The solver never reads an inactive parameter, but rows in the CSVs need a value.
INACTIVE_DEFAULTS = {
    "perturbation_kicks": 3,
    "reheat_interval": 20,
    "reheat_factor": 0.1,
}

PARAM_COLUMNS = [
    "initial_temperature", "cooling_rate", "min_temperature", "iterations_per_temp",
    "p_swap", "p_insert", "p_2opt", "p_oropt",
    "init_method", "restarts", "restart_strategy", "perturbation_kicks",
    "use_reheat", "reheat_interval", "reheat_factor",
]


def optuna_suggest_v2(trial) -> SAExtConfig:
    """Sample the v2 space with Optuna's define-by-run API (conditionals included)."""
    initial_temperature = trial.suggest_float("initial_temperature", 0.01, 5000.0, log=True)
    cooling_rate = trial.suggest_float("cooling_rate", 0.5, 0.9999)
    min_temperature = trial.suggest_float("min_temperature", 1e-6, 1.0, log=True)
    iterations_per_temp = trial.suggest_int("iterations_per_temp", 1, 500, log=True)
    p_swap = trial.suggest_float("p_swap", 0.0, 1.0)
    p_insert = trial.suggest_float("p_insert", 0.0, 1.0)
    p_2opt = trial.suggest_float("p_2opt", 0.0, 1.0)
    p_oropt = trial.suggest_float("p_oropt", 0.0, 1.0)
    init_method = trial.suggest_categorical("init_method", ["random", "nearest_neighbor"])
    restarts = trial.suggest_int("restarts", 0, 16)
    restart_strategy = trial.suggest_categorical("restart_strategy", ["fresh", "perturb_best"])
    if restart_strategy == "perturb_best":
        perturbation_kicks = trial.suggest_int("perturbation_kicks", 1, 8)
    else:
        perturbation_kicks = INACTIVE_DEFAULTS["perturbation_kicks"]
    use_reheat = trial.suggest_categorical("use_reheat", ["no", "yes"])
    if use_reheat == "yes":
        reheat_interval = trial.suggest_int("reheat_interval", 5, 100)
        reheat_factor = trial.suggest_float("reheat_factor", 0.001, 1.0, log=True)
    else:
        reheat_interval = INACTIVE_DEFAULTS["reheat_interval"]
        reheat_factor = INACTIVE_DEFAULTS["reheat_factor"]

    return SAExtConfig(
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        iterations_per_temp=iterations_per_temp,
        p_swap=p_swap, p_insert=p_insert, p_2opt=p_2opt, p_oropt=p_oropt,
        init_method=init_method,
        restarts=restarts,
        restart_strategy=restart_strategy,
        perturbation_kicks=perturbation_kicks,
        use_reheat=use_reheat,
        reheat_interval=reheat_interval,
        reheat_factor=reheat_factor,
    )


def build_configspace(seed: int):
    """The same space as a ConfigSpace object for SMAC (imports lazily so the
    Optuna-only arms run on machines without SMAC/ConfigSpace installed)."""
    from ConfigSpace import (Categorical, ConfigurationSpace, EqualsCondition,
                             Float, Integer)

    cs = ConfigurationSpace(seed=seed)
    initial_temperature = Float("initial_temperature", (0.01, 5000.0), log=True, default=10.0)
    cooling_rate = Float("cooling_rate", (0.5, 0.9999), default=0.95)
    min_temperature = Float("min_temperature", (1e-6, 1.0), log=True, default=1e-3)
    iterations_per_temp = Integer("iterations_per_temp", (1, 500), log=True, default=50)
    p_swap = Float("p_swap", (0.0, 1.0), default=0.25)
    p_insert = Float("p_insert", (0.0, 1.0), default=0.25)
    p_2opt = Float("p_2opt", (0.0, 1.0), default=0.25)
    p_oropt = Float("p_oropt", (0.0, 1.0), default=0.25)
    init_method = Categorical("init_method", ["random", "nearest_neighbor"], default="random")
    restarts = Integer("restarts", (0, 16), default=2)
    restart_strategy = Categorical("restart_strategy", ["fresh", "perturb_best"], default="fresh")
    perturbation_kicks = Integer("perturbation_kicks", (1, 8), default=3)
    use_reheat = Categorical("use_reheat", ["no", "yes"], default="no")
    reheat_interval = Integer("reheat_interval", (5, 100), default=20)
    reheat_factor = Float("reheat_factor", (0.001, 1.0), log=True, default=0.1)

    items = [initial_temperature, cooling_rate, min_temperature, iterations_per_temp,
             p_swap, p_insert, p_2opt, p_oropt,
             init_method, restarts, restart_strategy, perturbation_kicks,
             use_reheat, reheat_interval, reheat_factor,
             EqualsCondition(perturbation_kicks, restart_strategy, "perturb_best"),
             EqualsCondition(reheat_interval, use_reheat, "yes"),
             EqualsCondition(reheat_factor, use_reheat, "yes")]
    try:
        cs.add(items)                     # ConfigSpace >= 1.0
    except (TypeError, AttributeError):   # pragma: no cover - legacy 0.x API
        cs.add_hyperparameters(items[:15])
        cs.add_conditions(items[15:])
    return cs


def config_from_mapping(m) -> SAExtConfig:
    """Build an SAExtConfig from any dict-like mapping (plain dict, SMAC/ConfigSpace
    ``Configuration``, pandas row). Inactive conditional parameters fall back to
    ``INACTIVE_DEFAULTS``."""
    def get(name, default=None):
        try:
            v = m.get(name, default)
        except AttributeError:  # objects without .get (e.g. pandas rows use [])
            try:
                v = m[name]
            except KeyError:
                v = default
        return default if v is None else v

    return SAExtConfig(
        initial_temperature=float(get("initial_temperature")),
        cooling_rate=float(get("cooling_rate")),
        min_temperature=float(get("min_temperature")),
        iterations_per_temp=int(get("iterations_per_temp")),
        p_swap=float(get("p_swap")),
        p_insert=float(get("p_insert")),
        p_2opt=float(get("p_2opt")),
        p_oropt=float(get("p_oropt")),
        init_method=str(get("init_method")),
        restarts=int(get("restarts")),
        restart_strategy=str(get("restart_strategy")),
        perturbation_kicks=int(get("perturbation_kicks", INACTIVE_DEFAULTS["perturbation_kicks"])),
        use_reheat=str(get("use_reheat")),
        reheat_interval=int(get("reheat_interval", INACTIVE_DEFAULTS["reheat_interval"])),
        reheat_factor=float(get("reheat_factor", INACTIVE_DEFAULTS["reheat_factor"])),
    )


def config_key(config: SAExtConfig) -> tuple:
    """Hashable identity of a configuration (used for caching canonical evaluations
    and for grouping SMAC intensification calls by configuration)."""
    d = config.as_dict()
    return tuple(d[k] for k in PARAM_COLUMNS)
