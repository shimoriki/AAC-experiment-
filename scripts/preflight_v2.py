#!/usr/bin/env python
"""Static/environment checks for the v2 configurator × resampling profiles.

This command does not run SA or create result files. On a fresh Linux clone it
bootstraps the repository virtualenv, then verifies the SMAC stack. Windows is
intentionally limited to the Optuna arms because pyrfr/SMAC is not supported
there by this project.
"""

from __future__ import annotations

import importlib
import itertools
import os
import platform
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO / ".venv" / "bin" / "python"
BOOTSTRAP_GUARD = "AAC_V2_PREFLIGHT_BOOTSTRAPPED"


def _bootstrap_and_reexec(reason: str) -> None:
    """Create/repair the Linux venv once, then restart this exact preflight."""
    if os.environ.get("AAC_V2_NO_BOOTSTRAP") == "1":
        print(
            f"PREFLIGHT NEEDS SETUP: {reason}.\n"
            "Run: bash slurm/setup_env.sh",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if os.environ.get(BOOTSTRAP_GUARD) == "1":
        print(
            f"PREFLIGHT SETUP FAILED: {reason} remains after setup.\n"
            "Inspect the setup output above, then rerun: bash slurm/setup_env.sh",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"PREFLIGHT: {reason}; running one-time environment setup...", flush=True)
    env = dict(os.environ)
    env[BOOTSTRAP_GUARD] = "1"
    completed = subprocess.run(
        ["bash", str(REPO / "slurm" / "setup_env.sh")],
        cwd=REPO,
        env=env,
        check=False,
    )
    if completed.returncode != 0 or not VENV_PYTHON.is_file():
        print(
            "PREFLIGHT SETUP FAILED. Fix the setup error above and rerun:\n"
            "  bash slurm/setup_env.sh",
            file=sys.stderr,
        )
        raise SystemExit(completed.returncode or 2)
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )

# On Linux the v2 environment is authoritative. Re-enter it automatically when
# it exists, and bootstrap it on a fresh clone. This must
# happen before importing PyYAML/numpy/the project package so a bare system
# Python never fails with an opaque ModuleNotFoundError.
if platform.system() != "Windows":
    if not VENV_PYTHON.is_file():
        _bootstrap_and_reexec(".venv/bin/python is missing")
    # Do not compare resolve() results: venv/bin/python is commonly a symlink to
    # /usr/bin/python and would then look identical to a bare system interpreter.
    if Path(sys.executable).absolute().parent != VENV_PYTHON.absolute().parent:
        os.execv(str(VENV_PYTHON),
                 [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

try:
    import yaml
except ModuleNotFoundError as exc:
    if platform.system() != "Windows":
        _bootstrap_and_reexec(f"dependency {exc.name!r} is missing")
    print(f"PREFLIGHT NEEDS SETUP: dependency {exc.name!r} is missing.",
          file=sys.stderr)
    raise SystemExit(2) from None

sys.path.insert(0, str(REPO / "src"))

try:
    from aac_tsp.runner_v2 import (OPTIMIZERS_V2, PRIMARY_TEST_FAMILY_V2,
                                   RunConfigV2)
    from aac_tsp.naming_v2 import run_id_v2
    from aac_tsp.postprocess_v2 import CHECKPOINT_FRACTIONS, TEST_METRICS
    from aac_tsp.resampling_v2 import (RESAMPLINGS_V2,
                                       build_resampling_plan_v2)
    from aac_tsp.space_v2 import (active_param_columns, build_configspace,
                                  config_from_mapping, optuna_suggest_v2)
except ModuleNotFoundError as exc:
    if platform.system() != "Windows":
        _bootstrap_and_reexec(f"dependency {exc.name!r} is missing")
    print(f"PREFLIGHT NEEDS SETUP: dependency {exc.name!r} is missing.",
          file=sys.stderr)
    raise SystemExit(2) from None

PROFILES = {
    "bovsrs_resampling_signal": {
        "optimizers": list(OPTIMIZERS_V2),
        "resamplings": ["holdout", "cv5", "repeated_cv5", "bootstrap_oob"],
        "train_families": ["mixed"],
        "train_sizes": [25],
        "seeds": list(range(12)),
        "n_trials": 200,
        "max_steps": 10_000,
        "solver_repeats": 1,
        "test_solver_repeats": 3,
        "test_size": 100,
        "n_cities": 50,
        "tpe_startup_trials": 15,
        "smac_initial_configs": 15,
        "smac_retrain_after": 8,
        "smac_max_config_calls": 8,
        "cv_repeats": 3,
        "bootstrap_repeats": 10,
        "progress_interval": 20,
        "config_space": "large_fixed_2opt",
        "expected_runs": 192,
    },
    "bovsrs_resampling_train_sizes": {
        "optimizers": list(OPTIMIZERS_V2),
        "resamplings": ["holdout", "cv5", "repeated_cv5", "bootstrap_oob"],
        "conditions": [
            {"train_family": "mixed", "train_size": 10,
             "config_space": "large_fixed_2opt", "n_trials": 500},
            {"train_family": "mixed", "train_size": 25,
             "config_space": "large_fixed_2opt", "n_trials": 200},
            {"train_family": "mixed", "train_size": 50,
             "config_space": "large_fixed_2opt", "n_trials": 100},
        ],
        "seeds": list(range(12)),
        "max_steps": 10_000,
        "solver_repeats": 1,
        "test_solver_repeats": 3,
        "test_size": 100,
        "n_cities": 50,
        "tpe_startup_trials": 15,
        "smac_initial_configs": 15,
        "smac_retrain_after": 8,
        "smac_max_config_calls": 8,
        "cv_repeats": 3,
        "bootstrap_repeats": 10,
        "progress_interval": 20,
        "expected_runs": 576,
        "expected_budget_calls": 5_000,
    },
    "bovsrs_resampling_train_sizes_smoke": {
        "optimizers": list(OPTIMIZERS_V2),
        "resamplings": ["holdout", "cv5", "repeated_cv5", "bootstrap_oob"],
        "conditions": [
            {"train_family": "mixed", "train_size": 10,
             "config_space": "large_fixed_2opt", "n_trials": 50},
            {"train_family": "mixed", "train_size": 25,
             "config_space": "large_fixed_2opt", "n_trials": 20},
            {"train_family": "mixed", "train_size": 50,
             "config_space": "large_fixed_2opt", "n_trials": 10},
        ],
        "seeds": [0],
        "max_steps": 1_000,
        "solver_repeats": 1,
        "test_solver_repeats": 1,
        "test_size": 10,
        "n_cities": 50,
        "tpe_startup_trials": 4,
        "smac_initial_configs": 4,
        "smac_retrain_after": 4,
        "smac_max_config_calls": 4,
        "cv_repeats": 3,
        "bootstrap_repeats": 10,
        "progress_interval": 5,
        "expected_runs": 48,
        "expected_budget_calls": 500,
    },
    "bovsrs_resampling_conditions": {
        "optimizers": list(OPTIMIZERS_V2),
        "resamplings": ["holdout", "cv5", "repeated_cv5", "bootstrap_oob"],
        "conditions": [
            {"train_family": "mixed", "train_size": 8,
             "config_space": "large_fixed_2opt"},
            {"train_family": "mixed", "train_size": 25,
             "config_space": "large_fixed_2opt"},
            {"train_family": "mixed", "train_size": 25,
             "config_space": "full"},
            {"train_family": "uniform", "train_size": 25,
             "config_space": "large_fixed_2opt"},
        ],
        "seeds": list(range(8)),
        "n_trials": 150,
        "max_steps": 10_000,
        "solver_repeats": 1,
        "test_solver_repeats": 3,
        "test_size": 100,
        "n_cities": 50,
        "tpe_startup_trials": 15,
        "smac_initial_configs": 15,
        "smac_retrain_after": 8,
        "smac_max_config_calls": 8,
        "cv_repeats": 3,
        "bootstrap_repeats": 10,
        "progress_interval": 20,
        "expected_runs": 512,
    },
    "bovsrs_resampling_smoke": {
        "optimizers": list(OPTIMIZERS_V2),
        "resamplings": ["holdout", "cv5", "repeated_cv5", "bootstrap_oob"],
        "train_families": ["mixed"],
        "train_sizes": [8],
        "seeds": [0],
        "n_trials": 8,
        "max_steps": 1_000,
        "solver_repeats": 1,
        "test_solver_repeats": 1,
        "test_size": 10,
        "n_cities": 50,
        "tpe_startup_trials": 4,
        "smac_initial_configs": 4,
        "smac_retrain_after": 4,
        "smac_max_config_calls": 4,
        "cv_repeats": 3,
        "bootstrap_repeats": 10,
        "progress_interval": 4,
        "config_space": "large_fixed_2opt",
        "expected_runs": 16,
    },
    "bovsrs_signal": {
        "optimizers": list(OPTIMIZERS_V2),
        "train_families": ["mixed"],
        "train_sizes": [25],
        "seeds": list(range(12)),
        "n_trials": 120,
        "max_steps": 10_000,
        "solver_repeats": 1,
        "test_solver_repeats": 3,
        "test_size": 100,
        "n_cities": 50,
        "tpe_startup_trials": 15,
        "smac_initial_configs": 15,
        "smac_retrain_after": 8,
        "smac_max_config_calls": 8,
        "config_space": "large_fixed_2opt",
        "expected_runs": 48,
    },
    "bovsrs_generalization_check": {
        "optimizers": ["random", "optuna_tpe"],
        "train_families": ["mixed"],
        "train_sizes": [8, 25, 50],
        "seeds": list(range(8)),
        "n_trials": 100,
        "max_steps": 10_000,
        "solver_repeats": 1,
        "test_solver_repeats": 3,
        "test_size": 100,
        "n_cities": 50,
        "config_space": "large_fixed_2opt",
        "expected_runs": 48,
    },
    "bovsrs_conditions": {
        "optimizers": list(OPTIMIZERS_V2),
        "conditions": [
            {"train_family": "mixed", "train_size": 8,
             "config_space": "large_fixed_2opt"},
            {"train_family": "mixed", "train_size": 25,
             "config_space": "large_fixed_2opt"},
            {"train_family": "mixed", "train_size": 50,
             "config_space": "large_fixed_2opt"},
            {"train_family": "mixed", "train_size": 25,
             "config_space": "full"},
            {"train_family": "uniform", "train_size": 25,
             "config_space": "large_fixed_2opt"},
            {"train_family": "clustered", "train_size": 25,
             "config_space": "large_fixed_2opt"},
        ],
        "seeds": list(range(20)),
        "n_trials": 150,
        "max_steps": 10_000,
        "solver_repeats": 1,
        "test_solver_repeats": 3,
        "test_size": 100,
        "n_cities": 50,
        "tpe_startup_trials": 20,
        "smac_initial_configs": 20,
        "smac_retrain_after": 8,
        "smac_max_config_calls": 8,
        "progress_interval": 25,
        "expected_runs": 480,
    },
    "bovsrs_conditions_smoke": {
        "optimizers": list(OPTIMIZERS_V2),
        "conditions": [
            {"train_family": "mixed", "train_size": 8,
             "config_space": "large_fixed_2opt"},
            {"train_family": "mixed", "train_size": 8,
             "config_space": "full"},
        ],
        "seeds": [0, 1],
        "n_trials": 12,
        "max_steps": 2_000,
        "solver_repeats": 1,
        "test_solver_repeats": 1,
        "test_size": 10,
        "n_cities": 50,
        "tpe_startup_trials": 4,
        "smac_initial_configs": 4,
        "smac_retrain_after": 8,
        "smac_max_config_calls": 4,
        "progress_interval": 4,
        "expected_runs": 16,
    },
}


class _FixedTrial:
    """Minimal Optuna-like trial used to check which parameters are requested."""

    def __init__(self):
        self.names: list[str] = []

    def suggest_float(self, name, low, high, **kwargs):
        self.names.append(name)
        defaults = {
            "initial_temperature": 10.0,
            "cooling_rate": 0.95,
            "min_temperature": 1e-3,
            "reheat_factor": 0.1,
        }
        return defaults.get(name, (low + high) / 2)

    def suggest_int(self, name, low, high, **kwargs):
        self.names.append(name)
        return max(low, min(high, 2))

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        defaults = {"init_method": "random", "restart_strategy": "fresh",
                    "use_reheat": "no"}
        return defaults.get(name, choices[0])


def _check_profiles(errors: list[str]):
    for name, expected in PROFILES.items():
        path = REPO / "configs" / f"{name}.yaml"
        if not path.exists():
            errors.append(f"missing profile: {path}")
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if key in {"expected_runs", "expected_budget_calls"}:
                continue
            if cfg.get(key) != value:
                errors.append(f"{name}: {key}={cfg.get(key)!r}, expected {value!r}")
        seeds = cfg["seeds"] if "seeds" in cfg else list(range(cfg["n_seeds"]))
        if "conditions" in cfg:
            condition_records = cfg["conditions"]
        else:
            spaces = cfg.get("config_spaces", [cfg.get("config_space", "full")])
            condition_records = [
                {"train_family": family, "train_size": size,
                 "config_space": space}
                for family, size, space in itertools.product(
                    cfg["train_families"], cfg["train_sizes"], spaces)
            ]
        conditions = [(c["train_family"], int(c["train_size"]), c["config_space"])
                      for c in condition_records]
        resamplings = cfg.get("resamplings", [cfg.get("resampling", "full_train")])
        count = len(cfg["optimizers"]) * len(resamplings) * len(conditions) * len(seeds)
        if count != expected["expected_runs"]:
            errors.append(f"{name}: expands to {count}, expected {expected['expected_runs']}")
        else:
            print(f"OK profile {name}: {count} runs")
        run_ids = {
            run_id_v2(optimizer, family, size, seed, space, resampling)
            for optimizer in cfg["optimizers"]
            for resampling in resamplings
            for family, size, space in conditions
            for seed in seeds
        }
        if len(run_ids) != count:
            errors.append(f"{name}: run IDs collide ({len(run_ids)} unique for {count} runs)")
        if "expected_budget_calls" in expected:
            budgets = []
            for condition in condition_records:
                n_trials = int(condition.get("n_trials", cfg.get("n_trials", 0)))
                solver_repeats = int(condition.get(
                    "solver_repeats", cfg.get("solver_repeats", 1)))
                budget = n_trials * int(condition["train_size"]) * solver_repeats
                budgets.append(budget)
                if budget != expected["expected_budget_calls"]:
                    errors.append(
                        f"{name}: n={condition['train_size']} has {budget} SA calls, "
                        f"expected {expected['expected_budget_calls']}"
                    )
            if all(budget == expected["expected_budget_calls"] for budget in budgets):
                print(f"OK profile {name}: equal {expected['expected_budget_calls']} SA-call cap")


def _check_space(errors: list[str]):
    full_trial = _FixedTrial()
    optuna_suggest_v2(full_trial, "full")
    if not {"p_swap", "p_insert", "p_2opt", "p_oropt"}.issubset(full_trial.names):
        errors.append("full Optuna space no longer exposes all move weights")
    if len(active_param_columns("full")) != 15:
        errors.append("full v2 space no longer has 15 active parameters")

    trial = _FixedTrial()
    cfg = optuna_suggest_v2(trial, "large_fixed_2opt")
    moves = (cfg.p_swap, cfg.p_insert, cfg.p_2opt, cfg.p_oropt)
    if moves != (0.0, 0.0, 1.0, 0.0):
        errors.append(f"Optuna fixed-2opt move weights are {moves!r}")
    if any(name.startswith("p_") for name in trial.names):
        errors.append("Optuna fixed-2opt space still proposes move weights")
    active = active_param_columns("large_fixed_2opt")
    if any(name.startswith("p_") for name in active) or len(active) != 11:
        errors.append(f"fixed-2opt active parameter list is invalid: {active}")

    mapping = {name: value for name, value in cfg.as_dict().items()
               if not name.startswith("p_")}
    mapped = config_from_mapping(mapping, "large_fixed_2opt")
    if (mapped.p_swap, mapped.p_insert, mapped.p_2opt, mapped.p_oropt) != moves:
        errors.append("mapping view does not restore pure 2-opt weights")

    try:
        cs = build_configspace(0, "large_fixed_2opt")
    except ImportError as exc:
        if platform.system() == "Windows":
            print(f"WARN ConfigSpace unavailable on Windows: {exc}")
        else:
            errors.append(f"ConfigSpace unavailable: {exc}")
    else:
        names = set(cs.keys())
        if names.intersection({"p_swap", "p_insert", "p_2opt", "p_oropt"}):
            errors.append("SMAC fixed-2opt space still exposes move weights")
        if len(names) != 11:
            errors.append(f"SMAC fixed-2opt space has {len(names)} parameters, expected 11")
        full_names = set(build_configspace(0, "full").keys())
        if len(full_names) != 15 or not {
                "p_swap", "p_insert", "p_2opt", "p_oropt"}.issubset(full_names):
            errors.append("SMAC full space no longer exposes all 15 parameters")
        print("OK fixed-2opt views: Optuna, mapping, and ConfigSpace agree")


def _check_environment(errors: list[str]):
    for module in ("numpy", "pandas", "scipy", "yaml", "optuna", "numba"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            errors.append(f"missing Python dependency {module}: {exc}")
    for module in ("ConfigSpace", "smac"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            if platform.system() == "Windows":
                print(f"WARN {module} unavailable on Windows (expected): {exc}")
            else:
                errors.append(f"missing v2 dependency {module}: {exc}")

    data_dir = Path(os.environ.get("DATA_DIR", "/local/rohit/datasets/tsp"))
    candidates = [data_dir / "best_known" / "best_known.csv"]
    candidates.extend(
        data_dir / "instances" / f"{family}_50_{role}.npz"
        for family in ("uniform", "clustered", "mixed")
        for role in ("train", "test")
    )
    def is_readable_file(path: Path) -> bool:
        try:
            return path.is_file() and os.access(path, os.R_OK)
        except OSError:
            return False

    if all(is_readable_file(path) for path in candidates):
        import pandas as pd
        try:
            references = pd.read_csv(candidates[0])
            unique_ids = references["instance_id"].nunique()
        except (OSError, KeyError, pd.errors.ParserError) as exc:
            errors.append(f"could not validate best-known table {candidates[0]}: {exc}")
        else:
            if len(references) != 750 or unique_ids != 750:
                errors.append(
                    f"best-known table is incomplete at {candidates[0]}: "
                    f"rows={len(references)}, unique_ids={unique_ids} (expected 750)"
                )
            else:
                print(
                    f"OK complete TSP dataset detected at {data_dir} "
                    "(6 pools, 750 references)"
                )
    else:
        print(f"WARN TSP dataset missing/incomplete at {data_dir}; "
              "the launcher will generate it")


def _check_budget(errors: list[str]):
    cfg = RunConfigV2("random", "mixed", 25, 120, 0,
                      config_space="large_fixed_2opt", solver_repeats=1)
    if cfg.total_budget_calls != 3_000:
        errors.append(f"budget invariant failed: got {cfg.total_budget_calls}, expected 3000")
    else:
        print("OK equal-budget unit: 120 x 25 x 1 = 3000 optimization SA calls")
    train_ids = [f"train_{i}" for i in range(25)]
    expected_calls = {"holdout": 5, "cv5": 25, "repeated_cv5": 75}
    for method in RESAMPLINGS_V2:
        plan = build_resampling_plan_v2(train_ids, method, 0)
        calls = plan.calls_per_config(1)
        if calls < 1:
            errors.append(f"resampling plan {method} has no target calls")
        if method in expected_calls and calls != expected_calls[method]:
            errors.append(
                f"resampling plan {method} costs {calls}, expected {expected_calls[method]}"
            )
    if not errors:
        print("OK v2 resampling plans: holdout, 5-fold instances, repeated folds, bootstrap OOB")
    if PRIMARY_TEST_FAMILY_V2 != "mixed":
        errors.append(
            "v2 primary test family must stay fixed to mixed for cross-family contrasts"
        )
    else:
        print("OK primary unseen-test distribution: mixed for every training family")


def _check_workflow_contract(errors: list[str]):
    """Static guardrails for requirements established during the VM bring-up."""
    required_files = (
        "EXPERIMENT_COVERAGE.md",
        "configs/bovsrs_signal.yaml",
        "configs/bovsrs_generalization_check.yaml",
        "configs/bovsrs_conditions.yaml",
        "configs/bovsrs_conditions_smoke.yaml",
        "configs/bovsrs_resampling_signal.yaml",
        "configs/bovsrs_resampling_train_sizes.yaml",
        "configs/bovsrs_resampling_train_sizes_smoke.yaml",
        "configs/bovsrs_resampling_conditions.yaml",
        "configs/bovsrs_resampling_smoke.yaml",
        "scripts/check_v2_completion.py",
        "slurm/setup_env.sh",
        "slurm/run_local.sh",
    )
    for relative in required_files:
        if not (REPO / relative).is_file():
            errors.append(f"missing required workflow file: {relative}")

    expected_metrics = {
        "final_test_cost", "auc_test", "abs_generalization_gap",
        "final_relative_overtuning",
    }
    if not expected_metrics.issubset(TEST_METRICS):
        errors.append(
            f"analysis metrics lost required entries: {sorted(expected_metrics - set(TEST_METRICS))}"
        )
    if tuple(CHECKPOINT_FRACTIONS) != (0.25, 0.50, 0.75, 1.00):
        errors.append(f"budget checkpoints changed unexpectedly: {CHECKPOINT_FRACTIONS}")

    sources = {
        relative: (REPO / relative).read_text(encoding="utf-8")
        for relative in (
            "src/aac_tsp/runner_v2.py",
            "src/aac_tsp/postprocess_v2.py",
            "src/aac_tsp/plotting_v2.py",
            "scripts/main_v2.py",
            "slurm/setup_env.sh",
            "slurm/run_local.sh",
        )
    }
    runner = sources["src/aac_tsp/runner_v2.py"]
    feedback_lines = [
        line.strip() for line in runner.splitlines()
        if ".tell(" in line or "TrialValue(cost=" in line
    ]
    if any("test" in line.lower() for line in feedback_lines):
        errors.append(f"test value appears in configurator feedback: {feedback_lines}")
    if "study.tell(trial, cost)" not in runner:
        errors.append("Optuna feedback contract changed; expected validation variable `cost`")
    if '"test_cost": entry[f"test_cost_{PRIMARY_TEST_FAMILY_V2}"]' not in runner:
        errors.append("trajectory primary test cost is no longer tied to PRIMARY_TEST_FAMILY_V2")

    postprocess = sources["src/aac_tsp/postprocess_v2.py"]
    for output in ("budget_checkpoint_summary.csv", "condition_contrasts.csv",
                   "paired_vs_random.csv", "paired_resampling_tests.csv",
                   "configurator_resampling_interactions.csv",
                   "parameter_summary.csv"):
        if output not in postprocess:
            errors.append(f"post-processing output contract missing {output}")

    plotting = sources["src/aac_tsp/plotting_v2.py"]
    for output in (
        "trajectory_validation_vs_test.png",
        "trajectories_by_resampling.png",
        "ecdf_relative_overtuning.png",
        "relative_overtuning_by_train_size.png",
        "overtuning_frequency_by_resampling.png",
        "final_test_by_resampling.png",
        "generalization_gap_by_train_size.png",
        "runtime_by_resampling.png",
        "resampling_quality_runtime_tradeoff.png",
        "selected_params_stability.png",
        "composition_heatmap.png",
        "composition_by_resampling.png",
        "optimizer_comparison.png",
        "optimizer_comparison_by_resampling",
        "configurator_comparison.png",
        "configurator_comparison_by_resampling",
        "overtuning_frequency_over_budget_by_train_size.png",
        "relative_overtuning_over_budget_by_train_size.png",
        "relative_overtuning_eligibility_over_budget_by_train_size.png",
    ):
        if output not in plotting:
            errors.append(f"v2 plotting contract missing {output}")

    main_source = sources["scripts/main_v2.py"]
    for token in ("--log_file", "START optimizer=", "FAILED end_time=", "END optimizer="):
        if token not in main_source:
            errors.append(f"per-run logging contract missing {token!r}")

    setup_lines = sources["slurm/setup_env.sh"].splitlines()
    if any(line.strip().startswith("sudo ") for line in setup_lines):
        errors.append("setup_env.sh contains an executable sudo command")
    if "python3 -m virtualenv .venv" not in sources["slurm/setup_env.sh"]:
        errors.append("setup_env.sh lost its rootless virtualenv fallback")

    launcher = sources["slurm/run_local.sh"]
    for token in ('PROFILE="${PROFILE:-bovsrs_resampling_signal}"',
                  'DATA_DIR="${DATA_DIR:-/local/rohit/datasets/tsp}"',
                  "check_v2_completion.py", "xargs"):
        if token not in launcher:
            errors.append(f"local launcher contract missing {token!r}")

    for relative in (
        "src/aac_tsp/runner.py", "src/aac_tsp/runner_v2.py",
        "scripts/main.py", "scripts/main_v2.py",
        "scripts/create_experiments.py", "scripts/create_experiments_v2.py",
        "scripts/run_final.sh", "slurm/run_local.sh", "slurm/submit_all.sh",
    ):
        text = (REPO / relative).read_text(encoding="utf-8")
        if "/local/anmol" in text:
            errors.append(f"active VM path still points to /local/anmol: {relative}")

    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    ignored = {line.strip().lstrip("\ufeff") for line in gitignore}
    for entry in (".venv/", "venv/", "data/", "results/", "logs/"):
        if entry not in ignored:
            errors.append(f"large/generated path is no longer ignored: {entry}")

    if not errors:
        print("OK complete workflow contract: paths, logging, analysis, rootless setup, isolation")


def main() -> int:
    errors: list[str] = []
    _check_profiles(errors)
    _check_environment(errors)
    _check_space(errors)
    _check_budget(errors)
    _check_workflow_contract(errors)
    for path in ("slurm/run_local.sh", "scripts/create_experiments_v2.py",
                 "scripts/check_v2_completion.py"):
        if not (REPO / path).exists():
            errors.append(f"missing required command file: {path}")

    if errors:
        print("\nPREFLIGHT FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nPREFLIGHT PASSED: v2 profiles are ready; no experiment was run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
