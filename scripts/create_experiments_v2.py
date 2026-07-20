#!/usr/bin/env python
"""Expand a v2 profile (configs/<profile>.yaml) into the SLURM manifest.

Writes:
  slurm/experiments_v2.txt      one argument line per run, consumed by the SLURM
                                array job (slurm/02_run_array.sbatch) line-by-line
  scripts/run_experiments_v2.sh xargs fallback for machines without SLURM

smac_aac runs are listed first: they are by far the longest jobs, so starting them
early keeps the array's makespan short.
"""
import argparse
import itertools
from pathlib import Path
import sys

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aac_tsp.naming_v2 import run_id_v2  # noqa: E402

SCALAR_KEYS = ["n_trials", "solver_repeats", "test_solver_repeats", "test_size",
               "n_cities", "max_steps", "tpe_startup_trials", "smac_initial_configs",
               "smac_max_config_calls", "smac_retrain_after", "progress_interval",
               "cv_repeats", "bootstrap_repeats"]
CONDITION_KEYS = {"train_family", "train_size", "config_space"}


def expand_conditions(cfg: dict) -> list[dict]:
    """Return condition mappings, including optional per-condition scalars.

    A scalar such as ``n_trials`` may be overridden inside a condition.  This is
    needed for training-size comparisons with an equal SA-call cap, because the
    reference proposal count must shrink as ``train_size`` grows.
    """
    if "conditions" in cfg:
        conditions = []
        for i, condition in enumerate(cfg["conditions"]):
            missing = CONDITION_KEYS - set(condition)
            if missing:
                raise ValueError(f"condition {i} is missing keys: {sorted(missing)}")
            unknown = set(condition) - CONDITION_KEYS - set(SCALAR_KEYS)
            if unknown:
                raise ValueError(f"condition {i} has unknown keys: {sorted(unknown)}")
            normalized = dict(condition)
            normalized["train_family"] = str(condition["train_family"])
            normalized["train_size"] = int(condition["train_size"])
            normalized["config_space"] = str(condition["config_space"])
            conditions.append(normalized)
        identities = [tuple(condition[k] for k in
                            ("train_family", "train_size", "config_space"))
                      for condition in conditions]
        if len(set(identities)) != len(identities):
            raise ValueError("profile contains duplicate conditions")
        return conditions

    spaces = cfg.get("config_spaces")
    if spaces is None:
        spaces = [cfg.get("config_space", "full")]
    return [
        {"train_family": str(family), "train_size": int(size),
         "config_space": str(space)}
        for family, size, space in
        itertools.product(cfg["train_families"], cfg["train_sizes"], spaces)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="bovsrs_resampling_signal",
                    help="name of configs/<profile>.yaml (default: factorial signal profile)")
    ap.add_argument("--data_dir", default="/local/rohit/datasets/tsp")
    ap.add_argument("--out_dir", default="results/v2/raw")
    ap.add_argument("--smac_scratch", default="results/v2/smac_output")
    ap.add_argument("--manifest", default=str(REPO / "slurm" / "experiments_v2.txt"))
    ap.add_argument("--fallback", default=str(REPO / "scripts" / "run_experiments_v2.sh"),
                    help="path for the generated non-SLURM runner")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / f"{args.profile}.yaml").read_text())
    seeds = cfg["seeds"] if "seeds" in cfg else list(range(cfg["n_seeds"]))

    conditions = expand_conditions(cfg)
    resamplings = cfg.get("resamplings", [cfg.get("resampling", "full_train")])
    grid = [(optimizer, resampling, condition, seed)
            for optimizer in cfg["optimizers"]
            for resampling in resamplings
            for condition in conditions
            for seed in seeds]
    # longest jobs first
    grid.sort(key=lambda g: (g[0] != "smac_aac", g[0] != "smac_bo"))

    log_dir = Path(args.out_dir).parent / "logs"
    lines = []
    for optimizer, resampling, condition, seed in grid:
        family = condition["train_family"]
        size = condition["train_size"]
        space = condition["config_space"]
        scalar_values = {
            key: condition[key] if key in condition else cfg[key]
            for key in SCALAR_KEYS if key in condition or key in cfg
        }
        scalars = " ".join(f"--{key} {value}"
                           for key, value in scalar_values.items())
        run_id = run_id_v2(optimizer, family, size, seed, space, resampling)
        log_file = (log_dir / (run_id + ".log")).as_posix()
        lines.append(
            f"--optimizer {optimizer} --resampling {resampling} "
            f"--train_family {family} --train_size {size} "
            f"--seed {seed} --config_space {space} {scalars} "
            f"--data_dir {args.data_dir} --out_dir {args.out_dir} "
            f"--smac_scratch {args.smac_scratch} "
            f"--log_file {log_file}")

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n", newline="\n")

    # xargs fallback for machines without SLURM
    fallback = Path(args.fallback)
    fallback.parent.mkdir(parents=True, exist_ok=True)
    # Keep the generated fallback portable: launchers always cd to the repo root,
    # and an archive generated on Windows must still work after extraction on Linux.
    py = ".venv/bin/python"
    main_py = "scripts/main_v2.py"
    fallback.write_text(
        "#!/usr/bin/env bash\n# Fallback without SLURM:  "
        "tail -n +3 scripts/run_experiments_v2.sh | xargs -P 48 -I{} bash -c '{}'\n"
        + "\n".join(f"{py} {main_py} {ln}" for ln in lines) + "\n",
        newline="\n",
    )
    try:
        fallback.chmod(0o755)
    except OSError:
        pass

    n = len(lines)
    print(f"Wrote {n} runs to {manifest} (profile={args.profile})")
    print(f"Submit with:  sbatch --array=0-{n - 1}%64 slurm/02_run_array.sbatch")
    print("Or simply:    bash slurm/submit_all.sh")


if __name__ == "__main__":
    main()
