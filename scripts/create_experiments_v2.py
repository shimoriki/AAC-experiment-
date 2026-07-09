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

import yaml

REPO = Path(__file__).resolve().parent.parent

SCALAR_KEYS = ["n_trials", "solver_repeats", "test_solver_repeats", "test_size",
               "n_cities", "max_steps", "tpe_startup_trials", "smac_initial_configs",
               "smac_max_config_calls", "smac_retrain_after"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="bovsrs", help="name of configs/<profile>.yaml")
    ap.add_argument("--data_dir", default="/local/anmol/datasets/tsp")
    ap.add_argument("--out_dir", default="results/v2/raw")
    ap.add_argument("--smac_scratch", default="results/v2/smac_output")
    ap.add_argument("--manifest", default=str(REPO / "slurm" / "experiments_v2.txt"))
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / f"{args.profile}.yaml").read_text())
    seeds = cfg.get("seeds", list(range(cfg["n_seeds"])))

    grid = list(itertools.product(cfg["optimizers"], cfg["train_families"],
                                  cfg["train_sizes"], seeds))
    # longest jobs first
    grid.sort(key=lambda g: (g[0] != "smac_aac", g[0] != "smac_bo"))

    scalars = " ".join(f"--{k} {cfg[k]}" for k in SCALAR_KEYS if k in cfg)
    lines = []
    for optimizer, family, size, seed in grid:
        lines.append(
            f"--optimizer {optimizer} --train_family {family} --train_size {size} "
            f"--seed {seed} {scalars} "
            f"--data_dir {args.data_dir} --out_dir {args.out_dir} "
            f"--smac_scratch {args.smac_scratch}")

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n")

    # xargs fallback for machines without SLURM
    fallback = REPO / "scripts" / "run_experiments_v2.sh"
    py = str(REPO / ".venv" / "bin" / "python")
    main_py = str(REPO / "scripts" / "main_v2.py")
    fallback.write_text("#!/usr/bin/env bash\n# Fallback without SLURM:  "
                        "tail -n +3 scripts/run_experiments_v2.sh | xargs -P 48 -I{} bash -c '{}'\n"
                        + "\n".join(f"{py} {main_py} {ln}" for ln in lines) + "\n")
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
