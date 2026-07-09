#!/usr/bin/env python
"""Expand a profile (configs/<profile>.yaml) into scripts/run_experiments.sh.

Each line is one `python scripts/main.py ...` invocation. Run them with GNU parallel:
    cat scripts/run_experiments.sh | parallel -j 32
or sequentially with:  bash scripts/run_experiments.sh
"""
import argparse
import itertools
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke", help="name of configs/<profile>.yaml")
    ap.add_argument("--out", default=str(REPO / "scripts" / "run_experiments.sh"))
    ap.add_argument("--data_dir", default="/local/anmol/datasets/tsp")
    ap.add_argument("--out_dir", default=str(REPO / "results" / "raw"))
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / f"{args.profile}.yaml").read_text())
    # config_spaces is optional (defaults to ["full"] for older profiles). Each variant
    # writes to a per-variant subdir of out_dir so "full" and "fixed_2opt" never collide.
    config_spaces = cfg.get("config_spaces", ["full"])
    grid = itertools.product(cfg["optimizers"], cfg["resamplings"], cfg["families"],
                             cfg["train_sizes"], cfg["seeds"], config_spaces)
    py = str(REPO / ".venv" / "bin" / "python")
    main_py = str(REPO / "scripts" / "main.py")

    lines = []
    for optimizer, resampling, family, size, seed, cspace in grid:
        out_dir = f"{args.out_dir}/{cspace}"
        lines.append(
            f"{py} {main_py} --optimizer {optimizer} --resampling {resampling} "
            f"--config_space {cspace} "
            f"--train_family {family} --train_size {size} --seed {seed} "
            f"--bo_budget {cfg['bo_budget']} --n_cities {cfg['n_cities']} "
            f"--solver_repeats {cfg['solver_repeats']} --test_solver_repeats {cfg['test_solver_repeats']} "
            f"--test_size {cfg['test_size']} --max_steps {cfg['max_steps']} "
            f"--cv_repeats {cfg['cv_repeats']} --bootstrap_repeats {cfg['bootstrap_repeats']} "
            f"--data_dir {args.data_dir} --out_dir {out_dir}")

    out = Path(args.out)
    out.write_text("#!/usr/bin/env bash\nset -e\n" + "\n".join(lines) + "\n")
    out.chmod(0o755)
    print(f"Wrote {len(lines)} runs to {out} (profile={args.profile})")


if __name__ == "__main__":
    main()
