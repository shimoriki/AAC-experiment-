#!/usr/bin/env python
"""Run one v2 experiment cell (one configurator run) and write its outputs.

Thread env vars are pinned to 1 *before* importing numpy/numba so that SLURM array
tasks (one per core) do not oversubscribe the node.

Idempotent: if the run's sidecar JSON reports status=complete, the run is skipped
(use --force to re-run), so a failed/timed-out SLURM array can simply be resubmitted.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
from contextlib import contextmanager, redirect_stderr, redirect_stdout  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from aac_tsp.runner_v2 import OPTIMIZERS_V2, RunConfigV2, run_experiment_v2  # noqa: E402
from aac_tsp.resampling_v2 import RESAMPLINGS_V2  # noqa: E402
from aac_tsp.space_v2 import CONFIG_SPACES_V2  # noqa: E402


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def _stdio_log(path: str | None):
    """Mirror stdout/stderr to a per-run log while retaining console feedback."""
    if not path:
        yield
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as fh:
        out, err = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)
        with redirect_stdout(out), redirect_stderr(err):
            yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=list(OPTIMIZERS_V2), required=True)
    ap.add_argument("--train_family", choices=["uniform", "clustered", "mixed"], default="mixed")
    ap.add_argument("--train_size", type=int, default=40)
    ap.add_argument("--n_trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config_space", choices=CONFIG_SPACES_V2, default="full")
    ap.add_argument("--resampling", choices=RESAMPLINGS_V2, default="full_train")
    ap.add_argument("--n_cities", type=int, default=50)
    ap.add_argument("--solver_repeats", type=int, default=1)
    ap.add_argument("--test_solver_repeats", type=int, default=2)
    ap.add_argument("--test_size", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=30_000)
    ap.add_argument("--tpe_startup_trials", type=int, default=25)
    ap.add_argument("--smac_initial_configs", type=int, default=25)
    ap.add_argument("--smac_max_config_calls", type=int, default=0)
    ap.add_argument("--smac_retrain_after", type=int, default=8)
    ap.add_argument("--cv_repeats", type=int, default=3)
    ap.add_argument("--bootstrap_repeats", type=int, default=10)
    ap.add_argument("--progress_interval", type=int, default=20)
    ap.add_argument("--data_dir", default="/local/rohit/datasets/tsp")
    ap.add_argument("--out_dir", default="results/v2/raw")
    ap.add_argument("--smac_scratch", default="results/v2/smac_output")
    ap.add_argument("--log_file", default=None,
                    help="mirror stdout/stderr to this per-run log")
    ap.add_argument("--force", action="store_true", help="re-run even if already complete")
    args = ap.parse_args()

    force, log_file = args.force, args.log_file
    del args.force
    del args.log_file
    cfg = RunConfigV2(**vars(args))

    with _stdio_log(log_file):
        sidecar = Path(cfg.out_dir) / f"{cfg.run_id}.json"
        if not force and sidecar.exists():
            try:
                if json.loads(sidecar.read_text()).get("status") == "complete":
                    print(f"[{cfg.run_id}] already complete -> {sidecar}", flush=True)
                    return
            except (json.JSONDecodeError, OSError):
                pass  # unreadable sidecar: re-run

        failure_path = (Path(cfg.out_dir).parent / "failures" / cfg.run_id /
                        "FAILED.json")
        started = datetime.now().astimezone()
        t0 = time.perf_counter()
        print(
            f"[{cfg.run_id}] START optimizer={cfg.optimizer} "
            f"resampling={cfg.resampling} "
            f"train_family={cfg.train_family} train_size={cfg.train_size} "
            f"seed={cfg.seed} trials={cfg.n_trials} config_space={cfg.config_space} "
            f"start_time={started.isoformat()} output_json={sidecar}",
            flush=True,
        )
        try:
            path = run_experiment_v2(cfg)
        except Exception as exc:
            ended = datetime.now().astimezone()
            traceback_text = traceback.format_exc()
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps({
                "status": "failed",
                "run_id": cfg.run_id,
                "run_config": asdict(cfg),
                "start_time": started.isoformat(),
                "end_time": ended.isoformat(),
                "elapsed_sec": time.perf_counter() - t0,
                "output_json": str(sidecar),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback_text,
            }, indent=2), encoding="utf-8")
            print(
                f"[{cfg.run_id}] FAILED end_time={ended.isoformat()} "
                f"failure_json={failure_path}",
                file=sys.stderr,
                flush=True,
            )
            print(traceback_text, file=sys.stderr, flush=True)
            raise SystemExit(1)

        if failure_path.exists():
            failure_path.unlink()
        ended = datetime.now().astimezone()
        print(
            f"[{cfg.run_id}] END optimizer={cfg.optimizer} "
            f"resampling={cfg.resampling} "
            f"train_family={cfg.train_family} train_size={cfg.train_size} "
            f"seed={cfg.seed} trials={cfg.n_trials} end_time={ended.isoformat()} "
            f"elapsed_sec={time.perf_counter() - t0:.1f} output_json={path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
