#!/usr/bin/env python
"""Verify that every run in a v2 manifest has one complete raw JSON sidecar."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aac_tsp.naming_v2 import run_id_v2  # noqa: E402


def _arguments(line: str) -> dict[str, str]:
    tokens = shlex.split(line)
    parsed: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--") and i + 1 < len(tokens):
            parsed[token[2:]] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return parsed


def _run_id(args: dict[str, str]) -> str:
    required = ("optimizer", "train_family", "train_size", "seed", "config_space")
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError(f"manifest line missing arguments: {missing}")
    return run_id_v2(args["optimizer"], args["train_family"], int(args["train_size"]),
                     int(args["seed"]), args["config_space"],
                     args.get("resampling", "full_train"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="slurm/experiments_v2.txt")
    ap.add_argument("--raw_dir", required=True)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    expected = {_run_id(_arguments(line)) for line in lines}
    if len(expected) != len(lines):
        print(f"ERROR: manifest contains duplicate run IDs ({len(lines)} lines, "
              f"{len(expected)} unique)")
        return 1

    complete: set[str] = set()
    invalid: list[str] = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append(f"{path}: {exc}")
            continue
        run_id = payload.get("run_id")
        if payload.get("status") == "complete" and run_id:
            complete.add(str(run_id))
        else:
            invalid.append(f"{path}: status={payload.get('status')!r}")

    missing = sorted(expected - complete)
    unexpected = sorted(complete - expected)
    print(f"Completion: expected={len(expected)} complete={len(expected & complete)} "
          f"missing={len(missing)} unexpected={len(unexpected)} invalid={len(invalid)}")
    if missing:
        print("Missing runs:")
        for run_id in missing:
            print(f"  {run_id}")
    if unexpected:
        print("Unexpected complete runs in raw_dir:")
        for run_id in unexpected:
            print(f"  {run_id}")
    if invalid:
        print("Invalid JSON sidecars:")
        for item in invalid:
            print(f"  {item}")
    return int(bool(missing or unexpected or invalid))


if __name__ == "__main__":
    raise SystemExit(main())
