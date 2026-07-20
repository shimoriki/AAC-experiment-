"""Stable run identifiers for v2 experiment cells."""

from __future__ import annotations


def run_id_v2(optimizer: str, train_family: str, train_size: int, seed: int,
              config_space: str = "full", resampling: str = "full_train") -> str:
    """Return a collision-safe ID while preserving completed signal-run names.

    The original signal profile used ``large_fixed_2opt`` without a space suffix.
    Keep those IDs stable so its 48 completed runs remain resumable. Other search
    spaces receive a suffix, which allows fixed and full spaces in one result root.
    """
    base = f"{optimizer}_{train_family}_n{int(train_size)}_seed{int(seed)}"
    suffixes = []
    if config_space not in {"large_fixed_2opt", "fixed_2opt"}:
        suffixes.append(f"space-{config_space.replace('_', '-')}")
    if resampling != "full_train":
        suffixes.append(f"resampling-{resampling.replace('_', '-')}")
    return f"{base}_{'_'.join(suffixes)}" if suffixes else base
