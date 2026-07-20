"""Instance-resampling plans for the factorial v2 experiment.

The same plan is exposed in two views:

* fixed-estimator configurators evaluate a proposal on every subset and minimize
  the mean of subset means;
* SMAC-AAC receives one scenario instance per plan unit and may race across those
  units.  Unit weights preserve the mean-of-subset-means estimand when bootstrap
  OOB subsets have different sizes.

Test instances are deliberately absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from .resampling import make_resampling

RESAMPLINGS_V2 = (
    "full_train",
    "holdout",
    "cv5",
    "repeated_cv5",
    "bootstrap_oob",
)


@dataclass(frozen=True)
class ResamplingUnitV2:
    key: str
    instance_id: str
    subset_seed: int
    weight: float


@dataclass(frozen=True)
class ResamplingPlanV2:
    name: str
    subsets: tuple[tuple[tuple[str, ...], int], ...]

    @property
    def n_subsets(self) -> int:
        return len(self.subsets)

    @property
    def n_instance_evaluations(self) -> int:
        return sum(len(ids) for ids, _ in self.subsets)

    def calls_per_config(self, solver_repeats: int) -> int:
        return self.n_instance_evaluations * int(solver_repeats)

    def units(self) -> list[ResamplingUnitV2]:
        """Flatten the plan for SMAC while retaining subset-equal weighting."""
        total_units = self.n_instance_evaluations
        units: list[ResamplingUnitV2] = []
        for subset_index, (ids, subset_seed) in enumerate(self.subsets):
            if not ids:
                continue
            # The ordinary mean over all weighted unit costs then equals the mean
            # of subset means used by the fixed-estimator arms.
            weight = total_units / (self.n_subsets * len(ids))
            for instance_index, instance_id in enumerate(ids):
                units.append(ResamplingUnitV2(
                    key=f"subset{subset_index:03d}_instance{instance_index:03d}",
                    instance_id=instance_id,
                    subset_seed=int(subset_seed),
                    weight=float(weight),
                ))
        return units


def build_resampling_plan_v2(
    train_ids,
    name: str,
    seed: int,
    *,
    cv_repeats: int = 3,
    bootstrap_repeats: int = 10,
) -> ResamplingPlanV2:
    """Build one fixed, reproducible plan for a complete configurator run."""
    name = str(name).lower()
    if name not in RESAMPLINGS_V2:
        raise ValueError(
            f"unknown v2 resampling {name!r}; expected one of {RESAMPLINGS_V2}"
        )
    ids = list(train_ids)
    if not ids:
        raise ValueError("cannot build a resampling plan from an empty training pool")
    if name == "full_train":
        raw_plan = [(ids, int(seed))]
    else:
        raw_plan = make_resampling(
            name,
            cv_repeats=cv_repeats,
            bootstrap_repeats=bootstrap_repeats,
        ).make_plan(ids, int(seed))
    subsets = tuple((tuple(subset_ids), int(subset_seed))
                    for subset_ids, subset_seed in raw_plan if subset_ids)
    if not subsets:
        raise RuntimeError(f"resampling method {name!r} produced no evaluation subsets")
    return ResamplingPlanV2(name=name, subsets=subsets)
