"""Resampling strategies = estimators of configuration quality C-hat(theta).

In AAC there is no model trained on folds: a configuration is only *evaluated* on
instances. So a resampling strategy here defines *which* training instances enter
the validation estimate the configurator minimizes, and with how many independent
solver seeds. The strategies differ in variance (and cost), which is exactly what
drives -- or mitigates -- overtuning.

Each strategy produces an *evaluation plan*: a list of (subset_instance_ids,
subset_seed) pairs. The validation cost is the mean over subsets of the mean gap
within each subset. The plan is built once per run (fixed across all BO trials of
that run) so every configuration is judged on the same estimator.

Variance ordering (by design): holdout > 5-fold CV > repeated 5-fold CV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EvalPlan = list[tuple[list[str], int]]


@dataclass
class ResamplingStrategy:
    name: str

    def make_plan(self, train_ids: list[str], seed: int) -> EvalPlan:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class HoldoutResampling(ResamplingStrategy):
    val_frac: float = 0.2

    def make_plan(self, train_ids, seed):
        rng = np.random.default_rng(seed)
        ids = list(train_ids)
        rng.shuffle(ids)
        n_val = max(2, round(self.val_frac * len(ids)))
        subset = ids[:n_val]
        return [(subset, int(seed))]


@dataclass
class KFoldResampling(ResamplingStrategy):
    n_folds: int = 5

    def make_plan(self, train_ids, seed):
        rng = np.random.default_rng(seed)
        ids = list(train_ids)
        rng.shuffle(ids)
        k = min(self.n_folds, len(ids))
        folds = [list(f) for f in np.array_split(np.array(ids, dtype=object), k)]
        # one solver seed shared across folds
        return [(fold, int(seed)) for fold in folds if len(fold) > 0]


@dataclass
class RepeatedKFoldResampling(ResamplingStrategy):
    n_folds: int = 5
    n_repeats: int = 3

    def make_plan(self, train_ids, seed):
        plan: EvalPlan = []
        for r in range(self.n_repeats):
            rng = np.random.default_rng(seed + 1000 * (r + 1))
            ids = list(train_ids)
            rng.shuffle(ids)
            k = min(self.n_folds, len(ids))
            folds = [list(f) for f in np.array_split(np.array(ids, dtype=object), k)]
            subset_seed = int(seed) + (r + 1)  # distinct solver seed per repeat
            for fold in folds:
                if len(fold) > 0:
                    plan.append((fold, subset_seed))
        return plan


@dataclass
class BootstrapOOBResampling(ResamplingStrategy):
    n_repeats: int = 10

    def make_plan(self, train_ids, seed):
        ids = list(train_ids)
        n = len(ids)
        plan: EvalPlan = []
        for b in range(self.n_repeats):
            rng = np.random.default_rng(seed + 7919 * (b + 1))
            sampled = set(int(i) for i in rng.integers(0, n, size=n))
            oob = [ids[i] for i in range(n) if i not in sampled]
            if not oob:  # extremely unlikely, but guard
                continue
            plan.append((oob, int(seed) + 100 * (b + 1)))
        return plan


def make_resampling(name: str, **kwargs) -> ResamplingStrategy:
    name = name.lower()
    if name == "holdout":
        return HoldoutResampling(name="holdout", val_frac=kwargs.get("val_frac", 0.2))
    if name == "cv5":
        return KFoldResampling(name="cv5", n_folds=kwargs.get("n_folds", 5))
    if name == "repeated_cv5":
        return RepeatedKFoldResampling(name="repeated_cv5",
                                       n_folds=kwargs.get("n_folds", 5),
                                       n_repeats=kwargs.get("cv_repeats", 3))
    if name == "bootstrap_oob":
        return BootstrapOOBResampling(name="bootstrap_oob",
                                      n_repeats=kwargs.get("bootstrap_repeats", 10))
    raise ValueError(f"unknown resampling {name!r}")
