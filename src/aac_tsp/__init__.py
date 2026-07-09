"""AAC configuration-generalization / overtuning study.

TSP (problem) + Simulated Annealing (target algorithm) + Bayesian Optimization
(configurator), comparing resampling strategies for estimating configuration quality.
"""

__version__ = "0.1.0"

FAMILIES = ("uniform", "clustered", "mixed")
MOVE_TYPES = ("swap", "insert", "2opt")
MOVE_CODE = {"swap": 0, "insert": 1, "2opt": 2}
