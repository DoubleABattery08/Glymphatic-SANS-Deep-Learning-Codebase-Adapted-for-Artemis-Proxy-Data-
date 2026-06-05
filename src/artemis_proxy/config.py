"""Single source of truth for paths, the global seed, and tunable constants.

Every random number generator in the project is seeded from ``SEED`` via
``set_global_seed`` so that a re-run reproduces identical numbers. Constants that
affect results live here, each with a one-line rationale, rather than being
scattered as literals through the code.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED = 20260604

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

CFT70_RAW = DATA_RAW / "cft70"
NHANES_RAW = DATA_EXTERNAL / "nhanes"

# Subject-level cross-validation. Five folds keeps roughly seven test subjects
# per fold given ~37 echo subjects, enough to estimate a fold metric while
# preserving the grouped (no subject leakage) guarantee.
CV_N_SPLITS = 5

# Clustered bootstrap. Two thousand resamples of whole subjects gives stable
# percentile interval endpoints at the small subject counts here without an
# unnecessary runtime cost.
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_CI = 0.95


def set_global_seed(seed: int = SEED) -> None:
    """Seed every RNG the project touches.

    Sets the hash seed, the ``random`` and NumPy generators, and (if importable)
    PyTorch on CPU, including its deterministic-algorithm flag. Call once at the
    start of any entry point before data shuffling or model initialization.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
