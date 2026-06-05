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

# NHANES control tier. The 2017-2018 cycle is the last standalone public release
# with the standard single-cycle component files (suffix "_J"); using one named
# cycle keeps acquisition deterministic and unambiguous.
NHANES_CYCLE = "2017-2018"
# The 2017-2018 release lives under the cycle's first-year directory ("2017").
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

# Component files fetched, keyed by the role each plays in cohort construction.
# Stems exclude the ".XPT" extension; the "_J" cycle suffix is already part of
# each stem as published by CDC.
NHANES_COMPONENTS = {
    "DEMO_J": "demographics (age, sex)",
    "BMX_J": "body measures (weight, height, BMI, waist)",
    "BPX_J": "blood pressure",
    "SMQ_J": "smoking (cigarette use)",
    "PAQ_J": "physical activity",
    "DIQ_J": "diabetes questionnaire",
    "BPQ_J": "blood pressure and cholesterol diagnoses",
    "MCQ_J": "medical conditions (cardiovascular disease, cancer)",
}


# Primary outcome. Left-ventricular mass is the canonical cardiac response to
# bed-rest unloading and the cephalad fluid shift. The per-subject PRE->POST
# change is dichotomized at its own median: a distribution-driven, non-arbitrary
# cutpoint that yields balanced classes without tuning toward any performance
# target. The positive class is a marked reduction (change at or below median).
PRIMARY_OUTCOME_MEASURE = "LV mass"
OUTCOME_DICHOTOMIZE_AT = "median"

# Mechanism-constrained multi-task network. The trunk is deliberately tiny and
# strongly regularized because the sample is small; an over-parameterized model
# would memorize subjects rather than learn the adaptation signal.
MTL_HIDDEN_DIM = 8
MTL_DROPOUT = 0.25
MTL_WEIGHT_DECAY = 1e-2
MTL_LEARNING_RATE = 1e-2
# Full-batch training for a fixed budget keeps optimization deterministic given
# the seed; the data is small enough that mini-batching adds noise, not value.
MTL_EPOCHS = 300
MTL_LAMBDA_CLASS = 1.0
# Auxiliary loss weight. The mechanism constraint is the comparison between this
# value and zero (the single-task baseline), so the auxiliary contribution is
# directly measurable; 0.5 keeps the auxiliaries informative without overwhelming
# the primary objective.
MTL_LAMBDA_REG = 0.5

# Robustness sweep grid for the auxiliary weight. 0.0 is the single-task
# baseline; the configured MTL_LAMBDA_REG stays the headline and the rest probe
# whether any auxiliary contribution is stable across the weight rather than an
# artifact of one choice. The grid is reported in full and never selected post hoc.
LAMBDA_SWEEP = (0.0, 0.1, 0.25, 0.5, 1.0)

# Logistic elastic-net cross-check. A balanced l1/l2 mix and a modest inverse
# regularization strength suit a small, collinear feature set.
ELASTICNET_L1_RATIO = 0.5
ELASTICNET_C = 1.0


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
