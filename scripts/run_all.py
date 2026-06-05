"""Single entry point: run the full pipeline end to end, in order.

Stages run sequentially: NHANES acquisition, CFT70 data dictionary, cohort
construction and baseline imbalance, subject-level validation, control-tier
integration, and figure generation. The pipeline reproduces every reported table
and figure. The authentication-gated raw CFT70 package is optional because the
derived analysis extract is committed; NHANES is fetched by the first step and
requires network access.

Run from the repository root::

    python scripts/run_all.py
"""

from __future__ import annotations

import build_cohorts
import build_data_dictionary
import download_nhanes
import integrate_control
import make_figures
import run_calibration
import run_lambda_sweep
import run_multimodal_extension
import run_seed_stability
import run_transfer_stress
import run_validation
from artemis_proxy import config

_STEPS = (
    ("Acquire NHANES", download_nhanes.main),
    ("Catalogue CFT70 variables", build_data_dictionary.main),
    ("Build cohorts and baseline imbalance", build_cohorts.main),
    ("Subject-level validation", run_validation.main),
    ("Auxiliary-weight robustness sweep", run_lambda_sweep.main),
    ("Multi-seed stability", run_seed_stability.main),
    ("Immune-and-virus multi-modal extension", run_multimodal_extension.main),
    ("Small-sample transfer stress test", run_transfer_stress.main),
    ("Calibration and baseline-imbalance love plot", run_calibration.main),
    ("Control-tier integration", integrate_control.main),
    ("Render figures", make_figures.main),
)


def main() -> None:
    config.set_global_seed()
    for index, (label, step) in enumerate(_STEPS, start=1):
        print(f"\n[{index}/{len(_STEPS)}] {label}")
        step()
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
