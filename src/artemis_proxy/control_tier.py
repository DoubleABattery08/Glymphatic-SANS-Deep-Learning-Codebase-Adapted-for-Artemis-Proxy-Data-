"""Use the NHANES cohort as the reference distribution for the bed rest data.

The two-tier strategy contextualizes a small longitudinal target sample with a
large astronaut-like control population. NHANES is cross-sectional, so it cannot
supply trajectories; its role is to fix a stable location and scale for the
measures shared with the bed rest data. The leverage is concrete: a moment
estimated from hundreds of reference subjects has a far tighter sampling interval
than the same moment from the ~20 bed rest subjects, so standardizing to the
reference stabilizes small-sample inference.

Body weight is the only measure shared between the modalities present here
(NHANES carries no echocardiography), so the standardization is demonstrated on
that channel and its scope is stated plainly rather than overclaimed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from artemis_proxy import config


def _bootstrap_sd_halfwidth(values: np.ndarray, n_resamples: int) -> float:
    """Half-width of the percentile interval for the standard deviation.

    Quantifies how precisely a sample pins down the scale used for
    standardization; a larger sample yields a smaller half-width.
    """

    rng = np.random.default_rng(config.SEED)
    n = len(values)
    draws = [values[rng.integers(0, n, size=n)].std(ddof=1) for _ in range(n_resamples)]
    alpha = (1.0 - config.BOOTSTRAP_CI) / 2.0
    lower, upper = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return float((upper - lower) / 2.0)


def reference_standardization(
    bed_rest_weight: pd.Series, reference_weight: pd.Series
) -> pd.DataFrame:
    """Standardize bed rest body weight to the NHANES reference moments.

    Returns one summary row comparing the reference and bed rest scales and the
    standardized position of the bed rest cohort within the reference population.
    """

    bed_rest = bed_rest_weight.dropna().to_numpy()
    reference = reference_weight.dropna().to_numpy()
    ref_mean = reference.mean()
    ref_sd = reference.std(ddof=1)
    z = (bed_rest - ref_mean) / ref_sd
    percentile = 100.0 * (reference[:, None] < bed_rest[None, :]).mean(axis=0)
    return pd.DataFrame(
        [
            {
                "reference_n": len(reference),
                "reference_mean_kg": ref_mean,
                "reference_sd_kg": ref_sd,
                "reference_sd_ci_halfwidth": _bootstrap_sd_halfwidth(
                    reference, config.BOOTSTRAP_N_RESAMPLES
                ),
                "bed_rest_n": len(bed_rest),
                "bed_rest_mean_kg": bed_rest.mean(),
                "bed_rest_sd_kg": bed_rest.std(ddof=1),
                "bed_rest_sd_ci_halfwidth": _bootstrap_sd_halfwidth(
                    bed_rest, config.BOOTSTRAP_N_RESAMPLES
                ),
                "bed_rest_mean_z_to_reference": float(z.mean()),
                "bed_rest_median_percentile_in_reference": float(np.median(percentile)),
            }
        ]
    )
