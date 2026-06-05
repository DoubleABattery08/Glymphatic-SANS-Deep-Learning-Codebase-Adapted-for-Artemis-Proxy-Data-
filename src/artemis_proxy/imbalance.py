"""Quantify baseline covariate imbalance between countermeasure groups.

In small randomized samples, simple randomization frequently fails to balance
baseline covariates. The standardized mean difference (SMD) expresses each
imbalance in pooled standard-deviation units and is independent of sample size,
which is why it is preferred over a hypothesis test for flagging imbalance here.
A common rule of thumb treats |SMD| > 0.1 as a non-negligible imbalance that
warrants covariate adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMD_FLAG_THRESHOLD = 0.1


def standardized_mean_difference(treated: pd.Series, control: pd.Series) -> float:
    """SMD between two groups using the pooled standard deviation.

    Returns NaN when the pooled standard deviation is zero or undefined (for
    example a constant or all-missing covariate).
    """

    treated = treated.dropna()
    control = control.dropna()
    if len(treated) < 2 or len(control) < 2:
        return np.nan
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return float((treated.mean() - control.mean()) / pooled_sd)


def baseline_smd_table(
    table: pd.DataFrame,
    covariates: list[str],
    group_col: str = "is_countermeasure",
    phase: str = "PRE_TEST",
) -> pd.DataFrame:
    """One row per baseline covariate with group means and the SMD.

    Imbalance is assessed at the pre-intervention phase only, so the comparison
    reflects group composition rather than any treatment effect.
    """

    baseline = table[table["Test_Phase"] == phase]
    treated = baseline[baseline[group_col] == 1]
    control = baseline[baseline[group_col] == 0]
    rows = []
    for covariate in covariates:
        smd = standardized_mean_difference(treated[covariate], control[covariate])
        rows.append(
            {
                "covariate": covariate,
                "mean_countermeasure": treated[covariate].mean(),
                "mean_control": control[covariate].mean(),
                "n_countermeasure": int(treated[covariate].notna().sum()),
                "n_control": int(control[covariate].notna().sum()),
                "smd": smd,
                "imbalanced": (
                    bool(abs(smd) > SMD_FLAG_THRESHOLD) if not np.isnan(smd) else False
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("smd", key=lambda s: s.abs(), ascending=False)
