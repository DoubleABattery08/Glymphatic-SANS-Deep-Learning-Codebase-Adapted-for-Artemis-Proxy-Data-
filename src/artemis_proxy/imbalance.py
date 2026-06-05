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
import statsmodels.api as sm

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


def covariate_adjustment(
    subject_frame: pd.DataFrame,
    outcome_col: str,
    covariates: list[str],
    group_col: str = "is_countermeasure",
) -> pd.DataFrame:
    """Countermeasure association with the outcome, unadjusted vs adjusted.

    A linear probability model (OLS) is used rather than logistic regression: with
    36 subjects and an imbalanced covariate it is numerically stable, avoids
    separation, and yields a directly interpretable risk-difference coefficient.
    The reported quantity is the countermeasure coefficient before and after
    adjusting for the imbalanced baseline covariates, and the change between them.
    """

    fit_frame = subject_frame.dropna(subset=[outcome_col, group_col, *covariates])
    y = fit_frame[outcome_col].to_numpy(dtype=float)

    unadjusted = sm.OLS(y, sm.add_constant(fit_frame[[group_col]])).fit()
    adjusted = sm.OLS(y, sm.add_constant(fit_frame[[group_col, *covariates]])).fit()

    unadjusted_coef = float(unadjusted.params[group_col])
    adjusted_coef = float(adjusted.params[group_col])
    return pd.DataFrame(
        [
            {
                "model": "unadjusted",
                "countermeasure_coef": unadjusted_coef,
                "std_err": float(unadjusted.bse[group_col]),
                "n": int(fit_frame.shape[0]),
            },
            {
                "model": "adjusted_for_imbalanced_baseline",
                "countermeasure_coef": adjusted_coef,
                "std_err": float(adjusted.bse[group_col]),
                "n": int(fit_frame.shape[0]),
            },
            {
                "model": "change_from_adjustment",
                "countermeasure_coef": adjusted_coef - unadjusted_coef,
                "std_err": np.nan,
                "n": int(fit_frame.shape[0]),
            },
        ]
    )


def arm_sensitivity(
    subject_frame: pd.DataFrame,
    outcome_col: str,
    arm_col: str = "arm_granular",
) -> pd.DataFrame:
    """Positive-outcome rate per granular countermeasure arm.

    The granular arms (8-11 subjects each) are too small to model as the headline
    contrast, so this is a documented sensitivity view of the collapsed result.
    """

    grouped = subject_frame.groupby(arm_col)[outcome_col]
    return pd.DataFrame(
        {
            "n_subjects": grouped.size(),
            "positive_rate": grouped.mean(),
        }
    ).reset_index()
