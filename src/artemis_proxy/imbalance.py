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


def _weighted_smd(
    x_t: np.ndarray, w_t: np.ndarray, x_c: np.ndarray, w_c: np.ndarray
) -> float:
    """SMD between two weighted groups using the pooled weighted SD."""

    def wmean(x: np.ndarray, w: np.ndarray) -> float:
        return float(np.sum(w * x) / np.sum(w))

    def wvar(x: np.ndarray, w: np.ndarray, m: float) -> float:
        return float(np.sum(w * (x - m) ** 2) / np.sum(w))

    m_t, m_c = wmean(x_t, w_t), wmean(x_c, w_c)
    pooled = np.sqrt((wvar(x_t, w_t, m_t) + wvar(x_c, w_c, m_c)) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (m_t - m_c) / pooled


def love_plot_smd_table(
    table: pd.DataFrame,
    covariates: list[str],
    adjust_covariates: list[str],
    group_col: str = "is_countermeasure",
    phase: str = "PRE_TEST",
) -> pd.DataFrame:
    """Baseline SMD per covariate before and after inverse-probability weighting.

    The "after" column reweights subjects by a stabilized inverse-probability-of-
    treatment weight from a regularized logistic propensity model fit on
    ``adjust_covariates``, then recomputes the SMD. With this few subjects the
    weighting reduces but does not eliminate imbalance; the point of the love plot
    is precisely to make that residual imbalance visible, not to claim perfect
    balance. Returns one row per covariate with ``smd_before`` and ``smd_after``.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    baseline = table[table["Test_Phase"] == phase].dropna(
        subset=[group_col, *adjust_covariates]
    )
    treat = baseline[group_col].to_numpy(dtype=float)
    design = StandardScaler().fit_transform(baseline[adjust_covariates].to_numpy())
    propensity = (
        LogisticRegression(C=1.0, max_iter=5000)
        .fit(design, treat)
        .predict_proba(design)[:, 1]
    )
    propensity = np.clip(propensity, 0.05, 0.95)
    p_treated = float(treat.mean())
    # Stabilized ATE weights.
    weights = np.where(
        treat == 1, p_treated / propensity, (1 - p_treated) / (1 - propensity)
    )

    is_t = treat == 1
    rows = []
    for covariate in covariates:
        before = standardized_mean_difference(
            baseline.loc[is_t, covariate], baseline.loc[~is_t, covariate]
        )
        xt_all = baseline.loc[is_t, covariate].to_numpy(dtype=float)
        xc_all = baseline.loc[~is_t, covariate].to_numpy(dtype=float)
        wt, wc = weights[is_t], weights[~is_t]
        ft, fc = np.isfinite(xt_all), np.isfinite(xc_all)
        if ft.sum() < 2 or fc.sum() < 2:
            after = np.nan
        else:
            after = _weighted_smd(xt_all[ft], wt[ft], xc_all[fc], wc[fc])
        rows.append(
            {
                "covariate": covariate,
                "smd_before": before,
                "smd_after": after,
                "adjusted": covariate in adjust_covariates,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "smd_before", key=lambda s: s.abs(), ascending=False
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
