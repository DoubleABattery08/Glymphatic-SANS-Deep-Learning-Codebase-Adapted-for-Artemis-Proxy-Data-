"""Subject-level cross-validation, clustered bootstrap, and honest metrics.

Two guarantees are enforced in code rather than assumed:

* every observation from a subject is confined to one cross-validation fold, so
  the model cannot score by recognizing a subject across folds (verified by
  ``_assert_no_subject_leakage``);
* uncertainty is quantified by resampling whole subjects with replacement (the
  clustered bootstrap), never individual observations, preserving within-subject
  dependence.

Out-of-fold observation probabilities are aggregated to one prediction per
subject (the mean over that subject's observations) before any metric or interval
is computed, matching the subject-level unit of the primary outcome.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from artemis_proxy import config, model

Estimator = Callable[[], object]


class _ElasticNetAdapter:
    """Uniform fit/predict interface so the cross-check shares the CV harness."""

    def __init__(self) -> None:
        self._pipeline = model.build_elastic_net()

    def fit(self, X, y, Z, mask) -> _ElasticNetAdapter:  # noqa: N803
        self._pipeline.fit(X, y)
        return self

    def predict_proba(self, X):  # noqa: N803
        return self._pipeline.predict_proba(X)[:, 1]


ESTIMATORS: dict[str, Estimator] = {
    "multitask": lambda: model.MechanismConstrainedMTL(config.MTL_LAMBDA_REG),
    "single_task": lambda: model.MechanismConstrainedMTL(0.0),
    "elastic_net": _ElasticNetAdapter,
}


def _assert_no_subject_leakage(
    subjects: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    overlap = set(subjects[train_idx]) & set(subjects[test_idx])
    if overlap:
        raise AssertionError(f"Subject leakage across folds: {sorted(overlap)}")


def _aggregate_to_subject(
    subjects: np.ndarray, y: np.ndarray, prob: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame({"Subject": subjects, "y": y, "prob": prob})
    return frame.groupby("Subject", as_index=False).agg(
        y=("y", "first"), prob=("prob", "mean")
    )


def subject_cv_predictions(make_estimator: Estimator, arrays: dict) -> pd.DataFrame:
    """Out-of-fold subject-level predictions under grouped cross-validation."""

    subjects = arrays["subjects"]
    oof = np.full(len(arrays["y"]), np.nan)
    splitter = GroupKFold(n_splits=config.CV_N_SPLITS)
    for train_idx, test_idx in splitter.split(arrays["X"], arrays["y"], subjects):
        _assert_no_subject_leakage(subjects, train_idx, test_idx)
        estimator = make_estimator()
        estimator.fit(
            arrays["X"][train_idx],
            arrays["y"][train_idx],
            arrays["Z"][train_idx],
            arrays["mask"][train_idx],
        )
        oof[test_idx] = estimator.predict_proba(arrays["X"][test_idx])
    return _aggregate_to_subject(subjects, arrays["y"], oof)


def subject_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["y"].to_numpy()
    prob = predictions["prob"].to_numpy()
    return {
        "auc": roc_auc_score(y, prob),
        "accuracy": accuracy_score(y, prob >= 0.5),
        "brier": brier_score_loss(y, prob),
    }


def clustered_bootstrap_ci(
    predictions: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_resamples: int = config.BOOTSTRAP_N_RESAMPLES,
) -> tuple[float, float]:
    """Percentile CI from resampling whole subjects (rows) with replacement.

    Resamples that leave the statistic undefined (for example a single outcome
    class for AUC) are skipped; the interval is taken over the valid resamples.
    """

    rng = np.random.default_rng(config.SEED)
    n = len(predictions)
    estimates = []
    for _ in range(n_resamples):
        draw = predictions.iloc[rng.integers(0, n, size=n)]
        value = statistic(draw)
        if not np.isnan(value):
            estimates.append(value)
    alpha = (1.0 - config.BOOTSTRAP_CI) / 2.0
    lower, upper = np.percentile(estimates, [100 * alpha, 100 * (1 - alpha)])
    return float(lower), float(upper)


def safe_auc(frame: pd.DataFrame) -> float:
    """AUC of a subject-prediction frame, NaN when only one class is present."""

    if frame["y"].nunique() < 2:
        return np.nan
    return roc_auc_score(frame["y"], frame["prob"])


def auxiliary_group_difference(
    frame: pd.DataFrame, target: str, group_col: str = "is_countermeasure"
) -> dict[str, float]:
    """Subject-mean difference in an auxiliary target between arms, with CI.

    The auxiliary is first averaged within subject (so each subject contributes
    once), then the Control-minus-countermeasure mean difference is bootstrapped
    by resampling subjects.
    """

    per_subject = (
        frame.dropna(subset=[target])
        .groupby("Subject", as_index=False)
        .agg(value=(target, "mean"), group=(group_col, "first"))
    )

    def difference(sample: pd.DataFrame) -> float:
        control = sample.loc[sample["group"] == 0, "value"]
        treated = sample.loc[sample["group"] == 1, "value"]
        if len(control) < 1 or len(treated) < 1:
            return np.nan
        return float(control.mean() - treated.mean())

    point = difference(per_subject)
    rng = np.random.default_rng(config.SEED)
    n = len(per_subject)
    draws = []
    for _ in range(config.BOOTSTRAP_N_RESAMPLES):
        sample = per_subject.iloc[rng.integers(0, n, size=n)]
        value = difference(sample)
        if not np.isnan(value):
            draws.append(value)
    alpha = (1.0 - config.BOOTSTRAP_CI) / 2.0
    lower, upper = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return {
        "target": target,
        "control_minus_countermeasure": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }
