"""Stage 5: subject-level validation, bootstrap intervals, imbalance adjustment.

Writes four tables under ``results/tables`` and prints an honest summary:

* ``model_comparison.csv`` - subject-level AUC, accuracy, and Brier score for the
  multi-task model, the single-task baseline (auxiliary weight zero), and the
  elastic-net cross-check, each with a clustered-bootstrap 95% interval.
* ``mtl_vs_baseline.csv`` - the multi-task minus single-task AUC difference (the
  measured mechanism-constraint contribution) with a paired clustered-bootstrap
  interval.
* ``auxiliary_group_differences.csv`` - Control vs countermeasure differences on
  the auxiliary targets, with clustered-bootstrap intervals.
* ``imbalance_adjustment.csv`` and ``arm_sensitivity.csv`` - the countermeasure
  association before and after adjusting for the imbalanced baseline covariates,
  and the granular per-arm sensitivity view.

Run from the repository root::

    python scripts/run_validation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from artemis_proxy import cft70, config, imbalance, model, targets, validation


def _subject_baseline_frame(table: pd.DataFrame) -> pd.DataFrame:
    outcome = targets.primary_outcome(table)
    arms = cft70.subject_arm()
    baseline = table[table["Test_Phase"] == "PRE_TEST"][
        ["Subject", *cft70.ECHO_2D_FEATURES]
    ]
    return outcome.merge(arms, on="Subject", how="left").merge(
        baseline, on="Subject", how="left"
    )


def _paired_auc_difference_ci(
    mtl: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[float, float, float]:
    merged = mtl.merge(baseline, on="Subject", suffixes=("_mtl", "_base"))
    rng = np.random.default_rng(config.SEED)
    n = len(merged)
    point = roc_auc_score(merged["y_mtl"], merged["prob_mtl"]) - roc_auc_score(
        merged["y_base"], merged["prob_base"]
    )
    draws = []
    for _ in range(config.BOOTSTRAP_N_RESAMPLES):
        sample = merged.iloc[rng.integers(0, n, size=n)]
        if sample["y_mtl"].nunique() < 2:
            continue
        diff = roc_auc_score(sample["y_mtl"], sample["prob_mtl"]) - roc_auc_score(
            sample["y_base"], sample["prob_base"]
        )
        draws.append(diff)
    alpha = (1.0 - config.BOOTSTRAP_CI) / 2.0
    lower, upper = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return point, float(lower), float(upper)


def main() -> None:
    config.set_global_seed()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    table = cft70.build_analysis_table()
    frame = targets.build_modeling_frame(table)
    arrays = model.to_supervised(frame)

    predictions = {
        name: validation.subject_cv_predictions(make, arrays)
        for name, make in validation.ESTIMATORS.items()
    }

    rows = []
    for name, preds in predictions.items():
        metrics = validation.subject_metrics(preds)
        lower, upper = validation.clustered_bootstrap_ci(preds, validation.safe_auc)
        rows.append(
            {
                "model": name,
                **{k: round(v, 4) for k, v in metrics.items()},
                "auc_ci_lower": round(lower, 4),
                "auc_ci_upper": round(upper, 4),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.TABLES / "model_comparison.csv", index=False)

    point, lo, hi = _paired_auc_difference_ci(
        predictions["multitask"], predictions["single_task"]
    )
    pd.DataFrame(
        [
            {
                "auc_difference": round(point, 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
            }
        ]
    ).to_csv(config.TABLES / "mtl_vs_baseline.csv", index=False)

    aux_rows = [
        validation.auxiliary_group_difference(frame, target)
        for target in targets.AUXILIARY_TARGETS
    ]
    pd.DataFrame(aux_rows).round(4).to_csv(
        config.TABLES / "auxiliary_group_differences.csv", index=False
    )

    subject_frame = _subject_baseline_frame(table)
    smd = imbalance.baseline_smd_table(table, cft70.ECHO_2D_FEATURES)
    imbalanced = smd.loc[smd["imbalanced"], "covariate"].head(3).tolist()
    adjustment = imbalance.covariate_adjustment(subject_frame, "outcome", imbalanced)
    adjustment.round(4).to_csv(config.TABLES / "imbalance_adjustment.csv", index=False)
    imbalance.arm_sensitivity(subject_frame, "outcome").round(4).to_csv(
        config.TABLES / "arm_sensitivity.csv", index=False
    )

    print("Subject-level cross-validation: zero subject leakage verified per fold.")
    print(comparison.to_string(index=False))
    print(
        f"Multi-task minus single-task AUC: {point:+.4f} "
        f"(95% CI {lo:+.4f}, {hi:+.4f})"
    )
    print(f"Adjusted for imbalanced baseline covariates: {imbalanced}")


if __name__ == "__main__":
    main()
