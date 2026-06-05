"""Stage 5b: auxiliary-weight (lambda_reg) robustness sweep.

Runs the multi-task model across the configured ``config.LAMBDA_SWEEP`` grid and
reports, at each weight, the binary subject-level AUC and the continuous-outcome
MAE and R-squared, each expressed as a difference from the single-task baseline
(``lambda_reg = 0.0``). The point is to show whether the mechanism-constraint
contribution is stable across the auxiliary weight rather than an artifact of one
choice; the configured ``MTL_LAMBDA_REG`` remains the headline and no weight is
selected post hoc. Writes ``results/tables/lambda_sweep.csv``.

Uses the same subject-level GroupKFold (with the leakage assertion) as the main
validation. Point estimates only; uncertainty for the headline weight is reported
by the clustered bootstrap in ``run_validation.py``.

Run from the repository root::

    python scripts/run_lambda_sweep.py
"""

from __future__ import annotations

import pandas as pd

from artemis_proxy import cft70, config, model, targets, validation


def _evaluate(lambda_reg: float, arrays: dict) -> dict[str, float]:
    clf = validation.subject_cv_predictions(
        lambda: model.MechanismConstrainedMTL(lambda_reg, task="classification"),
        arrays,
    )
    reg = validation.subject_cv_regression(
        lambda: model.MechanismConstrainedMTL(lambda_reg, task="regression"),
        arrays,
    )
    return {
        "auc": validation.subject_metrics(clf)["auc"],
        **validation.regression_metrics(reg),
    }


def main() -> None:
    config.set_global_seed()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    table = cft70.load_analysis_table()
    frame = targets.build_modeling_frame(table)
    arrays = model.to_supervised(frame)

    metrics = {lam: _evaluate(lam, arrays) for lam in config.LAMBDA_SWEEP}
    baseline = metrics[0.0]
    rows = []
    for lam in config.LAMBDA_SWEEP:
        m = metrics[lam]
        rows.append(
            {
                "lambda_reg": lam,
                "is_headline": lam == config.MTL_LAMBDA_REG,
                "auc": round(m["auc"], 4),
                "auc_minus_single": round(m["auc"] - baseline["auc"], 4),
                "cont_mae": round(m["mae"], 4),
                # Positive = multi-task lowers error relative to the baseline.
                "cont_mae_improvement": round(baseline["mae"] - m["mae"], 4),
                "cont_r2": round(m["r2"], 4),
                "cont_r2_improvement": round(m["r2"] - baseline["r2"], 4),
            }
        )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(config.TABLES / "lambda_sweep.csv", index=False)

    print("Auxiliary-weight robustness sweep (single-task baseline at lambda=0.0):")
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
