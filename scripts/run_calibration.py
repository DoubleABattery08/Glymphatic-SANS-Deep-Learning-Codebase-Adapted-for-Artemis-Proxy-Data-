"""Stage 5f: calibration of the primary model and a before/after love plot.

Two complementary diagnostics:

* Calibration. Reports the calibration slope and intercept of the multi-task
  primary model (from a logistic regression of the outcome on the predicted
  log-odds) alongside the Brier score and a binned reliability summary. This
  matters for the operational-screening framing, where the probability itself is
  acted on rather than only its ranking. Writes ``calibration.csv`` and
  ``calibration_reliability.csv``.
* Baseline-imbalance love plot. Reports each baseline covariate's SMD before and
  after stabilized inverse-probability-of-treatment weighting on the imbalanced
  covariates, so the residual imbalance after adjustment is visible. Writes
  ``love_plot_smd.csv``.

Run from the repository root::

    python scripts/run_calibration.py
"""

from __future__ import annotations

import pandas as pd

from artemis_proxy import cft70, config, imbalance, model, targets, validation


def main() -> None:
    config.set_global_seed()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    table = cft70.load_analysis_table()
    frame = targets.build_modeling_frame(table)
    arrays = model.to_supervised(frame)

    predictions = validation.subject_cv_predictions(
        validation.ESTIMATORS["multitask"], arrays
    )
    summary, reliability = validation.calibration_assessment(predictions)
    pd.DataFrame([summary]).to_csv(config.TABLES / "calibration.csv", index=False)
    reliability.to_csv(config.TABLES / "calibration_reliability.csv", index=False)

    smd = imbalance.baseline_smd_table(table, cft70.ECHO_2D_FEATURES)
    adjust = smd.loc[smd["imbalanced"], "covariate"].head(3).tolist()
    love = imbalance.love_plot_smd_table(table, cft70.ECHO_2D_FEATURES, adjust)
    love.round(4).to_csv(config.TABLES / "love_plot_smd.csv", index=False)

    print("Calibration of the multi-task primary model:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(reliability.to_string(index=False))
    print(f"Love-plot adjustment covariates (stabilized IPTW): {adjust}")
    print(love.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
