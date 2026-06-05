"""Stage 5d: immune-and-virus multi-modal extension (exploratory).

Adds the third Artemis modality (immune biomarkers and latent-virus reactivation)
as additional masked auxiliary regression heads and reports the primary binary
outcome with and without them, alongside the single-task baseline, under the same
subject-level GroupKFold. The immune and virus targets have a weaker mechanistic
link to cardiac adaptation than the fluid and diastolic auxiliaries and carry
heavy, unequal missingness, so this is an honest multi-modal stress test of the
masked-loss machinery, not a claimed mechanism. If the extension degrades the
primary result, the cardiovascular-and-fluid set stays the headline. Writes
``results/tables/multimodal_extension.csv``.

Run from the repository root::

    python scripts/run_multimodal_extension.py
"""

from __future__ import annotations

import pandas as pd

from artemis_proxy import cft70, config, model, targets, validation


def _row(name: str, aux_targets: list[str], lambda_reg: float, frame) -> dict:
    arrays = model.to_supervised(frame, aux_targets)
    clf = validation.subject_cv_predictions(
        lambda: model.MechanismConstrainedMTL(lambda_reg, task="classification"),
        arrays,
    )
    reg = validation.subject_cv_regression(
        lambda: model.MechanismConstrainedMTL(lambda_reg, task="regression"),
        arrays,
    )
    metrics = validation.subject_metrics(clf)
    lo, hi = validation.clustered_bootstrap_ci(clf, validation.safe_auc)
    reg_metrics = validation.regression_metrics(reg)
    return {
        "model": name,
        "n_auxiliaries": len(aux_targets) if lambda_reg > 0 else 0,
        "auc": round(metrics["auc"], 4),
        "auc_ci_lower": round(lo, 4),
        "auc_ci_upper": round(hi, 4),
        "accuracy": round(metrics["accuracy"], 4),
        "brier": round(metrics["brier"], 4),
        "cont_mae": round(reg_metrics["mae"], 4),
        "cont_r2": round(reg_metrics["r2"], 4),
    }


def main() -> None:
    config.set_global_seed()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    table = cft70.load_analysis_table()
    frame = targets.build_modeling_frame(table)

    rows = [
        _row("single_task", targets.AUXILIARY_TARGETS, 0.0, frame),
        _row("multitask_core", targets.AUXILIARY_TARGETS, config.MTL_LAMBDA_REG, frame),
        _row(
            "multitask_extended",
            targets.EXTENDED_AUXILIARY_TARGETS,
            config.MTL_LAMBDA_REG,
            frame,
        ),
    ]
    extension = pd.DataFrame(rows)
    extension.to_csv(config.TABLES / "multimodal_extension.csv", index=False)

    print("Immune-and-virus multi-modal extension (exploratory):")
    print(extension.to_string(index=False))


if __name__ == "__main__":
    main()
