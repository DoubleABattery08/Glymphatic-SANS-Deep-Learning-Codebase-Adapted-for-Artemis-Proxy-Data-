"""Stage 5e: small-sample transfer stress test (the Artemis n=4 problem).

The operational Artemis use case is a model trained on a larger analog cohort and
applied to a handful of crew members. This mirrors that directly: it repeatedly
holds out ``config.TRANSFER_TEST_SIZE`` subjects as the test set, trains on the
remainder, and reports the performance distribution across
``config.TRANSFER_N_SPLITS`` deterministic splits, comparing the multi-task model
against the single-task baseline. AUC is undefined or unstable on so few test
subjects, so the test reports subject-level accuracy (binary outcome) and MAE
(continuous outcome) instead. The honest question is whether the mechanism
constraint helps most precisely when the sample is smallest. Writes
``results/tables/transfer_stress.csv``.

Subject-level holdout keeps a subject's correlated longitudinal rows entirely in
train or test, preserving the leakage control. Run from the repository root::

    python scripts/run_transfer_stress.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from artemis_proxy import cft70, config, model, targets


def _subject_predictions(
    lambda_reg: float, task: str, arrays: dict, train: np.ndarray, test: np.ndarray
) -> pd.DataFrame:
    target_key = "y" if task == "classification" else "y_continuous"
    est = model.MechanismConstrainedMTL(lambda_reg, task=task)
    est.fit(
        arrays["X"][train],
        arrays[target_key][train],
        arrays["Z"][train],
        arrays["mask"][train],
    )
    if task == "classification":
        pred = est.predict_proba(arrays["X"][test])
    else:
        pred = est.predict(arrays["X"][test])
    frame = pd.DataFrame(
        {"Subject": arrays["subjects"][test], "y": arrays[target_key][test], "p": pred}
    )
    return frame.groupby("Subject", as_index=False).agg(
        y=("y", "first"), p=("p", "mean")
    )


def main() -> None:
    config.set_global_seed()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    table = cft70.load_analysis_table()
    frame = targets.build_modeling_frame(table)
    arrays = model.to_supervised(frame)
    subjects = np.array(sorted(set(arrays["subjects"])))

    rng = np.random.default_rng(config.SEED)
    records = []
    for _ in range(config.TRANSFER_N_SPLITS):
        test_subjects = rng.choice(
            subjects, size=config.TRANSFER_TEST_SIZE, replace=False
        )
        test = np.isin(arrays["subjects"], test_subjects)
        train = ~test
        train_idx = np.flatnonzero(train)
        test_idx = np.flatnonzero(test)

        clf_mtl = _subject_predictions(
            config.MTL_LAMBDA_REG, "classification", arrays, train_idx, test_idx
        )
        clf_single = _subject_predictions(
            0.0, "classification", arrays, train_idx, test_idx
        )
        reg_mtl = _subject_predictions(
            config.MTL_LAMBDA_REG, "regression", arrays, train_idx, test_idx
        )
        reg_single = _subject_predictions(
            0.0, "regression", arrays, train_idx, test_idx
        )
        records.append(
            {
                "acc_mtl": float(np.mean((clf_mtl["p"] >= 0.5) == clf_mtl["y"])),
                "acc_single": float(
                    np.mean((clf_single["p"] >= 0.5) == clf_single["y"])
                ),
                "mae_mtl": float(np.mean(np.abs(reg_mtl["y"] - reg_mtl["p"]))),
                "mae_single": float(np.mean(np.abs(reg_single["y"] - reg_single["p"]))),
            }
        )
    splits = pd.DataFrame(records)

    summary = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "multitask_mean": round(splits["acc_mtl"].mean(), 4),
                "single_task_mean": round(splits["acc_single"].mean(), 4),
                "multitask_sd": round(splits["acc_mtl"].std(ddof=1), 4),
                # Higher accuracy is better, so MTL wins when it is at least as high.
                "multitask_win_fraction": round(
                    float(np.mean(splits["acc_mtl"] >= splits["acc_single"])), 3
                ),
            },
            {
                "metric": "mae",
                "multitask_mean": round(splits["mae_mtl"].mean(), 4),
                "single_task_mean": round(splits["mae_single"].mean(), 4),
                "multitask_sd": round(splits["mae_mtl"].std(ddof=1), 4),
                # Lower MAE is better, so MTL wins when its error is no larger.
                "multitask_win_fraction": round(
                    float(np.mean(splits["mae_mtl"] <= splits["mae_single"])), 3
                ),
            },
        ]
    )
    summary.to_csv(config.TABLES / "transfer_stress.csv", index=False)

    print(
        f"Small-sample transfer: {config.TRANSFER_N_SPLITS} random holdouts of "
        f"{config.TRANSFER_TEST_SIZE} subjects (mirrors Artemis crew size):"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
