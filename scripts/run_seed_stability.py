"""Stage 5c: multi-seed stability of the mechanism-constraint advantage.

The neural fits are deterministic given one seed, which leaves open whether the
modest multi-task advantage is a single-seed artifact. This repeats the
multi-task and single-task classification fits over ``config.SEED_STABILITY_N``
network seeds drawn deterministically from ``config.SEED`` (so the report itself
reproduces), under the same subject-level GroupKFold, and reports the
distribution of subject-level AUC for each model and of the paired multi-task
minus single-task gap (mean, standard deviation, and the fraction of seeds with a
positive gap). Writes ``results/tables/seed_stability.csv`` (per seed) and
``results/tables/seed_stability_summary.csv`` (aggregates).

Run from the repository root::

    python scripts/run_seed_stability.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from artemis_proxy import cft70, config, model, targets, validation


def _auc(lambda_reg: float, seed: int, arrays: dict) -> float:
    preds = validation.subject_cv_predictions(
        lambda: model.MechanismConstrainedMTL(
            lambda_reg, task="classification", seed=seed
        ),
        arrays,
    )
    return validation.subject_metrics(preds)["auc"]


def main() -> None:
    config.set_global_seed()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    table = cft70.load_analysis_table()
    frame = targets.build_modeling_frame(table)
    arrays = model.to_supervised(frame)

    seed_rng = np.random.default_rng(config.SEED)
    seeds = seed_rng.integers(0, 2**31 - 1, size=config.SEED_STABILITY_N)

    rows = []
    for seed in seeds:
        auc_mtl = _auc(config.MTL_LAMBDA_REG, int(seed), arrays)
        auc_single = _auc(0.0, int(seed), arrays)
        rows.append(
            {
                "seed": int(seed),
                "auc_multitask": round(auc_mtl, 4),
                "auc_single_task": round(auc_single, 4),
                "gap": round(auc_mtl - auc_single, 4),
            }
        )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(config.TABLES / "seed_stability.csv", index=False)

    gap = per_seed["gap"].to_numpy()
    positive_fraction = round(float(np.mean(gap > 0)), 3)
    summary = pd.DataFrame(
        [
            {
                "metric": "auc_multitask",
                "mean": round(per_seed["auc_multitask"].mean(), 4),
                "sd": round(per_seed["auc_multitask"].std(ddof=1), 4),
                "positive_gap_fraction": np.nan,
            },
            {
                "metric": "auc_single_task",
                "mean": round(per_seed["auc_single_task"].mean(), 4),
                "sd": round(per_seed["auc_single_task"].std(ddof=1), 4),
                "positive_gap_fraction": np.nan,
            },
            {
                "metric": "gap",
                "mean": round(gap.mean(), 4),
                "sd": round(gap.std(ddof=1), 4),
                "positive_gap_fraction": positive_fraction,
            },
        ]
    )
    summary.to_csv(config.TABLES / "seed_stability_summary.csv", index=False)

    print(f"Multi-seed stability over {config.SEED_STABILITY_N} network seeds:")
    print(per_seed.to_string(index=False))
    print(
        f"Mean gap {gap.mean():+.4f} (SD {gap.std(ddof=1):.4f}); "
        f"positive in {int(np.sum(gap > 0))}/{len(gap)} seeds."
    )


if __name__ == "__main__":
    main()
