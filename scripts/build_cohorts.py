"""Stage 2: assemble the analysis cohorts and report baseline imbalance.

Writes three artifacts:

* ``data/interim/cft70_analysis_table.csv`` - the derived, de-identified
  longitudinal extract (one row per subject and phase). It is committed so the
  modeling and figures reproduce without access to the authentication-gated raw
  package; it is a transformation of the raw files, not the raw files.
* ``data/processed/nhanes_control_cohort.csv`` - the astronaut-like reference
  cohort (regenerable from the public NHANES download).
* ``results/tables/baseline_smd.csv`` - standardized mean differences between
  countermeasure groups on baseline covariates.

Run from the repository root::

    python scripts/build_cohorts.py
"""

from __future__ import annotations

from artemis_proxy import cft70, config, imbalance, nhanes

BASELINE_COVARIATES = [*cft70.ECHO_2D_FEATURES, "body_weight_kg"]


def main() -> None:
    config.set_global_seed()
    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.TABLES.mkdir(parents=True, exist_ok=True)

    analysis_table = cft70.build_analysis_table()
    analysis_path = config.DATA_INTERIM / "cft70_analysis_table.csv"
    analysis_table.to_csv(analysis_path, index=False)

    cohort = nhanes.build_control_cohort()
    cohort_path = config.DATA_PROCESSED / "nhanes_control_cohort.csv"
    cohort.to_csv(cohort_path, index=False)

    smd = imbalance.baseline_smd_table(analysis_table, BASELINE_COVARIATES)
    smd_path = config.TABLES / "baseline_smd.csv"
    smd.to_csv(smd_path, index=False)

    n_imbalanced = int(smd["imbalanced"].sum())
    print(
        f"CFT70 analysis table: {analysis_table['Subject'].nunique()} subjects, "
        f"{len(analysis_table)} subject-phase rows -> {analysis_path}"
    )
    print(f"NHANES control cohort: {len(cohort)} subjects -> {cohort_path}")
    print(
        f"Baseline imbalance: {n_imbalanced} of {len(smd)} covariates with "
        f"|SMD| > {imbalance.SMD_FLAG_THRESHOLD} -> {smd_path}"
    )


if __name__ == "__main__":
    main()
