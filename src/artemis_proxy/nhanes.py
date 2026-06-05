"""Build an astronaut-like control cohort from NHANES 2017-2018.

NHANES is cross-sectional, so it provides a large, deeply phenotyped reference
distribution that standardizes the bed rest cardiovascular and anthropometric
measures; it is not a source of within-subject trajectories. The cohort is
restricted to generally healthy, active, non-smoking, age-appropriate adults
using documented variable codes, approximating the screened astronaut population.
"""

from __future__ import annotations

import pandas as pd

from artemis_proxy import config

# Astronaut corps span roughly the mid-30s to early-50s; this window keeps the
# reference population age-appropriate for an Artemis-like crew.
AGE_MIN_YEARS = 30
AGE_MAX_YEARS = 55

_DISEASE_EXCLUSION = {
    "DIQ_J": ["DIQ010"],  # diagnosed diabetes
    "BPQ_J": ["BPQ020"],  # diagnosed hypertension
    "MCQ_J": ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "MCQ220"],
}

_STANDARDIZATION_VARS = {
    "RIDAGEYR": "age_years",
    "BMXWT": "weight_kg",
    "BMXHT": "height_cm",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_cm",
}


def _read_xpt(stem: str) -> pd.DataFrame:
    path = config.NHANES_RAW / f"{stem}.XPT"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing NHANES file {path}. Run scripts/download_nhanes.py first."
        )
    return pd.read_sas(path, format="xport")


def _mean_blood_pressure(bpx: pd.DataFrame) -> pd.DataFrame:
    # A recorded BP of 0 is NHANES' code for an unobtained reading, not a true
    # zero, so each zero is masked before averaging the up-to-four readings.
    systolic = bpx[[f"BPXSY{i}" for i in range(1, 5)]].replace(0, pd.NA)
    diastolic = bpx[[f"BPXDI{i}" for i in range(1, 5)]].replace(0, pd.NA)
    return pd.DataFrame(
        {
            "SEQN": bpx["SEQN"],
            "systolic_bp": systolic.mean(axis=1),
            "diastolic_bp": diastolic.mean(axis=1),
        }
    )


def _is_non_smoker(smq: pd.DataFrame) -> pd.Series:
    # Never smoked 100 cigarettes (SMQ020 == 2) or, among ever-smokers, currently
    # not at all (SMQ040 == 3).
    never = smq["SMQ020"] == 2
    not_current = smq["SMQ040"] == 3
    return (never | not_current).reindex(smq.index, fill_value=False)


def _is_active(paq: pd.DataFrame) -> pd.Series:
    # Engages in vigorous (PAQ650) or moderate (PAQ665) recreational activity.
    return (paq["PAQ650"] == 1) | (paq["PAQ665"] == 1)


def build_control_cohort() -> pd.DataFrame:
    """Return the filtered astronaut-like NHANES cohort.

    Columns are the standardization variables plus sex and mean blood pressure.
    Each exclusion is applied explicitly so the filtering is auditable.
    """

    demo = _read_xpt("DEMO_J")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    bmx = _read_xpt("BMX_J")[["SEQN", *list(_STANDARDIZATION_VARS)[1:]]]
    bpx = _mean_blood_pressure(_read_xpt("BPX_J"))

    cohort = demo.merge(bmx, on="SEQN", how="inner").merge(bpx, on="SEQN", how="inner")

    smq = _read_xpt("SMQ_J")
    smq = smq.assign(_non_smoker=_is_non_smoker(smq))[["SEQN", "_non_smoker"]]
    paq = _read_xpt("PAQ_J")
    paq = paq.assign(_active=_is_active(paq))[["SEQN", "_active"]]
    cohort = cohort.merge(smq, on="SEQN", how="left").merge(paq, on="SEQN", how="left")

    keep = (
        cohort["RIDAGEYR"].between(AGE_MIN_YEARS, AGE_MAX_YEARS)
        & cohort["_non_smoker"].fillna(False)
        & cohort["_active"].fillna(False)
        & cohort[list(_STANDARDIZATION_VARS)[1:]].notna().all(axis=1)
        & cohort[["systolic_bp", "diastolic_bp"]].notna().all(axis=1)
    )

    for stem, columns in _DISEASE_EXCLUSION.items():
        flags = _read_xpt(stem)[["SEQN", *columns]]
        # A positive diagnosis is coded 1; missing answers are treated as absence
        # of a reported condition rather than as an exclusion.
        has_disease = (flags[columns] == 1).any(axis=1)
        diseased_ids = set(flags.loc[has_disease, "SEQN"])
        keep &= ~cohort["SEQN"].isin(diseased_ids)

    cohort = cohort.loc[keep].rename(columns=_STANDARDIZATION_VARS)
    cohort["sex"] = cohort["RIAGENDR"].map({1: "male", 2: "female"})
    output_cols = [
        "SEQN",
        "age_years",
        "sex",
        *list(_STANDARDIZATION_VARS.values())[1:],
        "systolic_bp",
        "diastolic_bp",
    ]
    return cohort[output_cols].reset_index(drop=True)
