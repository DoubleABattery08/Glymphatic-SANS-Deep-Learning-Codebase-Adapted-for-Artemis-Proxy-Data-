"""Define the primary outcome, the auxiliary targets, and the feature set.

Mechanism constraint. The auxiliary regression targets are quantities on the same
fluid-loading axis as the primary cardiac outcome, so jointly fitting them pushes
the shared representation toward genuine cardiovascular-adaptation signal rather
than subject-specific noise:

* ``aux_body_weight_kg`` and ``aux_net_water_balance_g`` index whole-body fluid
  status; head-down tilt drives a cephalad fluid shift and diuresis whose
  magnitude tracks the cardiac unloading that the outcome measures.
* ``aux_mitral_ea`` (mitral E/A), ``aux_ivrt`` (isovolumic relaxation time), and
  ``aux_tdi_e`` (TDI mitral annular E') are diastolic filling/loading indices
  that respond to the same preload change, without being the outcome itself.

Leakage is prevented explicitly: the outcome's own constituent (LV mass) and its
volumetric family (LVDV, LVSV) are never used as features or auxiliaries, and no
auxiliary target appears among the features. These properties are asserted in
``assert_no_leakage`` and the modeling frame fails loudly otherwise.

Exploratory multi-modal extension. ``EXPLORATORY_AUXILIARIES`` adds immune
(T-cell subset ratios, lymphocyte count) and latent-virus (CMV, EBV, VZV copy
number) targets so the demonstration genuinely ingests the third Artemis modality
under heavy, unequal missingness. These are flagged exploratory: their
mechanistic link to cardiac adaptation is weaker than the fluid and diastolic
auxiliaries, so they probe the framework's multi-modal and masked-loss capacity
rather than a claimed mechanism. The headline keeps the cardiovascular-and-fluid
set; the extension is reported separately so any effect on the primary result is
visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from artemis_proxy import cft70, config

AUXILIARY_TARGETS = [
    "aux_body_weight_kg",
    "aux_net_water_balance_g",
    "aux_mitral_ea",
    "aux_ivrt",
    "aux_tdi_e",
]

# Exploratory third-modality auxiliaries (immune biomarkers and latent-virus
# reactivation), used directly from the extract. Weaker mechanistic link than the
# core set; included to demonstrate multi-modal ingestion and masked-loss handling
# under heavy missingness, never as the headline.
EXPLORATORY_AUXILIARIES = [
    "immune_cd3+/cd4+",
    "immune_cd3+/cd8+",
    "immune_lymphocytes",
    "virus_cmv_copies",
    "virus_ebv_copies",
    "virus_vzv_copies",
]

EXTENDED_AUXILIARY_TARGETS = AUXILIARY_TARGETS + EXPLORATORY_AUXILIARIES

# Outcome family excluded from inputs to avoid leaking the label.
_OUTCOME_FAMILY = ["LV mass", "LVDV", "LVSV"]


def feature_columns() -> list[str]:
    """Model inputs: 2D structural/timing echo indices plus countermeasure arm."""

    return [*cft70.ECHO_2D_FEATURES, "is_countermeasure"]


def primary_outcome(table: pd.DataFrame) -> pd.DataFrame:
    """Per-subject binary cardiac-adaptation label and its continuous basis.

    Returns one row per subject with the PRE->POST change in the outcome measure
    and the dichotomized label. Subjects lacking either a PRE_TEST or POST_TEST
    measurement are dropped (the change is undefined for them).
    """

    measure = config.PRIMARY_OUTCOME_MEASURE
    wide = table.pivot_table(
        index="Subject", columns="Test_Phase", values=measure, observed=True
    )
    change = (wide["POST_TEST"] - wide["PRE_TEST"]).dropna()
    if config.OUTCOME_DICHOTOMIZE_AT != "median":
        raise ValueError("Only the median dichotomization rule is implemented.")
    threshold = change.median()
    outcome = (change <= threshold).astype(int)
    return pd.DataFrame(
        {
            "Subject": change.index,
            "lv_mass_change": change.to_numpy(),
            "outcome": outcome.to_numpy(),
        }
    )


def _add_auxiliary_targets(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    frame["aux_body_weight_kg"] = frame["body_weight_kg"]
    frame["aux_net_water_balance_g"] = frame["net_water_balance_g"]
    # Mitral E/A is undefined when the A wave is absent or zero; such cases become
    # missing and are masked rather than producing a spurious ratio.
    a_wave = frame["Mitral A Wave Velocity"].replace(0, np.nan)
    frame["aux_mitral_ea"] = frame["Mitral E Wave Velocity"] / a_wave
    frame["aux_ivrt"] = frame["Isovolumic Relaxation Time"]
    frame["aux_tdi_e"] = frame["TDI Mitral Annular E"]
    return frame


def assert_no_leakage() -> None:
    """Fail loudly if outcome constituents or targets contaminate the features."""

    features = set(feature_columns())
    contaminants = (
        set(_OUTCOME_FAMILY) | set(AUXILIARY_TARGETS) | set(EXPLORATORY_AUXILIARIES)
    )
    overlap = features & contaminants
    if overlap:
        raise AssertionError(
            f"Feature set leaks outcome or auxiliary information: {sorted(overlap)}"
        )
    if config.PRIMARY_OUTCOME_MEASURE in cft70.ECHO_2D_FEATURES:
        raise AssertionError("Primary outcome measure is present among features.")


def build_modeling_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Observation-level frame: features, masked auxiliaries, replicated outcome.

    One row per (Subject, Test_Phase) observation that carries the echo feature
    set. The subject-level outcome is replicated across that subject's
    observations; subject-level cross-validation therefore confines all of a
    subject's correlated rows to a single fold.
    """

    assert_no_leakage()
    outcome = primary_outcome(table)
    frame = _add_auxiliary_targets(table)
    frame = frame.merge(outcome, on="Subject", how="inner")
    feature_cols = feature_columns()
    complete_features = frame[feature_cols].notna().all(axis=1)
    frame = frame[complete_features].reset_index(drop=True)
    keep = [
        "Subject",
        "Test_Phase",
        *feature_cols,
        *AUXILIARY_TARGETS,
        *EXPLORATORY_AUXILIARIES,
        "lv_mass_change",
        "outcome",
    ]
    return frame[keep]
