"""Load and clean the CFT70 bed rest modalities into one longitudinal table.

The unit of observation is ``(Subject, Test_Phase)``: each subject contributes up
to three correlated observations (PRE_TEST, IN_TEST, POST_TEST). This repeated
structure is what makes subject-level cross-validation and the clustered
bootstrap necessary rather than cosmetic. ``Subject`` is the grouping unit for
all resampling.

Modalities cover overlapping but unequal subject sets, so the table is built by
left-joining every modality onto the echocardiography backbone and is left
deliberately sparse; missingness is handled downstream by masked auxiliary
losses, never by dropping subjects to a complete-case intersection.
"""

from __future__ import annotations

import pandas as pd

from artemis_proxy import config

PHASE_ORDER = ["PRE_TEST", "IN_TEST", "POST_TEST"]

# The latent-virus salivary/plasma files label the longitudinal axis Before /
# During / After; these map onto the shared three-phase axis used everywhere else.
TIME_PERIOD_TO_PHASE = {
    "Before": "PRE_TEST",
    "During": "IN_TEST",
    "After": "POST_TEST",
}

KEYS = ["Subject", "Test_Phase"]

# 3D echo structural measures. LV mass is the basis of the primary outcome and
# LVDV/LVSV are its volumetric family; they are kept here but excluded from the
# feature set downstream to prevent outcome leakage.
ECHO_3D_MEASURES = ["LV mass", "LVDV", "LVSV"]

# 2D echo diastolic/loading indices used as mechanism-constrained auxiliary
# regression targets (fluid-loading axis), none of which is the outcome itself.
ECHO_2D_DIASTOLIC = [
    "Mitral E Wave Velocity",
    "Mitral A Wave Velocity",
    "Isovolumic Relaxation Time",
    "TDI Mitral Annular E",
]

# 2D echo structural/timing indices available as model inputs (features). The
# diastolic indices above are deliberately omitted here because they serve as
# auxiliary targets, and using a target as a feature would leak it.
ECHO_2D_FEATURES = [
    "Left Ventricular Diameter Diastole",
    "Left Ventricular Diameter Systole",
    "Ejection Time",
    "Isovolumic Contraction Time",
    "Mitral Deceleration Time",
    "LVOT Velocity Time Integral",
    "TDI Left Heart Rate",
]

# A compact, robust immune summary (percent subsets reliably present in the panel)
# attached only as exploratory secondary heads; its mechanistic link to cardiac
# adaptation is weak and it is flagged as such, not claimed as a mechanism.
IMMUNE_MARKERS = ["LYMPHOCYTES (%)", "CD3+/CD4+ (%)", "CD3+/CD8+ (%)"]

VIRUS_FILES = {
    "CMV": ("BRSMLVR_CFT70_CMV.csv", "CMV copies/ng urinary DNA", "Test_Phase"),
    "EBV": ("BRSMLVR_CFT70_EBV.csv", "EBV copies/ng Salivary DNA", "Time_Period"),
    "VZV": ("BRSMLVR_CFT70_VZV.csv", "VZV copies/ng Salivary DNA", "Time_Period"),
}


def _read(name: str) -> pd.DataFrame:
    path = config.CFT70_RAW / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing CFT70 file {path}. See the README for download and "
            "placement instructions (data/raw/cft70/)."
        )
    return pd.read_csv(path)


def _aggregate_phase(frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Average replicate rows (multiple days/analyzers) to one row per key.

    CFT70 records several echo readings per phase (different days, or different
    analyzers for the 3D volumes); the phase-level mean is the stable summary.
    """

    present = [c for c in value_cols if c in frame.columns]
    return frame.groupby(KEYS, as_index=False)[present].mean(numeric_only=True)


def load_echo_2d() -> pd.DataFrame:
    frame = _read("BRSMCF_CFT70_2D_Echo_All.csv")
    return _aggregate_phase(frame, ECHO_2D_FEATURES + ECHO_2D_DIASTOLIC)


def load_echo_3d() -> pd.DataFrame:
    frame = _read("BRSMCF_CFT70_3D_Echo_All.csv")
    return _aggregate_phase(frame, ECHO_3D_MEASURES)


def load_fluid() -> pd.DataFrame:
    """Per-phase fluid status: mean body weight and mean net daily water balance.

    Net balance is the recorded daily overage/shortage in grams (intake minus
    goal); its phase mean indexes fluid loading under head-down tilt.
    """

    frame = _read("CRF_FARUWATERIO_CFT70_FINAL_daily.csv").rename(
        columns={
            "Subject_Weight_kg": "body_weight_kg",
            "Daily Water Overage/Shortage - Grams": "net_water_balance_g",
        }
    )
    return _aggregate_phase(frame, ["body_weight_kg", "net_water_balance_g"])


def load_immune_summary() -> pd.DataFrame:
    frame = _read("BRSMIMMUNE_CFT70_obsv_All.csv")
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    marker = frame[frame["Unit"].isin(IMMUNE_MARKERS)]
    wide = marker.pivot_table(
        index=KEYS, columns="Unit", values="Value", aggfunc="mean"
    ).reset_index()
    rename = {m: f"immune_{m.split(' ')[0].lower()}" for m in IMMUNE_MARKERS}
    return wide.rename(columns=rename)


def load_virus() -> pd.DataFrame:
    """Mean viral copies per subject and phase for CMV, EBV, VZV.

    EBV/VZV record repeated trials per phase; the phase mean is used. CMV is
    sampled at PRE/POST only, so its IN_TEST values are simply absent (masked
    downstream rather than imputed).
    """

    merged: pd.DataFrame | None = None
    for virus, (name, value_col, phase_col) in VIRUS_FILES.items():
        frame = _read(name)
        if phase_col == "Time_Period":
            frame["Test_Phase"] = frame["Time_Period"].map(TIME_PERIOD_TO_PHASE)
        agg = frame.groupby(KEYS, as_index=False)[value_col].mean(numeric_only=True)
        agg = agg.rename(columns={value_col: f"virus_{virus.lower()}_copies"})
        merged = agg if merged is None else merged.merge(agg, on=KEYS, how="outer")
    return merged


def subject_arm() -> pd.DataFrame:
    """Map each subject to its countermeasure arm from the echo Treatment field.

    The echo file is the most populated modality (37 subjects) and the only one
    carrying the granular four-arm labels, so it is the canonical source. Arm
    labels differ across files (echo: Control/Exercise A/Exercise B/Flywheel;
    immune: Control/Exercise/Exercise + Testosterone; virus: Control/Exercise
    A/Exercise B); rather than assume they are interchangeable, the headline
    contrast collapses to Control vs any countermeasure (per-arm counts of 8-11
    are too small to model separately), while the granular arm is retained for a
    documented sensitivity analysis.
    """

    frame = _read("BRSMCF_CFT70_2D_Echo_All.csv")
    arm = (
        frame[["Subject", "Treatment"]]
        .dropna()
        .drop_duplicates(subset="Subject")
        .rename(columns={"Treatment": "arm_granular"})
    )
    arm["is_countermeasure"] = (arm["arm_granular"] != "Control").astype(int)
    return arm


def build_analysis_table() -> pd.DataFrame:
    """Assemble the longitudinal multi-modal table keyed by (Subject, Test_Phase).

    Echocardiography is the backbone (it defines the primary outcome and the
    feature set); fluid, immune, and virus modalities are left-joined and may be
    absent for many subject-phases by design.
    """

    table = load_echo_2d().merge(load_echo_3d(), on=KEYS, how="outer")
    for modality in (load_fluid(), load_immune_summary(), load_virus()):
        table = table.merge(modality, on=KEYS, how="left")
    table = table.merge(subject_arm(), on="Subject", how="left")
    table["Test_Phase"] = pd.Categorical(
        table["Test_Phase"], categories=PHASE_ORDER, ordered=True
    )
    return table.sort_values(KEYS).reset_index(drop=True)


ANALYSIS_TABLE_PATH = config.DATA_INTERIM / "cft70_analysis_table.csv"


def has_raw() -> bool:
    """Whether the authentication-gated raw echo backbone is present locally."""

    return (config.CFT70_RAW / "BRSMCF_CFT70_2D_Echo_All.csv").exists()


def load_analysis_table() -> pd.DataFrame:
    """Return the analysis table, preferring the committed derived extract.

    The extract reproduces the modeling and figures without the gated raw
    package; it is rebuilt from raw only when the extract is absent. The ordered
    phase categorical is restored either way so downstream sorting is stable.
    """

    if ANALYSIS_TABLE_PATH.exists():
        table = pd.read_csv(ANALYSIS_TABLE_PATH)
    elif has_raw():
        table = build_analysis_table()
    else:
        raise FileNotFoundError(
            f"Neither the committed extract {ANALYSIS_TABLE_PATH} nor the raw "
            f"CFT70 package in {config.CFT70_RAW} is available."
        )
    table["Test_Phase"] = pd.Categorical(
        table["Test_Phase"], categories=PHASE_ORDER, ordered=True
    )
    return table.sort_values(KEYS).reset_index(drop=True)
