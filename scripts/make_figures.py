"""Stage 7: render the reported figures from the committed tables and extract.

Produces five PNGs under ``results/figures``: baseline imbalance, the outcome
dichotomization, the model comparison with bootstrap intervals, the per-arm
sensitivity, and the bed rest weight against the NHANES reference. All inputs are
the committed derived extract and result tables, so the figures reproduce without
the gated raw package.

Run from the repository root::

    python scripts/make_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from artemis_proxy import cft70, config, imbalance, targets  # noqa: E402

_DPI = 150


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=_DPI)
    plt.close(fig)


def figure_baseline_imbalance() -> None:
    smd = pd.read_csv(config.TABLES / "baseline_smd.csv").sort_values("smd")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(smd["covariate"], smd["smd"], color="#3b6ea5")
    for threshold in (-imbalance.SMD_FLAG_THRESHOLD, imbalance.SMD_FLAG_THRESHOLD):
        ax.axvline(threshold, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("Standardized mean difference (countermeasure - control)")
    ax.set_title("Baseline covariate imbalance at PRE_TEST")
    _save(fig, "baseline_imbalance.png")


def figure_outcome_distribution() -> None:
    outcome = targets.primary_outcome(cft70.load_analysis_table())
    threshold = outcome["lv_mass_change"].median()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(outcome["lv_mass_change"], bins=12, color="#9ec3e0", edgecolor="white")
    ax.axvline(
        threshold,
        color="#a5343b",
        linestyle="--",
        label=f"median threshold = {threshold:.1f} g",
    )
    ax.set_xlabel("PRE -> POST change in LV mass (g)")
    ax.set_ylabel("Subjects")
    ax.set_title("Primary outcome: continuous change and its dichotomization")
    ax.legend()
    _save(fig, "outcome_distribution.png")


def figure_model_comparison() -> None:
    comparison = pd.read_csv(config.TABLES / "model_comparison.csv")
    lower = comparison["auc"] - comparison["auc_ci_lower"]
    upper = comparison["auc_ci_upper"] - comparison["auc"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        comparison["model"],
        comparison["auc"],
        yerr=[lower, upper],
        capsize=6,
        color="#3b6ea5",
    )
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1, label="chance")
    ax.set_ylabel("Subject-level AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model comparison with clustered-bootstrap 95% intervals")
    ax.legend()
    _save(fig, "model_comparison.png")


def figure_arm_sensitivity() -> None:
    arms = pd.read_csv(config.TABLES / "arm_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(arms["arm_granular"], arms["positive_rate"], color="#3b6ea5")
    for bar, n in zip(bars, arms["n_subjects"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={n}",
            ha="center",
        )
    ax.set_ylabel("Marked LV-mass reduction rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Outcome rate by countermeasure arm (sensitivity view)")
    _save(fig, "arm_sensitivity.png")


def figure_control_standardization() -> None:
    table = cft70.load_analysis_table()
    bed_rest = (
        table[table["Test_Phase"] == "PRE_TEST"]
        .groupby("Subject")["body_weight_kg"]
        .mean()
        .dropna()
    )
    cohort = pd.read_csv(config.DATA_PROCESSED / "nhanes_control_cohort.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        cohort["weight_kg"],
        bins=30,
        density=True,
        color="#ccd9e6",
        label=f"NHANES reference (n={len(cohort)})",
    )
    for weight in bed_rest:
        ax.axvline(weight, color="#a5343b", alpha=0.5, linewidth=1)
    ax.axvline(
        weight,
        color="#a5343b",
        alpha=0.5,
        linewidth=1,
        label=f"bed rest subjects (n={len(bed_rest)})",
    )
    ax.set_xlabel("Body weight (kg)")
    ax.set_ylabel("Reference density")
    ax.set_title("Bed rest baseline weight against the astronaut-like reference")
    ax.legend()
    _save(fig, "control_standardization.png")


def main() -> None:
    config.set_global_seed()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    figure_baseline_imbalance()
    figure_outcome_distribution()
    figure_model_comparison()
    figure_arm_sensitivity()
    figure_control_standardization()
    print(f"Wrote 5 figures to {config.FIGURES}")


if __name__ == "__main__":
    main()
