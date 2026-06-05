"""Stage 6: anchor the bed rest anthropometry to the NHANES reference tier.

Standardizes the bed rest baseline body weight to the astronaut-like NHANES
cohort and records the comparison in ``results/tables/control_standardization.csv``,
making the large-population leverage explicit (a much tighter scale estimate from
the reference than from the small bed rest sample).

Run from the repository root::

    python scripts/integrate_control.py
"""

from __future__ import annotations

import pandas as pd

from artemis_proxy import cft70, config, control_tier


def main() -> None:
    config.set_global_seed()
    config.TABLES.mkdir(parents=True, exist_ok=True)

    table = cft70.build_analysis_table()
    bed_rest_weight = (
        table[table["Test_Phase"] == "PRE_TEST"]
        .groupby("Subject")["body_weight_kg"]
        .mean()
    )
    cohort = pd.read_csv(config.DATA_PROCESSED / "nhanes_control_cohort.csv")

    summary = control_tier.reference_standardization(
        bed_rest_weight, cohort["weight_kg"]
    )
    out_path = config.TABLES / "control_standardization.csv"
    summary.round(4).to_csv(out_path, index=False)

    row = summary.iloc[0]
    print(
        f"NHANES reference weight: {row['reference_mean_kg']:.1f} kg "
        f"(sd {row['reference_sd_kg']:.1f}, n {int(row['reference_n'])}); "
        f"scale CI half-width {row['reference_sd_ci_halfwidth']:.2f} kg."
    )
    print(
        f"Bed rest baseline weight: {row['bed_rest_mean_kg']:.1f} kg "
        f"(sd {row['bed_rest_sd_kg']:.1f}, n {int(row['bed_rest_n'])}); "
        f"scale CI half-width {row['bed_rest_sd_ci_halfwidth']:.2f} kg."
    )
    print(
        f"Bed rest cohort sits at mean z {row['bed_rest_mean_z_to_reference']:+.2f}, "
        f"median {row['bed_rest_median_percentile_in_reference']:.0f}th percentile "
        "of the reference -> {out}".format(out=out_path)
    )


if __name__ == "__main__":
    main()
