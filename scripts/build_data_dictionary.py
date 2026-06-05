"""Catalogue the CFT70 variables actually present in the local package.

Scans every CSV under ``data/raw/cft70`` and writes a per-column data dictionary
to ``data/interim/cft70_data_dictionary.csv`` (committed, since it is metadata
rather than restricted raw data). This documents the variables, types, coverage,
and subject counts that downstream stages rely on, built from the files as
provided rather than from assumptions.

Run from the repository root::

    python scripts/build_data_dictionary.py
"""

from __future__ import annotations

import pandas as pd

from artemis_proxy import config

# Files excluded from the analysis and the reason, recorded for transparency.
_EXCLUDED = {
    "MR035G_AG_PILOT_DXA_SCREEN": "single-subject screening DXA fragment",
}


def _exclusion_reason(name: str) -> str:
    for prefix, reason in _EXCLUDED.items():
        if name.startswith(prefix):
            return reason
    return ""


def describe_file(path) -> pd.DataFrame:
    """Return one row per column with type and coverage for a single CSV."""

    frame = pd.read_csv(path)
    subject_col = next(
        (c for c in ("Subject", "GroupLabel") if c in frame.columns), None
    )
    n_subjects = frame[subject_col].nunique() if subject_col == "Subject" else pd.NA
    return pd.DataFrame(
        {
            "file": path.name,
            "excluded_reason": _exclusion_reason(path.name),
            "n_rows": len(frame),
            "n_subjects": n_subjects,
            "column": frame.columns,
            "dtype": [str(frame[c].dtype) for c in frame.columns],
            "n_nonnull": [int(frame[c].notna().sum()) for c in frame.columns],
            "n_unique": [int(frame[c].nunique(dropna=True)) for c in frame.columns],
        }
    )


def main() -> None:
    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(config.CFT70_RAW.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CFT70 CSV files found in {config.CFT70_RAW}. "
            "See the README for download and placement instructions."
        )
    dictionary = pd.concat(
        (describe_file(path) for path in csv_paths), ignore_index=True
    )
    out_path = config.DATA_INTERIM / "cft70_data_dictionary.csv"
    dictionary.to_csv(out_path, index=False)
    n_files = dictionary["file"].nunique()
    n_excluded = dictionary.loc[dictionary["excluded_reason"] != "", "file"].nunique()
    print(
        f"Catalogued {n_files} CFT70 files "
        f"({n_excluded} flagged as excluded) to {out_path}"
    )


if __name__ == "__main__":
    main()
