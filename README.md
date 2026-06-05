# Artemis II Proxy-Data Demonstration (Component 2)

Reproducible code demonstration for the NASA Artemis II Human Research Data
Methodology Challenge. It applies a fixed, established methodology to terrestrial
spaceflight-analog proxy data and reports the result honestly, with uncertainty.

This README is filled out fully in the final reproducibility pass. The sections
below mark the structure that the staged work populates.

## What this repository is

Component 2 of a five-part submission: a fully functional, documented,
reproducible application of the submission's methodology to public proxy data.
The other components (methodology description, application narrative, supporting
documentation, superlative criterion) are written documents and are not in this
repository.

## Methodology summary

Established methods translated from other domains, not an invention:

- **mechanism-constrained multi-task learning** — a shared-representation model
  that predicts a primary outcome while regressing mechanistically related
  auxiliary targets, so the auxiliary tasks regularize the small-*n* model;
- **subject-level cross-validation** — grouped cross-validation with verified
  zero subject leakage;
- **clustered bootstrap** — resampling whole subjects with replacement;
- **baseline-imbalance handling** — standardized mean differences across
  countermeasure arms with covariate adjustment and a sensitivity analysis.

The written components use glymphatics/SANS from brain MRI as the worked example.
This demonstration uses no imaging; it instantiates the same method on bed rest
cardiovascular physiology, with the cephalad fluid-shift axis as the thread back
to the SANS application. No result here is a SANS or ocular finding.

## Data provenance and acquisition

- **CFT70 bed rest (NASA LSDA).** Manual download and placement instructions:
  to be documented in Stage 1.
- **NHANES control cohort (CDC).** Deterministic acquisition script: to be
  documented in Stage 1.

## Environment setup

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.13.2, CPU only. Pinned versions are in `requirements.txt`.

## Reproduce everything

Exact ordered commands: to be documented in Stage 7.

## Expected outputs

Files, figures, tables, expected metric values, runtime, and hardware: to be
documented in Stage 7.

## Interpretation and limitations

A plain, honest reading of the results and explicit limitations (small sample,
no imaging, what cannot be concluded): to be documented in Stage 7.

## License and citation

MIT License (see `LICENSE`). Citation details: to be documented in Stage 7.
