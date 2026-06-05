# Plan: Component 2 — Proxy-Data Code Demonstration

## What this is

This repository is the reproducible code demonstration (Component 2) for a
submission to NASA's Artemis II Human Research Data Methodology Challenge. It
applies a fixed methodology to terrestrial spaceflight-analog proxy data and
reports, honestly and with uncertainty, whatever that data yields. It is a
reproducibility package, not a tutorial and not an accuracy showcase.

The methodology is fixed by the written components of the submission and is not
modified here. Its named parts are:

- **mechanism-constrained multi-task learning** — a shared-representation model
  that predicts a primary outcome while regressing mechanistically related
  auxiliary targets, so the auxiliary tasks act as a data-dependent regularizer
  in the small-*n* regime;
- **subject-level cross-validation** — grouped CV in which every observation
  from a subject stays in one fold, with programmatically verified zero subject
  leakage;
- **clustered bootstrap** — uncertainty quantification that resamples whole
  subjects with replacement, preserving within-subject dependence;
- **baseline-imbalance handling** — quantifying countermeasure-arm imbalance
  with standardized mean differences and addressing it by covariate adjustment,
  reporting what the adjustment changes.

The worked example in the written components is glymphatic/SANS from brain MRI.
This demonstration uses **no imaging**. It instantiates the identical method on
bed rest physiology, with the cephalad fluid-shift axis as the thread back to
the SANS application. Nothing here predicts or implies a SANS or ocular finding.

## Data

Two tiers, mirroring the intended Artemis II use.

**Primary target — CFT70 bed rest campaign (NASA LSDA).** A head-down tilt
flight-analog study with a countermeasure-arm design and longitudinal repeated
measures across `PRE_TEST`, `IN_TEST`, `POST_TEST`. Join keys across modalities
are `Subject` (unit of analysis and resampling group), `Test_Phase`, and
`BR_Day`. Verified files and their use:

- Cardiovascular echocardiography: `BRSMCF_CFT70_2D_Echo_All.csv` (diastolic and
  structural indices) and `BRSMCF_CFT70_3D_Echo_All.csv` (LV mass, diastolic and
  systolic volumes). 37 subjects, all three phases, carries the `Treatment` arm.
  Home of the primary outcome.
- Fluid balance: `CRF_FARUWATERIO_CFT70_FINAL_daily.csv` (24 subjects; daily
  body weight, total water intake, goal, overage/shortage), with the `_obsv`
  companion. Direct fluid-status signal for the auxiliary targets.
- Immune panel: `BRSMIMMUNE_CFT70_obsv_All.csv` (long; pivot to wide) and
  `_avgs_All.csv`. 29 subjects, carries `GroupLabel`. Exploratory auxiliary only.
- Latent virus: `BRSMLVR_CFT70_{CMV,EBV,VZV}.csv`. 29 subjects, carries
  `GroupLabel`. EBV/VZV use `Time_Period` (`Before`/`During`/`After`) mapped onto
  the three phases; CMV already uses `Test_Phase`. Exploratory auxiliary only.
- Bone QCT: `BRSMQCT_CFT70_{Left,Right}_Hip.csv`. Only 5 subjects, keyed by date.
  Optional and heavily incomplete; will not drive anything.
- Vitals `CRF_CFT70_MEALS_*.csv`: only three subjects present; supplementary.

Dropped: all `MR035G_AG_PILOT_DXA_SCREEN_*` files (single-row screening
fragments); the countermeasure structure they were meant to supply already
exists, more richly, in the CFT70 `Treatment` field.

Two wrinkles handled explicitly:

1. **Inconsistent arm labels.** Echo: `Control`, `Exercise A`, `Exercise B`,
   `Flywheel`. Immune: `Control`, `Exercise`, `Exercise + Testosterone`. Virus:
   `Control`, `Exercise A`, `Exercise B`. Build one explicit subject-to-arm
   mapping; for the headline contrast collapse to `Control` vs `any
   countermeasure` (per-arm counts are too small to model separately); retain
   granular arms for a documented sensitivity analysis.
2. **Multi-modal missingness.** Modalities cover overlapping but unequal subject
   sets (37/29/24/5). Do not force a complete-case intersection. Use **masked
   auxiliary losses**: each auxiliary head contributes to the loss only for
   samples where its target is observed.

**Control tier — NHANES (CDC).** Fetched by a deterministic script; filtered to
an astronaut-like cohort (healthy, active, non-smoking, age-appropriate) with
documented variable codes and cycles. NHANES is cross-sectional, so it is the
large reference distribution that standardizes the bed rest cardiovascular and
anthropometric measures and demonstrates the "leverage a large dataset as
baseline" mechanism — not a source of within-subject trajectories.

## Modeling spec (fixed)

- Primary loss: binary cross-entropy. Auxiliary loss: masked mean squared error.
  `L_total = lambda_class * L_class + lambda_reg * L_reg`; `lambda_reg = 0`
  recovers the single-task baseline that isolates the auxiliary contribution.
- Primary outcome (binary): a cardiovascular-adaptation label derived from the
  echo `PRE_TEST` to `POST_TEST` change (a marked reduction in LV mass or stroke
  volume, the canonical cardiac response to bed-rest unloading). Threshold chosen
  and justified from the observed distribution; the underlying continuous change
  is reported alongside the label so the dichotomization is transparent.
- Auxiliary regression targets (the mechanism constraint): fluid-status measures
  (body-weight change, net water balance) and diastolic/loading cardiac indices
  that are **not** components of the outcome (mitral E/A, isovolumic relaxation
  time, TDI E'). Outcome constituents are never used as auxiliaries; this is
  asserted in code to prevent leakage.
- Optional secondary heads: immune summary and latent-virus reactivation as
  additional masked heads, flagged exploratory (weaker mechanistic link),
  included only if they do not degrade the primary result; comparison reported.
- Transparent cross-check: a strongly regularized logistic elastic net, so
  conclusions do not rest on a single flexible estimator.

## Reproducibility and code rules

Single centralized seed; all RNGs set; pinned dependencies; one config location
for hyperparameters and thresholds, each with a one-line rationale. `black` and
`ruff` enforced. Comment the why, not the what. Assert the claimed properties:
zero subject leakage across folds, no target leakage into features, fail loudly
on violated data assumptions. One documented entry point runs acquisition,
preprocessing, modeling, validation, and figures in order.

Data provenance: raw LSDA CFT70 files sit behind portal authentication, so
`data/raw/` is git-ignored and the README documents exact manual download and
placement. The processed analysis table (a derived tabular extract) is committed
under `data/interim/` where licensing and size permit, so modeling and figures
reproduce even if a reviewer cannot re-download the raw package. NHANES is
fetched by script and also git-ignored.

## Staged execution (commit at the end of each stage)

- **Stage 0 — Scaffolding and reproducibility harness.** Repository layout,
  pinned environment, seed/config module, license, `.gitignore`, README
  skeleton, formatter/linter config.
- **Stage 1 — Data acquisition and provenance.** Deterministic NHANES download
  script; documented CFT70 placement and provenance; a data dictionary built
  from the variables actually present; raw/interim directory structure.
- **Stage 2 — Preprocessing and cohort construction.** Build the astronaut-like
  NHANES control cohort with documented filters. Clean the CFT70 longitudinal
  table; define subjects, timepoints, and the harmonized countermeasure arms;
  compute and report baseline (`PRE_TEST`) imbalance via standardized mean
  differences.
- **Stage 3 — Outcome and auxiliary targets.** Define the binary primary outcome
  and the mechanistically linked auxiliary targets, each with written rationale,
  and verify no outcome constituent leaks into the auxiliaries or features.
- **Stage 4 — Model and baseline.** Implement the mechanism-constrained
  multi-task learner with masked auxiliary losses, the single-task baseline
  (`lambda_reg = 0`), and the elastic-net cross-check.
- **Stage 5 — Validation.** Subject-level `GroupKFold` with verified zero
  leakage; clustered bootstrap confidence intervals; baseline-imbalance
  adjustment and the granular-arm sensitivity analysis; honest metric reporting.
- **Stage 6 — Control integration.** Use the NHANES cohort as the reference
  distribution that standardizes the bed rest measures, demonstrating the
  small-*n*-via-large-population leverage mechanism.
- **Stage 7 — Results, figures, final reproducibility pass.** Generate all
  reported figures and tables; run a clean-clone end-to-end check; finalize the
  README with exact commands, expected outputs, expected metric values, runtime,
  and limitations; tag a submission release.

## Honesty and consistency

With four-to-thirty-seven subjects per modality, effects may be modest; they are
reported with uncertainty and never tuned to a target. No result is presented as
a SANS or ocular finding, and no number from the prior MRI study appears here.
The through-line is a proven small-sample, longitudinal, multi-modal methodology
demonstrated rigorously on a spaceflight analog and contextualized by a large
astronaut-like control, ready to transfer to the four-person Artemis II crew.
