# Artemis II Proxy-Data Demonstration (Component 2)

Reproducible code demonstration for the NASA Artemis II Human Research Data
Methodology Challenge. It applies a fixed, established methodology to terrestrial
spaceflight-analog proxy data and reports the result honestly, with uncertainty.
It is a reproducibility package, not a tutorial and not an accuracy showcase.

## What this repository is

Component 2 of a five-part submission: a fully functional, documented,
reproducible application of the submission's methodology to public proxy data.
The other components (methodology description, application narrative, supporting
documentation, superlative criterion) are written documents and are not in this
repository. The demonstration predicts a bed rest cardiovascular-adaptation
outcome; it never predicts or implies a SANS or ocular outcome, and it imports no
results from any prior imaging study.

## Methodology summary

Established methods translated from other domains, not an invention:

- **mechanism-constrained multi-task learning** — a compact shared-representation
  network predicts the primary outcome while regressing mechanistically related
  auxiliary targets through a shared trunk. The combined objective is
  `L_total = lambda_class * L_class + lambda_reg * L_reg`, with binary
  cross-entropy for the primary head and masked mean squared error for the
  auxiliary heads. Setting `lambda_reg = 0` recovers a single-task baseline, so
  the auxiliary (mechanism-constraint) contribution is measured directly
  (Caruana 1997; Baxter 2000; Ruder 2017).
- **subject-level cross-validation** — grouped `GroupKFold` on subject id, with a
  programmatic assertion of zero subject overlap across folds, so the model
  cannot score by recognizing a subject across folds (Varoquaux 2018; Vabalas et
  al. 2019).
- **clustered bootstrap** — resampling whole subjects with replacement,
  preserving within-subject dependence (Cameron et al. 2008; Efron & Tibshirani
  1993).
- **baseline-imbalance handling** — standardized mean differences across
  countermeasure arms, covariate adjustment, a granular-arm sensitivity analysis,
  and a before/after inverse-probability-weighting love plot.
- **robustness and small-sample stress tests** — an auxiliary-weight
  (`lambda_reg`) sweep, multi-seed stability of the mechanism-constraint gap, a
  calibration assessment, and an n=4 transfer stress test that mirrors the
  Artemis crew-size use case (train on a larger analog cohort, test on four).

The written components use glymphatics/SANS from brain MRI as the worked example.
This demonstration uses no imaging; it instantiates the same method on bed rest
cardiovascular physiology, with the cephalad fluid-shift axis as the thread back
to the SANS application.

### Outcome, auxiliaries, and the mechanism constraint

- **Primary outcome (binary headline):** a marked PRE_TEST -> POST_TEST reduction
  in left-ventricular (LV) mass, the canonical cardiac response to bed-rest
  unloading. The per-subject continuous change is dichotomized at its observed
  median (a distribution-driven, non-arbitrary cutpoint).
- **Continuous companion:** the same `lv_mass_change` is also regressed directly
  (the framework's primary head can be a classifier or a regressor) under the
  identical subject-level cross-validation. The median split discards information,
  so the continuous analysis is the better-powered confirmation of whether the
  mechanism constraint helps; both are reported even when modest.
- **Auxiliary regression targets (mechanism constraint):** whole-body fluid
  status (body weight, net water balance) and diastolic loading indices (mitral
  E/A, isovolumic relaxation time, TDI mitral annular E'), all on the same
  fluid-loading axis as the outcome.
- **Exploratory third modality:** immune biomarkers (T-cell subset ratios,
  lymphocyte count) and latent-virus reactivation (CMV, EBV, VZV copy number) can
  be added as further masked auxiliary heads. Their mechanistic link to cardiac
  adaptation is weaker, so they are reported separately as a multi-modal /
  masked-loss stress test under heavy missingness, never as the headline.
- **Leakage prevention:** the outcome's own constituents (LV mass, LVDV, LVSV)
  are never used as features or auxiliaries, and no auxiliary target appears
  among the features; this is asserted in code.
- **Multi-modal missingness:** modalities cover unequal subject sets, handled by
  masked auxiliary losses (a target contributes only where observed) rather than
  by dropping subjects to a complete-case intersection.

## Data provenance and acquisition

### CFT70 bed rest (NASA LSDA / NLSP) — manual, authentication-gated

The CFT70 head-down tilt bed rest package is distributed through the NASA Life
Sciences Data Archive, which requires registration and sits behind portal
authentication. It is **not redistributed** here (`data/raw/` is git-ignored). To
obtain it:

1. Sign in to the NASA LSDA / NLSP portal and locate the CFT70 ("Spaceflight
   Standard Measures", UTMB bed rest) study.
2. Download the study CSV exports.
3. Place every CSV directly under `data/raw/cft70/` (no subfolders), keeping the
   original file names (for example `BRSMCF_CFT70_2D_Echo_All.csv`).

The variables expected in each file are catalogued in
`data/interim/cft70_data_dictionary.csv`. **Reviewers without portal access do
not need the raw package:** the derived analysis extract
`data/interim/cft70_analysis_table.csv` is committed, and all modeling, tables,
and figures reproduce from it. The extract is a transformation and aggregation of
the raw files (one row per subject and phase), not the raw files themselves.

### NHANES control cohort (CDC) — scripted, public

NHANES is fully public and fetched deterministically; the script records each
file's source URL, byte size, and SHA-256 in `data/external/nhanes/manifest.json`.
Cycle: NHANES 2017-2018. Components: `DEMO_J`, `BMX_J`, `BPX_J`, `SMQ_J`,
`PAQ_J`, `DIQ_J`, `BPQ_J`, `MCQ_J` (about 19 MB). Files land in
`data/external/nhanes/` (git-ignored). The astronaut-like cohort filters to ages
30-55, non-smokers, recreationally active adults, with no reported diabetes,
hypertension, cardiovascular disease, or cancer.

## Environment setup

Requires CPython 3.11 or newer (developed on 3.13.2, verified installing on
3.12; Windows). CPU only; no GPU.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

On macOS/Linux use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

## Reproduce everything

One documented entry point runs acquisition, preprocessing, modeling, validation,
control integration, and figures in order:

```
.venv\Scripts\python.exe scripts/run_all.py
```

Equivalently, `make all`, or the numbered steps individually:

```
.venv\Scripts\python.exe scripts/download_nhanes.py
.venv\Scripts\python.exe scripts/build_data_dictionary.py
.venv\Scripts\python.exe scripts/build_cohorts.py
.venv\Scripts\python.exe scripts/run_validation.py
.venv\Scripts\python.exe scripts/run_lambda_sweep.py
.venv\Scripts\python.exe scripts/run_seed_stability.py
.venv\Scripts\python.exe scripts/run_multimodal_extension.py
.venv\Scripts\python.exe scripts/run_transfer_stress.py
.venv\Scripts\python.exe scripts/run_calibration.py
.venv\Scripts\python.exe scripts/integrate_control.py
.venv\Scripts\python.exe scripts/make_figures.py
```

The pipeline is deterministic (single seed in `src/artemis_proxy/config.py`); a
re-run reproduces identical numbers. NHANES acquisition needs network access; the
raw CFT70 package is optional (see above).

## Expected outputs

Tables in `results/tables/` and figures in `results/figures/`:

| Artifact | Content |
| --- | --- |
| `baseline_smd.csv`, `baseline_imbalance.png` | Baseline covariate imbalance |
| `model_comparison.csv`, `model_comparison.png` | Subject-level AUC per model |
| `mtl_vs_baseline.csv` | Mechanism-constraint AUC contribution |
| `continuous_outcome.csv`, `continuous_mtl_vs_baseline.csv` | Continuous-outcome companion (MAE, R-squared) |
| `lambda_sweep.csv` | Auxiliary-weight robustness sweep |
| `seed_stability.csv`, `seed_stability_summary.csv` | Multi-seed stability of the gap |
| `multimodal_extension.csv` | Immune-and-virus exploratory extension |
| `transfer_stress.csv` | n=4 small-sample transfer stress test |
| `calibration.csv`, `calibration_reliability.csv`, `calibration.png` | Calibration of the primary model |
| `love_plot_smd.csv`, `love_plot.png` | SMD before/after IPTW adjustment |
| `auxiliary_group_differences.csv` | Auxiliary differences by arm |
| `imbalance_adjustment.csv` | Countermeasure association, adjusted |
| `arm_sensitivity.csv`, `arm_sensitivity.png` | Per-arm outcome rates |
| `control_standardization.csv`, `control_standardization.png` | NHANES anchoring |
| `outcome_distribution.png` | Outcome change and threshold |
| `data/processed/nhanes_control_cohort.csv` | Astronaut-like cohort |

Expected metric values (a reviewer can confirm a match):

- CFT70 modeling cohort: 36 subjects with a defined outcome (18 positive), 108
  subject-phase observations; NHANES control cohort: 526 subjects.
- Baseline imbalance: 6 of 8 covariates with |SMD| > 0.1 (largest: isovolumic
  contraction time, SMD 0.72).
- Subject-level AUC: multi-task 0.713, single-task 0.654, elastic-net 0.670.
- Mechanism-constraint contribution (multi-task minus single-task AUC): +0.059,
  clustered-bootstrap 95% interval (-0.019, +0.153).
- Continuous companion (`lv_mass_change` regressed under the same CV): multi-task
  MAE 10.36 g and R-squared -0.116, single-task MAE 10.89 g and R-squared -0.173;
  paired MAE improvement +0.53 g (95% interval -0.35, +1.46). Both R-squared are
  negative — neither model beats the mean predictor at this sample size — but the
  multi-task model is consistently the less-poor of the two, agreeing in direction
  with the binary headline.
- Auxiliary-weight sweep: every non-zero `lambda_reg` in {0.1, 0.25, 0.5, 1.0}
  beats single-task on AUC (+0.05 to +0.08) and on continuous MAE/R-squared, so
  the advantage is not an artifact of the configured 0.5.
- Multi-seed stability (10 seeds): mean multi-task-minus-single-task AUC gap
  +0.032 (SD 0.035), positive in 8 of 10 seeds.
- Immune-and-virus extension (11 auxiliaries): AUC unchanged at 0.713 but slightly
  worse accuracy/Brier/continuous-MAE, so the cardiovascular-and-fluid set stays
  the headline and the extension is a documented multi-modal stress test.
- Small-sample transfer (200 holdouts of 4 subjects): multi-task beats or ties
  single-task on subject accuracy in 96.5% of splits (mean 0.624 vs 0.588) and on
  continuous MAE in 59% (10.75 vs 11.04 g).
- Calibration of the primary model: slope 0.71 (mild over-confidence), intercept
  0.13, Brier 0.219; the reliability bins track the diagonal.
- Countermeasure association with the outcome: coefficient -0.327 unadjusted,
  -0.365 adjusted for the imbalanced baseline covariates. Stabilized IPTW on the
  three imbalanced covariates shrinks their |SMD| (0.72/0.40/0.37 -> 0.31/0.06/
  0.21) with visible residual imbalance.
- Per-arm marked-reduction rate: Control 0.73, Exercise A 0.43, Exercise B 0.40,
  Flywheel 0.38.
- Bed rest baseline weight 77.0 kg sits at mean z -0.17 (median 50th percentile)
  of the NHANES reference (80.6 kg); the reference pins the scale far more
  tightly (sd interval half-width 1.8 kg vs 3.2 kg).

Approximate runtime: roughly 20-25 minutes on a modern laptop CPU after
installation, dominated by the n=4 transfer stress test (200 retrained
split-pairs) and the subject-level validation; the individual core steps each run
in a few minutes.

## Interpretation

Bed rest reduced LV mass in most subjects (median change -10.4 g), consistent
with cardiac unloading under head-down tilt. Countermeasure subjects were
markedly less likely to show a large LV-mass reduction than controls (0.73 in
controls vs 0.38-0.43 across the active arms), the expected protective effect of
exercise and artificial-gravity countermeasures, and adjusting for the imbalanced
baseline slightly strengthened this association. The mechanism-constrained
multi-task model edged out the single-task baseline and the elastic-net
cross-check on subject-level AUC, and its advantage over the baseline was
positive but not separable from zero at this sample size.

The polish-pass robustness checks tell a consistent, deliberately modest story.
The better-powered continuous regression agrees in direction (multi-task lower
error) while both models trail the mean predictor, so the benefit is real in
sign but small in size. The advantage holds across every auxiliary weight in the
sweep and is positive in 8 of 10 random seeds, so it is not a single-weight or
single-seed artifact. It is most pronounced exactly where it matters for Artemis:
in the n=4 transfer stress test the multi-task model wins or ties the single-task
baseline on accuracy in 96.5% of crew-sized holdouts. The exploratory immune and
virus heads neither help nor materially hurt the headline, demonstrating
multi-modal ingestion under heavy missingness without overclaiming. The primary
model is reasonably calibrated (slope 0.71), and the bed rest subjects were
anthropometrically typical of the astronaut-like reference population.

## Limitations

- **Small sample.** Roughly 36 subjects drive the primary outcome; all intervals
  are wide and modest effects are reported as such. The mechanism-constraint
  benefit is suggestive, not established here.
- **No imaging.** The proxy data is tabular physiology, not the brain MRI of the
  worked example; the demonstration establishes the method in the small-*n*,
  longitudinal, multi-modal regime, not a SANS or ocular result.
- **Cross-sectional control tier.** NHANES informs baselines and standardization
  only, and shares only body weight with the bed rest measures (it carries no
  echocardiography), so the control anchoring is limited to anthropometry.
- **Arm-label harmonization.** Countermeasure labels differ across modalities and
  are collapsed to Control vs any countermeasure for the headline contrast; the
  granular arms are underpowered and shown only as a sensitivity view.

## License and citation

MIT License (see `LICENSE`). If you use this code, please cite the Artemis II
Human Research Data Methodology Challenge submission of which it is Component 2,
and the NASA LSDA CFT70 study and NHANES 2017-2018 as the data sources.
