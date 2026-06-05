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
  countermeasure arms, covariate adjustment, and a granular-arm sensitivity
  analysis.

The written components use glymphatics/SANS from brain MRI as the worked example.
This demonstration uses no imaging; it instantiates the same method on bed rest
cardiovascular physiology, with the cephalad fluid-shift axis as the thread back
to the SANS application.

### Outcome, auxiliaries, and the mechanism constraint

- **Primary outcome (binary):** a marked PRE_TEST -> POST_TEST reduction in
  left-ventricular (LV) mass, the canonical cardiac response to bed-rest
  unloading. The per-subject continuous change is dichotomized at its observed
  median (a distribution-driven, non-arbitrary cutpoint), and the continuous
  change is reported alongside the label.
- **Auxiliary regression targets (mechanism constraint):** whole-body fluid
  status (body weight, net water balance) and diastolic loading indices (mitral
  E/A, isovolumic relaxation time, TDI mitral annular E'), all on the same
  fluid-loading axis as the outcome.
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

Requires CPython 3.13 (developed on 3.13.2, Windows). CPU only; no GPU.

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
- Countermeasure association with the outcome: coefficient -0.327 unadjusted,
  -0.365 adjusted for the imbalanced baseline covariates.
- Per-arm marked-reduction rate: Control 0.73, Exercise A 0.43, Exercise B 0.40,
  Flywheel 0.38.
- Bed rest baseline weight 77.0 kg sits at mean z -0.17 (median 50th percentile)
  of the NHANES reference (80.6 kg); the reference pins the scale far more
  tightly (sd interval half-width 1.8 kg vs 3.2 kg).

Approximate runtime: under three minutes on a modern laptop CPU after
installation, dominated by the NHANES download (~19 MB) and validation.

## Interpretation

Bed rest reduced LV mass in most subjects (median change -10.4 g), consistent
with cardiac unloading under head-down tilt. Countermeasure subjects were
markedly less likely to show a large LV-mass reduction than controls (0.73 in
controls vs 0.38-0.43 across the active arms), the expected protective effect of
exercise and artificial-gravity countermeasures, and adjusting for the imbalanced
baseline slightly strengthened this association. The mechanism-constrained
multi-task model edged out the single-task baseline and the elastic-net
cross-check on subject-level AUC, and its advantage over the baseline was
positive but not separable from zero at this sample size. The bed rest subjects
were anthropometrically typical of the astronaut-like reference population.

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
