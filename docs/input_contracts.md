# Input contracts

## Prediction matrices

The explicit feature order is in `configs/conventional_features.json`.
`configs/approved_inputs.example.json` describes already prepared matrices.
Paths in a config resolve relative to the config itself.

Each matrix has one row per participant with these fields:

| Field | Meaning |
| --- | --- |
| patient_id | Stable unique pseudonymous join key; internal/external keys disjoint |
| center | Internal center or external A/B evaluation stratum; not a predictor |
| time_months | First target/competing event or censoring interval from radiotherapy completion, bounded at 60 |
| endpoint_code | 0 censored, 1 LRR, 2 preceding DM, 3 reliably documented preceding death |
| age, t_stage, n_stage | Numeric clinical inputs; the source's plausibility and imputation rules are retained |
| gtv_volume_cm3 | Primary GTV volume; the preprocessor uses log(1+volume) |
| formal__ring15__q_hl | Single prespecified EITN fraction |
| 1848 named conventional fields | Numeric descriptors; missingness is handled inside training folds |

At the horizon, a target event is positive; a competing event or documented
event-free observation through the horizon is negative; earlier censoring gets
zero evaluation weight. The API variable `time_after_landmark`
is an API alias for `time_months` in prediction; no new landmark is introduced.

Published settings require internal n=372 and external n=197. Membership is
provided explicitly; the package does not reconstruct physician QC or select
participants using model outcomes. The external phenotype cohort is larger
(203) and must not be substituted for the prediction input.

`prepare_inputs.py` accepts a separate approved-source config with
`conventional_features`, optional `corrections`, and `cohorts`. Each cohort has
`clinical`, `features` and `expected_n`. Clinical input fields use
`endpoint_time_months`, `endpoint_code`, `age`, `t_stage`, `n_stage`,
`gtv_volume_cm3`, `patient_id`, and externally `center_B`. Features have the
same join key plus the explicit image and EITN columns. Corrections, when
provided, have population (internal/external), patient_id and any verified
age/t_stage/n_stage values; blank correction cells never erase a source value.
The supplied frozen membership/order is preserved.

## Registered images

`extract_eitn.py --input-dir ... --output ...` requires eight NIfTI files:
mr_pre_t1, mr_pre_t1ce, mr_pre_t2, mr_post_t1, mr_post_t1ce, mr_post_t2,
gtv_mr_pre_gtvnx and ct (each `.nii.gz`). All must already share the planning
CT native 3D grid and affine. MRI and contour resampling are upstream tasks.
Six MRI inputs remain mandatory regardless of registration-reference fallback.

The extractor preserves empirical-CDF matching, outside-GTV pretreatment
intensity reference and fallback, one-voxel masked Gaussian smoothing, physical
shells, strict peritumoral -300<HU<300 filtering, and the six-volume common-valid
denominator. The core has no CT-HU restriction. It returns per-ROI counts,
coverage, EITN/control fractions and threshold variants plus automatic QC;
physician registration QC remains a separate provided decision.

## Fixed-prediction evaluation

One row per participant: patient_id, center, time_months, endpoint_code and
probability columns M1/M2/M3/M4. `evaluate_predictions.py` never trains a model.
The seed used by that CLI targets the main external evaluation. The Python
`evaluate` function accepts the internal/external seed explicitly.

## Association input

One row per source participant/ROI, including explicit binary
phenotype_eligible and association_eligible flags, with population (internal/external), center,
patient_id, roi, q_hl, age, t_stage, n_stage, gtv_volume_cm3,
lrr/dm/pfs_time_months and lrr/dm/pfs_event, plus lrr_competing_code.
Clinical endpoint adjudication and competing-event ordering precede this input.
The six ROI means/SDs are estimated in the full internal phenotype cohort and
then fixed externally. Main Cox fits use complete cases; external pooled fits
are stratified by center; External B estimates are unadjusted as in the study.
The 3/6-month sensitivity analyses exclude all endpoint/competing/censoring
times through the landmark and reset time there. CIF uses the fixed 0.1515
display threshold and tied-time simultaneous event handling.

For expanded adjustment, `paired_adjustment` accepts the already curated
time/event/standardized-EITN fields and extra covariates. The core comparison
uses exactly the same complete cases as the expanded fit. Internal extras are
sex_male, three recorded histology dummies, induction_cycles, and optionally
log_ebv_pre=log(1+baseline EBV). Externally only sex adjustment is exposed.
Threshold/compatibility associations use the primary internal ROI SD, not a new
variant-specific SD. These supplementary input fields are not created from
unreviewed clinical notes by this package.


## Main phenotype and broader sensitivity populations

The final main phenotype and association populations are 517 internal and 203
external participants (external A=124, B=79; 16 LRR events). The approved source
export retains the 206-person pre-restriction external population to reproduce
the explicitly labeled broader sensitivity. For main spatial descriptions,
select phenotype_eligible=1; for main Cox/landmark/CIF, select
association_eligible=1. These flags currently identify the same 203 external
participants. Never report all 206 as the current main phenotype population.

`prepare_associations.py` joins approved files through the private config's
association section: internal_long, internal_clinical, external_manifest,
external_features and excluded_external_keys. Supply the three approved
external exclusion keys privately; the public example deliberately leaves
them empty. This adapter accepts the 517/206 main-and-sensitivity source export and
checks 203 eligible external participants. An already curated 517/203 main
matrix can instead be passed directly to run_associations.py; it cannot
reproduce the 206-person sensitivity without the additional source rows.
The correction overlay is shared with prediction preparation. No private
exclusion keys are included in the package.

## Published prediction membership

The published prediction workflow uses explicitly supplied 372/197 cohorts.
The input adapter preserves approved participant membership and ordering;
it does not derive inclusion from outcomes or performance. Cohort definitions
and exploratory design limitations are described in the manuscript. Applying
the workflow to a different population requires a separately labeled evaluation
and corresponding reference outputs.
