# Analysis scope and reproducibility

## Implemented analysis

The software implements registered-image EITN extraction across six spatial
regions, threshold fractions, adjusted and landmark Cox analyses, paired
expanded adjustment, tied-time Aalen–Johansen cumulative incidence, four-model
nested development, frozen inference and conditional IPCW evaluation.

EITN is computed on each participant's planning-CT grid. The six MRI inputs,
ROI geometry, intensity references and denominator rules are specified in
input_contracts.md. Main phenotype and association populations are 517/203;
prediction populations are 372/197. Broader association and automatic-QC
results are labeled as sensitivity analyses in the aggregate references.

## Training and evaluation

The internal development pipeline shares outer and inner partitions across
models. Outer-fold seeds are 20261015+h+10000*repeat; inner seeds are
outer_seed*10+fold. Full-internal tuning uses 20261015+900000+h. No external
record enters preprocessing, selection or fitting. Calibration regression is
an evaluation calculation and does not update external predictions.

The manuscript describes the exploratory study design, analysis eligibility,
reporting choices and clinical interpretation. Statistical association,
predictive discrimination, calibration and clinical utility remain distinct.
A successful synthetic run verifies software execution rather than clinical
validity or registration accuracy.

## Inputs supplied upstream

Clinical endpoint adjudication, DICOM conversion, registration, registration
review, conventional radiomics/dose features and clinical harmonization are
provided as approved inputs. The software accepts registered NIfTI volumes
or explicit analysis-ready tables. It does not perform clinical care.

## Reference outputs

The results directory provides aggregate metrics, associations, fixed
six-month CIF grid values and 14 manuscript table blocks. Original patient
illustrations and participant-level outputs are not included. plot_results.py
produces an aggregate AUC summary from the supplied reference estimates.

Method fingerprints in provenance/source_provenance.json and result-source
fingerprints in provenance/result_sources.json support traceability without
exposing private identifiers or filesystem locations. FILE_MANIFEST.json seals
the distributed files. Hashes establish integrity; clinical access follows the
institution's approval process.
