# Running the analysis workflows

Activate the Python 3.11 environment described in environment.md. Every output
destination must be new and outside the repository. Use approved storage for
clinical inputs, fitted models and participant-level outputs.

## Registered-image extraction

```sh
python scripts/extract_eitn.py --input-dir /path/to/registered_nifti --output /path/to/new_eitn_output
```

The eight NIfTI inputs must share the planning-CT grid and affine. The script
exports region features and image-QC information. See input_contracts.md for
filenames, intensity masks, six MRI requirements and physical ROI definitions.

## Prediction development and frozen evaluation

```sh
python scripts/prepare_inputs.py --config /path/to/approved_sources.json --output /path/to/new_prediction_inputs
python scripts/run_analysis.py --config /path/to/new_prediction_inputs/inputs.json --output /path/to/new_prediction_results
```

The preparation step joins explicit approved membership, clinical inputs and
features. The clinical workflow requires 1848 conventional features and
372/197 internal/external participants. All fitting uses internal data. Six
partitions of five outer/four inner folds assess internal performance; full
internal models are frozen before external prediction.

| Output | Contents |
| --- | --- |
| `predictions.csv` | Participant probabilities by model, horizon and partition |
| `metrics.csv` | IPCW AUC, Brier, calibration and other metrics |
| `paired_comparisons.csv` | Paired performance differences and intervals |
| `tuning.csv` | Training-fold model selection results |
| `fitted_models.joblib` | Preprocessing and fitted estimators |
| `run.json` | Cohort sizes, horizons, folds and evaluation settings |

Generated clinical model files and prediction tables remain in approved
storage and are separate from the public aggregate reference results.

## Fixed-prediction evaluation

```sh
python scripts/evaluate_predictions.py --input /path/to/approved_predictions.csv --horizon 36 --output /path/to/new_evaluation
```

Use --horizon 60 for the second reported horizon. Input rows require outcomes,
center strata and M1–M4 probabilities. This entry evaluates fixed predictions
without training or updating a model.

## Association and cumulative incidence

```sh
python scripts/prepare_associations.py --config /path/to/approved_association_sources.json --output /path/to/new_association_inputs
python scripts/run_associations.py --input /path/to/new_association_inputs/phenotypes.csv --output /path/to/new_association_results
```

Main population flags identify the 517/203 phenotype and association cohorts.
A broader 206-person external source matrix also permits the separately labeled
sensitivity. Outputs include Cox and landmark estimates, internal reference
scales, cumulative incidence and sex-adjusted comparisons when the required
inputs are supplied. The broader sensitivity is exported only when additional
external source rows are present.

## Aggregate visualization

```sh
task_dir=$(mktemp -d /tmp/npc_eitn_plots.XXXXXX)
python scripts/plot_results.py --output "$task_dir/summary"
```

This entry reads only the included aggregate references and writes PNG/SVG
AUC summaries. It requires no clinical inputs.
