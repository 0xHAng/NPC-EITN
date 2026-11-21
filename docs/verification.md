# Software verification

The project combines automated software checks with numerical comparison to
aggregate analysis references.

| Check | Coverage | Result |
| --- | --- | --- |
| Numerical, input and leakage tests | 25 tests | Passed |
| Synthetic end-to-end execution | Registered-image extraction and all four models | Completed |
| Frozen-model reload | 12 model/horizon groups | Maximum probability difference 1.1e-16 |
| Independent weighted ROC AUC | 24 primary groups | Difference below 1e-12 |
| Shared-cohort frozen probabilities | Four models at two horizons | Identical |
| Cox and landmark calculation | 120 groups | Maximum HR difference 2.22e-15 |
| CIF calculation | 44 fixed-grid estimates and intervals | Agreement within 1e-12 |
| Prediction reference comparison | 56 aggregate groups | Agreement within 1e-13 |
| Manuscript table references | 4 main and 10 supplementary tables | Matched |

Run the checks from the repository root:

```sh
python scripts/check_package.py
python -B -m unittest discover -s tests -v
task_dir=$(mktemp -d /tmp/npc_eitn_verify.XXXXXX)
python scripts/run_demo.py --output "$task_dir/demo"
```

The package checker validates file hashes, population references, model
coverage and version metadata. Tests cover endpoint ties, censoring weights,
Brier denominators, physical image geometry, common masks, fold separation,
training-only preprocessing, input keys and output protection.

GitHub Actions repeats the package checks, tests and synthetic workflow on
Linux/Python 3.11. Version assets are generated only after successful checks.
A deterministic ZIP contains only files in the sealed manifest and excludes
private configuration, participant-level data, fitted clinical models and
cache files.

The synthetic example uses artificial data and reduced settings. Clinical
reference comparisons evaluate numerical consistency; they do not establish
independent clinical validation, registration accuracy or clinical utility.
Clinical reproduction requires the approved inputs specified in input_contracts.md.
