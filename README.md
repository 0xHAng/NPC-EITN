# NPC EITN

Reproducibility software for *A spatial MRI treatment response phenotype
associated with locoregional recurrence in nasopharyngeal carcinoma*.

**Version 0.2.3** · [中文文档](README.zh-CN.md) ·
[Releases](https://github.com/0xHAng/NPC-EITN/releases) · [Citation](CITATION.md)

The project provides registered-image EITN extraction, spatial and outcome
association analysis, and internally developed four-model prediction with
frozen external evaluation. It includes runnable command-line tools, a pinned
Python environment, synthetic examples, automated tests and aggregate reference
results. Clinical data are supplied separately through the approved access process.

## Install

Use Python 3.11 on macOS or Linux. From the repository root:

```sh
python3.11 -m venv /tmp/npc_eitn_env
. /tmp/npc_eitn_env/bin/activate
python -m pip install -r requirements-lock.txt
```

See [environment](docs/environment.md) for numerical dependencies and platform details.

## Run the synthetic example

```sh
python scripts/check_package.py
python -B -m unittest discover -s tests -v
task_dir=$(mktemp -d /tmp/npc_eitn_demo.XXXXXX)
python scripts/run_demo.py --output "$task_dir/results"
```

The example generates artificial image arrays, features and follow-up records,
then runs extraction and all four models. It uses two outer/two inner folds,
one partition, reduced tuning grids and the 36-month horizon to demonstrate
execution. Synthetic outputs are separate from the clinical reference results.
Output destinations must be new directories outside the repository.

## Analysis workflows

| Task | Entry point | Input |
| --- | --- | --- |
| Extract six-region EITN | `scripts/extract_eitn.py` | Registered NIfTI volumes |
| Prepare prediction matrices | `scripts/prepare_inputs.py` | Approved clinical and feature tables |
| Develop and evaluate M1–M4 | `scripts/run_analysis.py` | Explicit analysis-ready cohorts |
| Evaluate fixed predictions | `scripts/evaluate_predictions.py` | Outcome rows and M1–M4 probabilities |
| Prepare association inputs | `scripts/prepare_associations.py` | Approved phenotype and outcome tables |
| Fit Cox, landmark and CIF analyses | `scripts/run_associations.py` | Curated participant/ROI rows |
| Plot aggregate AUC results | `scripts/plot_results.py` | Included aggregate references |

[Workflow examples and output files](docs/workflows.md) explain every entry.
[Input contracts](docs/input_contracts.md) define the required fields, image
geometry, cohort membership and endpoint coding.

## Scientific configuration

EITN uses pre- and post-radiotherapy T1-weighted, contrast-enhanced T1-weighted
and T2-weighted MRI on the patient's planning-CT grid. Post-radiotherapy MRI is
obtained within three months. Outcome time starts at radiotherapy completion,
with 60-month administrative censoring. The main phenotype and association
populations are 517 internal/203 external; prediction populations are 372/197.
The broader 206-person external association is a separate sensitivity analysis.

| Model | Inputs and estimator |
| --- | --- |
| M1 | Four clinical variables with IPCW logistic ridge |
| M2 | Clinical variables, conventional imaging and EITN with joint screening and Elastic Net |
| M3 | Clinical variables and conventional imaging Elastic Net, without EITN |
| M4 | M3's selected imaging and absolute penalty, with separately regularized EITN |

The conventional pool contains 1848 descriptors. Preprocessing, feature
selection and tuning operate within internal training folds. Clinical settings
use six shared five-outer/four-inner partitions at 36 and 60 months. External
predictions use frozen preprocessing and fitted coefficients. IPCW uses G(t-),
a 0.05 floor and the full-cohort Brier denominator. Conditional intervals
resample patients with fixed predictions. The study is exploratory; design,
reporting choices and limitations are described in the accompanying manuscript.

## Project structure

| Directory | Contents |
| --- | --- |
| `src/npc_eitn/` | Image, statistical, modeling and evaluation modules |
| `scripts/` | Command-line workflows and package verification |
| `configs/` | Feature definitions and example configurations |
| `results/` | Aggregate reference estimates and 14 manuscript tables |
| `provenance/` | Method and reference fingerprints |
| `tests/` | Numerical, input-validation and leakage checks |
| `docs/` | Installation, inputs, workflows, scope and verification |

[Verification](docs/verification.md) describes the checks. The release workflow
runs the package checks, 25 tests and synthetic pipeline before packaging a
versioned ZIP. `FILE_MANIFEST.json` records file integrity.

## Scope and access

The pipeline begins with registered images or curated tables. DICOM conversion,
registration, clinical adjudication and upstream conventional-feature extraction
are supplied before execution; see [scope](docs/scope_and_provenance.md).
Participant-level data, images and fitted clinical models are not distributed.
Generated clinical outputs belong in the institution's approved storage.

See [code availability](docs/availability.md), [citation](CITATION.md) and
[reuse terms](LICENSE_STATUS.md). Dependencies retain their own licenses.
