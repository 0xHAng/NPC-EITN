# Environment

Release verification used CPython 3.11.16 on macOS-27.0.1-arm64-arm-64bit. Core numerical
versions are listed below; requirements-lock.txt pins the full tested environment.
Use the pinned requirements for reproducible software checks.

```json
{
  "numpy": "2.4.6",
  "pandas": "2.3.3",
  "scipy": "1.17.1",
  "sklearn": "1.8.0",
  "lifelines": "0.30.3",
  "joblib": "1.6.0",
  "nibabel": "5.5.0",
  "skimage": "0.26.0",
  "matplotlib": "3.11.2"
}
```

From the repository root on macOS/Linux:

```sh
python3.11 -m venv /tmp/npc_eitn_env
. /tmp/npc_eitn_env/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/check_package.py
python -B -m unittest discover -s tests -v
```

Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for deterministic single-thread
verification if needed. Matplotlib is
needed only for summary plotting; nibabel is used by the registered-NIfTI CLI.
No dependencies are vendored. External clinical sources are not downloaded by
any script; no repository entry communicates with a patient database or service.

GitHub Actions runs the tests and synthetic workflow on Linux/Python 3.11.
