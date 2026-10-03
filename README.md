# Insurance cross-sell: production ML pipeline

Binary classification: will a health-insurance customer also buy vehicle insurance?
Dataset: Kaggle Playground S4E7 (11.5M rows, ~12% positives, metric F1).

The project was refactored from a team research notebook
([`notebooks/01_research_notebook.ipynb`](notebooks/01_research_notebook.ipynb))
into an installable, tested package with a CLI, linters, pre-commit and Docker.

Team "Страховщики": Роман Осипов, Дмитрий Мартынов, Дмитрий Иванков.

## Quick start

```bash
poetry install                      # creates .venv inside the project from poetry.lock
poetry run pre-commit install       # enable git hooks

poetry run insurance download       # cache the dataset in data/raw/ (632 MB, git-ignored, checksum-verified)
poetry run insurance train          # fit all models on the 3% stratified sample
poetry run insurance predict --model-path artifacts/lightgbm/model.joblib \
    --input new_customers.csv --output predictions.csv
```

Requirements: Python 3.10-3.13 and [Poetry](https://python-poetry.org/) 2.x
(`pipx install poetry` or `pip install --user poetry`).
The EDA-only libraries (`phik`, `klib`, `shap`, `matplotlib`) are an optional group:
`poetry install --with eda`.

## No hours-long retraining

The original notebook spent hours on GridSearchCV/Optuna. The best hyperparameters from
those runs are stored in [`config.BEST_PARAMS`](src/insurance_cross_sell/config.py), so
`insurance train` only **fits each final model once** (seconds to a few minutes on the 3%
sample). Searching is a separate, optional command:

```bash
poetry run insurance tune --model lightgbm --method optuna --n-trials 30
poetry run insurance train --model lightgbm --params-file artifacts/best_params_lightgbm.json
```

Quick smoke run on a slice of the data: `insurance train --nrows 50000 --sample-frac 1.0`.

`insurance download` fetches the 632 MB file in 32 parallel range requests by default
(`--workers`): on some networks a single connection to GitHub LFS is throttled to tens of
KB/s, while parallel connections reach the full bandwidth. The file is verified by SHA-256.

## Layout

```
src/insurance_cross_sell/
  config.py    constants, column groups, tuned hyperparameters
  data.py      download, load, schema validation, cleaning, stratified sample/split
  features.py  ThresholdFrequencyEncoder, CategoryCaster, preprocessors
  models.py    pipeline per model: logreg, random_forest, lightgbm, catboost
  evaluate.py  metrics, feature importance
  bundle.py    pipeline + threshold + metadata, saved with joblib
  train.py     training workflow        tuning.py  optional grid / Optuna search
  predict.py   batch inference          cli.py     `insurance` command
tests/         pytest suite (synthetic data, no download needed)
notebooks/     01 archived research notebook (outputs stripped), 02 demo that uses the package
docs/          results of the original runs
```

## Quality tooling

| Tool | Purpose | Run |
|---|---|---|
| Poetry | dependency management, lock file, in-project `.venv` | `poetry install` |
| ruff | lint + format | `poetry run ruff check .` |
| mypy | static types | `poetry run mypy` |
| pre-commit | ruff, mypy, nbstripout, whitespace/EOF, large-file guard on every commit | `poetry run pre-commit run --all-files` |
| pytest | unit and end-to-end tests | `poetry run pytest` |
| GitHub Actions | pre-commit + tests + Docker build on push/PR | `.github/workflows/ci.yml` |

### About the virtual environment and Git

`.venv/` is **not** committed (hundreds of MB, tied to one OS and path). What is committed
is everything needed to recreate it exactly: `pyproject.toml`, `poetry.lock` and
`poetry.toml` (`in-project = true`). A fresh clone is ready with `poetry install`.

## Security note

Models are stored with `joblib` (pickle). Loading a pickle can execute arbitrary code, so
only load `model.joblib` files that you produced yourself or got from a trusted source.

## Docker

```bash
docker build -t insurance-cross-sell .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
    insurance-cross-sell train --model lightgbm
```

## Changes relative to the notebook

Bugs found during the refactor and how they are handled:

- **Threshold**: `ModelFactory(threshold=0.69)` was ignored (hard-coded 0.5). Now one
  explicit `THRESHOLD` in config, overridable with `--threshold`.
- **Feature importance names**: labels came from `X_train.columns` while the
  `ColumnTransformer` reorders columns. Names now come from `get_feature_names_out()`,
  so importances must be re-read (the notebook's "Annual_Premium 50%" is likely
  `Previously_Insured`).
- **Categorical handling in LightGBM/CatBoost** is now explicit (`category` dtype /
  `cat_features`) instead of relying on `remainder="passthrough"`.
- **Logistic regression** had identical "Fixed / Grid / Optuna" results in the notebook
  because `class_weight='balanced'` was hard-coded in all three; it is now a single model
  with one tuned configuration.
- **Memory**: the CSV is read in 1M-row chunks and subsampled chunk by chunk, so the 11.5M-row
  frame is never held in memory (the notebook loaded it entirely).
- **Data URL**: the `media.githubusercontent.com/.../refs/heads/main/...` link from the
  notebook returns 404 now; `config.DATA_URL` uses the working LFS link and the download is
  verified by SHA-256.
- **Custom transformer** lives in the package, so saved models load in any process.
- **Optuna** uses the public sklearn scoring API instead of `scorer._score_func`; the
  Random Forest search space no longer includes the degenerate
  `min_weight_fraction_leaf` up to 0.5.
- Input validation at inference, logging instead of `print`, no global state.

Because categorical handling changed, final scores can differ slightly from the notebook;
see [`docs/RESULTS.md`](docs/RESULTS.md) for the original numbers to compare against.
