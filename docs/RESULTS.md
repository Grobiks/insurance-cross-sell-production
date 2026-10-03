# Results of the original notebook runs

Setup: stratified 3% sample (345,144 rows), train 258,858 / test 86,286, threshold 0.5,
metric F1. Source: outputs of `01_research_notebook.ipynb` before they were stripped.

## Test set (F1 / ROC-AUC)

| Model | Fixed | GridSearchCV | Optuna |
|---|---|---|---|
| LogisticRegression (elasticnet, balanced) | 0.401 / 0.832 | 0.401 / 0.832 | 0.401 / 0.832 |
| RandomForest | 0.201 / 0.841 (train F1 0.998: overfit) | 0.438 / 0.857 | 0.434 / 0.859 |
| LightGBM | 0.443 / 0.868 | 0.474 / 0.871 | **0.475 / 0.871** |
| CatBoost | 0.450 / 0.871 | 0.469 / 0.868 | 0.465 / 0.864 |

## Search cost (why parameters are stored instead of re-searched)

- GridSearchCV: from about 2 up to about 22 minutes per model
- Optuna (30 trials): from about 5 up to about 17 minutes per model
- All of the above on the 3% sample; the full dataset would take far longer

## Best parameters

See `BEST_PARAMS` in `src/insurance_cross_sell/config.py`.
Winner: LightGBM from Optuna (CV F1 0.4751, test F1 0.4754).

## Notes from the original EDA

- Strongest signals: `Vehicle_Damage` (+0.54), `Previously_Insured` (+0.52), `Age` (+0.31),
  `Policy_Sales_Channel` (+0.27), `Region_Code` (+0.15) by phik correlation.
- No missing values or duplicates; one row with `Region_Code = 39.2` is dropped.

## Reproduced with the refactored package

`poetry run insurance train` on the real dataset (632 MB, SHA-256 verified), 3% stratified
sample, threshold 0.5. The whole run (reading 11.5M rows in chunks, fitting four models)
takes under two minutes because the tuned hyperparameters are reused.

| Model | Test F1 / ROC-AUC (package) | Notebook (best variant) | Fit time |
|---|---|---|---|
| LogisticRegression | 0.403 / 0.836 | 0.401 / 0.832 | 14.5 s |
| RandomForest | 0.444 / 0.861 | 0.438 / 0.857 (Grid) | 2.0 s |
| LightGBM | **0.477 / 0.873** | 0.475 / 0.871 (Optuna) | 6.2 s |
| CatBoost | 0.472 / 0.868 | 0.465 / 0.864 (Optuna) | 11.6 s |

The numbers agree with the notebook within about 0.007 F1; the small differences come from
a different random subsample and, for CatBoost, from declaring `Region_Code` and
`Policy_Sales_Channel` as categorical features.

### Feature importance fix confirmed

The notebook reported `Annual_Premium` as the top CatBoost feature (50%). With names taken
from the transformed columns the top features are `Previously_Insured` (47%) and
`Vehicle_Damage` (20%), consistent with the EDA correlations. The old labels were shifted
by the `ColumnTransformer` column reordering.
