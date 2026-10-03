from insurance_cross_sell import config
from insurance_cross_sell.tuning import optuna_search


def test_optuna_search_returns_params_and_score(raw_df):
    X = raw_df[config.FEATURE_COLUMNS]
    y = raw_df[config.TARGET]

    best, score = optuna_search("logreg", X, y, n_trials=2, folds=2)

    assert set(best) == {"C", "l1_ratio"}
    assert 0 <= score <= 1
