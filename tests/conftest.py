"""Shared fixtures: a small synthetic dataset with the real schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_cross_sell import config

# Tiny hyperparameters so every model fits in about a second.
FAST_PARAMS = {
    "logreg": {"C": 1.0, "l1_ratio": 0.5, "max_iter": 50},
    "random_forest": {"n_estimators": 10, "max_depth": 5, "max_samples": 0.5},
    "lightgbm": {"n_estimators": 20, "num_leaves": 8, "max_depth": 4},
    "catboost": {"iterations": 20, "depth": 4},
}


def make_synthetic(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Rows shaped like the Kaggle data.

    `Response` depends on damage and prior insurance.
    """
    rng = np.random.default_rng(seed)
    damage = rng.choice(["Yes", "No"], n)
    insured = rng.integers(0, 2, n)
    age = rng.integers(20, 86, n)
    logit = -2.5 + 2.5 * (damage == "Yes") - 2.0 * insured + 0.02 * (age - 40)
    response = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": age,
            "Driving_License": rng.choice([0, 1], n, p=[0.02, 0.98]),
            "Region_Code": rng.choice([8.0, 28.0, 41.0, 46.0, 3.0, 11.0], n),
            "Previously_Insured": insured,
            "Vehicle_Age": rng.choice(config.ORDINAL_ORDER, n),
            "Vehicle_Damage": damage,
            "Annual_Premium": rng.normal(30000, 15000, n).clip(2500, 540000),
            "Policy_Sales_Channel": rng.choice(
                [26.0, 124.0, 152.0, 160.0, 7.0], n
            ),
            "Vintage": rng.integers(10, 300, n),
            config.TARGET: response,
        },
        index=pd.RangeIndex(n, name=config.ID_COLUMN),
    )
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture
def csv_path(tmp_path, raw_df):
    path = tmp_path / "train.csv"
    raw_df.to_csv(path)
    return path


@pytest.fixture(scope="session")
def fast_params() -> dict[str, dict]:
    return FAST_PARAMS
