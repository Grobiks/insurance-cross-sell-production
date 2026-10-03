"""Custom transformers and preprocessors.

These classes live in the package (not in a notebook) so that a pickled model can be
loaded from any process that has `insurance_cross_sell` installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from . import config


class ThresholdFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its relative frequency in the training data.

    Categories rarer than `threshold` are merged into a single "other" bucket whose
    value is their combined frequency. Categories unseen during `fit` get that value
    too, so inference never fails on new codes.
    """

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> ThresholdFrequencyEncoder:
        self.columns_ = list(X.columns)
        self.freq_maps_: dict[str, dict[str, float]] = {}
        self.other_freq_: dict[str, float] = {}
        for column in self.columns_:
            freqs = X[column].astype(str).value_counts(normalize=True)
            common = freqs[freqs >= self.threshold]
            self.freq_maps_[column] = {str(k): float(v) for k, v in common.items()}
            self.other_freq_[column] = float(freqs[freqs < self.threshold].sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = {}
        for column in self.columns_:
            values = X[column].astype(str).map(self.freq_maps_[column])
            encoded[column] = values.fillna(self.other_freq_[column]).astype(float)
        return pd.DataFrame(encoded, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Turn columns into categorical features for models with native category support.

    `as_str=False` produces a pandas `category` dtype with the categories fixed at fit
    time (LightGBM); `as_str=True` produces plain strings (CatBoost `cat_features`).
    Values unseen during `fit` become NaN / "nan".
    """

    def __init__(self, as_str: bool = False) -> None:
        self.as_str = as_str

    def fit(self, X: pd.DataFrame, y: Any = None) -> CategoryCaster:
        self.columns_ = list(X.columns)
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in self.columns_}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for column in self.columns_:
            if self.as_str:
                out[column] = X[column].astype(str).astype(object)
            else:
                out[column] = pd.Series(
                    pd.Categorical(X[column], categories=self.categories_[column]),
                    index=X.index,
                )
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


def build_encoded_preprocessor() -> ColumnTransformer:
    """Fully numeric features for linear models and random forest."""
    return ColumnTransformer(
        transformers=[
            ("premium", RobustScaler(), config.OUTLIER_NUM_COLS),
            ("numeric", StandardScaler(), config.NUM_COLS),
            ("frequency", ThresholdFrequencyEncoder(threshold=0.01), config.HIGH_CARDINALITY_COLS),
            (
                "binary",
                OneHotEncoder(drop="if_binary", sparse_output=False, dtype=np.int8),
                config.BINARY_COLS,
            ),
            (
                "ordinal",
                OrdinalEncoder(categories=[config.ORDINAL_ORDER], dtype=np.int8),
                [config.ORDINAL_COL],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_native_preprocessor(cat_as_str: bool) -> ColumnTransformer:
    """Features for gradient boosting: no scaling, high-cardinality codes stay categorical."""
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", [*config.OUTLIER_NUM_COLS, *config.NUM_COLS]),
            ("binary", OrdinalEncoder(dtype=np.int8), config.BINARY_COLS),
            (
                "ordinal",
                OrdinalEncoder(categories=[config.ORDINAL_ORDER], dtype=np.int8),
                [config.ORDINAL_COL],
            ),
            ("categorical", CategoryCaster(as_str=cat_as_str), config.HIGH_CARDINALITY_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
