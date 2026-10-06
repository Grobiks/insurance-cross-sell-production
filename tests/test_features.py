import numpy as np
import pandas as pd

from insurance_cross_sell import config
from insurance_cross_sell.features import (
    CategoryCaster,
    ThresholdFrequencyEncoder,
    build_encoded_preprocessor,
    build_native_preprocessor,
)


def test_frequency_encoder_merges_rare_and_handles_unseen():
    x = pd.DataFrame({"c": ["a"] * 90 + ["b"] * 9 + ["z"]})
    enc = ThresholdFrequencyEncoder(threshold=0.05).fit(x)

    out = enc.transform(pd.DataFrame({"c": ["a", "b", "z", "never-seen"]}))

    assert out["c"].tolist() == [0.9, 0.09, 0.01, 0.01]


def test_frequency_encoder_keeps_index():
    x = pd.DataFrame({"c": [1.0, 2.0, 2.0]}, index=[10, 11, 12])
    out = ThresholdFrequencyEncoder(0.0).fit_transform(x)
    assert out.index.tolist() == [10, 11, 12]


def test_category_caster_fixes_categories_at_fit_time():
    caster = CategoryCaster().fit(pd.DataFrame({"r": [1.0, 2.0, 3.0]}))
    out = caster.transform(pd.DataFrame({"r": [3.0, 99.0]}))

    assert str(out["r"].dtype) == "category"
    assert list(out["r"].cat.categories) == [1.0, 2.0, 3.0]
    assert out["r"].isna().tolist() == [False, True]  # unseen code -> NaN


def test_encoded_preprocessor_is_numeric_and_named(raw_df):
    x = raw_df[config.FEATURE_COLUMNS]
    out = build_encoded_preprocessor().fit_transform(x)

    assert len(out) == len(x)
    assert not out.isna().any().any()
    assert all(np.issubdtype(t, np.number) for t in out.dtypes)
    assert {
        "Annual_Premium",
        "Vehicle_Age",
        "Gender_Male",
        "Vehicle_Damage_Yes",
    } <= set(out.columns)


def test_native_preprocessor_keeps_categoricals(raw_df):
    x = raw_df[config.FEATURE_COLUMNS]
    out = build_native_preprocessor(cat_as_str=False).fit_transform(x)

    assert str(out["Region_Code"].dtype) == "category"
    assert str(out["Policy_Sales_Channel"].dtype) == "category"
    assert out["Gender"].isin([0, 1]).all()
