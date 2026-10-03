import pytest

from insurance_cross_sell import config
from insurance_cross_sell.data import (
    SchemaError,
    clean,
    load_sample,
    split_xy,
    stratified_sample,
    validate_features,
)


def test_clean_drops_anomalous_region(raw_df):
    df = raw_df.copy()
    df.loc[0, "Region_Code"] = config.ANOMALOUS_REGION_CODE

    cleaned = clean(df)

    assert 0 not in cleaned.index
    assert len(cleaned) == len(df) - 1


def test_stratified_sample_preserves_balance(raw_df):
    sample = stratified_sample(raw_df, 0.5)

    assert len(sample) == pytest.approx(len(raw_df) * 0.5, abs=2)
    assert sample[config.TARGET].mean() == pytest.approx(raw_df[config.TARGET].mean(), abs=0.01)


def test_split_is_stratified_and_disjoint(raw_df):
    X_train, X_test, y_train, y_test = split_xy(raw_df)

    assert set(X_train.index).isdisjoint(X_test.index)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)
    assert list(X_train.columns) == config.FEATURE_COLUMNS


def test_load_sample_reads_in_chunks_and_stays_stratified(csv_path, raw_df):
    sample = load_sample(csv_path, frac=0.5, chunksize=300)

    assert len(sample) == pytest.approx(len(raw_df) * 0.5, abs=15)
    assert sample[config.TARGET].mean() == pytest.approx(raw_df[config.TARGET].mean(), abs=0.02)
    assert sample["Age"].dtype == "int16"


def test_load_sample_drops_anomalous_region_across_chunks(tmp_path, raw_df):
    df = raw_df.copy()
    df.loc[1500, "Region_Code"] = config.ANOMALOUS_REGION_CODE
    path = tmp_path / "anomaly.csv"
    df.to_csv(path)

    sample = load_sample(path, frac=1.0, chunksize=500)

    assert 1500 not in sample.index
    assert len(sample) == len(df) - 1


def test_load_sample_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="insurance download"):
        load_sample(tmp_path / "nope.csv")


def test_validate_rejects_missing_column(raw_df):
    with pytest.raises(SchemaError, match="Missing required columns"):
        validate_features(raw_df.drop(columns=["Age"]))


def test_validate_rejects_unknown_category(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Gender"] = "Other"
    with pytest.raises(SchemaError, match="Gender"):
        validate_features(bad)


def test_validate_rejects_missing_values(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Annual_Premium"] = None
    with pytest.raises(SchemaError, match="Missing values"):
        validate_features(bad)


def test_validate_rejects_negative_numbers(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Annual_Premium"] = -1
    with pytest.raises(SchemaError, match="negative"):
        validate_features(bad)
