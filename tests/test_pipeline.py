import json
import subprocess
import sys

import pandas as pd
import pytest

from insurance_cross_sell import config
from insurance_cross_sell.bundle import ModelBundle
from insurance_cross_sell.cli import main
from insurance_cross_sell.evaluate import feature_importance
from insurance_cross_sell.models import build_pipeline
from insurance_cross_sell.train import fit_and_evaluate, prepare_split, save_run


@pytest.mark.parametrize("name", config.MODEL_NAMES)
def test_every_model_fits_and_predicts(name, raw_df, fast_params):
    X = raw_df[config.FEATURE_COLUMNS]
    y = raw_df[config.TARGET]

    pipeline = build_pipeline(name, fast_params[name], n_jobs=1).fit(X, y)
    proba = pipeline.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("svm")


def test_tuned_params_are_applied():
    pipeline = build_pipeline("lightgbm")
    clf = pipeline.named_steps["classifier"]
    assert clf.num_leaves == config.BEST_PARAMS["lightgbm"]["num_leaves"]


def test_models_learn_signal(csv_path, fast_params):
    split = prepare_split(csv_path, sample_frac=1.0)
    bundle = fit_and_evaluate("lightgbm", split, fast_params["lightgbm"])
    assert bundle.metrics["test"]["roc_auc"] > 0.75


def test_feature_importance_names_match_transformed_columns(raw_df, fast_params):
    X = raw_df[config.FEATURE_COLUMNS]
    pipeline = build_pipeline("lightgbm", fast_params["lightgbm"]).fit(X, raw_df[config.TARGET])

    importance = feature_importance(pipeline)

    assert set(importance["feature"]) == set(pipeline[:-1].get_feature_names_out())
    # The synthetic target is driven by damage / prior insurance, not by sex.
    top = set(importance["feature"].head(3))
    assert {"Vehicle_Damage", "Previously_Insured"} <= top


def test_saved_model_loads_in_fresh_process_and_predicts(tmp_path, csv_path, raw_df, fast_params):
    """Guards against pickling custom classes defined in __main__ / a notebook."""
    split = prepare_split(csv_path, sample_frac=1.0)
    bundle = fit_and_evaluate("catboost", split, fast_params["catboost"])
    run_dir = save_run(bundle, tmp_path / "artifacts")
    sample = raw_df.drop(columns=[config.TARGET]).head(5)
    sample.to_csv(tmp_path / "new.csv")

    code = (
        "from insurance_cross_sell.bundle import ModelBundle; import pandas as pd, sys;"
        "b = ModelBundle.load(sys.argv[1]);"
        "df = pd.read_csv(sys.argv[2], index_col='id');"
        "print(len(b.predict(df)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(run_dir / "model.joblib"), str(tmp_path / "new.csv")],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "5"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "feature_importance.csv").exists()


def test_bundle_rejects_bad_input(tmp_path, raw_df, fast_params):
    X = raw_df[config.FEATURE_COLUMNS]
    pipeline = build_pipeline("logreg", fast_params["logreg"]).fit(X, raw_df[config.TARGET])
    bundle = ModelBundle("logreg", pipeline)
    path = bundle.save(tmp_path / "m.joblib")

    loaded = ModelBundle.load(path)
    out = loaded.predict(X.head(3))
    assert list(out.columns) == ["probability", "prediction"]

    with pytest.raises(ValueError, match="Missing required columns"):
        loaded.predict(X.drop(columns=["Age"]))


def test_cli_train_then_predict(tmp_path, csv_path, raw_df, capsys, monkeypatch, fast_params):
    from insurance_cross_sell import config as cfg

    # keep the CLI run fast: shrink the stored best params
    monkeypatch.setitem(cfg.BEST_PARAMS, "lightgbm", fast_params["lightgbm"])
    out_dir = tmp_path / "out"

    assert (
        main(
            [
                "train",
                "--model",
                "lightgbm",
                "--data-path",
                str(csv_path),
                "--sample-frac",
                "1.0",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    metrics = json.loads((out_dir / "lightgbm" / "metrics.json").read_text())
    assert 0 <= metrics["test"]["f1"] <= 1

    new = raw_df.drop(columns=[config.TARGET]).head(4)
    new.to_csv(tmp_path / "new.csv")
    assert (
        main(
            [
                "predict",
                "--model-path",
                str(out_dir / "lightgbm" / "model.joblib"),
                "--input",
                str(tmp_path / "new.csv"),
                "--output",
                str(tmp_path / "pred.csv"),
            ]
        )
        == 0
    )
    pred = pd.read_csv(tmp_path / "pred.csv")
    assert len(pred) == 4
    assert set(pred["prediction"]) <= {0, 1}


def test_cli_rejects_params_file_for_all_models(tmp_path):
    params = tmp_path / "p.json"
    params.write_text("{}")
    with pytest.raises(SystemExit):
        main(["train", "--params-file", str(params)])
