"""Training: load -> clean -> sample -> split -> fit -> evaluate -> persist."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .bundle import ModelBundle
from .data import load_sample, split_xy
from .evaluate import compute_metrics, feature_importance
from .models import build_pipeline

logger = logging.getLogger(__name__)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_split(
    data_path: Path = config.DATA_PATH,
    sample_frac: float = config.SAMPLE_FRAC,
    nrows: int | None = None,
    random_state: int = config.RANDOM_STATE,
) -> Split:
    """Read, clean, stratified-subsample and split the data (done once for all models)."""
    df = load_sample(data_path, sample_frac, random_state, nrows)
    X_train, X_test, y_train, y_test = split_xy(df, random_state=random_state)
    logger.info("train=%s test=%s positive rate=%.3f", X_train.shape, X_test.shape, y_train.mean())
    return Split(X_train, X_test, y_train, y_test)


def fit_and_evaluate(
    model_name: str,
    split: Split,
    params: dict[str, Any] | None = None,
    threshold: float = config.THRESHOLD,
    random_state: int = config.RANDOM_STATE,
) -> ModelBundle:
    """Fit one model with tuned (or overridden) parameters and score it on train and test."""
    pipeline = build_pipeline(model_name, params, random_state=random_state)

    start = time.perf_counter()
    pipeline.fit(split.X_train, split.y_train)
    fit_seconds = time.perf_counter() - start

    metrics: dict[str, Any] = {
        "train": compute_metrics(
            split.y_train, pipeline.predict_proba(split.X_train)[:, 1], threshold
        ),
        "test": compute_metrics(
            split.y_test, pipeline.predict_proba(split.X_test)[:, 1], threshold
        ),
        "fit_seconds": round(fit_seconds, 2),
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
    }
    logger.info(
        "%s: test F1=%.4f AUC=%.4f (fit %.1fs)",
        model_name,
        metrics["test"]["f1"],
        metrics["test"]["roc_auc"],
        fit_seconds,
    )
    return ModelBundle(
        model_name=model_name, pipeline=pipeline, threshold=threshold, metrics=metrics
    )


def save_run(bundle: ModelBundle, out_dir: Path) -> Path:
    """Persist model, metrics and feature importance under `out_dir/<model_name>/`."""
    run_dir = Path(out_dir) / bundle.model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle.save(run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(bundle.metrics, indent=2), encoding="utf-8")
    feature_importance(bundle.pipeline).to_csv(run_dir / "feature_importance.csv", index=False)
    return run_dir


def train_models(
    model_names: list[str],
    data_path: Path = config.DATA_PATH,
    sample_frac: float = config.SAMPLE_FRAC,
    out_dir: Path = config.ARTIFACTS_DIR,
    threshold: float = config.THRESHOLD,
    params: dict[str, Any] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Train several models on one shared split and return a comparison table."""
    split = prepare_split(data_path, sample_frac, nrows)
    rows = []
    for name in model_names:
        bundle = fit_and_evaluate(name, split, params, threshold)
        save_run(bundle, out_dir)
        rows.append(
            {"model": name, **bundle.metrics["test"], "fit_seconds": bundle.metrics["fit_seconds"]}
        )
    table = pd.DataFrame(rows).set_index("model").round(4)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out_dir) / "comparison.csv")
    return table
