"""Metrics and feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from . import config


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    threshold: float = config.THRESHOLD,
) -> dict[str, float]:
    """Precision, recall, F1 at `threshold`, and ROC-AUC."""
    pred = (proba >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return importance per *transformed* feature, with its name.

    The research notebook labelled importances with the raw
    `X_train.columns`, but the ColumnTransformer reorders columns, so
    the names were wrong. Using `get_feature_names_out()` keeps names
    and values aligned.
    """
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    else:
        values = np.abs(np.asarray(classifier.coef_, dtype=float)).ravel()
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
