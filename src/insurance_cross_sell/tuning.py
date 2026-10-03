"""Optional hyperparameter search (GridSearchCV / Optuna).

Not needed to reproduce the project: the best parameters from the original long runs
are stored in `config.BEST_PARAMS`. Use this only to re-tune, e.g. on more data.
"""

from __future__ import annotations

import logging
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from . import config
from .models import build_pipeline

logger = logging.getLogger(__name__)

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logreg": {"C": [0.001, 0.01, 0.1, 1, 10, 100], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "random_forest": {
        "n_estimators": [50],
        "max_depth": [8, 10],
        "min_samples_leaf": [50, 70],
        "max_features": ["sqrt", 0.5],
        "max_samples": [0.1, 0.2],
    },
    "lightgbm": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7],
        "num_leaves": [31, 63],
        "scale_pos_weight": [3, 5],
    },
    "catboost": {
        "iterations": [100, 200],
        "learning_rate": [0.05, 0.1],
        "depth": [6, 8],
        "l2_leaf_reg": [3, 5],
        "scale_pos_weight": [1, 3],
    },
}


def _suggest(name: str, trial: optuna.Trial) -> dict[str, Any]:
    if name == "logreg":
        return {
            "C": trial.suggest_float("C", 0.001, 100, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "max_features": trial.suggest_float("max_features", 0.1, 0.9),
            "max_samples": trial.suggest_float("max_samples", 0.1, 0.5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100, log=True),
        }
    if name == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 20),
        }
    return {
        "iterations": trial.suggest_int("iterations", 100, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1),
        "depth": trial.suggest_int("depth", 6, 8),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 3, 5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 3),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }


def _cv(folds: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)


def grid_search(
    name: str, X: pd.DataFrame, y: pd.Series, folds: int = config.CV_FOLDS
) -> tuple[dict[str, Any], float]:
    """Exhaustive search over `PARAM_GRIDS[name]`, scored by CV F1."""
    pipeline = build_pipeline(name, n_jobs=1)
    grid = {f"classifier__{k}": v for k, v in PARAM_GRIDS[name].items()}
    search = GridSearchCV(pipeline, grid, scoring="f1", cv=_cv(folds), n_jobs=-1, verbose=1)
    search.fit(X, y)
    best = {k.removeprefix("classifier__"): v for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


def optuna_search(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
    folds: int = config.CV_FOLDS,
) -> tuple[dict[str, Any], float]:
    """TPE search over the space in `_suggest`, scored by CV F1 (train data only)."""

    def objective(trial: optuna.Trial) -> float:
        pipeline = build_pipeline(name, _suggest(name, trial))
        return float(cross_val_score(pipeline, X, y, scoring="f1", cv=_cv(folds)).mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best %s CV F1=%.4f params=%s", name, study.best_value, study.best_params)
    return dict(study.best_params), float(study.best_value)
