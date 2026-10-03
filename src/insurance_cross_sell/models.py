"""Model registry: one sklearn `Pipeline` (preprocessor + classifier) per model name."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .features import build_encoded_preprocessor, build_native_preprocessor


def build_pipeline(
    name: str,
    params: dict[str, Any] | None = None,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = -1,
) -> Pipeline:
    """Build an unfitted pipeline for `name`; `params` override the tuned defaults."""
    if name not in config.BEST_PARAMS:
        raise ValueError(f"Unknown model {name!r}; choose from {config.MODEL_NAMES}")
    merged = {**config.BEST_PARAMS[name], **(params or {})}

    classifier: Any
    if name == "logreg":
        preprocessor = build_encoded_preprocessor()
        classifier = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            class_weight="balanced",
            random_state=random_state,
            **{"max_iter": 1000, "tol": 1e-4, **merged},
        )
    elif name == "random_forest":
        preprocessor = build_encoded_preprocessor()
        classifier = RandomForestClassifier(
            random_state=random_state, n_jobs=n_jobs, bootstrap=True, **merged
        )
    elif name == "lightgbm":
        preprocessor = build_native_preprocessor(cat_as_str=False)
        # NB: as in the notebook, `subsample` has no effect without `subsample_freq`;
        # kept as-is so the tuned parameters reproduce the reported scores.
        classifier = LGBMClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
            importance_type="gain",
            **merged,
        )
    else:  # catboost (imported lazily: heavy optional-at-import dependency)
        from catboost import CatBoostClassifier

        preprocessor = build_native_preprocessor(cat_as_str=True)
        classifier = CatBoostClassifier(
            random_seed=random_state,
            thread_count=n_jobs,
            verbose=0,
            allow_writing_files=False,
            loss_function="Logloss",
            boosting_type="Plain",
            bootstrap_type="Bernoulli",
            subsample=0.8,
            cat_features=config.HIGH_CARDINALITY_COLS,
            **merged,
        )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
