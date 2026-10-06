"""Project-wide constants: paths, column groups, tuned parameters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Repository root (src/insurance_cross_sell/config.py -> ../..).
# Override with INSURANCE_PROJECT_ROOT when the package is installed
# as a regular (non-editable) wheel.
PROJECT_ROOT = Path(
    os.environ.get(
        "INSURANCE_PROJECT_ROOT", Path(__file__).resolve().parents[2]
    )
)

# --- data -------------------------------------------------------------
# The old `media.githubusercontent.com/.../refs/heads/main/...` link
# from the notebook now returns 404; this one follows the Git LFS
# redirect.
DATA_URL = (
    "https://github.com/taysumova/urfu_ml/raw/main/data_sources/train.csv"
)
# From the Git LFS pointer of that file: used to verify the download.
DATA_SIZE_BYTES = 662_779_095
DATA_SHA256 = (
    "2bd6bd083cdfb9194dce39e521c4bbf4ca7a8fe8a221af35bb94a242d28b4517"
)
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET = "Response"
ID_COLUMN = "id"

# The notebook trains on a stratified 3% subsample (~345k rows).
SAMPLE_FRAC = 0.03
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 3  # used only by the optional tuning commands

# Decision threshold applied to predicted probabilities.
# NB: the research notebook accepted `threshold=0.69` but silently
# used 0.5.
THRESHOLD = 0.5

# --- features ---------------------------------------------------------
HIGH_CARDINALITY_COLS = ["Region_Code", "Policy_Sales_Channel"]
BINARY_COLS = [
    "Gender",
    "Vehicle_Damage",
    "Previously_Insured",
    "Driving_License",
]
ORDINAL_COL = "Vehicle_Age"
ORDINAL_ORDER = ["< 1 Year", "1-2 Year", "> 2 Years"]
OUTLIER_NUM_COLS = ["Annual_Premium"]
NUM_COLS = ["Age", "Vintage"]

FEATURE_COLUMNS = [
    *HIGH_CARDINALITY_COLS,
    *BINARY_COLS,
    ORDINAL_COL,
    *OUTLIER_NUM_COLS,
    *NUM_COLS,
]

# Allowed values of categorical inputs (checked at inference).
ALLOWED_VALUES: dict[str, set[Any]] = {
    "Gender": {"Male", "Female"},
    "Vehicle_Damage": {"Yes", "No"},
    "Vehicle_Age": set(ORDINAL_ORDER),
    "Driving_License": {0, 1},
    "Previously_Insured": {0, 1},
}

# One row with this Region_Code is an anomaly in the data (see EDA).
ANOMALOUS_REGION_CODE = 39.2

# --- tuned hyperparameters --------------------------------------------
# Taken from the notebook runs on the 3% sample, so the final models
# can be re-fitted in seconds instead of repeating hours of
# grid/Optuna search.
# Keys are bare estimator parameters (no `classifier__` prefix).
BEST_PARAMS: dict[str, dict[str, Any]] = {
    # GridSearchCV best: C=100, l1_ratio=0.1, elasticnet
    "logreg": {"C": 100, "l1_ratio": 0.1},
    # GridSearchCV best (test F1 0.439)
    "random_forest": {
        "n_estimators": 50,
        "max_depth": 10,
        "max_features": 0.5,
        "max_samples": 0.1,
        "min_samples_leaf": 50,
        "class_weight": "balanced",
    },
    # Optuna best (CV F1 0.4751, test F1 0.475)
    "lightgbm": {
        "num_leaves": 98,
        "max_depth": 9,
        "learning_rate": 0.017023810747318804,
        "n_estimators": 753,
        "subsample": 0.9233625854586776,
        "colsample_bytree": 0.9398069633632562,
        "reg_alpha": 0.1565772322629042,
        "reg_lambda": 0.35805661012144113,
        "scale_pos_weight": 3.5183751581022755,
    },
    # Optuna best (CV F1 0.4673, test F1 0.465)
    "catboost": {
        "iterations": 171,
        "learning_rate": 0.06439694378125348,
        "depth": 7,
        "l2_leaf_reg": 4,
        "scale_pos_weight": 2.966658829058813,
        "colsample_bylevel": 0.5350507509284778,
        "random_strength": 0.2317976866851252,
        "border_count": 202,
    },
}

MODEL_NAMES = list(BEST_PARAMS)
