"""Batch inference from a saved model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .bundle import ModelBundle


def predict_file(model_path: Path, input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Score a CSV with the raw feature columns and write probability + prediction."""
    bundle = ModelBundle.load(model_path)
    df = pd.read_csv(input_csv)
    if config.ID_COLUMN in df.columns:
        df = df.set_index(config.ID_COLUMN)
    result = bundle.predict(df)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv)
    return result
