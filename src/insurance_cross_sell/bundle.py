"""A fitted pipeline packaged with everything needed for inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from . import __version__, config
from .data import validate_features


@dataclass
class ModelBundle:
    """A fitted pipeline with its threshold and metadata."""

    model_name: str
    pipeline: Pipeline
    threshold: float = config.THRESHOLD
    feature_columns: list[str] = field(
        default_factory=lambda: list(config.FEATURE_COLUMNS)
    )
    metrics: dict[str, Any] = field(default_factory=dict)
    package_version: str = __version__

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate raw input; return probability and 0/1 per row."""
        validate_features(df)
        proba = self.pipeline.predict_proba(df[self.feature_columns])[:, 1]
        return pd.DataFrame(
            {
                "probability": proba,
                "prediction": (proba >= self.threshold).astype(int),
            },
            index=df.index,
        )

    def save(self, path: Path) -> Path:
        """Write the bundle to `path` with joblib; return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> ModelBundle:
        """Read a bundle from `path`; reject other pickled objects."""
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle
