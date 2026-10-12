"""Automated retraining cycle with quality gates and a model registry.

One call of `run_cycle` walks through the stages described in
`Model_retrain_BPMN.md`: checks of the new data, training of a
challenger, quality gates against the production model (champion),
approval, registration of the version, deployment with a smoke test
and rollback on failure. Every cycle is written to a JSON report and
problems go to a notification log.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import config
from .bundle import ModelBundle
from .data import SchemaError, clean, validate_features
from .evaluate import compute_metrics, feature_importance
from .train import Split, fit_and_evaluate

logger = logging.getLogger(__name__)

NUMERIC_PROFILE = ["Age", "Annual_Premium", "Vintage"]
CATEGORICAL_PROFILE = [
    "Gender",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Previously_Insured",
]
AGE_BINS = [0, 29, 39, 49, 59, 200]
AGE_LABELS = ["20-29", "30-39", "40-49", "50-59", "60+"]


@dataclass
class GateConfig:
    """Thresholds of the data and quality checks."""

    min_auc: float = 0.80
    max_auc_drop: float = 0.002
    max_f1_drop: float = 0.005
    max_slice_auc_drop: float = 0.02
    positive_rate: float = 0.123
    positive_rate_tolerance: float = 0.03
    min_rows: int = 1_000
    psi_drift: float = 0.2
    min_slice_rows: int = 200
    expected_top_features: tuple[str, ...] = (
        "Vehicle_Damage",
        "Previously_Insured",
    )


@dataclass
class GateResult:
    """Outcome of one check."""

    name: str
    passed: bool
    detail: str


@dataclass
class CycleReport:
    """Everything that happened in one retraining cycle."""

    cycle: int
    trigger: str
    started_at: str
    data_files: list[str]
    status: str = "started"
    version: str | None = None
    champion: str | None = None
    production_after: str | None = None
    approved_by: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, float] = field(default_factory=dict)
    gates: list[GateResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_gates(self) -> list[GateResult]:
        """Return the checks that did not pass."""
        return [g for g in self.gates if not g.passed]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- drift -----------------------------------------------------------


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Return the population stability index of two share vectors."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def data_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Describe feature distributions for later drift checks."""
    profile: dict[str, Any] = {}
    for col in NUMERIC_PROFILE:
        quantiles = np.quantile(df[col], np.linspace(0, 1, 11)[1:-1])
        cuts = np.unique(quantiles)
        bins = np.searchsorted(cuts, df[col].to_numpy(), side="right")
        shares = np.bincount(bins, minlength=len(cuts) + 1) / len(df)
        profile[col] = {
            "kind": "numeric",
            "cuts": cuts.tolist(),
            "shares": shares.tolist(),
        }
    for col in CATEGORICAL_PROFILE:
        freqs = df[col].astype(str).value_counts(normalize=True)
        profile[col] = {
            "kind": "categorical",
            "shares": {str(k): float(v) for k, v in freqs.items()},
        }
    return profile


def drift_report(
    profile: dict[str, Any], df: pd.DataFrame
) -> dict[str, float]:
    """Return PSI of each profiled feature of `df` against `profile`."""
    out: dict[str, float] = {}
    for col, spec in profile.items():
        if spec["kind"] == "numeric":
            cuts = np.asarray(spec["cuts"])
            bins = np.searchsorted(cuts, df[col].to_numpy(), side="right")
            actual = np.bincount(bins, minlength=len(cuts) + 1) / len(df)
            expected = np.asarray(spec["shares"])
        else:
            exp = spec["shares"]
            act = df[col].astype(str).value_counts(normalize=True)
            keys = sorted(set(exp) | {str(k) for k in act.index})
            expected = np.array([exp.get(k, 0.0) for k in keys])
            actual = np.array([float(act.get(k, 0.0)) for k in keys])
        out[col] = round(psi(expected, actual), 4)
    return out


# --- registry --------------------------------------------------------


class ModelRegistry:
    """File-based model registry with a production pointer.

    Layout: `registry.json` (versions and their statuses),
    `<version>/model.joblib`, `cycles/cycle_NNN.json` (cycle reports)
    and `notifications.log`.
    """

    def __init__(self, root: Path) -> None:
        """Open (or create) a registry in the folder `root`."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"

    def _read(self) -> dict[str, Any]:
        if self.index_path.exists():
            return dict(json.loads(self.index_path.read_text("utf-8")))
        return {"production": None, "versions": []}

    def _write(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @property
    def versions(self) -> list[dict[str, Any]]:
        """Return all registered versions, oldest first."""
        return list(self._read()["versions"])

    def production_version(self) -> str | None:
        """Return the version that is currently in production."""
        prod = self._read()["production"]
        return str(prod) if prod else None

    def get(self, version: str) -> dict[str, Any]:
        """Return the metadata of `version`."""
        for entry in self._read()["versions"]:
            if entry["version"] == version:
                return dict(entry)
        raise KeyError(version)

    def load_bundle(self, version: str) -> ModelBundle:
        """Load the model of `version`."""
        return ModelBundle.load(self.root / version / "model.joblib")

    def register(
        self, bundle: ModelBundle, status: str, meta: dict[str, Any]
    ) -> str:
        """Save a model as a new version with the given status."""
        index = self._read()
        version = f"v{len(index['versions']) + 1}"
        bundle.save(self.root / version / "model.joblib")
        index["versions"].append(
            {
                "version": version,
                "status": status,
                "created_at": _now(),
                **meta,
            }
        )
        self._write(index)
        return version

    def _set_status(self, index: dict[str, Any], version: str, status: str):
        for entry in index["versions"]:
            if entry["version"] == version:
                entry["status"] = status

    def promote(self, version: str) -> str | None:
        """Make `version` production; return the previous production."""
        index = self._read()
        previous = index["production"]
        if previous:
            self._set_status(index, previous, "archived")
        self._set_status(index, version, "production")
        index["production"] = version
        self._write(index)
        return str(previous) if previous else None

    def rollback(self, failed: str, previous: str | None) -> None:
        """Mark `failed` as rolled back and restore `previous`."""
        index = self._read()
        self._set_status(index, failed, "rolled_back")
        index["production"] = previous
        if previous:
            self._set_status(index, previous, "production")
        self._write(index)

    def notify(self, level: str, message: str) -> None:
        """Append a notification for the people on duty."""
        line = f"{_now()} [{level}] {message}\n"
        with open(self.root / "notifications.log", "a", encoding="utf-8") as f:
            f.write(line)
        logger.log(getattr(logging, level, logging.WARNING), message)

    def next_cycle(self) -> int:
        """Return the number of the next retraining cycle."""
        cycles = self.root / "cycles"
        return (
            len(list(cycles.glob("cycle_*.json"))) + 1
            if cycles.exists()
            else 1
        )

    def save_report(self, report: CycleReport) -> Path:
        """Write the cycle report as JSON."""
        path = self.root / "cycles" / f"cycle_{report.cycle:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


# --- checks ----------------------------------------------------------


def check_data(df: pd.DataFrame, gates: GateConfig) -> list[GateResult]:
    """Validate a new batch before any training."""
    results = []
    try:
        validate_features(df)
        if config.TARGET not in df.columns:
            raise SchemaError(f"Missing target column {config.TARGET!r}")
        bad = set(df[config.TARGET].dropna().unique()) - {0, 1}
        if bad or df[config.TARGET].isna().any():
            raise SchemaError("Target must contain only 0 and 1")
        results.append(GateResult("schema", True, "columns and values ok"))
    except SchemaError as err:
        results.append(GateResult("schema", False, str(err)))
        return results

    results.append(
        GateResult(
            "min_rows",
            len(df) >= gates.min_rows,
            f"{len(df)} rows (minimum {gates.min_rows})",
        )
    )
    rate = float(df[config.TARGET].mean())
    results.append(
        GateResult(
            "positive_rate",
            abs(rate - gates.positive_rate) <= gates.positive_rate_tolerance,
            f"{rate:.3f} (expected {gates.positive_rate:.3f} "
            f"+/- {gates.positive_rate_tolerance:.3f})",
        )
    )
    return results


def slice_auc(
    y: pd.Series, proba: np.ndarray, x: pd.DataFrame, min_rows: int = 200
) -> dict[str, float]:
    """ROC-AUC inside groups by gender, age group and vehicle age."""
    groups = {
        "Gender": x["Gender"].astype(str),
        "Age": pd.cut(x["Age"], AGE_BINS, labels=AGE_LABELS).astype(str),
        "Vehicle_Age": x["Vehicle_Age"].astype(str),
    }
    out: dict[str, float] = {}
    y_arr = np.asarray(y)
    for name, values in groups.items():
        for value in sorted(values.unique()):
            mask = (values == value).to_numpy()
            if mask.sum() < min_rows or len(np.unique(y_arr[mask])) < 2:
                continue
            auc = roc_auc_score(y_arr[mask], proba[mask])
            out[f"{name}={value}"] = round(float(auc), 4)
    return out


def score(
    bundle: ModelBundle, x: pd.DataFrame, y: pd.Series, min_slice_rows: int
) -> dict[str, Any]:
    """Metrics of a model on the given data, including slices."""
    proba = bundle.pipeline.predict_proba(x)[:, 1]
    metrics: dict[str, Any] = compute_metrics(y, proba, bundle.threshold)
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics["slices"] = slice_auc(y, proba, x, min_slice_rows)
    return metrics


def check_quality(
    challenger: dict[str, Any],
    champion: dict[str, Any] | None,
    bundle: ModelBundle,
    control: pd.DataFrame,
    gates: GateConfig,
) -> list[GateResult]:
    """Quality gates for the challenger (see Model_retrain_BPMN.md)."""
    auc = challenger["roc_auc"]
    results = [
        GateResult(
            "min_auc",
            auc >= gates.min_auc,
            f"ROC-AUC {auc:.4f} (minimum {gates.min_auc:.2f})",
        )
    ]
    if champion is not None:
        auc_drop = champion["roc_auc"] - auc
        f1_drop = champion["f1"] - challenger["f1"]
        results.append(
            GateResult(
                "vs_champion_auc",
                auc_drop <= gates.max_auc_drop,
                f"{auc:.4f} vs champion {champion['roc_auc']:.4f} "
                f"(allowed drop {gates.max_auc_drop})",
            )
        )
        results.append(
            GateResult(
                "vs_champion_f1",
                f1_drop <= gates.max_f1_drop,
                f"{challenger['f1']:.4f} vs champion {champion['f1']:.4f} "
                f"(allowed drop {gates.max_f1_drop})",
            )
        )
        drops = {
            name: champion["slices"][name] - value
            for name, value in challenger["slices"].items()
            if name in champion["slices"]
        }
        worst = max(drops, key=lambda k: drops[k]) if drops else None
        worst_drop = drops[worst] if worst else 0.0
        results.append(
            GateResult(
                "slices",
                worst_drop <= gates.max_slice_auc_drop,
                f"largest drop {worst_drop:+.4f} in {worst} "
                f"(allowed {gates.max_slice_auc_drop})",
            )
        )

    expected = gates.expected_top_features
    if expected:
        importance = feature_importance(bundle.pipeline)
        top = importance["feature"].head(len(expected)).tolist()
        top_ok = all(any(n.startswith(f) for f in expected) for n in top)
        results.append(
            GateResult("top_features", top_ok, f"top features: {top}")
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = bundle.save(Path(tmp) / "model.joblib")
        reloaded = ModelBundle.load(path)
        same = np.allclose(
            reloaded.predict(control)["probability"],
            bundle.predict(control)["probability"],
        )
    results.append(
        GateResult("load_test", bool(same), "saved model reloads and predicts")
    )
    return results


# --- the cycle -------------------------------------------------------


def run_cycle(
    data_files: Sequence[Path],
    registry: ModelRegistry,
    trigger: str = "manual",
    approver: Callable[[CycleReport], str | None] | None = None,
    model_name: str = "lightgbm",
    params: dict[str, Any] | None = None,
    threshold: float = config.THRESHOLD,
    gates: GateConfig | None = None,
    simulate_deploy_failure: bool = False,
    random_state: int = config.RANDOM_STATE,
) -> CycleReport:
    """Run one retraining cycle; the last file is the new data batch.

    Earlier files form the rest of the training window. 25% of the new
    batch is held out: both the challenger and the production model
    are scored on it. `approver` returns the name of the person who
    approved publication or None to reject; without it nothing is
    published.
    """
    gates = gates or GateConfig()
    report = CycleReport(
        cycle=registry.next_cycle(),
        trigger=trigger,
        started_at=_now(),
        data_files=[str(f) for f in data_files],
        champion=registry.production_version(),
    )
    frames = [pd.read_csv(f, index_col=config.ID_COLUMN) for f in data_files]
    new = frames[-1]
    report.data = {
        "new_batch_rows": len(new),
        "new_batch_positive_rate": round(float(new[config.TARGET].mean()), 4),
    }

    # 2-3. data checks
    report.gates.extend(check_data(new, gates))
    if report.champion and not report.failed_gates:
        profile = registry.get(report.champion).get("profile")
        if profile:
            report.drift = drift_report(profile, new)
    if report.failed_gates:
        report.status = "rejected_data"
        registry.notify(
            "ERROR",
            f"cycle {report.cycle}: data check failed "
            f"({'; '.join(g.detail for g in report.failed_gates)}); "
            f"production stays {report.champion}",
        )
        return _finish(report, registry)

    new = clean(new)
    new_train, holdout = train_test_split(
        new,
        test_size=0.25,
        stratify=new[config.TARGET],
        random_state=random_state,
    )
    history = [clean(f) for f in frames[:-1]]
    train_df = pd.concat([*history, new_train])
    split = Split(
        train_df[config.FEATURE_COLUMNS],
        holdout[config.FEATURE_COLUMNS],
        train_df[config.TARGET],
        holdout[config.TARGET],
    )
    report.data.update(
        {"train_rows": len(train_df), "holdout_rows": len(holdout)}
    )

    # 4. challenger
    bundle = fit_and_evaluate(
        model_name, split, params, threshold, random_state
    )
    challenger = score(
        bundle, split.X_test, split.y_test, gates.min_slice_rows
    )
    report.metrics["challenger"] = challenger
    champion = None
    if report.champion:
        champion_bundle = registry.load_bundle(report.champion)
        champion = score(
            champion_bundle, split.X_test, split.y_test, gates.min_slice_rows
        )
        report.metrics["champion"] = champion

    # 5. quality gates
    control = split.X_test.head(100)
    report.gates.extend(
        check_quality(challenger, champion, bundle, control, gates)
    )
    meta = {
        "cycle": report.cycle,
        "trigger": trigger,
        "model_name": model_name,
        "threshold": threshold,
        "metrics": {k: v for k, v in challenger.items() if k != "slices"},
        "train_rows": len(train_df),
        "data_files": report.data_files,
        "profile": data_profile(split.X_train),
    }
    if report.failed_gates:
        report.version = registry.register(bundle, "rejected", meta)
        report.status = "rejected_quality"
        registry.notify(
            "WARNING",
            f"cycle {report.cycle}: {report.version} rejected by quality "
            f"gates ({', '.join(g.name for g in report.failed_gates)}); "
            f"production stays {report.champion}",
        )
        return _finish(report, registry)

    # 6. approval (user task)
    report.approved_by = approver(report) if approver else None
    if not report.approved_by:
        report.version = registry.register(bundle, "rejected", meta)
        report.status = "not_approved"
        registry.notify(
            "INFO",
            f"cycle {report.cycle}: {report.version} passed all checks "
            "but was not approved",
        )
        return _finish(report, registry)

    # 7. registration and 8. deployment
    meta["approved_by"] = report.approved_by
    report.version = registry.register(bundle, "candidate", meta)
    previous = registry.promote(report.version)
    try:
        if simulate_deploy_failure:
            raise RuntimeError("simulated failure of the post-deploy check")
        deployed = registry.load_bundle(report.version)
        deployed.predict(control)
    except Exception as err:  # any failure: roll back
        registry.rollback(report.version, previous)
        report.status = "rolled_back"
        registry.notify(
            "ERROR",
            f"cycle {report.cycle}: deploy of {report.version} failed "
            f"({err}); rolled back to {previous}",
        )
        return _finish(report, registry)

    report.status = "deployed"
    registry.notify(
        "INFO",
        f"cycle {report.cycle}: {report.version} deployed "
        f"(was {previous}), ROC-AUC {challenger['roc_auc']:.4f}",
    )
    return _finish(report, registry)


def _finish(report: CycleReport, registry: ModelRegistry) -> CycleReport:
    report.production_after = registry.production_version()
    registry.save_report(report)
    return report


def check_drift(
    data_file: Path, registry: ModelRegistry, gates: GateConfig | None = None
) -> tuple[dict[str, float], bool]:
    """Return PSI of a new batch against production training data."""
    gates = gates or GateConfig()
    prod = registry.production_version()
    if prod is None:
        return {}, False
    profile = registry.get(prod)["profile"]
    df = pd.read_csv(data_file, index_col=config.ID_COLUMN)
    report = drift_report(profile, df)
    return report, max(report.values()) > gates.psi_drift
