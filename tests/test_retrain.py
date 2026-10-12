"""Retraining cycle: every branch of Model_retrain_BPMN.md."""

import json

import numpy as np
import pytest

from conftest import make_synthetic
from insurance_cross_sell import config
from insurance_cross_sell.cli import main
from insurance_cross_sell.retrain import (
    GateConfig,
    ModelRegistry,
    data_profile,
    drift_report,
    run_cycle,
)

# Synthetic data is easier and more balanced than the real one.
LENIENT = GateConfig(
    min_auc=0.6,
    max_auc_drop=0.05,
    max_f1_drop=0.1,
    max_slice_auc_drop=0.3,
    positive_rate=0.3,
    positive_rate_tolerance=0.3,
    min_rows=500,
    min_slice_rows=50,
    expected_top_features=(),
)


def write_batch(tmp_path, name, seed, change=None, n=3000):
    df = make_synthetic(n, seed)
    df.index = df.index + seed * 100_000
    df.index.name = config.ID_COLUMN
    if change is not None:
        df = change(df)
    path = tmp_path / f"{name}.csv"
    df.to_csv(path)
    return path


def approve(report):
    return "tester"


@pytest.fixture
def deployed(tmp_path, fast_params):
    """Registry with v1 in production after a first successful cycle."""
    registry = ModelRegistry(tmp_path / "models")
    b1 = write_batch(tmp_path, "b1", 1)
    report = run_cycle(
        [b1],
        registry,
        approver=approve,
        params=fast_params["lightgbm"],
        gates=LENIENT,
    )
    return registry, b1, report


def cycle(registry, files, fast_params, **kwargs):
    return run_cycle(
        files,
        registry,
        params=fast_params["lightgbm"],
        gates=LENIENT,
        **{"approver": approve, **kwargs},
    )


def test_first_cycle_deploys_version_one(deployed):
    registry, _, report = deployed

    assert report.status == "deployed"
    assert report.version == "v1"
    assert registry.production_version() == "v1"
    assert report.champion is None
    assert (registry.root / "cycles" / "cycle_001.json").exists()
    assert "v1 deployed" in (registry.root / "notifications.log").read_text(
        encoding="utf-8"
    )


def test_second_cycle_compares_with_champion(tmp_path, deployed, fast_params):
    registry, b1, _ = deployed
    b2 = write_batch(tmp_path, "b2", 2)

    report = cycle(registry, [b1, b2], fast_params, trigger="new_data")

    assert report.champion == "v1"
    assert {"challenger", "champion"} <= set(report.metrics)
    assert report.status == "deployed"
    assert registry.production_version() == "v2"
    assert registry.get("v1")["status"] == "archived"


def test_bad_data_is_rejected_before_training(tmp_path, deployed, fast_params):
    registry, b1, _ = deployed

    def corrupt(df):
        df.loc[df.index[:30], "Gender"] = "Unknown"
        return df

    bad = write_batch(tmp_path, "bad", 3, corrupt)

    report = cycle(registry, [b1, bad], fast_params)

    assert report.status == "rejected_data"
    assert report.version is None
    assert len(registry.versions) == 1
    assert registry.production_version() == "v1"


def test_broken_labels_fail_quality_gate(tmp_path, deployed, fast_params):
    registry, _, _ = deployed
    rng = np.random.default_rng(0)

    def shuffle_labels(df):
        df[config.TARGET] = rng.permutation(df[config.TARGET].to_numpy())
        return df

    noisy = write_batch(tmp_path, "noisy", 4, shuffle_labels)

    report = cycle(registry, [noisy], fast_params)

    assert report.status == "rejected_quality"
    assert "min_auc" in {g.name for g in report.failed_gates}
    assert registry.get(report.version)["status"] == "rejected"
    assert registry.production_version() == "v1"


def test_without_approval_nothing_is_published(
    tmp_path, deployed, fast_params
):
    registry, b1, _ = deployed
    b2 = write_batch(tmp_path, "b2", 2)

    report = cycle(registry, [b1, b2], fast_params, approver=lambda r: None)

    assert report.status == "not_approved"
    assert registry.production_version() == "v1"


def test_failed_deploy_rolls_back(tmp_path, deployed, fast_params):
    registry, b1, _ = deployed
    b2 = write_batch(tmp_path, "b2", 2)

    report = cycle(
        registry, [b1, b2], fast_params, simulate_deploy_failure=True
    )

    assert report.status == "rolled_back"
    assert registry.production_version() == "v1"
    assert registry.get("v1")["status"] == "production"
    assert registry.get(report.version)["status"] == "rolled_back"


def test_psi_detects_shift():
    df = make_synthetic(3000, 5)
    profile = data_profile(df)
    shifted = df.assign(Age=(df["Age"] + 25).clip(upper=85))

    same = drift_report(profile, df)
    moved = drift_report(profile, shifted)

    assert max(same.values()) < 0.01
    assert moved["Age"] > 0.2


def test_cli_monitor_reports_drift(tmp_path, deployed):
    registry, _, _ = deployed
    drifted = write_batch(
        tmp_path, "drift", 6, lambda d: d.assign(Age=(d["Age"] + 25) % 86)
    )

    assert (
        main(
            [
                "monitor",
                "--data",
                str(drifted),
                "--registry",
                str(registry.root),
            ]
        )
        == 2
    )
    report = json.loads(
        (registry.root / "cycles" / "cycle_001.json").read_text("utf-8")
    )
    assert report["status"] == "deployed"
