from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils.train_logging import (
    add_curriculum_metrics,
    add_horizon_metrics_to_log_data,
    add_joint_loss_metrics,
    build_epoch_logging_bundle,
    compute_horizon_metric_series,
    format_horizon_status_lines,
    format_joint_loss_components_status,
)


def test_add_joint_loss_metrics_only_adds_present_keys() -> None:
    log_data: dict[str, float] = {}
    metrics = {
        "loss_ww": 1.0,
        "loss_ww_raw": 0.9,
        "loss_ww_weighted": 0.1,
        "loss_sir": 2.0,
    }

    add_joint_loss_metrics(log_data=log_data, split_prefix="val", metrics=metrics)

    assert log_data == {
        "loss_val_ww": 1.0,
        "loss_val_ww_raw": 0.9,
        "loss_val_ww_weighted": 0.1,
        "loss_val_sir": 2.0,
    }


def test_compute_horizon_metric_series_weekly_uses_median() -> None:
    horizon_metrics = compute_horizon_metric_series(
        aggregation="weekly",
        mae_per_h=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
        rmse_per_h=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
    )

    assert horizon_metrics == [("w1", 4.0, 5.0), ("w2", 9.0, 10.0)]


def test_horizon_helpers_build_log_data_and_status_lines() -> None:
    horizon_metrics = [("h1", 0.2, 0.3), ("h2", 0.4, 0.5)]
    log_data: dict[str, float] = {}

    add_horizon_metrics_to_log_data(
        log_data=log_data,
        split_prefix="test",
        horizon_metrics=horizon_metrics,
    )
    status_lines = format_horizon_status_lines(
        prefix="Test",
        horizon_metrics=horizon_metrics,
    )

    assert log_data == {
        "mae_test_h1": 0.2,
        "rmse_test_h1": 0.3,
        "mae_test_h2": 0.4,
        "rmse_test_h2": 0.5,
    }
    assert status_lines == [
        "Test MAE_h1: 0.200000 | RMSE_h1: 0.300000",
        "Test MAE_h2: 0.400000 | RMSE_h2: 0.500000",
    ]


def test_format_joint_loss_components_status_requires_core_keys() -> None:
    assert format_joint_loss_components_status({"loss_ww": 1.0}) is None

    components = format_joint_loss_components_status(
        {
            "loss_ww": 1.0,
            "loss_hosp": 2.0,
            "loss_sir": 3.0,
            "loss_ww_weighted": 0.1,
            "loss_hosp_weighted": 0.2,
            "loss_sir_weighted": 0.3,
            "loss_cases": 4.0,
            "loss_cases_weighted": 0.4,
        }
    )

    assert components is not None
    assert "WW=1" in components
    assert "Hosp=2" in components
    assert "SIR=3" in components
    assert "Cases=4" in components


def test_add_curriculum_metrics_respects_flags() -> None:
    sampler = SimpleNamespace(state=SimpleNamespace(max_sparsity=0.35, synth_ratio=0.6))
    log_data: dict[str, float] = {}

    add_curriculum_metrics(
        log_data=log_data,
        curriculum_sampler=sampler,
        key_suffix="step",
        include_synth_ratio=False,
    )
    add_curriculum_metrics(
        log_data=log_data,
        curriculum_sampler=sampler,
        key_suffix="epoch",
        include_synth_ratio=True,
    )

    assert log_data["train_sparsity_step"] == 0.35
    assert "train_synth_ratio_step" not in log_data
    assert log_data["train_sparsity_epoch"] == 0.35
    assert log_data["train_synth_ratio_epoch"] == 0.6


def test_build_epoch_logging_bundle_includes_payload_and_status_lines() -> None:
    sampler = SimpleNamespace(state=SimpleNamespace(max_sparsity=0.2, synth_ratio=0.4))
    metrics = {
        "mae": 0.1,
        "rmse": 0.2,
        "smape": 0.3,
        "r2": 0.4,
        "loss_ww": 1.0,
        "loss_hosp": 2.0,
        "loss_sir": 3.0,
        "loss_ww_weighted": 0.1,
        "loss_hosp_weighted": 0.2,
        "loss_sir_weighted": 0.3,
        "mae_per_h": [0.11, 0.12],
        "rmse_per_h": [0.21, 0.22],
    }

    log_data, status_lines = build_epoch_logging_bundle(
        split_name="val",
        loss=1.23,
        metrics=metrics,
        epoch=5,
        aggregation="daily",
        curriculum_sampler=sampler,
    )

    assert log_data["epoch"] == 5
    assert log_data["loss_val"] == 1.23
    assert log_data["mae_val_h1"] == 0.11
    assert log_data["rmse_val_h2"] == 0.22
    assert log_data["mae_val_mixed_horizon"] == pytest.approx(0.115)
    assert log_data["rmse_val_mixed_horizon"] == pytest.approx(0.215)
    assert log_data["train_sparsity_epoch"] == 0.2
    assert log_data["train_synth_ratio_epoch"] == 0.4

    assert status_lines[0].startswith("Val loss:")
    assert any("Val loss components:" in line for line in status_lines)
    assert any("Val MAE_h1" in line for line in status_lines)
    assert any("Val MAE_mixed" in line for line in status_lines)
