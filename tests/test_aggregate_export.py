from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.aggregate_export import (
    build_main_model_joint_observation_aggregate,
    build_main_model_target_aggregate,
    write_main_model_aggregate_csvs,
)


def test_build_main_model_target_aggregate_schema_and_mapping() -> None:
    eval_metrics = {
        "mae_hosp_log1p_per_100k": 1.0,
        "rmse_hosp_log1p_per_100k": 2.0,
        "r2_hosp_log1p_per_100k": 0.2,
        "observed_count_hosp": 11,
        "mae_ww_log1p_per_100k": 1.1,
        "rmse_ww_log1p_per_100k": 2.1,
        "r2_ww_log1p_per_100k": 0.21,
        "observed_count_ww": 12,
        "mae_cases_log1p_per_100k": 1.2,
        "rmse_cases_log1p_per_100k": 2.2,
        "r2_cases_log1p_per_100k": 0.22,
        "observed_count_cases": 13,
        "mae_deaths_log1p_per_100k": 1.3,
        "rmse_deaths_log1p_per_100k": 2.3,
        "r2_deaths_log1p_per_100k": 0.23,
        "observed_count_deaths": 14,
    }

    df = build_main_model_target_aggregate(eval_metrics=eval_metrics)
    expected_columns = [
        "model",
        "target",
        "folds",
        "mae_median",
        "mae_iqr",
        "rmse_median",
        "rmse_iqr",
        "r2_median",
        "r2_iqr",
        "observed_count_median",
        "observed_count_iqr",
    ]
    assert list(df.columns) == expected_columns
    assert list(df["target"]) == ["hospitalizations", "wastewater", "cases", "deaths"]
    assert (df["model"] == "epiforecaster").all()
    assert (df["folds"] == 1).all()
    assert (df["mae_iqr"] == 0.0).all()
    assert (df["observed_count_iqr"] == 0.0).all()

    ww_row = df[df["target"] == "wastewater"].iloc[0]
    assert ww_row["mae_median"] == pytest.approx(1.1)
    assert ww_row["observed_count_median"] == pytest.approx(12.0)


def test_build_main_model_target_aggregate_missing_metrics_are_nan() -> None:
    df = build_main_model_target_aggregate(eval_metrics={})
    assert df["mae_median"].isna().all()
    assert df["mae_iqr"].isna().all()
    assert df["observed_count_median"].isna().all()
    assert df["observed_count_iqr"].isna().all()


def test_build_main_model_joint_observation_aggregate_excludes_sird_supervision() -> (
    None
):
    eval_metrics = {
        "loss_ww": 1.0,
        "loss_hosp": 2.0,
        "loss_cases": 3.0,
        "loss_deaths": 4.0,
        "loss_ww_weighted": 10.0,
        "loss_hosp_weighted": 20.0,
        "loss_cases_weighted": 30.0,
        "loss_deaths_weighted": 40.0,
        "loss_sird_supervision_weighted": 99999.0,
        "observed_count_ww": 7,
        "observed_count_hosp": 8,
        "observed_count_cases": 9,
        "observed_count_deaths": 10,
    }
    df = build_main_model_joint_observation_aggregate(eval_metrics=eval_metrics)

    expected_columns = [
        "model",
        "folds",
        "joint_obs_loss_total_median",
        "joint_obs_loss_total_iqr",
        "joint_loss_ww_median",
        "joint_loss_ww_iqr",
        "joint_loss_hosp_median",
        "joint_loss_hosp_iqr",
        "joint_loss_cases_median",
        "joint_loss_cases_iqr",
        "joint_loss_deaths_median",
        "joint_loss_deaths_iqr",
        "joint_loss_ww_weighted_median",
        "joint_loss_ww_weighted_iqr",
        "joint_loss_hosp_weighted_median",
        "joint_loss_hosp_weighted_iqr",
        "joint_loss_cases_weighted_median",
        "joint_loss_cases_weighted_iqr",
        "joint_loss_deaths_weighted_median",
        "joint_loss_deaths_weighted_iqr",
        "joint_observed_count_ww_median",
        "joint_observed_count_ww_iqr",
        "joint_observed_count_hosp_median",
        "joint_observed_count_hosp_iqr",
        "joint_observed_count_cases_median",
        "joint_observed_count_cases_iqr",
        "joint_observed_count_deaths_median",
        "joint_observed_count_deaths_iqr",
    ]
    assert list(df.columns) == expected_columns
    row = df.iloc[0]
    assert row["joint_obs_loss_total_median"] == pytest.approx(100.0)
    assert row["joint_loss_hosp_weighted_median"] == pytest.approx(20.0)
    assert row["joint_loss_hosp_weighted_iqr"] == pytest.approx(0.0)
    assert row["joint_observed_count_cases_median"] == pytest.approx(9.0)


def test_write_main_model_aggregate_csvs_writes_expected_paths(tmp_path: Path) -> None:
    eval_metrics = {
        "mae_hosp_log1p_per_100k": 1.0,
        "rmse_hosp_log1p_per_100k": 2.0,
        "r2_hosp_log1p_per_100k": 0.2,
        "observed_count_hosp": 1,
        "mae_ww_log1p_per_100k": 1.0,
        "rmse_ww_log1p_per_100k": 2.0,
        "r2_ww_log1p_per_100k": 0.2,
        "observed_count_ww": 1,
        "mae_cases_log1p_per_100k": 1.0,
        "rmse_cases_log1p_per_100k": 2.0,
        "r2_cases_log1p_per_100k": 0.2,
        "observed_count_cases": 1,
        "mae_deaths_log1p_per_100k": 1.0,
        "rmse_deaths_log1p_per_100k": 2.0,
        "r2_deaths_log1p_per_100k": 0.2,
        "observed_count_deaths": 1,
        "loss_ww": 1.0,
        "loss_hosp": 2.0,
        "loss_cases": 3.0,
        "loss_deaths": 4.0,
        "loss_ww_weighted": 1.0,
        "loss_hosp_weighted": 2.0,
        "loss_cases_weighted": 3.0,
        "loss_deaths_weighted": 4.0,
    }

    artifacts = write_main_model_aggregate_csvs(
        run_dir=tmp_path,
        split="val",
        eval_metrics=eval_metrics,
    )
    assert "val_main_model_aggregate_metrics" in artifacts
    assert "val_main_model_joint_loss_aggregate" in artifacts
    for path in artifacts.values():
        assert path.exists()

    target_df = pd.read_csv(artifacts["val_main_model_aggregate_metrics"])
    joint_df = pd.read_csv(artifacts["val_main_model_joint_loss_aggregate"])
    assert not target_df.empty
    assert not joint_df.empty
