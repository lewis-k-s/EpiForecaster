from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.analyze_crossval import (
    analyze_crossval_campaign,
    collect_crossval_runs,
    parse_crossval_experiment_name,
    validate_crossval_run_consistency,
)


def _write_crossval_run(
    root: Path,
    campaign_id: str,
    run_id: str,
    *,
    seed: int,
    learning_rate: float = 1e-3,
    metric_offset: float = 0.0,
) -> Path:
    run_dir = root / f"crossval__{campaign_id}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "training": {
            "seed": seed,
            "epochs": 50,
            "split_strategy": "node",
            "learning_rate": learning_rate,
        },
        "output": {
            "wandb_group": f"group_{campaign_id}",
            "wandb_tags": ["crossval"],
            "experiment_name": f"crossval__{campaign_id}",
        },
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    target_metrics = pd.DataFrame(
        [
            {
                "model": "epiforecaster",
                "target": "cases",
                "folds": 1,
                "mae_median": 1.0 + metric_offset,
                "mae_iqr": 0.0,
                "rmse_median": 2.0 + metric_offset,
                "rmse_iqr": 0.0,
                "smape_median": 3.0 + metric_offset,
                "smape_iqr": 0.0,
                "r2_median": 0.5 + metric_offset,
                "r2_iqr": 0.0,
                "observed_count_median": 10 + metric_offset,
                "observed_count_iqr": 0.0,
            },
            {
                "model": "epiforecaster",
                "target": "wastewater",
                "folds": 1,
                "mae_median": 4.0 + metric_offset,
                "mae_iqr": 0.0,
                "rmse_median": 5.0 + metric_offset,
                "rmse_iqr": 0.0,
                "smape_median": 6.0 + metric_offset,
                "smape_iqr": 0.0,
                "r2_median": 0.2 + metric_offset,
                "r2_iqr": 0.0,
                "observed_count_median": 20 + metric_offset,
                "observed_count_iqr": 0.0,
            },
        ]
    )
    target_metrics.to_csv(run_dir / "test_main_model_aggregate_metrics.csv", index=False)

    joint_metrics = pd.DataFrame(
        [
            {
                "model": "epiforecaster",
                "folds": 1,
                "joint_obs_loss_total_median": 1.5 + metric_offset,
                "joint_obs_loss_total_iqr": 0.0,
                "joint_loss_ww_median": 0.3 + metric_offset,
                "joint_loss_ww_iqr": 0.0,
                "joint_loss_hosp_median": 0.4 + metric_offset,
                "joint_loss_hosp_iqr": 0.0,
            }
        ]
    )
    joint_metrics.to_csv(
        run_dir / "test_main_model_joint_loss_aggregate.csv",
        index=False,
    )

    granular_rows = pd.DataFrame(
        [
            {
                "split": "test",
                "target": "cases",
                "node_id": 1,
                "region_id": "08001",
                "region_label": "Region A",
                "window_start": 0,
                "window_start_date": "2024-01-01",
                "horizon": 1,
                "target_index": 3,
                "target_date": "2024-01-04",
                "observed": 10.0,
                "abs_error": 1.0 + metric_offset,
                "sq_error": (1.0 + metric_offset) ** 2,
                "smape_num": 2.0 + metric_offset,
                "smape_den": 20.0,
            },
            {
                "split": "test",
                "target": "wastewater",
                "node_id": 2,
                "region_id": "08002",
                "region_label": "Region B",
                "window_start": 0,
                "window_start_date": "2024-01-01",
                "horizon": 2,
                "target_index": 4,
                "target_date": "2024-01-05",
                "observed": 5.0,
                "abs_error": 2.0 + metric_offset,
                "sq_error": (2.0 + metric_offset) ** 2,
                "smape_num": 4.0 + metric_offset,
                "smape_den": 10.0,
            },
        ]
    )
    granular_path = run_dir / "test_granular_metrics.csv"
    granular_rows.to_csv(granular_path, index=False)
    granular_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "split": "test",
                "training_seed": seed,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_parse_crossval_experiment_name() -> None:
    assert parse_crossval_experiment_name("crossval__camp123") == "camp123"
    assert parse_crossval_experiment_name("mn5_ablation__camp123__baseline") is None


def test_collect_crossval_runs_filters_campaign(tmp_path: Path) -> None:
    _write_crossval_run(tmp_path, "camp_a", "run_a", seed=42)
    _write_crossval_run(tmp_path, "camp_b", "run_b", seed=43)

    runs, campaigns = collect_crossval_runs(tmp_path, "camp_a")

    assert campaigns == {"camp_a"}
    assert len(runs) == 1
    assert runs[0].campaign_id == "camp_a"
    assert runs[0].seed == 42


def test_validate_crossval_run_consistency_accepts_seed_only_variation(
    tmp_path: Path,
) -> None:
    _write_crossval_run(tmp_path, "camp_ok", "run_1", seed=42)
    _write_crossval_run(tmp_path, "camp_ok", "run_2", seed=43)
    runs, _ = collect_crossval_runs(tmp_path, "camp_ok")

    _config, fingerprint = validate_crossval_run_consistency(
        runs,
        assert_consistent_config=True,
    )

    assert isinstance(fingerprint, str)
    assert fingerprint


def test_validate_crossval_run_consistency_rejects_config_drift(tmp_path: Path) -> None:
    _write_crossval_run(
        tmp_path,
        "camp_drift",
        "run_1",
        seed=42,
        learning_rate=1e-3,
    )
    _write_crossval_run(
        tmp_path,
        "camp_drift",
        "run_2",
        seed=43,
        learning_rate=5e-4,
    )
    runs, _ = collect_crossval_runs(tmp_path, "camp_drift")

    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        validate_crossval_run_consistency(
            runs,
            assert_consistent_config=True,
        )


def test_analyze_crossval_campaign_writes_expected_outputs(tmp_path: Path) -> None:
    _write_crossval_run(tmp_path, "camp_z", "run_1", seed=42, metric_offset=0.0)
    _write_crossval_run(tmp_path, "camp_z", "run_2", seed=43, metric_offset=0.2)

    output_dir = tmp_path / "reports"
    artifacts = analyze_crossval_campaign(
        training_dir=tmp_path,
        campaign_id="camp_z",
        split="test",
        output_dir=output_dir,
        assert_consistent_config=True,
    )

    target_df = pd.read_csv(artifacts["target_aggregate_metrics"])
    cases_row = target_df[target_df["target"] == "cases"].iloc[0]
    assert cases_row["folds"] == 2
    assert cases_row["mae_mean"] == pytest.approx(1.1)
    assert cases_row["mae_std"] == pytest.approx((0.02) ** 0.5)

    joint_df = pd.read_csv(artifacts["joint_aggregate_metrics"])
    assert joint_df.iloc[0]["folds"] == 2
    assert joint_df.iloc[0]["joint_obs_loss_total_mean"] == pytest.approx(1.6)

    metadata = json.loads(Path(artifacts["crossval_metadata"]).read_text())
    assert metadata["campaign_id"] == "camp_z"
    assert metadata["fold_count"] == 2
    assert metadata["seeds"] == [42, 43]
