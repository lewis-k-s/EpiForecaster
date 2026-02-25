from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.analyze_ablations import (
    aggregate_ablation_metrics,
    collect_ablation_runs,
    compute_baseline_deltas,
    parse_experiment_name,
)


def _write_run(
    root: Path,
    experiment: str,
    run_id: str,
    *,
    seed: int,
    epochs: int = 50,
    split_strategy: str = "node",
    learning_rate: float = 1e-3,
    metric_offset: float = 0.0,
) -> Path:
    run_dir = root / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "training": {
            "seed": seed,
            "epochs": epochs,
            "split_strategy": split_strategy,
            "learning_rate": learning_rate,
        },
        "output": {
            "wandb_group": "group",
            "wandb_tags": ["mn5"],
            "experiment_name": experiment,
        },
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    metrics = pd.DataFrame(
        [
            {
                "target": "cases",
                "mae_median": 1.0 + metric_offset,
                "rmse_median": 2.0 + metric_offset,
                "smape_median": 3.0 + metric_offset,
                "r2_median": 0.5 + metric_offset,
            }
        ]
    )
    metrics.to_csv(run_dir / "test_main_model_aggregate_metrics.csv", index=False)
    return run_dir


def test_parse_experiment_name_supports_new_and_legacy() -> None:
    assert parse_experiment_name("mn5_ablation__camp123__baseline") == (
        "camp123",
        "baseline",
    )
    assert parse_experiment_name("mn5_ablation_no_sir_loss") == (None, "no_sir_loss")
    assert parse_experiment_name("not_an_ablation") is None


def test_collect_ablation_runs_filters_by_campaign(tmp_path: Path) -> None:
    _write_run(tmp_path, "mn5_ablation__camp_a__baseline", "run_1", seed=42)
    _write_run(tmp_path, "mn5_ablation__camp_b__baseline", "run_2", seed=42)

    ablation_runs, campaigns = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_a",
    )

    assert campaigns == {"camp_a"}
    assert sorted(ablation_runs.keys()) == ["baseline"]
    assert len(ablation_runs["baseline"]) == 1
    assert ablation_runs["baseline"][0].campaign_id == "camp_a"


def test_aggregate_rejects_mixed_epochs(tmp_path: Path) -> None:
    _write_run(tmp_path, "mn5_ablation__camp_x__baseline", "run_1", seed=42, epochs=50)
    _write_run(tmp_path, "mn5_ablation__camp_x__baseline", "run_2", seed=43, epochs=10)

    ablation_runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_x",
    )
    with pytest.raises(ValueError, match="training.epochs mismatch"):
        aggregate_ablation_metrics(ablation_runs, assert_consistent_config=True)


def test_aggregate_rejects_non_seed_config_drift(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "mn5_ablation__camp_x__baseline",
        "run_1",
        seed=42,
        learning_rate=1e-3,
    )
    _write_run(
        tmp_path,
        "mn5_ablation__camp_x__baseline",
        "run_2",
        seed=43,
        learning_rate=5e-4,
    )

    ablation_runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_x",
    )
    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        aggregate_ablation_metrics(ablation_runs, assert_consistent_config=True)


def test_aggregate_accepts_seed_only_variation(tmp_path: Path) -> None:
    _write_run(tmp_path, "mn5_ablation__camp_y__baseline", "run_1", seed=42)
    _write_run(tmp_path, "mn5_ablation__camp_y__baseline", "run_2", seed=43)

    ablation_runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_y",
    )
    df = aggregate_ablation_metrics(ablation_runs, assert_consistent_config=True)

    assert len(df) == 1
    assert df.iloc[0]["n_runs"] == 2


def test_compute_baseline_deltas_includes_absolute_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "ablation": "baseline",
                "target": "cases",
                "mae_median_mean": 2.0,
                "rmse_median_mean": 4.0,
                "smape_median_mean": 10.0,
                "r2_median_mean": 0.0,
            },
            {
                "ablation": "no_sir_loss",
                "target": "cases",
                "mae_median_mean": 3.0,
                "rmse_median_mean": 5.0,
                "smape_median_mean": 12.0,
                "r2_median_mean": 0.2,
            },
        ]
    )

    deltas = compute_baseline_deltas(df, baseline_name="baseline")
    row = deltas.iloc[0]

    assert row["mae_median_delta_abs"] == pytest.approx(1.0)
    assert row["rmse_median_delta_abs"] == pytest.approx(1.0)
    assert row["smape_median_delta_abs"] == pytest.approx(2.0)
    assert row["r2_median_delta_abs"] == pytest.approx(0.2)

    assert row["mae_median_delta_pct"] == pytest.approx(50.0)
    assert row["rmse_median_delta_pct"] == pytest.approx(25.0)
    assert row["smape_median_delta_pct"] == pytest.approx(20.0)
    assert "r2_median_delta_pct" not in deltas.columns


def test_end_to_end_campaign_aggregation(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "mn5_ablation__camp_z__baseline",
        "run_1",
        seed=42,
        metric_offset=0.0,
    )
    _write_run(
        tmp_path,
        "mn5_ablation__camp_z__baseline",
        "run_2",
        seed=43,
        metric_offset=0.1,
    )
    _write_run(
        tmp_path,
        "mn5_ablation__camp_z__no_sir_loss",
        "run_1",
        seed=42,
        metric_offset=0.2,
    )
    _write_run(
        tmp_path,
        "mn5_ablation__camp_z__no_sir_loss",
        "run_2",
        seed=43,
        metric_offset=0.3,
    )
    # Different campaign should be excluded by filter.
    _write_run(
        tmp_path,
        "mn5_ablation__camp_other__baseline",
        "run_1",
        seed=42,
        metric_offset=9.0,
    )

    ablation_runs, campaigns = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_z",
    )
    assert campaigns == {"camp_z"}
    assert sorted(ablation_runs) == ["baseline", "no_sir_loss"]

    aggregated = aggregate_ablation_metrics(
        ablation_runs,
        assert_consistent_config=True,
    )
    deltas = compute_baseline_deltas(aggregated, baseline_name="baseline")

    assert set(aggregated["ablation"]) == {"baseline", "no_sir_loss"}
    assert (aggregated["n_runs"] == 2).all()
    assert len(deltas) == 1
    assert deltas.iloc[0]["ablation"] == "no_sir_loss"
