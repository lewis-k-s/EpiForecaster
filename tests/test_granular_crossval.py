from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from dataviz.granular_crossval import CrossvalGranularRun, analyze_crossval_granular


def _write_granular_run(
    root: Path,
    run_name: str,
    *,
    seed: int,
    split: str = "test",
    abs_error_h1: float = 1.0,
    abs_error_h2: float = 2.0,
    duplicate_first_row: bool = False,
    write_sidecar: bool = True,
) -> CrossvalGranularRun:
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    granular_path = run_dir / f"{split}_granular_metrics.csv"
    rows = [
        {
            "split": split,
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
            "abs_error": abs_error_h1,
            "sq_error": abs_error_h1**2,
            "smape_num": abs_error_h1 * 2.0,
            "smape_den": 20.0,
        },
        {
            "split": split,
            "target": "cases",
            "node_id": 1,
            "region_id": "08001",
            "region_label": "Region A",
            "window_start": 0,
            "window_start_date": "2024-01-01",
            "horizon": 2,
            "target_index": 4,
            "target_date": "2024-01-05",
            "observed": 12.0,
            "abs_error": abs_error_h2,
            "sq_error": abs_error_h2**2,
            "smape_num": abs_error_h2 * 2.0,
            "smape_den": 24.0,
        },
    ]
    if duplicate_first_row:
        rows.append(dict(rows[0]))
    pd.DataFrame(rows).to_csv(granular_path, index=False)

    if write_sidecar:
        granular_path.with_suffix(".csv.meta.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "split": split,
                    "training_seed": seed,
                }
            ),
            encoding="utf-8",
        )

    return CrossvalGranularRun(
        fold=0 if seed == 42 else 1,
        seed=seed,
        run_dir=run_dir,
        granular_csv_path=granular_path,
    )


def test_analyze_crossval_granular_writes_expected_tables_and_plots(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runs = [
        _write_granular_run(tmp_path, "run_1", seed=42, abs_error_h1=1.0, abs_error_h2=2.0),
        _write_granular_run(tmp_path, "run_2", seed=43, abs_error_h1=3.0, abs_error_h2=4.0),
    ]

    with caplog.at_level(logging.INFO):
        artifacts = analyze_crossval_granular(
            runs=runs,
            split="test",
            output_dir=tmp_path / "reports",
        )

    horizon_df = pd.read_csv(artifacts["tables"]["horizon_aggregates"])
    horizon_one = horizon_df[horizon_df["horizon"] == 1].iloc[0]
    assert horizon_one["folds"] == 2
    assert horizon_one["mae_mean"] == pytest.approx(2.0)
    assert horizon_one["mae_std"] == pytest.approx(2**0.5)

    region_fold_df = pd.read_csv(artifacts["tables"]["region_fold_metrics"])
    assert sorted(region_fold_df["fold"].unique()) == [0, 1]
    assert set(artifacts["tables"]) == {
        "target_fold_metrics",
        "target_aggregates",
        "horizon_fold_metrics",
        "horizon_aggregates",
        "region_fold_metrics",
        "region_aggregates",
        "time_fold_metrics",
        "time_aggregates",
        "region_time_fold_metrics",
        "region_time_aggregates",
    }
    assert any("Loaded granular run:" in record.message for record in caplog.records)
    assert any("aggregate-build" in record.message for record in caplog.records)
    assert any(
        "Crossval granular analysis complete:" in record.message
        for record in caplog.records
    )

    for plot_path in artifacts["plots"].values():
        assert Path(plot_path).exists()


def test_analyze_crossval_granular_fails_for_missing_sidecar(tmp_path: Path) -> None:
    run = _write_granular_run(
        tmp_path,
        "run_missing_sidecar",
        seed=42,
        write_sidecar=False,
    )

    with pytest.raises(FileNotFoundError, match="Missing granular metadata sidecar"):
        analyze_crossval_granular(
            runs=[run],
            split="test",
            output_dir=tmp_path / "reports",
        )


def test_analyze_crossval_granular_fails_for_duplicate_keys(tmp_path: Path) -> None:
    run = _write_granular_run(
        tmp_path,
        "run_duplicate",
        seed=42,
        duplicate_first_row=True,
    )

    with pytest.raises(ValueError, match="duplicate granular keys"):
        analyze_crossval_granular(
            runs=[run],
            split="test",
            output_dir=tmp_path / "reports",
        )


def test_analyze_crossval_granular_fails_for_split_mismatch(tmp_path: Path) -> None:
    run = _write_granular_run(
        tmp_path,
        "run_val",
        seed=42,
        split="val",
    )

    with pytest.raises(ValueError, match="sidecar split mismatch"):
        analyze_crossval_granular(
            runs=[run],
            split="test",
            output_dir=tmp_path / "reports",
        )
