"""Test canonical report exports and temporal source-cohort reuse."""

import json

import pandas as pd
import pytest

from evaluation import run_report_evidence
from tests.native_fixtures import _trainer_config


@pytest.mark.parametrize("strategy", ["region", "temporal"])
def test_example_exports_keep_masks_selection_and_period_identity(
    tmp_path, monkeypatch, strategy
):
    config = _trainer_config(tmp_path)
    config.model.input_window_length = 28
    config.model.forecast_horizon = 14
    if strategy == "region":
        contract = tmp_path / "fold.json"
        contract.write_text(json.dumps({"schema": "native_source_sets.v3"}))
        config.native.fold_manifest = str(contract)
        config.native.fold_index = 0
    dates = pd.date_range("2021-01-01", periods=100)
    history = pd.DataFrame(
        [
            {
                "target": "cases",
                "source_id": source,
                "input_date": date,
                "date": date,
                "observed": True,
                "censored": False,
                "observed_raw": 1.0,
                "observed_score": 0.2,
                "excess": None,
            }
            for source in ("001", "002", "003", "004")
            for date in dates
        ]
    )
    monkeypatch.setattr(
        run_report_evidence, "load_example_history", lambda *args, **kwargs: history
    )
    for split, starts, sources in (
        ("val", (0, 10), ("001", "002", "003", "004")),
        ("test", (40, 50), ("001", "002", "003")),
    ):
        rows = pd.DataFrame(
            [
                {
                    "target": "cases",
                    "source_type": "municipality",
                    "source_id": source,
                    "window_start": start,
                    "window_start_date": dates[start],
                    "target_date": dates[start + 28 + day],
                    "target_index": start + 28 + day,
                    "wis": float(source) if split == "val" else 10 - float(source),
                    "observed_raw": 1.0,
                }
                for source in sources
                for start in starts
                for day in range(10)
            ]
        )
        rows.to_parquet(tmp_path / f"{split}_granular.parquet", index=False)
        paths = run_report_evidence.export_native_report_examples(
            config, tmp_path, split
        )
        assert len(paths) == 5
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / f"{split}_example_history.parquet"), history
        )
    val = pd.read_csv(tmp_path / "val_example_selection.csv", dtype={"source_id": str})
    test = pd.read_csv(
        tmp_path / "test_example_selection.csv", dtype={"source_id": str}
    )
    assert val.source_id.tolist() == ["001", "003", "004"]
    assert val.window_start.tolist() == [10] * 3
    assert test.window_start.eq(50).all()
    availability = pd.read_csv(tmp_path / "test_example_availability.csv")
    if strategy == "temporal":
        assert test.source_id.tolist() == ["001", "003"]
        assert test.example_rank.tolist() == ["low", "middle"]
        assert test.source_mean_wis.tolist() == [1, 3]
        assert test.selection_split.eq("val").all()
        assert availability.missing_cohort_sources.tolist() == [1]
        (tmp_path / "val_example_selection.csv").unlink()
        run_report_evidence.export_native_report_examples(config, tmp_path, "test")
        assert pd.read_csv(tmp_path / "test_example_selection.csv").empty
        assert pd.read_csv(
            tmp_path / "test_example_availability.csv"
        ).cohort_available.tolist() == [False]
    else:
        assert test.source_id.tolist() == ["003", "002", "001"]
        assert test.selection_split.eq("test").all()
        assert availability.missing_cohort_sources.tolist() == [0]


def test_fixed_cohort_is_deterministic_under_row_order_and_test_scores():
    from evaluation.native_forecast_examples import select_native_forecast_examples

    rows = pd.DataFrame(
        [
            {
                "target": "cases",
                "source_type": "municipality",
                "source_id": source,
                "window_start": start,
                "wis": score,
                "target_index": start + day,
            }
            for source, score in (("001", 10), ("002", 0.1))
            for start in (10, 20, 30)
            for day in (0, 1)
        ]
    )
    cohort = pd.DataFrame(
        [
            {
                "target": "cases",
                "source_type": "municipality",
                "source_id": "001",
                "source_mean_wis": 0.2,
                "example_rank": "low",
            }
        ]
    )
    _, original = select_native_forecast_examples(
        rows, input_window_length=28, source_cohort=cohort
    )
    rows["wis"] = 100 - rows.wis
    _, permuted = select_native_forecast_examples(
        rows.sample(frac=1, random_state=4),
        input_window_length=28,
        source_cohort=cohort,
    )
    pd.testing.assert_frame_equal(original, permuted)
    assert original.source_id.tolist() == ["001"]
    assert original.window_start.tolist() == [20]
