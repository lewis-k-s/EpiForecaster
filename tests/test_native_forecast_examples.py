import pandas as pd
import pytest

from evaluation.native_forecast_examples import select_native_forecast_examples


def test_examples_keep_native_identity_and_choose_complete_middle_window():
    rows = pd.DataFrame(
        [
            {
                "target": target,
                "source_type": "abs",
                "source_id": source,
                "window_start": origin,
                "wis": score,
                "target_index": origin + day,
            }
            for target in ("deaths", "hospitalizations")
            for source, score in (("001", 0.1), ("002", 0.2), ("003", 0.3))
            for origin, count in ((10, 1), (20, 2), (30, 2), (40, 2))
            for day in range(count)
        ]
    )
    selected, manifest = select_native_forecast_examples(rows, input_window_length=28)
    assert len(manifest) == 6
    assert set(manifest.window_start) == {30}
    assert set(manifest.origin_index) == {57}
    assert manifest.source_id.tolist() == ["001", "002", "003"] * 2
    assert set(selected.example_rank) == {"low", "middle", "high"}
    assert len(selected) == 12
    shuffled, _ = select_native_forecast_examples(
        rows.sample(frac=1, random_state=4), input_window_length=28
    )
    assert sorted(shuffled.target_index) == sorted(selected.target_index)


def test_examples_do_not_duplicate_a_single_source():
    rows = pd.DataFrame(
        [
            {
                "target": "cases",
                "source_type": "municipality",
                "source_id": "08019",
                "window_start": 10,
                "wis": 0.1,
            }
        ]
    )
    selected, manifest = select_native_forecast_examples(rows, input_window_length=28)
    assert len(selected) == len(manifest) == 1
    with pytest.raises(ValueError, match="Missing"):
        select_native_forecast_examples(
            rows.drop(columns="source_id"), input_window_length=28
        )


def test_support_filters_sparse_windows_before_ranking():
    rows = pd.DataFrame(
        [
            {
                "target": "wastewater",
                "source_type": "edar",
                "source_id": source,
                "window_start": start,
                "wis": score,
                "target_index": start + day,
            }
            for source, start, score in (("A::N1", 10, 0.1), ("B::N1", 20, 0.3))
            for day in range(2)
        ]
    )
    support = pd.DataFrame(
        [
            {
                "target": "wastewater",
                "source_id": source,
                "window_start": start,
                "eligible": eligible,
                "history_observed": count,
                "history_uncensored": count,
                "history_nonzero": count,
                "forecast_nonzero": 2,
                "origin_date": "2021-04-01",
            }
            for source, start, eligible, count in (
                ("A::N1", 10, False, 1),
                ("B::N1", 20, True, 4),
            )
        ]
    )
    selected, manifest = select_native_forecast_examples(
        rows, input_window_length=28, support=support
    )
    assert set(selected.source_id) == {"B::N1"}
    assert manifest.history_uncensored.tolist() == [4]


def test_support_counts_only_real_history_samples_and_event_days():
    from evaluation.native_example_support import eligible_windows, window_support

    dates = pd.date_range("2021-01-01", periods=42)
    history = pd.DataFrame(
        {
            "target": "wastewater",
            "source_id": "A::N1",
            "input_date": dates,
            "observed": [i in (0, 7, 14, 21) for i in range(42)],
            "censored": [i == 21 for i in range(42)],
            "observed_raw": 10.0,
        }
    )
    forecasts = pd.DataFrame(
        {
            "target": ["wastewater"] * 2,
            "source_id": ["A::N1"] * 2,
            "window_start": [0] * 2,
            "window_start_date": [dates[0]] * 2,
            "target_date": [dates[28], dates[35]],
            "observed_raw": [3.0, 4.0],
        }
    )
    support = eligible_windows(window_support(forecasts, history, 28))
    assert support.history_observed.tolist() == [4]
    assert support.history_uncensored.tolist() == [3]
    assert not support.eligible.iloc[0]
    history["censored"] = False
    assert eligible_windows(window_support(forecasts, history, 28)).eligible.iloc[0]


def test_death_support_requires_nonzero_events_not_just_zero_filled_days():
    from evaluation.native_example_support import eligible_windows

    support = pd.DataFrame(
        {
            "target": ["deaths", "deaths"],
            "history_observed": [28, 28],
            "history_uncensored": [28, 28],
            "history_nonzero": [1, 14],
            "forecast_observed": [14, 14],
            "forecast_nonzero": [1, 7],
        }
    )
    assert eligible_windows(support).eligible.tolist() == [False, True]


def test_history_loader_preserves_masks_and_native_units(monkeypatch):
    from pathlib import Path

    import numpy as np
    import xarray as xr

    from evaluation.native_example_support import load_example_history

    dates = pd.date_range("2021-01-01", periods=2)
    data = xr.Dataset(
        {
            "cases": (
                ("run_id", "date", "region_id"),
                np.array([[[np.log1p(10.0)], [0.0]]]),
            ),
            "cases_mask": (
                ("run_id", "date", "region_id"),
                np.array([[[True], [False]]]),
            ),
            "population": (("region_id",), [10000]),
        },
        coords={"run_id": [0], "date": dates, "region_id": ["001"]},
    )
    monkeypatch.setattr(xr, "open_zarr", lambda path: data)
    forecasts = pd.DataFrame(
        {"target": ["cases"], "source_id": ["001"], "population": [10000]}
    )
    h = load_example_history(Path("/unused"), forecasts)
    assert h.observed_raw.iloc[0] == pytest.approx(1.0)
    assert h.observed_score.iloc[0] == pytest.approx(np.log1p(10.0))
    assert pd.isna(h.observed_raw.iloc[1])
    assert h.observed.tolist() == [True, False]
