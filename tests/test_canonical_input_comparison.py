from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from dataviz.canonical_input_comparison import (
    SourceSpec,
    generate_canonical_input_comparison,
    load_canonical_source,
    make_canonical_input_comparison_figure,
    restrict_sources_to_shared_time_range,
    select_comparison_windows,
)


def _write_dataset(
    path: Path,
    *,
    run_ids: list[str],
    dates: pd.DatetimeIndex,
    region_ids: list[str],
    offset: float,
) -> None:
    shape = (len(run_ids), len(dates), len(region_ids))
    time = np.arange(len(dates), dtype=np.float32).reshape(1, len(dates), 1)
    region = np.arange(len(region_ids), dtype=np.float32).reshape(1, 1, len(region_ids))
    values = time + region + offset
    values = np.broadcast_to(values, shape).astype(np.float32)
    mask = np.ones(shape, dtype=np.float32)
    age = np.linspace(0.0, 1.0, len(dates), dtype=np.float32).reshape(1, len(dates), 1)
    age = np.broadcast_to(age, shape).astype(np.float32)
    source_mask = np.array([True, True, False])[: len(region_ids)]

    ds = xr.Dataset(
        data_vars={
            "cases": (("run_id", TEMPORAL_COORD, REGION_COORD), values),
            "cases_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), mask),
            "cases_age": (("run_id", TEMPORAL_COORD, REGION_COORD), age),
            "hospitalizations": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                values + 10.0,
            ),
            "hospitalizations_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), mask),
            "hospitalizations_age": (("run_id", TEMPORAL_COORD, REGION_COORD), age),
            "deaths": (("run_id", TEMPORAL_COORD, REGION_COORD), values + 20.0),
            "deaths_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), mask),
            "deaths_age": (("run_id", TEMPORAL_COORD, REGION_COORD), age),
            "edar_biomarker_N1": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                values + 30.0,
            ),
            "edar_biomarker_N1_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                mask,
            ),
            "edar_biomarker_N1_age": (("run_id", TEMPORAL_COORD, REGION_COORD), age),
            "edar_biomarker_N1_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                np.zeros(shape, dtype=np.float32),
            ),
            "edar_biomarker_N2": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                values + 31.0,
            ),
            "edar_biomarker_N2_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                mask,
            ),
            "edar_biomarker_N2_age": (("run_id", TEMPORAL_COORD, REGION_COORD), age),
            "edar_biomarker_N2_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                np.zeros(shape, dtype=np.float32),
            ),
            "edar_biomarker_IP4": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                values + 32.0,
            ),
            "edar_biomarker_IP4_mask": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                mask,
            ),
            "edar_biomarker_IP4_age": (("run_id", TEMPORAL_COORD, REGION_COORD), age),
            "edar_biomarker_IP4_censor": (
                ("run_id", TEMPORAL_COORD, REGION_COORD),
                np.zeros(shape, dtype=np.float32),
            ),
            "edar_has_source": ((REGION_COORD,), source_mask.astype(np.int8)),
            "population": ((REGION_COORD,), np.array([1000, 1200, 900])[: len(region_ids)]),
            "valid_targets": (
                ("run_id", REGION_COORD),
                np.ones((len(run_ids), len(region_ids)), dtype=np.int8),
            ),
        },
        coords={
            "run_id": np.asarray(run_ids, dtype=object),
            TEMPORAL_COORD: dates,
            REGION_COORD: np.asarray(region_ids, dtype=object),
        },
    )
    ds.to_zarr(path, zarr_format=2)


def test_select_comparison_windows_uses_shared_regions_and_overlap(tmp_path: Path) -> None:
    real_path = tmp_path / "real.zarr"
    synth_path = tmp_path / "synth.zarr"
    _write_dataset(
        real_path,
        run_ids=["real"],
        dates=pd.date_range("2020-03-01", periods=8, freq="D"),
        region_ids=["08001", "08002", "08003"],
        offset=0.0,
    )
    _write_dataset(
        synth_path,
        run_ids=["0_Baseline", "1_Baseline"],
        dates=pd.date_range("2020-02-28", periods=10, freq="D"),
        region_ids=["08001", "08002", "09000"],
        offset=100.0,
    )

    sources = [
        load_canonical_source(SourceSpec(real_path, "real", "real")),
        load_canonical_source(SourceSpec(synth_path, "1_Baseline", "synth")),
    ]
    try:
        restricted, overlap_dates = restrict_sources_to_shared_time_range(sources)
        selections = select_comparison_windows(
            restricted,
            overlap_dates=overlap_dates,
            num_samples=2,
            window_length=4,
            requested_region_ids=None,
            require_biomarker_source=True,
            seed=0,
        )
    finally:
        for source in sources:
            source.dataset.close()

    assert len(selections) == 2
    assert {selection.region_id for selection in selections} == {"08001", "08002"}
    assert all(len(selection.window_dates) == 4 for selection in selections)
    assert all(str(selection.window_dates[0].date()) >= "2020-03-01" for selection in selections)


def test_make_canonical_input_comparison_figure_returns_expected_axes() -> None:
    window_dates = pd.date_range("2020-03-01", periods=5, freq="D")
    grouped_samples = [
        [
            {
                "source_label": "real",
                "region_id": "08001",
                "window_dates": window_dates,
                "comparison_score": 20.0,
                "cases_series": np.arange(5, dtype=np.float32),
                "cases_age": np.linspace(0, 1, 5, dtype=np.float32),
                "cases_obs_mask_full": np.ones(5, dtype=np.float32),
                "hosp_series": np.arange(5, dtype=np.float32) + 10,
                "hosp_age": np.linspace(0, 1, 5, dtype=np.float32),
                "hosp_obs_mask_full": np.ones(5, dtype=np.float32),
                "deaths_series": np.arange(5, dtype=np.float32) + 20,
                "deaths_age": np.linspace(0, 1, 5, dtype=np.float32),
                "deaths_obs_mask_full": np.ones(5, dtype=np.float32),
                "biomarkers": {
                    "N1": {
                        "value": np.arange(5, dtype=np.float32) + 30,
                        "mask": np.ones(5, dtype=np.float32),
                        "censor": np.zeros(5, dtype=np.float32),
                        "age": np.linspace(0, 1, 5, dtype=np.float32),
                    }
                },
                "ww_series": np.arange(5, dtype=np.float32) + 30,
                "ww_obs_mask_full": np.ones(5, dtype=np.float32),
            },
            {
                "source_label": "synth",
                "region_id": "08001",
                "window_dates": window_dates,
                "comparison_score": 20.0,
                "cases_series": np.arange(5, dtype=np.float32) + 1,
                "cases_age": np.linspace(0, 1, 5, dtype=np.float32),
                "cases_obs_mask_full": np.ones(5, dtype=np.float32),
                "hosp_series": np.arange(5, dtype=np.float32) + 11,
                "hosp_age": np.linspace(0, 1, 5, dtype=np.float32),
                "hosp_obs_mask_full": np.ones(5, dtype=np.float32),
                "deaths_series": np.arange(5, dtype=np.float32) + 21,
                "deaths_age": np.linspace(0, 1, 5, dtype=np.float32),
                "deaths_obs_mask_full": np.ones(5, dtype=np.float32),
                "biomarkers": {
                    "N1": {
                        "value": np.arange(5, dtype=np.float32) + 31,
                        "mask": np.ones(5, dtype=np.float32),
                        "censor": np.zeros(5, dtype=np.float32),
                        "age": np.linspace(0, 1, 5, dtype=np.float32),
                    }
                },
                "ww_series": np.arange(5, dtype=np.float32) + 31,
                "ww_obs_mask_full": np.ones(5, dtype=np.float32),
            },
        ]
    ]

    fig = make_canonical_input_comparison_figure(grouped_samples, window_length=5)
    try:
        assert fig is not None
        assert len(fig.axes) == 8
    finally:
        fig.clf()


def test_generate_canonical_input_comparison_writes_outputs(tmp_path: Path) -> None:
    left_path = tmp_path / "left.zarr"
    right_path = tmp_path / "right.zarr"
    dates = pd.date_range("2020-03-01", periods=7, freq="D")
    regions = ["08001", "08002", "08003"]
    _write_dataset(
        left_path,
        run_ids=["real"],
        dates=dates,
        region_ids=regions,
        offset=0.0,
    )
    _write_dataset(
        right_path,
        run_ids=["0_Baseline"],
        dates=dates,
        region_ids=regions,
        offset=50.0,
    )

    output_dir = tmp_path / "artifacts"
    selections, artifacts = generate_canonical_input_comparison(
        source_specs=[
            SourceSpec(left_path, "real", "real"),
            SourceSpec(right_path, "0_Baseline", "synth"),
        ],
        output_dir=output_dir,
        num_samples=1,
        window_length=5,
        region_ids=["08001"],
        seed=0,
    )

    assert len(selections) == 1
    assert selections[0].region_id == "08001"
    assert artifacts["figure"].exists()
    assert artifacts["summary"].exists()
