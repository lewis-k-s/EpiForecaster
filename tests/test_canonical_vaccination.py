from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from dataviz.canonical_vaccination import (
    MASK_VAR,
    RATE_VAR,
    compute_daily_summary,
    compute_municipality_summary,
    generate_report,
    ordered_geo_regions,
    select_choropleth_dates,
    vaccination_period,
)
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD


def _tiny_dataset() -> xr.Dataset:
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    regions = ["08001", "08002", "08003"]
    values = np.zeros((20, 3), dtype=np.float32)
    values[:, 0] = np.linspace(0.0, 0.60, 20)
    values[:, 1] = np.linspace(0.0, 0.30, 20)
    values[:, 2] = np.linspace(0.0, 0.09, 20)
    mask = np.zeros((20, 3), dtype=bool)
    mask[2:18, 0] = True
    mask[4:18, 1] = True
    mask[7:17, 2] = True
    population = np.array([100.0, 200.0, 300.0], dtype=np.float32)

    return xr.Dataset(
        {
            RATE_VAR: (
                (TEMPORAL_COORD, REGION_COORD),
                values,
            ),
            MASK_VAR: (
                (TEMPORAL_COORD, REGION_COORD),
                mask,
            ),
            "vaccination_rate_age": (
                (TEMPORAL_COORD, REGION_COORD),
                np.ones((20, 3), dtype=np.uint8),
            ),
            "population": ((REGION_COORD,), population),
        },
        coords={TEMPORAL_COORD: dates, REGION_COORD: regions},
    )


def _tiny_geojson(path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "id": ["08001", "08002"],
            "name": ["A", "B"],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def test_vaccination_period_uses_mask_any_observation() -> None:
    ds = _tiny_dataset()
    start, end = vaccination_period(ds[MASK_VAR])

    assert start == pd.Timestamp("2021-01-03")
    assert end == pd.Timestamp("2021-01-18")


def test_select_choropleth_dates_returns_16_evenly_spaced_dates() -> None:
    dates = pd.date_range("2021-01-03", periods=32, freq="D")
    selected = select_choropleth_dates(
        dates,
        pd.Timestamp("2021-01-03"),
        pd.Timestamp("2021-02-03"),
        n=16,
    )

    assert len(selected) == 16
    assert selected[0] == pd.Timestamp("2021-01-03")
    assert selected[-1] == pd.Timestamp("2021-02-03")
    assert all(a <= b for a, b in zip(selected, selected[1:]))


def test_summary_stats_include_weighted_rate_and_milestones() -> None:
    ds = _tiny_dataset()
    start, end = vaccination_period(ds[MASK_VAR])
    ds = ds.sel({TEMPORAL_COORD: slice(start, end)})

    daily = compute_daily_summary(ds[RATE_VAR], ds[MASK_VAR], ds["population"])
    municipality = compute_municipality_summary(
        ds[RATE_VAR],
        ds[MASK_VAR],
        period_start=start,
    )

    first_values = ds[RATE_VAR].isel({TEMPORAL_COORD: 0}).values
    expected_weighted = np.average(first_values, weights=ds["population"].values)
    assert np.isclose(daily.iloc[0]["population_weighted_rate"], expected_weighted)
    assert daily.iloc[0]["observed_municipality_count"] == 1

    row = municipality.set_index("region_id").loc["08001"]
    assert np.isclose(row["final_rate"], ds[RATE_VAR].isel({TEMPORAL_COORD: -1, REGION_COORD: 0}).item())
    assert row["first_observed_date"] == "2021-01-03"
    assert row["last_observed_date"] == "2021-01-18"
    assert row["days_to_10_percent"] == 2.0
    assert row["days_to_25_percent"] == 6.0
    assert row["days_to_50_percent"] == 14.0


def test_ordered_geo_regions_preserves_order_and_reports_missing() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "id": ["08002", "08001"],
            "geometry": [box(1, 0, 2, 1), box(0, 0, 1, 1)],
        },
        crs="EPSG:4326",
    )

    ordered, missing = ordered_geo_regions(gdf, ["08001", "08003", "08002"])

    assert ordered["region_id"].tolist() == ["08001", "08002"]
    assert missing == ["08003"]


def test_generate_report_writes_expected_artifacts(tmp_path: Path) -> None:
    ds = _tiny_dataset().expand_dims(run_id=["real"])
    dataset_path = tmp_path / "vaccination.zarr"
    ds.to_zarr(dataset_path, mode="w", zarr_format=2)
    geojson_path = _tiny_geojson(tmp_path / "regions.geojson")
    output_dir = tmp_path / "report"

    artifacts = generate_report(
        dataset_path=dataset_path,
        output_dir=output_dir,
        geojson=geojson_path,
        run_id="real",
        max_scatter_points=1_000,
    )

    for path in artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    daily = pd.read_csv(artifacts["daily_summary"])
    municipality = pd.read_csv(artifacts["municipality_summary"], dtype={"region_id": str})
    assert len(daily) == 16
    assert set(municipality["region_id"]) == {"08001", "08002", "08003"}
