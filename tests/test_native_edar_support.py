"""Tests for source-native EDAR catchment and point relations."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box
import xarray as xr

from data.native_registry import SOURCE_ABS, SOURCE_COMARCA, SOURCE_MUNICIPALITY
from data.preprocess.native_sources.edar import (
    build_catchment_edges,
    build_point_location_edges,
)

pytestmark = pytest.mark.epiforecaster


def _overlap_row(
    municipality_id: str, target_type: str, target_id: str, fraction: float
) -> dict[str, object]:
    return {
        "src_type": SOURCE_MUNICIPALITY,
        "src_id": municipality_id,
        "dst_type": target_type,
        "dst_id": target_id,
        "frac_of_src_pop": fraction,
    }


def test_catchment_population_is_composed_from_contribution_and_overlap() -> None:
    contribution = xr.DataArray(
        np.array([[1.0, 0.5], [0.0, 0.25]]),
        dims=("edar_id", "home"),
        coords={"edar_id": ["E1", "E2"], "home": ["M1", "M2"]},
    )
    overlap = pd.DataFrame(
        [
            _overlap_row("M1", SOURCE_ABS, "A1", 0.6),
            _overlap_row("M1", SOURCE_ABS, "A2", 0.4),
            _overlap_row("M2", SOURCE_ABS, "A1", 1.0),
            _overlap_row("M1", SOURCE_COMARCA, "C1", 1.0),
            _overlap_row("M2", SOURCE_COMARCA, "C2", 1.0),
        ]
    )

    edges, report = build_catchment_edges(
        contribution, {"M1": 100.0, "M2": 200.0}, overlap
    )

    e1 = edges[edges["edar_id"] == "E1"]
    e1_muni = e1[e1["target_type"] == SOURCE_MUNICIPALITY].set_index("target_id")
    assert e1_muni.loc["M1", "served_population_est"] == 100.0
    assert e1_muni.loc["M2", "served_population_est"] == 100.0
    assert e1_muni["frac_of_edar_catchment"].sum() == pytest.approx(1.0)

    e1_abs = e1[e1["target_type"] == SOURCE_ABS].set_index("target_id")
    assert e1_abs.loc["A1", "served_population_est"] == 160.0
    assert e1_abs.loc["A2", "served_population_est"] == 40.0
    assert e1_abs["frac_of_edar_catchment"].sum() == pytest.approx(1.0)

    e1_comarca = e1[e1["target_type"] == SOURCE_COMARCA]
    assert sorted(e1_comarca["served_population_est"].tolist()) == [100.0, 100.0]
    assert report.edar_count == 2
    assert report.abs_allocation_fraction_min == pytest.approx(1.0)
    assert report.comarca_allocation_fraction_min == pytest.approx(1.0)
    assert report.missing_abs_municipalities == ()
    assert report.missing_comarca_municipalities == ()


def test_missing_admin_mapping_is_reported_without_renormalization() -> None:
    contribution = xr.DataArray(
        np.array([[0.5, 0.5]]),
        dims=("edar_id", "home"),
        coords={"edar_id": ["E1"], "home": ["M1", "M2"]},
    )
    overlap = pd.DataFrame(
        [
            _overlap_row("M1", SOURCE_ABS, "A1", 1.0),
            _overlap_row("M1", SOURCE_COMARCA, "C1", 1.0),
        ]
    )

    edges, report = build_catchment_edges(
        contribution, {"M1": 100.0, "M2": 100.0}, overlap
    )

    abs_edges = edges[edges["target_type"] == SOURCE_ABS]
    assert abs_edges["frac_of_edar_catchment"].sum() == pytest.approx(0.5)
    assert report.abs_allocation_fraction_min == pytest.approx(0.5)
    assert report.comarca_allocation_fraction_min == pytest.approx(0.5)
    assert report.missing_abs_municipalities == ("M2",)
    assert report.missing_comarca_municipalities == ("M2",)


def test_point_locations_are_independent_and_have_no_nearest_fallback() -> None:
    points = gpd.GeoDataFrame(
        {"edar_id": ["E1", "E2"]},
        geometry=[Point(0.5, 0.5), Point(2.5, 0.5)],
        crs="EPSG:3042",
    )
    municipalities = gpd.GeoDataFrame(
        {"municipality_id": ["M1"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:3042",
    )
    abs_layer = gpd.GeoDataFrame(
        {"abs_id": ["A1", "A2"]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
        crs="EPSG:3042",
    )

    edges, report = build_point_location_edges(
        points,
        {
            SOURCE_MUNICIPALITY: (municipalities, "municipality_id"),
            SOURCE_ABS: (abs_layer, "abs_id"),
        },
    )

    assert set(edges["edar_id"]) == {"E1", "E2"}
    assert report.missing_location_ids[SOURCE_MUNICIPALITY] == ("E2",)
    assert report.missing_location_ids[SOURCE_ABS] == ()
    assert not (
        (edges["edar_id"] == "E2")
        & (edges["target_type"] == SOURCE_MUNICIPALITY)
    ).any()
