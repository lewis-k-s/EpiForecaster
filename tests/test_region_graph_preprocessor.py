import pandas as pd

from data.preprocess.region_graph_preprocessor import (
    RegionGraphPreprocessConfig,
    RegionGraphPreprocessor,
)


def test_region_graph_preprocessor_defaults_to_first_available_mobility_week() -> None:
    preprocessor = RegionGraphPreprocessor(RegionGraphPreprocessConfig())
    dates = pd.date_range("2020-02-14", "2021-05-09", freq="D")

    start, end = preprocessor._resolve_temporal_bounds(pd.DatetimeIndex(dates))

    assert start.date().isoformat() == "2020-02-14"
    assert end.date().isoformat() == "2020-02-20"


def test_region_graph_preprocessor_uses_week_after_explicit_start_date() -> None:
    preprocessor = RegionGraphPreprocessor(
        RegionGraphPreprocessConfig(start_date="2020-03-01")
    )
    dates = pd.date_range("2020-02-14", "2021-05-09", freq="D")

    start, end = preprocessor._resolve_temporal_bounds(pd.DatetimeIndex(dates))

    assert start.date().isoformat() == "2020-03-01"
    assert end.date().isoformat() == "2020-03-07"


def test_region_graph_preprocessor_uses_week_before_explicit_end_date() -> None:
    preprocessor = RegionGraphPreprocessor(
        RegionGraphPreprocessConfig(end_date="2020-03-07")
    )
    dates = pd.date_range("2020-02-14", "2021-05-09", freq="D")

    start, end = preprocessor._resolve_temporal_bounds(pd.DatetimeIndex(dates))

    assert start.date().isoformat() == "2020-03-01"
    assert end.date().isoformat() == "2020-03-07"


def test_region_graph_preprocessor_respects_explicit_date_range() -> None:
    preprocessor = RegionGraphPreprocessor(
        RegionGraphPreprocessConfig(
            start_date="2021-04-10",
            end_date="2021-05-09",
        )
    )
    dates = pd.date_range("2020-02-14", "2021-05-09", freq="D")

    start, end = preprocessor._resolve_temporal_bounds(pd.DatetimeIndex(dates))

    assert start.date().isoformat() == "2021-04-10"
    assert end.date().isoformat() == "2021-05-09"
