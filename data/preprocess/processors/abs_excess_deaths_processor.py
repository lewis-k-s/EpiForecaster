"""Build daily ABS all-cause mortality and a pre-pandemic excess baseline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d

STUDY_START = pd.Timestamp("2020-03-01")
STUDY_END = pd.Timestamp("2021-05-09")
BASELINE_YEARS = (2017, 2018, 2019)
BASELINE_SIGMA_DAYS = 14
SOURCE_COLUMNS = (
    "data",
    "codi_abs",
    "nom_abs",
    "sexe",
    "grup_edat",
    "index_socioeconomic",
    "casos",
)


def _calendar_baseline(daily: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Apply the report's 2017-19 day-of-year baseline to each ABS."""
    years = [
        daily.reindex(pd.date_range(f"{year}-01-01", f"{year}-12-31"))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        for year in BASELINE_YEARS
    ]
    curve = gaussian_filter1d(
        np.mean(years, axis=0), sigma=BASELINE_SIGMA_DAYS, axis=0, mode="wrap"
    )
    day_index = dates.dayofyear.to_numpy() - 1
    leap = dates.is_leap_year & (dates.month > 2)
    day_index[leap] -= 1
    expected = curve[day_index].copy()
    feb_29 = (dates.month == 2) & (dates.day == 29)
    expected[feb_29] = (curve[58] + curve[59]) / 2
    return expected.astype(np.float32)


def build_abs_excess_deaths(
    frame: pd.DataFrame,
    abs_ids: list[str],
    *,
    start: pd.Timestamp = STUDY_START,
    end: pd.Timestamp = STUDY_END,
) -> xr.Dataset:
    """Return a source-aligned store with counts, baseline, and signed excess."""
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"ABS mortality source missing columns: {sorted(missing)}")
    if frame.loc[:, list(SOURCE_COLUMNS)].isna().any().any():
        raise ValueError("ABS mortality source contains missing required values")
    if (frame["casos"] <= 0).any():
        raise ValueError("ABS mortality source counts must be positive")
    key = ["data", "codi_abs", "sexe", "grup_edat", "index_socioeconomic"]
    if frame.duplicated(key).any():
        raise ValueError("ABS mortality source has duplicate demographic keys")
    if frame["data"].min() > pd.Timestamp("2017-01-01") or frame["data"].max() < end:
        raise ValueError("ABS mortality source does not cover the baseline and study")
    if len(abs_ids) != len(set(abs_ids)):
        raise ValueError("ABS graph IDs must be unique")

    dates = pd.date_range(start, end)
    source_ids = set(frame["codi_abs"].astype(str))
    available = np.asarray([source_id in source_ids for source_id in abs_ids])
    panel = (
        frame.groupby(["data", "codi_abs"], sort=True)["casos"]
        .sum()
        .unstack("codi_abs")
    )
    panel.columns = panel.columns.astype(str)
    history = panel.reindex(
        index=pd.date_range("2017-01-01", end), columns=abs_ids, fill_value=0
    ).fillna(0)
    observed = history.reindex(dates).to_numpy(dtype=np.float32)
    expected = _calendar_baseline(history, dates)
    excess = observed - expected
    mask = np.broadcast_to(available[None, :], observed.shape).copy()
    age = np.where(mask, 1, 14).astype(np.uint8)
    return xr.Dataset(
        {
            "value": (("date", "abs_id"), observed),
            "expected": (("date", "abs_id"), expected),
            "excess": (("date", "abs_id"), excess),
            "mask": (("date", "abs_id"), mask),
            "age": (("date", "abs_id"), age),
        },
        coords={"date": dates, "abs_id": abs_ids},
        attrs={
            "schema": "obs_deaths_abs_excess",
            "schema_version": 1,
            "source_type": "abs",
            "head": "deaths",
            "observed_definition": "all-cause deaths by ABS and death date",
            "expected_definition": "2017-19 mean calendar day, 14-day circular Gaussian",
            "excess_definition": "observed minus expected; signed deaths per day",
            "zero_fill_policy": "no source row for an ABS-day is treated as zero deaths",
            "unmatched_source_ids": ",".join(sorted(source_ids - set(abs_ids))),
            "unmatched_graph_ids": ",".join(sorted(set(abs_ids) - source_ids)),
        },
    )


def build_store(source_csv: str | Path, graph_path: str | Path, output: str | Path) -> Path:
    """Write a new versioned store; refuse to replace an existing path."""
    source_csv = Path(source_csv)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to replace ABS excess store: {output}")
    frame = pd.read_csv(
        source_csv,
        dtype={"codi_abs": "string"},
        parse_dates=["data"],
        dayfirst=True,
    )
    graph = xr.open_zarr(graph_path)
    abs_ids = sorted(
        str(source_id)
        for source_id, source_type in zip(
            graph["source_id"].values, graph["node_type"].values
        )
        if str(source_type) == "abs"
    )
    result = build_abs_excess_deaths(frame, abs_ids)
    digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    result.attrs["source_sha256"] = digest
    result.attrs["source_file"] = source_csv.name
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_zarr(output, mode="w", zarr_format=3)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path, default=Path("data/processed/native_support/native_graph.zarr")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/native_support/obs_deaths_abs_excess.zarr"),
    )
    args = parser.parse_args()
    print(build_store(args.source_csv, args.graph, args.output))


if __name__ == "__main__":
    main()
