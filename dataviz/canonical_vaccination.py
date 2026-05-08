"""Visualize canonical municipality-level COVID-19 vaccination coverage."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.append(sys_path)

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402
from utils.plotting import Style, save_figure  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_GEOJSON = Path("data/files/geo/fl_municipios_catalonia.geojson")
DEFAULT_OUTPUT_DIR = Path("outputs/reports/canonical_vaccination")
RATE_VAR = "vaccination_rate"
MASK_VAR = "vaccination_rate_mask"
AGE_VAR = "vaccination_rate_age"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load training config YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def dataset_path_from_config(config: dict[str, Any]) -> Path:
    """Extract canonical dataset path from a training-style config."""
    dataset_path = config.get("dataset", {}).get("path")
    if dataset_path is None:
        dataset_path = config.get("data", {}).get("dataset_path")
    if dataset_path is None:
        dataset_path = config.get("preprocessing", {}).get("output_path")
    if dataset_path is None:
        raise ValueError("Could not find dataset path in config")
    return Path(dataset_path)


def load_vaccination_dataset(dataset_path: Path, run_id: str = "real") -> xr.Dataset:
    """Open canonical Zarr dataset and select vaccination variables."""
    dataset = xr.open_zarr(dataset_path)
    if "run_id" in dataset.dims or "run_id" in dataset.coords:
        dataset = dataset.sel(run_id=run_id)

    if RATE_VAR not in dataset:
        raise ValueError(f"Dataset missing required variable: {RATE_VAR}")
    if MASK_VAR not in dataset:
        logger.warning("Dataset missing %s; using finite vaccination_rate values", MASK_VAR)
        dataset[MASK_VAR] = np.isfinite(dataset[RATE_VAR])

    return dataset


def vaccination_period(mask_da: xr.DataArray) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return first and last date with any vaccination source observation."""
    if TEMPORAL_COORD not in mask_da.dims:
        raise ValueError(f"{MASK_VAR} must include {TEMPORAL_COORD!r} dimension")

    mask = mask_da
    if "run_id" in mask.dims:
        mask = mask.squeeze("run_id", drop=True)
    non_time_dims = [dim for dim in mask.dims if dim != TEMPORAL_COORD]
    any_observed = mask.astype(bool).any(dim=non_time_dims)
    observed_dates = pd.to_datetime(mask[TEMPORAL_COORD].values[any_observed.values])
    if len(observed_dates) == 0:
        raise ValueError("No vaccination observations found in vaccination_rate_mask")
    return pd.Timestamp(observed_dates[0]), pd.Timestamp(observed_dates[-1])


def select_choropleth_dates(
    dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    n: int = 16,
) -> list[pd.Timestamp]:
    """Pick evenly spaced dataset dates within the vaccination period."""
    period_dates = dates[(dates >= start) & (dates <= end)]
    if len(period_dates) == 0:
        raise ValueError("No dataset dates fall inside vaccination period")
    if len(period_dates) <= n:
        return [pd.Timestamp(date) for date in period_dates]

    positions = np.linspace(0, len(period_dates) - 1, n)
    indices = np.rint(positions).astype(int)
    indices[0] = 0
    indices[-1] = len(period_dates) - 1
    return [pd.Timestamp(period_dates[idx]) for idx in indices]


def _as_2d_array(data: xr.DataArray) -> np.ndarray:
    """Return data as (date, region) array."""
    arr = data.transpose(TEMPORAL_COORD, REGION_COORD).values
    return np.asarray(arr, dtype=float)


def compute_daily_summary(
    rate_da: xr.DataArray,
    mask_da: xr.DataArray,
    population_da: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Compute date-level vaccination summary statistics."""
    values = _as_2d_array(rate_da)
    mask_values = mask_da.transpose(TEMPORAL_COORD, REGION_COORD).values.astype(bool)
    dates = pd.to_datetime(rate_da[TEMPORAL_COORD].values)
    population = None
    if population_da is not None:
        population = population_da.reindex({REGION_COORD: rate_da[REGION_COORD]}).values
        population = np.asarray(population, dtype=float)

    rows = []
    for idx, date in enumerate(dates):
        row_values = values[idx]
        finite = row_values[np.isfinite(row_values)]
        weighted_rate = np.nan
        if population is not None:
            valid = np.isfinite(row_values) & np.isfinite(population) & (population > 0)
            if valid.any():
                weighted_rate = float(np.average(row_values[valid], weights=population[valid]))

        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "mean_rate": float(np.mean(finite)) if finite.size else np.nan,
                "median_rate": float(np.median(finite)) if finite.size else np.nan,
                "p10_rate": float(np.percentile(finite, 10)) if finite.size else np.nan,
                "p25_rate": float(np.percentile(finite, 25)) if finite.size else np.nan,
                "p75_rate": float(np.percentile(finite, 75)) if finite.size else np.nan,
                "p90_rate": float(np.percentile(finite, 90)) if finite.size else np.nan,
                "max_rate": float(np.max(finite)) if finite.size else np.nan,
                "observed_municipality_count": int(mask_values[idx].sum()),
                "population_weighted_rate": weighted_rate,
            }
        )
    return pd.DataFrame(rows)


def _milestone_days(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    period_start: pd.Timestamp,
    threshold: float,
) -> float:
    reached = np.flatnonzero(np.isfinite(series) & (series >= threshold))
    if reached.size == 0:
        return np.nan
    reached_date = pd.Timestamp(dates[int(reached[0])])
    return float((reached_date - period_start).days)


def compute_municipality_summary(
    rate_da: xr.DataArray,
    mask_da: xr.DataArray,
    *,
    period_start: pd.Timestamp,
) -> pd.DataFrame:
    """Compute per-municipality vaccination summaries."""
    values = _as_2d_array(rate_da)
    mask_values = mask_da.transpose(TEMPORAL_COORD, REGION_COORD).values.astype(bool)
    dates = pd.to_datetime(rate_da[TEMPORAL_COORD].values)
    region_ids = [str(region_id) for region_id in rate_da[REGION_COORD].values]
    n_days = values.shape[0]

    rows = []
    for idx, region_id in enumerate(region_ids):
        series = values[:, idx]
        mask_series = mask_values[:, idx]
        observed_idx = np.flatnonzero(mask_series)

        if observed_idx.size:
            first_observed = pd.Timestamp(dates[int(observed_idx[0])]).date().isoformat()
            last_observed = pd.Timestamp(dates[int(observed_idx[-1])]).date().isoformat()
        else:
            first_observed = None
            last_observed = None

        rows.append(
            {
                "region_id": region_id,
                "final_rate": float(series[-1]) if np.isfinite(series[-1]) else np.nan,
                "peak_rate": float(np.nanmax(series)) if np.isfinite(series).any() else np.nan,
                "first_observed_date": first_observed,
                "last_observed_date": last_observed,
                "observed_day_count": int(mask_series.sum()),
                "source_coverage_fraction": float(mask_series.sum() / max(n_days, 1)),
                "days_to_10_percent": _milestone_days(series, dates, period_start, 0.10),
                "days_to_25_percent": _milestone_days(series, dates, period_start, 0.25),
                "days_to_50_percent": _milestone_days(series, dates, period_start, 0.50),
            }
        )
    return pd.DataFrame(rows)


def ordered_geo_regions(
    regions_gdf: gpd.GeoDataFrame,
    region_ids: list[str],
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Order municipality polygons by canonical region IDs and report missing IDs."""
    if "id" not in regions_gdf.columns:
        raise ValueError("GeoJSON must contain an 'id' property")

    regions = regions_gdf.copy()
    regions["id"] = regions["id"].astype(str)
    regions = regions.drop_duplicates(subset="id", keep="first").set_index("id")
    missing_ids = [region_id for region_id in region_ids if region_id not in regions.index]
    present_ids = [region_id for region_id in region_ids if region_id in regions.index]
    ordered = regions.loc[present_ids].copy()
    ordered["region_id"] = present_ids
    return ordered.reset_index(drop=True), missing_ids


def choropleth_vmax(values: np.ndarray) -> float:
    """Round max rate up to next five percentage points, capped at 1.0."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    rounded = np.ceil(float(finite.max()) / 0.05) * 0.05
    return float(min(max(rounded, 0.05), 1.0))


def plot_vaccination_scatter(
    daily_summary: pd.DataFrame,
    rate_da: xr.DataArray,
    output_path: Path,
    *,
    max_scatter_points: int,
) -> None:
    """Plot municipality-date vaccination rates with summary overlays."""
    dates = pd.to_datetime(rate_da[TEMPORAL_COORD].values)
    values = _as_2d_array(rate_da)
    date_grid = np.repeat(dates.to_numpy(), values.shape[1])
    value_flat = values.reshape(-1)
    valid = np.isfinite(value_flat)
    date_grid = date_grid[valid]
    value_flat = value_flat[valid]

    if len(value_flat) > max_scatter_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(value_flat), size=max_scatter_points, replace=False)
        date_grid = date_grid[keep]
        value_flat = value_flat[keep]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(date_grid, value_flat * 100.0, s=7, alpha=0.06, color="tab:blue", linewidths=0)

    summary_dates = pd.to_datetime(daily_summary["date"])
    ax.plot(
        summary_dates,
        daily_summary["median_rate"] * 100.0,
        color="black",
        linewidth=2.0,
        label="Median",
    )
    ax.fill_between(
        summary_dates,
        daily_summary["p25_rate"].to_numpy(dtype=float) * 100.0,
        daily_summary["p75_rate"].to_numpy(dtype=float) * 100.0,
        color="black",
        alpha=0.16,
        label="IQR",
    )
    if daily_summary["population_weighted_rate"].notna().any():
        ax.plot(
            summary_dates,
            daily_summary["population_weighted_rate"] * 100.0,
            color="tab:orange",
            linewidth=2.0,
            label="Population-weighted mean",
        )

    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("Date")
    ax.set_ylabel("Vaccination rate (%)")
    ax.set_title("Municipality Vaccination Rates Over Time")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    save_figure(fig, output_path, dpi=Style.DPI, log_msg="Saved vaccination scatter")


def plot_choropleth_grid(
    rate_da: xr.DataArray,
    regions_gdf: gpd.GeoDataFrame,
    snapshot_dates: list[pd.Timestamp],
    output_path: Path,
) -> None:
    """Plot a 4x4 grid of municipality vaccination choropleths."""
    region_ids = [str(region_id) for region_id in rate_da[REGION_COORD].values]
    ordered_gdf, _ = ordered_geo_regions(regions_gdf, region_ids)
    rate_by_region = rate_da.transpose(TEMPORAL_COORD, REGION_COORD).to_pandas()
    vmax = choropleth_vmax(rate_by_region.loc[snapshot_dates].to_numpy())

    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = "YlGnBu"

    for ax, snapshot_date in zip(axes.flat, snapshot_dates):
        values = rate_by_region.loc[snapshot_date]
        plot_gdf = ordered_gdf.copy()
        plot_gdf["vaccination_rate"] = [
            float(values.get(region_id, np.nan)) for region_id in plot_gdf["region_id"]
        ]
        plot_gdf.plot(
            column="vaccination_rate",
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidth=0.05,
            edgecolor="white",
            missing_kwds={"color": "lightgrey"},
        )
        ax.set_title(pd.Timestamp(snapshot_date).date().isoformat(), fontsize=10)
        ax.set_axis_off()

    for ax in axes.flat[len(snapshot_dates) :]:
        ax.set_axis_off()

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    cbar.set_label("Vaccination rate (%)")
    ticks = np.linspace(0.0, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick * 100:.0f}" for tick in ticks])
    fig.suptitle("Municipality Vaccination Rates Across the Vaccination Period", y=0.995)
    save_figure(fig, output_path, dpi=Style.DPI, bbox_inches="tight", log_msg="Saved vaccination choropleth grid")


def write_summary_json(
    path: Path,
    *,
    dataset_path: Path,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    snapshot_dates: list[pd.Timestamp],
    missing_geo_ids: list[str],
    daily_summary: pd.DataFrame,
    municipality_summary: pd.DataFrame,
    age_da: xr.DataArray | None = None,
) -> None:
    """Write compact report metadata and headline stats."""
    final_daily = daily_summary.iloc[-1]
    final_age_mean = None
    final_age_max = None
    if age_da is not None:
        final_age = age_da.transpose(TEMPORAL_COORD, REGION_COORD).isel(
            {TEMPORAL_COORD: -1}
        )
        final_age_values = np.asarray(final_age.values, dtype=float)
        final_age_values = final_age_values[np.isfinite(final_age_values)]
        if final_age_values.size:
            final_age_mean = float(np.mean(final_age_values))
            final_age_max = float(np.max(final_age_values))

    payload = {
        "dataset_path": str(dataset_path),
        "vaccination_period": {
            "start": period_start.date().isoformat(),
            "end": period_end.date().isoformat(),
            "days": int((period_end - period_start).days + 1),
        },
        "snapshot_dates": [date.date().isoformat() for date in snapshot_dates],
        "n_municipalities": int(len(municipality_summary)),
        "missing_geo_ids": missing_geo_ids,
        "headline_stats": {
            "final_mean_rate": float(final_daily["mean_rate"]),
            "final_median_rate": float(final_daily["median_rate"]),
            "final_population_weighted_rate": (
                None
                if pd.isna(final_daily["population_weighted_rate"])
                else float(final_daily["population_weighted_rate"])
            ),
            "final_max_rate": float(final_daily["max_rate"]),
            "municipalities_reaching_50_percent": int(
                municipality_summary["days_to_50_percent"].notna().sum()
            ),
            "final_mean_age_days": final_age_mean,
            "final_max_age_days": final_age_max,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote vaccination summary to %s", path)


def generate_report(
    *,
    dataset_path: Path,
    output_dir: Path,
    geojson: Path,
    run_id: str,
    max_scatter_points: int,
) -> dict[str, Path]:
    """Generate the canonical vaccination dataviz report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_vaccination_dataset(dataset_path, run_id=run_id)
    try:
        period_start, period_end = vaccination_period(dataset[MASK_VAR])
        period_slice = {TEMPORAL_COORD: slice(period_start, period_end)}
        rate_da = dataset[RATE_VAR].sel(period_slice)
        mask_da = dataset[MASK_VAR].sel(period_slice)
        age_da = dataset[AGE_VAR].sel(period_slice) if AGE_VAR in dataset else None
        population_da = dataset["population"] if "population" in dataset else None

        daily_summary = compute_daily_summary(rate_da, mask_da, population_da)
        municipality_summary = compute_municipality_summary(
            rate_da,
            mask_da,
            period_start=period_start,
        )

        daily_path = output_dir / "vaccination_daily_summary.csv"
        municipality_path = output_dir / "vaccination_municipality_summary.csv"
        daily_summary.to_csv(daily_path, index=False)
        municipality_summary.to_csv(municipality_path, index=False)

        scatter_path = output_dir / "vaccination_rate_over_time_scatter.png"
        plot_vaccination_scatter(
            daily_summary,
            rate_da,
            scatter_path,
            max_scatter_points=max_scatter_points,
        )

        regions_gdf = gpd.read_file(geojson)
        region_ids = [str(region_id) for region_id in rate_da[REGION_COORD].values]
        _, missing_geo_ids = ordered_geo_regions(regions_gdf, region_ids)
        dates = pd.DatetimeIndex(pd.to_datetime(rate_da[TEMPORAL_COORD].values))
        snapshot_dates = select_choropleth_dates(dates, period_start, period_end, n=16)
        choropleth_path = output_dir / "vaccination_choropleth_grid.png"
        plot_choropleth_grid(rate_da, regions_gdf, snapshot_dates, choropleth_path)

        summary_path = output_dir / "vaccination_summary.json"
        write_summary_json(
            summary_path,
            dataset_path=dataset_path,
            period_start=period_start,
            period_end=period_end,
            snapshot_dates=snapshot_dates,
            missing_geo_ids=missing_geo_ids,
            daily_summary=daily_summary,
            municipality_summary=municipality_summary,
            age_da=age_da,
        )

        return {
            "daily_summary": daily_path,
            "municipality_summary": municipality_path,
            "scatter": scatter_path,
            "choropleth_grid": choropleth_path,
            "summary_json": summary_path,
        }
    finally:
        dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Canonical Zarr dataset")
    parser.add_argument("--config", type=Path, default=None, help="Training config with data.dataset_path")
    parser.add_argument("--run-id", default="real", help="Run ID to select when dataset has run_id")
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON, help="Municipality GeoJSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=1_000_000,
        help="Maximum municipality-date scatter points before deterministic downsampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.dataset is None:
        if args.config is None:
            raise ValueError("Provide either --dataset or --config")
        args.dataset = dataset_path_from_config(load_config(args.config))

    artifacts = generate_report(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        geojson=args.geojson,
        run_id=args.run_id,
        max_scatter_points=args.max_scatter_points,
    )
    for name, path in artifacts.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
