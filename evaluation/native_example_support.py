"""Export history and support checks for native qualitative forecast examples."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_example_history(
    root: Path,
    forecasts: pd.DataFrame,
    *,
    native_root: Path | None = None,
    cases_path: Path | None = None,
    wastewater_filename: str = "obs_wastewater.zarr",
    deaths_filename: str = "obs_deaths_abs_excess.zarr",
) -> pd.DataFrame:
    """Read observed history from the pinned input stores, preserving missingness."""
    outputs = []
    for target in forecasts.target.unique():
        sources = forecasts[forecasts.target == target]
        if target == "cases":
            dataset = xr.open_zarr(cases_path or root / "real_with_id.zarr")
        else:
            filename = {
                "deaths": deaths_filename,
                "hospitalizations": "obs_admissions.zarr",
                "wastewater": wastewater_filename,
            }[target]
            dataset = xr.open_zarr((native_root or root / "native_support") / filename)
        with dataset:
            for source in sources.source_id.unique():
                population = float(
                    sources[sources.source_id == source].population.iloc[0]
                )
                if target == "cases":
                    data = dataset.sel(region_id=source).isel(run_id=0)
                    mask = data.cases_mask.values.astype(bool)
                    raw = (
                        np.expm1(data.cases.values.astype(float)) * population / 100000
                    )
                    dates = data.date.values
                    censored = np.zeros(len(dates), dtype=bool)
                    excess = np.full(len(dates), np.nan)
                elif target == "wastewater":
                    plant, assay = source.split("::")
                    data = dataset.sel(edar_id=plant, assay=assay)
                    mask = data.observed_mask.values.astype(bool)
                    raw = data.concentration_raw.values
                    dates = data.date.values
                    censored = data.censor_code.values == 1
                    excess = np.full(len(dates), np.nan)
                else:
                    data = dataset.sel(abs_id=source)
                    mask = data["mask"].values.astype(bool)
                    raw = data.value.values
                    # Preserve the model's week-start input convention.
                    # Forecast exports use week-end target dates instead.
                    dates = data.date.values
                    censored = np.zeros(len(dates), dtype=bool)
                    excess = (
                        data.excess.values
                        if target == "deaths"
                        else np.full(len(dates), np.nan)
                    )
                score = (
                    np.log1p(raw)
                    if target == "wastewater"
                    else np.log1p(raw / population * 100000)
                )
                outputs.append(
                    pd.DataFrame(
                        {
                            "target": target,
                            "source_id": source,
                            "date": pd.to_datetime(dates),
                            "input_date": pd.to_datetime(data.date.values),
                            "observed_raw": np.where(mask, raw, np.nan),
                            "observed_score": np.where(mask, score, np.nan),
                            "observed": mask,
                            "censored": censored & mask,
                            "excess": np.where(mask, excess, np.nan),
                        }
                    )
                )
    return pd.concat(outputs, ignore_index=True)


def window_support(
    forecasts: pd.DataFrame, history: pd.DataFrame, length: int
) -> pd.DataFrame:
    """Count real observations and event days; never count imputed daily values."""
    result = []
    for (target, source, start), window in forecasts.groupby(
        ["target", "source_id", "window_start"], sort=True
    ):
        start_date = pd.Timestamp(window.window_start_date.iloc[0])
        origin = start_date + pd.Timedelta(days=length - 1)
        h = history[
            (history.target == target)
            & (history.source_id == source)
            & (history.input_date >= start_date)
            & (history.input_date <= origin)
            & history.observed
        ]
        exact = h[~h.censored]
        result.append(
            {
                "target": target,
                "source_id": source,
                "window_start": start,
                "history_observed": len(h),
                "history_uncensored": len(exact),
                "history_nonzero": int((exact.observed_raw > 0).sum()),
                "forecast_observed": window.target_date.nunique(),
                "forecast_nonzero": int((window.observed_raw > 0).sum()),
                "origin_date": origin,
            }
        )
    return pd.DataFrame(result)


def eligible_windows(support: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit head-specific thresholds for history-conditioned examples."""
    result = support.copy()
    rules = {
        "cases": (21, 10, 0, 0),
        "deaths": (28, 14, 14, 7),
        "hospitalizations": (4, 2, 3, 1),
        "wastewater": (4, 2, 0, 0),
    }
    keep = []
    for row in result.itertuples(index=False):
        history, future, positive_history, positive_future = rules[row.target]
        count = (
            row.history_uncensored
            if row.target == "wastewater"
            else row.history_observed
        )
        keep.append(
            count >= history
            and row.forecast_observed >= future
            and row.history_nonzero >= positive_history
            and row.forecast_nonzero >= positive_future
        )
    result["eligible"] = keep
    return result
