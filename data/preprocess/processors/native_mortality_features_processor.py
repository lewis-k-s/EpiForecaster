"""Add fixed expected mortality and strictly past observed-count summaries."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class MortalityFeaturesConfig:
    short_window_days: int = 7
    long_window_days: int = 14
    lag_days: int = 1

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.short_window_days,
                self.long_window_days,
                self.lag_days,
            )
        ):
            raise ValueError("Mortality windows and lag must be positive integers")
        if self.short_window_days >= self.long_window_days:
            raise ValueError("Short mortality window must be smaller than long window")


MORTALITY_FEATURE_VARIABLES = (
    "mortality_mean_short",
    "mortality_mean_long",
    "mortality_coverage_short",
    "mortality_coverage_long",
)


def add_native_mortality_features(
    parent: xr.Dataset, config: MortalityFeaturesConfig
) -> xr.Dataset:
    """Preserve daily targets; summarize observed days ending at t - lag."""
    if parent.attrs.get("schema") != "obs_deaths_abs_excess":
        raise ValueError("Mortality parent has an unexpected schema")
    dates = np.asarray(parent.date.values).astype("datetime64[D]")
    if len(dates) < 1 or np.any(np.diff(dates) != np.timedelta64(1, "D")):
        raise ValueError("Mortality features require a consecutive daily grid")
    for name in ("value", "mask", "age", "expected", "excess"):
        if parent[name].dims != ("date", "abs_id"):
            raise ValueError(f"Mortality {name} must have date and abs_id dimensions")
    values = np.asarray(parent.value.values, dtype=np.float64)
    mask = np.asarray(parent["mask"].values, dtype=bool)
    expected = np.asarray(parent.expected.values)
    if (
        not np.isfinite(values[mask]).all()
        or np.any(values[mask] < 0)
        or not np.isfinite(expected).all()
        or np.any(expected < 0)
        or not np.allclose(parent.excess.values, values - expected, atol=1e-4)
    ):
        raise ValueError("Mortality parent has invalid counts or baseline")
    result = parent.copy(deep=True)
    count_sum = np.vstack(
        [np.zeros(values.shape[1]), np.cumsum(np.where(mask, values, 0), axis=0)]
    )
    support_sum = np.vstack([np.zeros(values.shape[1]), np.cumsum(mask, axis=0)])
    end = np.maximum(np.arange(len(dates)) - config.lag_days + 1, 0)
    for label, window in (
        ("short", config.short_window_days),
        ("long", config.long_window_days),
    ):
        start = np.maximum(end - window, 0)
        support = support_sum[end] - support_sum[start]
        total = count_sum[end] - count_sum[start]
        mean = np.divide(total, support, out=np.zeros_like(total), where=support > 0)
        result[f"mortality_mean_{label}"] = (
            ("date", "abs_id"),
            mean.astype(np.float32),
        )
        result[f"mortality_coverage_{label}"] = (
            ("date", "abs_id"),
            (support / window).astype(np.float32),
        )
    result.attrs.update(
        mortality_features_version=1,
        mortality_short_window_days=config.short_window_days,
        mortality_long_window_days=config.long_window_days,
        mortality_lag_days=config.lag_days,
        mortality_mean_units="observed deaths per observed day",
        mortality_coverage_definition="observed days divided by full window length",
    )
    return result


def qa_native_mortality_features(
    parent: xr.Dataset, output: xr.Dataset, config: MortalityFeaturesConfig
) -> dict:
    """Check preservation and the full persisted causal feature contract."""
    preserved = output[list(parent.data_vars)].copy(deep=False)
    preserved.attrs = parent.attrs.copy()
    xr.testing.assert_identical(parent, preserved)
    expected = add_native_mortality_features(parent, config)
    xr.testing.assert_identical(expected, output)
    return {
        "qa_passed": True,
        "parent_variables_preserved": list(parent.data_vars),
        "feature_variables": list(MORTALITY_FEATURE_VARIABLES),
        "short_window_days": config.short_window_days,
        "long_window_days": config.long_window_days,
        "lag_days": config.lag_days,
        "observed_days": int(parent["mask"].sum().item()),
    }


def write_native_mortality_features(dataset: xr.Dataset, output: Path) -> None:
    """Write only to a new versioned destination."""
    if output.exists():
        raise FileExistsError(f"Mortality output already exists: {output}")
    dataset.drop_encoding().to_zarr(output, mode="w-", zarr_format=2)
