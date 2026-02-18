"""
Processor for temporal covariates (day-of-week, holidays).

This module generates temporal covariate features for epidemiological forecasting,
including cyclic day-of-week encoding and holiday indicators.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import TEMPORAL_COORD, PreprocessingConfig, TemporalCovariatesConfig


class TemporalCovariatesProcessor:
    """
    Generates temporal covariate features for time series forecasting.

    Features are computed as:
    - Day-of-week sin/cos: sin(2π * day_of_week / 7), cos(2π * day_of_week / 7)
    - Holiday indicator: 1.0 if date is a holiday, 0.0 otherwise

    Output shape: (time, covariate_dim) where covariate_dim = 2 (dow) + 1 (holiday) = 3
    """

    def __init__(self, config: PreprocessingConfig):
        if config.temporal_covariates is None:
            raise ValueError(
                "temporal_covariates config is required for TemporalCovariatesProcessor"
            )
        self.config = config
        self.tc_config: TemporalCovariatesConfig = config.temporal_covariates
        self._holiday_dates: set[str] | None = None

    def _load_holiday_calendar(self) -> set[str]:
        """Load holiday dates from CSV file."""
        if self._holiday_dates is not None:
            return self._holiday_dates

        if not self.tc_config.holiday_calendar_file:
            return set()

        holiday_path = Path(self.tc_config.holiday_calendar_file)
        if not holiday_path.exists():
            raise FileNotFoundError(f"Holiday calendar file not found: {holiday_path}")

        df = pd.read_csv(holiday_path)
        if "date" not in df.columns:
            raise ValueError(
                f"Holiday calendar must have 'date' column. Got: {df.columns.tolist()}"
            )

        dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        self._holiday_dates = set(dates.tolist())
        print(f"  Loaded {len(self._holiday_dates)} holiday dates from {holiday_path}")
        return self._holiday_dates

    def process(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> xr.DataArray:
        """
        Generate temporal covariates for the given date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (exclusive, or last date in the series)

        Returns:
            xr.DataArray with shape (time, covariate_dim) and dims (date, covariate)
        """
        print("Processing temporal covariates...")

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        holiday_dates = set()
        if self.tc_config.include_holidays:
            holiday_dates = self._load_holiday_calendar()

        features = []
        feature_names = []

        if self.tc_config.include_day_of_week:
            dow = np.array([d.dayofweek for d in date_range])
            dow_sin = np.sin(2 * math.pi * dow / 7)
            dow_cos = np.cos(2 * math.pi * dow / 7)
            features.append(dow_sin.reshape(-1, 1))
            features.append(dow_cos.reshape(-1, 1))
            feature_names.extend(["dow_sin", "dow_cos"])

        if self.tc_config.include_holidays:
            is_holiday = np.array(
                [
                    1.0 if d.strftime("%Y-%m-%d") in holiday_dates else 0.0
                    for d in date_range
                ]
            )
            features.append(is_holiday.reshape(-1, 1))
            feature_names.append("is_holiday")

        data = np.concatenate(features, axis=1).astype(np.float16)

        da = xr.DataArray(
            data,
            dims=[TEMPORAL_COORD, "covariate"],
            coords={
                TEMPORAL_COORD: date_range,
                "covariate": feature_names,
            },
            name="temporal_covariates",
        )

        n_holidays = (
            int(da.sel(covariate="is_holiday").sum().item())
            if self.tc_config.include_holidays
            else 0
        )
        print(f"  Generated temporal covariates: shape={da.shape}")
        print(f"  Date range: {date_range.min()} to {date_range.max()}")
        print(f"  Holidays in range: {n_holidays}")

        return da
