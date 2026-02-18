"""
Processor for Catalonia COVID-19 deaths data from pre-aggregated municipality CSV files.

This module handles the conversion of deaths data from municipality-level
data (pre-aggregated by polygon overlap), including Kalman smoothing on a
daily grid while preserving an observation mask for true measurement days.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.statespace.structural import UnobservedComponents

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from .edar_processor import _KalmanFilter


class DeathsProcessor:
    """
    Converts Catalonia deaths data to xarray Dataset.

    Deaths are reported at municipality level. This processor:
    - Loads deaths data from pre-aggregated municipality CSV
    - Returns xarray Dataset with deaths variable
    """

    DEATHS_FILE = "deaths_municipality.csv"

    COLUMN_MAPPING = {
        "Data defunció": "date",
        "municipality_code": "municipality_code",
        "municipality_name": "municipality_name",
        "defuncions_muni": "deaths",  # Already weighted by polygon overlap
    }

    def __init__(self, config: PreprocessingConfig):
        """Initialize the deaths processor."""
        self.config = config

    def _load_raw_data(self, data_dir: Path) -> pd.DataFrame:
        """Load raw deaths data from CSV."""
        deaths_file = data_dir / self.DEATHS_FILE

        if not deaths_file.exists():
            raise FileNotFoundError(f"Deaths file not found: {deaths_file}")

        print(f"  Loading deaths from {deaths_file}")

        # Load the CSV file with string dtype for municipality_code to preserve leading zeros
        df = pd.read_csv(deaths_file, dtype={"municipality_code": str})

        # Rename columns
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Drop rows where municipality_code is NaN or empty string
        df = df[
            df["municipality_code"].notna() & (df["municipality_code"] != "")
        ]

        # Municipality codes are already strings from dasymetric_mob output
        # Ensure they are strings
        df["municipality_code"] = df["municipality_code"].astype(str)

        # Parse dates from DD/MM/YYYY format
        df["date"] = (
            pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
            .dt.tz_localize(None)
            .dt.floor("D")
        )

        # Remove invalid rows
        df = df.dropna(subset=["date", "deaths"])

        # Remove negative deaths if any
        df = df[df["deaths"] >= 0]

        print(f"  Loaded {len(df):,} death records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique municipalities: {df['municipality_code'].nunique()}")

        return df

    def _aggregate_to_municipality_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate deaths to municipality-date level."""
        aggregated = (
            df.groupby(["date", "municipality_code"], dropna=False)["deaths"]
            .sum()
            .reset_index()
        )

        return aggregated

    def _fit_kalman_params(self, series: pd.Series) -> tuple[float, float]:
        """Fit Kalman process/measurement variances from positive observations."""
        series = series.where(series > 0)
        series_log = pd.Series(np.log(series), index=series.index)

        if series_log.dropna().empty:
            raise ValueError("No finite observations to fit Kalman params")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(series_log, level="local level")
            result = model.fit(disp=False)

        params = dict(zip(result.param_names, result.params, strict=False))
        process_var = max(float(params.get("sigma2.level", 0.0)), 1e-6)
        measurement_var = max(float(params.get("sigma2.irregular", 0.0)), 1e-6)
        return process_var, measurement_var

    def _apply_kalman_smoothing(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Smooth deaths time series and interpolate missing days per municipality."""
        print("  Applying Kalman smoothing to deaths...")

        fallback_process = float(
            self.config.validation_options.get("process_var", 0.05)
        )
        fallback_measure = float(
            self.config.validation_options.get("measurement_var", 0.5)
        )

        smoothed_records: list[dict[str, object]] = []
        for muni_code in daily_df["municipality_code"].unique():
            muni_data = daily_df[daily_df["municipality_code"] == muni_code].copy()
            muni_data = muni_data.set_index("date").sort_index()

            # Preserve true observation mask before smoothing/interpolation.
            observed = muni_data["deaths"].notna().to_numpy()
            values = muni_data["deaths"].to_numpy()

            fit_series = pd.Series(values, index=muni_data.index).where(
                lambda s: s > 0
            )
            try:
                process_var, measurement_var = self._fit_kalman_params(fit_series)
            except Exception as e:
                print(f"    ! Falling back to configured variances for {muni_code}: {e}")
                process_var = fallback_process
                measurement_var = fallback_measure

            kf = _KalmanFilter(
                process_var=process_var,
                measurement_var=measurement_var,
            )
            filtered_values, flags = kf.filter_series(values)

            for i, date in enumerate(muni_data.index):
                smoothed_records.append(
                    {
                        "date": date,
                        "municipality_code": muni_code,
                        "deaths": float(np.exp(filtered_values[i])),
                        "deaths_missing_flag": float(flags[i]),
                        "deaths_observed": float(observed[i]),
                    }
                )

        smoothed_df = pd.DataFrame(smoothed_records)
        non_finite = (~np.isfinite(smoothed_df["deaths"])).sum()
        if non_finite > 0:
            print(f"  Warning: {non_finite} non-finite values after deaths smoothing")
            smoothed_df["deaths"] = smoothed_df["deaths"].replace(
                [np.inf, -np.inf], np.nan
            )

        print(
            f"  Smoothing complete for {smoothed_df['municipality_code'].nunique()} municipalities"
        )
        return smoothed_df

    def _create_mask_and_age_channels(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Create deaths mask and age channels from observation availability."""
        daily_df = daily_df.copy()
        if "deaths_observed" in daily_df.columns:
            daily_df["deaths_mask"] = (daily_df["deaths_observed"] > 0.5).astype(float)
        else:
            daily_df["deaths_mask"] = daily_df["deaths"].notna().astype(float)

        age_series_list = []
        for muni_code in daily_df["municipality_code"].unique():
            muni_data = daily_df[
                daily_df["municipality_code"] == muni_code
            ].sort_values("date")
            muni_mask = muni_data["deaths_mask"] > 0.5

            groups = muni_mask.cumsum()
            age = muni_data.groupby(groups).cumcount() + 1

            first_true_idx = muni_mask.idxmax() if muni_mask.any() else None
            if first_true_idx is not None:
                age.loc[muni_data["date"] < muni_data.loc[first_true_idx, "date"]] = 14
            else:
                age[:] = 14

            muni_data["deaths_age"] = age.clip(upper=14).astype(float)
            age_series_list.append(muni_data)

        return pd.concat(age_series_list, ignore_index=True)

    def process(
        self,
        data_dir: str | Path,
        apply_smoothing: bool = True,
    ) -> xr.Dataset:
        """
        Process deaths data into xarray Dataset.

        Args:
            data_dir: Directory containing deaths CSV file
            apply_smoothing: Whether to apply Kalman smoothing/interpolation

        Returns:
            xarray Dataset with deaths variable
        """
        print("Processing Catalonia deaths data")

        data_dir = Path(data_dir)

        # Load raw data
        deaths_df = self._load_raw_data(data_dir)

        # Filter to config date range
        deaths_df = deaths_df[
            (deaths_df["date"] >= self.config.start_date)
            & (deaths_df["date"] <= self.config.end_date)
        ]

        if deaths_df.empty:
            print("  Warning: No death data in specified date range")
            # Return empty dataset with proper structure
            date_range = pd.date_range(
                start=self.config.start_date, end=self.config.end_date, freq="D"
            )
            deaths_da = xr.DataArray(
                np.zeros((len(date_range), 0)),
                dims=[TEMPORAL_COORD, REGION_COORD],
                coords={TEMPORAL_COORD: date_range, REGION_COORD: []},
            )
            mask_da = xr.DataArray(
                np.zeros((len(date_range), 0)),
                dims=[TEMPORAL_COORD, REGION_COORD],
                coords={TEMPORAL_COORD: date_range, REGION_COORD: []},
            )
            age_da = xr.DataArray(
                np.ones((len(date_range), 0)),
                dims=[TEMPORAL_COORD, REGION_COORD],
                coords={TEMPORAL_COORD: date_range, REGION_COORD: []},
            )
            return xr.Dataset(
                {"deaths": deaths_da, "deaths_mask": mask_da, "deaths_age": age_da}
            )

        # Aggregate to municipality-day
        muni_deaths = self._aggregate_to_municipality_day(deaths_df)

        # Create sparse pivot (NaN where unobserved)
        pivot = muni_deaths.pivot_table(
            index="date",
            columns="municipality_code",
            values="deaths",
            aggfunc="first",
        )

        # Reindex to complete date range
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )
        pivot = pivot.reindex(date_range)

        # Rename to standard coordinate names
        pivot.columns.name = REGION_COORD
        pivot.index.name = TEMPORAL_COORD

        # Convert to long format for channel construction
        daily_df = pivot.reset_index().melt(
            id_vars=[TEMPORAL_COORD],
            var_name="municipality_code",
            value_name="deaths",
        )

        if apply_smoothing:
            daily_df = self._apply_kalman_smoothing(daily_df)

        daily_df = self._create_mask_and_age_channels(daily_df)

        # Pivot values/mask/age back to wide format
        pivot_deaths = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="deaths",
            aggfunc="first",
        ).reindex(date_range)
        pivot_mask = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="deaths_mask",
            aggfunc="max",
        ).reindex(date_range, fill_value=0)
        pivot_age = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="deaths_age",
            aggfunc="max",
        ).reindex(date_range, fill_value=14)

        pivot_deaths.columns.name = REGION_COORD
        pivot_deaths.index.name = TEMPORAL_COORD
        pivot_mask.columns.name = REGION_COORD
        pivot_mask.index.name = TEMPORAL_COORD
        pivot_age.columns.name = REGION_COORD
        pivot_age.index.name = TEMPORAL_COORD

        # Convert to xarray
        deaths_da = xr.DataArray(
            pivot_deaths.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot_deaths.index,
                REGION_COORD: pivot_deaths.columns.astype(str),
            },
        )
        mask_da = xr.DataArray(
            pivot_mask.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot_mask.index,
                REGION_COORD: pivot_mask.columns.astype(str),
            },
        )
        age_da = xr.DataArray(
            pivot_age.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot_age.index,
                REGION_COORD: pivot_age.columns.astype(str),
            },
        )

        result_ds = xr.Dataset(
            {"deaths": deaths_da, "deaths_mask": mask_da, "deaths_age": age_da}
        )

        print(
            f"  Processed deaths: {deaths_da.sizes[TEMPORAL_COORD]} dates x {deaths_da.sizes[REGION_COORD]} regions"
        )
        print(f"  Total deaths: {int(deaths_da.sum())}")

        return result_ds
