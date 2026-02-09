"""
Processor for COVID-19 hospitalization data from pre-aggregated municipality CSV files.

This module handles the conversion of hospitalization data from weekly municipality-level
to daily municipality-level time series. It includes:
- Loading and parsing weekly hospitalization data (pre-aggregated by polygon overlap)
- Weekly to daily resampling with distribution smoothing
- Tobit-Kalman filtering for censored/missing data
- Output as xarray Dataset with consistent dims and coordinates
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.statespace.structural import UnobservedComponents

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from .edar_processor import _KalmanFilter


class HospitalizationsProcessor:
    """
    Converts weekly municipality-level hospitalization data to daily xarray Dataset.

    Processing pipeline:
    1. Load pre-aggregated weekly hospitalization data (municipality level)
    2. Aggregate by municipality/week
    3. Resample weekly to daily using equal distribution
    4. Apply Tobit-Kalman smoothing for censored data
    5. Output xarray Dataset with dims (run_id, date, region_id)
    """

    HOSP_FILE = "hospitalizations_municipality.csv"

    COLUMN_MAPPING = {
        "setmana_epidemiologica": "epi_week",
        "any": "year",
        "data_inici": "week_start",  # DD/MM/YYYY format
        "data_final": "week_end",  # DD/MM/YYYY format
        "municipality_code": "municipality_code",
        "municipality_name": "municipality_name",
        "casos_muni": "hospitalizations",  # Already weighted by polygon overlap
    }

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the hospitalizations processor.

        Args:
            config: Preprocessing configuration with date ranges
        """
        self.config = config

    def _load_raw_data(self, data_dir: Path) -> pd.DataFrame:
        """Load raw hospitalization data from CSV."""
        hosp_file = data_dir / self.HOSP_FILE

        if not hosp_file.exists():
            raise FileNotFoundError(f"Hospitalization file not found: {hosp_file}")

        print(f"  Loading hospitalizations from {hosp_file}")

        # Load the CSV file with string dtype for municipality_code to preserve leading zeros
        df = pd.read_csv(hosp_file, dtype={"municipality_code": str})

        # Rename columns
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Drop rows where municipality_code is NaN or empty string
        df = df[
            df["municipality_code"].notna() & (df["municipality_code"] != "")
        ]

        # Municipality codes are already strings from dasymetric_mob output
        # Ensure they are strings
        df["municipality_code"] = df["municipality_code"].astype(str)

        # Parse week start dates (DD/MM/YYYY format)
        df["week_start"] = pd.to_datetime(
            df["week_start"], format="%d/%m/%Y", errors="coerce"
        ).dt.tz_localize(None)

        # Remove rows with invalid data
        df = df.dropna(subset=["week_start", "hospitalizations"])

        # Remove negative values
        df = df[df["hospitalizations"] >= 0]

        print(f"  Loaded {len(df):,} hospitalization records")
        print(f"  Week range: {df['week_start'].min()} to {df['week_start'].max()}")
        print(f"  Unique municipalities: {df['municipality_code'].nunique()}")

        return df

    def _aggregate_to_municipality_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate hospitalizations to municipality-week level."""
        # Validate required columns exist
        required_cols = ["week_start", "municipality_code", "hospitalizations"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for aggregation: {missing_cols}\n"
                f"Available columns: {sorted(df.columns)}"
            )

        aggregated = (
            df.groupby(["week_start", "municipality_code"], dropna=False)[
                "hospitalizations"
            ]
            .sum()
            .reset_index()
        )

        print(f"  Aggregated to {len(aggregated)} municipality-week records")
        return aggregated

    def _resample_weekly_to_daily(self, muni_weekly: pd.DataFrame) -> pd.DataFrame:
        """
        Resample weekly hospitalization data to daily.

        Strategy: Distribute each week's total equally across 7 days.
        This preserves the total while creating a daily series.

        Also computes integer age channel:
        - Age = 1 on week start (original observation day)
        - Age = 2-7 on subsequent days (interpolated from weekly)
        - Age resets to 1 at the start of each new week
        """
        daily_records = []

        for muni_code in muni_weekly["municipality_code"].unique():
            muni_data = muni_weekly[
                muni_weekly["municipality_code"] == muni_code
            ].copy()
            muni_data = muni_data.set_index("week_start").sort_index()

            # Distribute weekly values evenly across 7 days
            for week_start, row in muni_data.iterrows():
                weekly_total = row["hospitalizations"]
                daily_value = weekly_total / 7.0

                # Create 7 daily records with integer age
                # Age 1 = week start (original data), Age 2-7 = interpolated
                for day_offset in range(7):
                    daily_date = week_start + pd.Timedelta(days=day_offset)
                    age = day_offset + 1  # Integer age: 1 to 7
                    daily_records.append(
                        {
                            "date": daily_date,
                            "municipality_code": muni_code,
                            "hospitalizations": daily_value,
                            "age": age,  # Integer age since last observation
                        }
                    )

        daily_df = pd.DataFrame(daily_records)

        # Aggregate by date (in case weeks overlap)
        # For age, use min (prioritize younger age if weeks overlap)
        daily_df = (
            daily_df.groupby(["date", "municipality_code"])
            .agg(
                {
                    "hospitalizations": "sum",
                    "age": "min",  # Use youngest age if overlapping
                }
            )
            .reset_index()
        )

        print(f"  Resampled to {len(daily_df):,} daily records with integer age")

        return daily_df

    def _fit_kalman_params(self, series: pd.Series) -> tuple[float, float]:
        """
        Fit Kalman filter parameters from time series data.

        Uses statsmodels UnobservedComponents to fit a local level model
        and extract the process and measurement variances.

        Args:
            series: Time series of hospitalization values

        Returns:
            Tuple of (process_var, measurement_var)
        """
        # Filter to positive values for log transform
        series = series.where(series > 0)
        series_log = pd.Series(np.log(series), index=series.index)

        if series_log.dropna().empty:
            raise ValueError("No finite observations to fit Kalman params")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(series_log, level="local level")
            result = model.fit(disp=False)

        params = dict(zip(result.param_names, result.params, strict=False))
        process_var = float(params.get("sigma2.level", 0.0))
        measurement_var = float(params.get("sigma2.irregular", 0.0))

        # Ensure positive variances
        process_var = max(process_var, 1e-6)
        measurement_var = max(measurement_var, 1e-6)

        return process_var, measurement_var

    def _apply_kalman_smoothing(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard Kalman filtering for time series smoothing.

        Unlike wastewater data, hospitalizations don't have detection limits,
        so we use standard Kalman (not Tobit-Kalman) for smoothing.
        """
        print("  Applying Kalman smoothing...")

        fallback_process = float(
            self.config.validation_options.get("process_var", 0.05)
        )
        fallback_measure = float(
            self.config.validation_options.get("measurement_var", 0.5)
        )

        smoothed_records = []

        for muni_code in daily_df["municipality_code"].unique():
            muni_data = daily_df[daily_df["municipality_code"] == muni_code].copy()
            muni_data = muni_data.set_index("date").sort_index()

            # Get values as numpy array
            values = muni_data["hospitalizations"].values

            # Fit Kalman parameters from data (mask zero/negative for fitting)
            fit_series = pd.Series(values, index=muni_data.index)
            fit_series = fit_series.where(fit_series > 0)

            try:
                process_var, measurement_var = self._fit_kalman_params(fit_series)
            except (ValueError, RuntimeError):
                process_var = fallback_process
                measurement_var = fallback_measure

            # Initialize Kalman filter with fitted params
            kf = _KalmanFilter(
                process_var=process_var,
                measurement_var=measurement_var,
            )

            # Apply filter
            filtered_values, flags = kf.filter_series(values)

            # Create smoothed records
            for i, date in enumerate(muni_data.index):
                smoothed_records.append(
                    {
                        "date": date,
                        "municipality_code": muni_code,
                        "hospitalizations": np.exp(
                            filtered_values[i]
                        ),  # Back-transform from log space
                        "hospitalizations_log": filtered_values[i],
                        "missing_flag": flags[i],  # 0=normal, 2=missing
                    }
                )

        smoothed_df = pd.DataFrame(smoothed_records)

        # Check for non-finite values
        non_finite = (~np.isfinite(smoothed_df["hospitalizations"])).sum()
        if non_finite > 0:
            print(f"  Warning: {non_finite} non-finite values after smoothing")
            smoothed_df["hospitalizations"] = smoothed_df["hospitalizations"].replace(
                [np.inf, -np.inf], np.nan
            )

        print(
            f"  Smoothing complete for {smoothed_df['municipality_code'].nunique()} municipalities"
        )

        return smoothed_df

    def _create_mask_and_age_channels(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mask and age channels matching EDAR/cases conventions.

        Mask: 1.0 ONLY if it's an original weekly measurement (age=1) and not missing.
        Age: Days since last ACTUAL observation (age=1 means today was measured).
        """
        daily_df = daily_df.copy()

        # Identify actual measurements: must be week start AND not marked as missing by Kalman
        is_week_start = (
            (daily_df["age"] == 1) if "age" in daily_df.columns else pd.Series(True, index=daily_df.index)
        )
        if "missing_flag" in daily_df.columns:
            is_not_missing = daily_df["missing_flag"] < 1.5
        else:
            is_not_missing = daily_df["hospitalizations"].notna()

        # Mask: 1.0 for actual measurements, 0.0 for all interpolated/missing values
        mask = is_week_start & is_not_missing
        daily_df["hospitalizations_mask"] = mask.astype(float)

        # Recompute age to correctly track staleness across missing weeks
        # Age = days since last measurement (mask == True)
        age_series_list = []
        for muni_code in daily_df["municipality_code"].unique():
            muni_data = daily_df[daily_df["municipality_code"] == muni_code].sort_values("date")
            muni_mask = muni_data["hospitalizations_mask"] > 0.5
            
            # Use pandas to calculate days since last True
            # 1. Create a group ID that increments each time a measurement is seen
            groups = muni_mask.cumsum()
            # 2. Count days within each group
            # If no measurement seen yet, groups will be 0.
            age = muni_data.groupby(groups).cumcount() + 1
            
            # Handle the leading zeros (before first measurement) - age should be max
            # Actually, groups.cumsum() for False, False, True, False starts with 0, 0, 1, 1
            # We want to detect the segment before the first True.
            first_true_idx = muni_mask.idxmax() if muni_mask.any() else None
            if first_true_idx is not None:
                # Set age to a large value (e.g. 14) for dates before first measurement
                age.loc[muni_data["date"] < muni_data.loc[first_true_idx, "date"]] = 14
            else:
                age[:] = 14
                
            muni_data["hospitalizations_age"] = age.clip(upper=14).astype(float)
            age_series_list.append(muni_data)
            
        return pd.concat(age_series_list, ignore_index=True)

    def process(
        self,
        data_dir: str | Path,
        apply_smoothing: bool = True,
    ) -> xr.Dataset:
        """
        Process hospitalization data into xarray Dataset.

        Args:
            data_dir: Directory containing hospitalization CSV file
            apply_smoothing: Whether to apply Kalman smoothing

        Returns:
            xarray Dataset with variables:
            - hospitalizations: Daily hospitalization counts (run_id, date, region_id)
            - hospitalizations_mask: Data availability mask
            - hospitalizations_age: Age channel for temporal tracking
        """
        print("Processing Catalonia hospitalization data")

        data_dir = Path(data_dir)

        # Load raw data
        hosp_df = self._load_raw_data(data_dir)

        if hosp_df.empty:
            print("  Warning: No hospitalization data found")
            return self._create_empty_dataset()

        # Filter to config date range (using week start)
        hosp_df = hosp_df[
            (hosp_df["week_start"] >= self.config.start_date)
            & (hosp_df["week_start"] <= self.config.end_date)
        ]

        if hosp_df.empty:
            print("  Warning: No hospitalization data in specified date range")
            return self._create_empty_dataset()

        # Aggregate to municipality-week
        muni_weekly = self._aggregate_to_municipality_week(hosp_df)

        # Resample to daily
        daily_df = self._resample_weekly_to_daily(muni_weekly)

        # Apply Kalman smoothing if requested
        if apply_smoothing:
            daily_df = self._apply_kalman_smoothing(daily_df)

        # Create mask and age channels
        daily_df = self._create_mask_and_age_channels(daily_df)

        # Crop to exact date range
        daily_df = daily_df[
            (daily_df["date"] >= self.config.start_date)
            & (daily_df["date"] <= self.config.end_date)
        ]

        # Pivot to wide format for xarray conversion
        pivot = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="hospitalizations",
            aggfunc="sum",
        )

        # Reindex to complete date range
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )
        pivot = pivot.reindex(date_range)

        # Rename to standard coordinate names
        pivot.columns.name = REGION_COORD
        pivot.index.name = TEMPORAL_COORD

        # Convert to xarray
        hosp_da = xr.DataArray(
            pivot.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot.index,
                REGION_COORD: pivot.columns.astype(str),
            },
            name="hospitalizations",
        )

        # Add run_id dimension (real data)
        hosp_da = hosp_da.expand_dims(run_id=["real"])

        # Create mask DataArray
        mask_pivot = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="hospitalizations_mask",
            aggfunc="max",  # Use max to preserve any valid observation
        ).reindex(date_range)

        # Rename columns to match REGION_COORD
        mask_pivot.columns.name = REGION_COORD
        mask_pivot.index.name = TEMPORAL_COORD

        mask_da = xr.DataArray(
            mask_pivot.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: mask_pivot.index,
                REGION_COORD: mask_pivot.columns.astype(str),
            },
            name="hospitalizations_mask",
        )
        mask_da = mask_da.expand_dims(run_id=["real"])

        # Create age DataArray
        age_pivot = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="hospitalizations_age",
            aggfunc="max",
        ).reindex(date_range)

        # Rename columns to match REGION_COORD
        age_pivot.columns.name = REGION_COORD
        age_pivot.index.name = TEMPORAL_COORD

        age_da = xr.DataArray(
            age_pivot.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: age_pivot.index,
                REGION_COORD: age_pivot.columns.astype(str),
            },
            name="hospitalizations_age",
        )
        age_da = age_da.expand_dims(run_id=["real"])

        # Merge into dataset
        result_ds = xr.Dataset(
            {
                "hospitalizations": hosp_da,
                "hospitalizations_mask": mask_da,
                "hospitalizations_age": age_da,
            }
        )

        print(
            f"  Processed hospitalizations: {hosp_da.sizes[TEMPORAL_COORD]} dates x "
            f"{hosp_da.sizes[REGION_COORD]} regions"
        )
        print(f"  Total hospitalizations: {float(hosp_da.sum()):.0f}")

        return result_ds

    def _create_empty_dataset(self) -> xr.Dataset:
        """Create empty dataset with proper structure when no data available."""
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )

        hosp_da = xr.DataArray(
            np.zeros((1, len(date_range), 0)),
            dims=["run_id", TEMPORAL_COORD, REGION_COORD],
            coords={"run_id": ["real"], TEMPORAL_COORD: date_range, REGION_COORD: []},
            name="hospitalizations",
        )

        mask_da = xr.DataArray(
            np.zeros((1, len(date_range), 0)),
            dims=["run_id", TEMPORAL_COORD, REGION_COORD],
            coords={"run_id": ["real"], TEMPORAL_COORD: date_range, REGION_COORD: []},
            name="hospitalizations_mask",
        )

        age_da = xr.DataArray(
            np.ones((1, len(date_range), 0)),
            dims=["run_id", TEMPORAL_COORD, REGION_COORD],
            coords={"run_id": ["real"], TEMPORAL_COORD: date_range, REGION_COORD: []},
            name="hospitalizations_age",
        )

        return xr.Dataset(
            {
                "hospitalizations": hosp_da,
                "hospitalizations_mask": mask_da,
                "hospitalizations_age": age_da,
            }
        )
