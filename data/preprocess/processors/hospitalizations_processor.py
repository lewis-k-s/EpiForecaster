"""
Processor for COVID-19 hospitalization data from pre-aggregated municipality CSV files.

This module handles the conversion of hospitalization data from weekly municipality-level
to daily municipality-level time series. It includes:
- Loading and parsing weekly hospitalization data (pre-aggregated by polygon overlap)
- Weekly to daily resampling: the weekly total sits on its report date, weeks
  without a row inside the source span are observed zeros, other days are filled
  causally
- Output as xarray Dataset with consistent dims and coordinates
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from ..smoothing import fill_clinical_gaps


class HospitalizationsProcessor:
    """
    Converts weekly municipality-level hospitalization data to daily xarray Dataset.

    Processing pipeline:
    1. Load pre-aggregated weekly hospitalization data (municipality level)
    2. Aggregate by municipality/week
    3. Resample weekly to daily with sparse observations (NaN on non-report days)
    4. Fill the non-report days causally
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
        df = df[df["municipality_code"].notna() & (df["municipality_code"] != "")]

        # Municipality codes are already strings from dasymetric_mob output
        # Ensure they are strings
        df["municipality_code"] = df["municipality_code"].astype(str)

        # Parse week start dates (DD/MM/YYYY format)
        df["week_start"] = pd.to_datetime(
            df["week_start"], format="%d/%m/%Y", errors="coerce"
        ).dt.tz_localize(None)

        df["week_start"] += pd.Timedelta(
            days=self.config.hospitalizations_reporting_lag_days
        )

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
        Resample weekly hospitalization data to daily with sparse observations.

        The weekly total is placed on its report date and the other days are NaN.
        The source lists positive totals only, so a week without a row inside
        the source's week span is an observed zero.
        """
        # Rename week_start to date for consistency with other processors
        muni_weekly = muni_weekly.rename(columns={"week_start": "date"})

        # Pivot to wide format (municipalities as columns) for easier resampling
        pivot = muni_weekly.pivot(
            index="date", columns="municipality_code", values="hospitalizations"
        )

        report_dates = pd.date_range(pivot.index.min(), pivot.index.max(), freq="7D")
        if not pivot.index.isin(report_dates).all():
            raise ValueError("Hospitalization report dates are not on a 7-day grid")
        pivot = pivot.reindex(report_dates).fillna(0.0).rename_axis("date")

        # Resample to daily frequency - creates NaN for days without observations
        pivot_daily = pivot.resample("D").asfreq()

        # Melt back to long format
        daily_df = pivot_daily.reset_index().melt(
            id_vars=["date"],
            var_name="municipality_code",
            value_name="hospitalizations",
        )

        print(
            f"  Resampled to {len(daily_df):,} daily records "
            f"({daily_df['hospitalizations'].notna().sum():,} observations, "
            f"{daily_df['hospitalizations'].isna().sum():,} to be filled)"
        )

        return daily_df

    def _fill_gaps(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Fill non-report days causally; every finite total is an observation."""
        method = self.config.smoothing.clinical_method
        print(f"  Filling non-report hospitalization days (method={method})...")
        return fill_clinical_gaps(
            daily_df,
            "hospitalizations",
            self.config.smoothing,
            fallback_process_var=float(
                self.config.validation_options.get("process_var", 0.05)
            ),
            fallback_measurement_var=float(
                self.config.validation_options.get("measurement_var", 0.5)
            ),
        )

    def _create_mask_and_age_channels(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mask and age channels matching EDAR/cases conventions.

        Mask: 1.0 ONLY if it was an original observation (not a filled day).
        Age: Days since last ACTUAL observation (age=1 means today was measured).
        """
        daily_df = daily_df.copy()
        daily_df["hospitalizations_mask"] = (
            daily_df["hospitalizations_observed"] > 0.5
        ).astype(float)

        # Age: Days since last ACTUAL observation
        age_series_list = []
        for muni_code in daily_df["municipality_code"].unique():
            muni_data = daily_df[
                daily_df["municipality_code"] == muni_code
            ].sort_values("date")
            muni_mask = muni_data["hospitalizations_mask"] > 0.5

            # Use pandas to calculate days since last True
            # 1. Create a group ID that increments each time a measurement is seen
            groups = muni_mask.cumsum()
            # 2. Count days within each group
            age = muni_data.groupby(groups).cumcount() + 1

            # Handle the leading period (before first measurement) - age should be max
            first_true_idx = muni_mask.idxmax() if muni_mask.any() else None
            if first_true_idx is not None:
                # Set age to max (14) for dates before first measurement
                age.loc[muni_data["date"] < muni_data.loc[first_true_idx, "date"]] = 14
            else:
                age[:] = 14

            muni_data["hospitalizations_age"] = age.clip(upper=14).astype(float)
            age_series_list.append(muni_data)

        return pd.concat(age_series_list, ignore_index=True)

    def process(
        self,
        data_dir: str | Path,
    ) -> xr.Dataset:
        """
        Process hospitalization data into xarray Dataset.

        Args:
            data_dir: Directory containing hospitalization CSV file

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

        # Resample to daily (sparse observations)
        daily_df = self._resample_weekly_to_daily(muni_weekly)

        daily_df = self._fill_gaps(daily_df)

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
