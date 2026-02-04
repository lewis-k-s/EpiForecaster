"""
Processor for Catalonia COVID-19 deaths data from pre-aggregated municipality CSV files.

This module handles the conversion of deaths data from municipality-level
data (pre-aggregated by polygon overlap).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig


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

    def process(
        self,
        data_dir: str | Path,
    ) -> xr.Dataset:
        """
        Process deaths data into xarray Dataset.

        Args:
            data_dir: Directory containing deaths CSV file

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
            return xr.Dataset({"deaths": deaths_da})

        # Aggregate to municipality-day
        muni_deaths = self._aggregate_to_municipality_day(deaths_df)

        # Create pivot table
        pivot = muni_deaths.pivot_table(
            index="date",
            columns="municipality_code",
            values="deaths",
            aggfunc="sum",
            fill_value=0,
        )

        # Reindex to complete date range
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )
        pivot = pivot.reindex(date_range, fill_value=0)

        # Rename to standard coordinate names
        pivot.columns.name = REGION_COORD
        pivot.index.name = TEMPORAL_COORD

        # Convert to xarray
        deaths_da = xr.DataArray(
            pivot.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot.index,
                REGION_COORD: pivot.columns.astype(str),
            },
        )

        result_ds = xr.Dataset({"deaths": deaths_da})

        print(
            f"  Processed deaths: {deaths_da.sizes[TEMPORAL_COORD]} dates x {deaths_da.sizes[REGION_COORD]} regions"
        )
        print(f"  Total deaths: {int(deaths_da.sum())}")

        return result_ds
