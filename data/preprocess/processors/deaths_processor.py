"""
Processor for Catalonia COVID-19 deaths data.

This module handles the conversion of deaths data from official Catalonia
open data sources. Deaths are reported at comarca level.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from .municipality_mapping_processor import MunicipalityMappingProcessor


class DeathsProcessor:
    """
    Converts Catalonia deaths data to xarray Dataset.

    Deaths are reported at comarca level. This processor:
    - Loads deaths data
    - Optionally allocates deaths to municipalities using population weights
    - Returns xarray Dataset with deaths variable
    """

    DEATHS_FILE = (
        "Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
    )

    COLUMN_MAPPING = {
        "Data defunció": "date_of_death",
        "Codi Comarca": "comarca_code",
        "Nombre defuncions": "n_deaths",
    }

    DTYPES = {
        "Data defunció": str,  # DD/MM/YYYY format
        "Codi Comarca": str,  # Preserve leading zeros
        "Nombre defuncions": int,
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

        df = pd.read_csv(
            deaths_file,
            dtype=self.DTYPES,  # type: ignore[arg-type]
            usecols=list(self.COLUMN_MAPPING.keys()),
        )

        # Rename columns
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Parse dates from DD/MM/YYYY format
        df["date_of_death"] = (
            pd.to_datetime(df["date_of_death"], format="%d/%m/%Y", errors="coerce")
            .dt.tz_localize(None)
            .dt.floor("D")
        )

        # Rename to standard date column
        df = df.rename(columns={"date_of_death": "date"})

        # Remove invalid rows
        df = df.dropna(subset=["date", "comarca_code", "n_deaths"])

        # Remove negative deaths if any
        df = df[df["n_deaths"] >= 0]

        print(f"  Loaded {len(df):,} death records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique comarcas: {df['comarca_code'].nunique()}")

        return df

    def _aggregate_to_comarca_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate deaths to comarca-date level (summing across sex)."""
        aggregated = (
            df.groupby(["date", "comarca_code"], dropna=False)["n_deaths"]
            .sum()
            .reset_index()
        )

        return aggregated

    def _allocate_to_municipalities(
        self,
        comarca_deaths: pd.DataFrame,
        population_df: pd.DataFrame,
        mapping_processor: MunicipalityMappingProcessor,
    ) -> pd.DataFrame:
        """
        Allocate comarca-level deaths to municipalities using population weights.

        Args:
            comarca_deaths: DataFrame with deaths per comarca per day
            population_df: DataFrame with population per municipality
            mapping_processor: For municipality->comarca mapping

        Returns:
            DataFrame with deaths per municipality per day
        """
        # Load mapping
        mapping = mapping_processor.load_mapping()

        # Merge mapping with population
        merged = mapping.merge(
            population_df,
            left_on="municipality_code",
            right_on=REGION_COORD,
            how="left",
        )

        # Calculate comarca population for weighting
        merged["comarca_population"] = merged.groupby("comarca_code")[
            "population"
        ].transform("sum")
        merged["weight"] = merged["population"] / merged["comarca_population"]

        # Merge with deaths data
        deaths_with_weights = comarca_deaths.merge(
            merged[["comarca_code", "municipality_code", "weight"]],
            on="comarca_code",
            how="left",
        )

        # Allocate deaths
        deaths_with_weights["n_deaths_allocated"] = (
            deaths_with_weights["n_deaths"] * deaths_with_weights["weight"]
        )

        # Aggregate to municipality level
        municipal_deaths = (
            deaths_with_weights.groupby(["date", "municipality_code"])[
                "n_deaths_allocated"
            ]
            .sum()
            .reset_index()
            .rename(columns={"n_deaths_allocated": "n_deaths"})
        )

        return municipal_deaths

    def process(
        self,
        data_dir: str | Path,
        population_df: pd.DataFrame | None = None,
        allocate_to_municipalities: bool = False,
        mapping_processor: MunicipalityMappingProcessor | None = None,
    ) -> xr.Dataset:
        """
        Process deaths data into xarray Dataset.

        Args:
            data_dir: Directory containing deaths CSV file
            population_df: Optional population data for municipality allocation
            allocate_to_municipalities: If True, allocate deaths to municipalities
            mapping_processor: Required if allocate_to_municipalities=True

        Returns:
            xarray Dataset with deaths variable
        """
        print("Processing Catalonia deaths data")

        if allocate_to_municipalities and mapping_processor is None:
            raise ValueError(
                "mapping_processor required when allocate_to_municipalities=True"
            )

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

        # Aggregate to comarca-day
        comarca_deaths = self._aggregate_to_comarca_day(deaths_df)

        # Optionally allocate to municipalities
        if allocate_to_municipalities and population_df is not None:
            processed_deaths = self._allocate_to_municipalities(
                comarca_deaths,
                population_df,
                mapping_processor,  # type: ignore
            )
            region_col = "municipality_code"
        else:
            # Keep at comarca level
            processed_deaths = comarca_deaths.rename(
                columns={"comarca_code": "region_id"}
            )
            region_col = "region_id"

        # Create pivot table
        pivot = processed_deaths.pivot_table(
            index="date",
            columns=region_col,
            values="n_deaths",
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
