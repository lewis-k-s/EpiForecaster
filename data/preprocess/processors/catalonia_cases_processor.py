"""
Processor for Catalonia official COVID-19 case data.

This module handles the conversion of official Catalonia open data CSV files
into canonical xarray formats. It processes daily case counts at municipality
level and produces output compatible with the existing pipeline.

Data source: Registre de casos de COVID-19 a Catalunya per municipi i sexe
https://datos.gob.es/ca/catalogo/a09002970-registro-de-test-de-covid-19-realizados-en-catalunya-segregacion-por-sexo-y-municipio
"""

from pathlib import Path

import pandas as pd
import xarray as xr

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig


class CataloniaCasesProcessor:
    """
    Converts Catalonia official COVID case data to xarray Dataset.

    This processor handles:
    - Loading cases from official Catalonia CSV
    - Preserving leading zeros in municipality codes
    - Converting DD/MM/YYYY dates to datetime
    - Aggregating across sex and test type
    - Optional test-type-specific processing (PCR vs TAR)

    Output format matches existing CasesProcessor for drop-in compatibility.
    """

    CASES_FILE = "Registre_de_casos_de_COVID-19_a_Catalunya_per_municipi_i_sexe.csv"

    # Column mapping from Catalan to English
    COLUMN_MAPPING = {
        "TipusCasData": "date",
        "MunicipiCodi": "municipality_code",
        "TipusCasDescripcio": "test_type",
        "NumCasos": "n_cases",
    }

    # Data types to preserve leading zeros
    DTYPES = {
        "TipusCasData": str,
        "MunicipiCodi": str,  # Critical: preserve leading zeros
        "ComarcaCodi": str,
        "SexeCodi": str,
        "NumCasos": int,
    }

    # Test type categorization for inferred testing rate
    TEST_TYPE_CATEGORIES = {
        "Positiu TAR": "TAR",
        "Positiu per Test Ràpid": "TAR",
        "Positiu PCR": "PCR",
        "PCR probable": "PCR",
        "Positiu per ELISA": "ELISA",
        "Epidemiològic": "EPI",
    }

    def __init__(self, config: PreprocessingConfig):
        """Initialize the Catalonia cases processor."""
        self.config = config
        self.validation_options = config.validation_options

    def _load_raw_data(self, data_dir: Path) -> pd.DataFrame:
        """Load raw cases data from CSV."""
        cases_file = data_dir / self.CASES_FILE

        if not cases_file.exists():
            raise FileNotFoundError(f"Cases file not found: {cases_file}")

        print(f"  Loading cases from {cases_file}")

        # Read with explicit dtypes to preserve leading zeros
        df = pd.read_csv(  # type: ignore[call-arg]
            cases_file,
            dtype=self.DTYPES,
            usecols=list(self.DTYPES.keys()) + ["TipusCasDescripcio"],
        )

        # Rename columns to English
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Parse dates from DD/MM/YYYY format
        df["date"] = (
            pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
            .dt.tz_localize(None)
            .dt.floor("D")
        )

        # Remove invalid rows
        df = df.dropna(subset=["date", "municipality_code", "n_cases"])

        # Filter to valid cases only (remove empty municipality codes)
        df = df[df["municipality_code"] != ""]

        # Remove negative cases if any
        df = df[df["n_cases"] >= 0]

        print(f"  Loaded {len(df):,} case records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique municipalities: {df['municipality_code'].nunique()}")

        return df

    def _categorize_test_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standardized test type category column."""
        df["test_category"] = df["test_type"].map(self.TEST_TYPE_CATEGORIES)  # type: ignore[arg-type]
        # Keep original test type as is for unmapped categories
        df["test_category"] = df["test_category"].fillna(df["test_type"])
        return df

    def _aggregate_by_municipality(
        self, df: pd.DataFrame, by_test_type: bool = False
    ) -> pd.DataFrame:
        """Aggregate cases to municipality-date level."""
        group_cols = ["date", "municipality_code"]
        if by_test_type:
            group_cols.append("test_category")

        aggregated = df.groupby(group_cols, dropna=False)["n_cases"].sum().reset_index()

        return aggregated

    def process(self, data_dir: str | Path, by_test_type: bool = False) -> xr.Dataset:
        """
        Process Catalonia cases data into canonical xarray Dataset.

        Args:
            data_dir: Directory containing cases CSV file
            by_test_type: If True, return separate variables for PCR/TAR cases

        Returns:
            xarray Dataset with structure:
            - cases: total cases (date, region_id)
            - cases_pcr: PCR cases (optional, if by_test_type=True)
            - cases_tar: TAR cases (optional, if by_test_type=True)
        """
        print("Processing Catalonia official case data")

        data_dir = Path(data_dir)

        # Load raw data
        cases_df = self._load_raw_data(data_dir)

        # Add test type categories
        cases_df = self._categorize_test_types(cases_df)

        # Filter to config date range
        cases_df = cases_df[
            (cases_df["date"] >= self.config.start_date)
            & (cases_df["date"] <= self.config.end_date)
        ]

        if cases_df.empty:
            raise ValueError("No case data in specified date range")

        # Aggregate to municipality level
        aggregated = self._aggregate_by_municipality(
            cases_df, by_test_type=by_test_type
        )

        # Create pivot table: date x municipality_code
        if by_test_type:
            # Create separate pivot for each test type
            pivot_pcr = aggregated[aggregated["test_category"] == "PCR"].pivot_table(
                index="date",
                columns="municipality_code",
                values="n_cases",
                aggfunc="sum",
                fill_value=0,
            )

            pivot_tar = aggregated[aggregated["test_category"] == "TAR"].pivot_table(
                index="date",
                columns="municipality_code",
                values="n_cases",
                aggfunc="sum",
                fill_value=0,
            )

            # Total cases pivot
            pivot_total = aggregated.pivot_table(
                index="date",
                columns="municipality_code",
                values="n_cases",
                aggfunc="sum",
                fill_value=0,
            )
        else:
            pivot_total = aggregated.pivot_table(
                index="date",
                columns="municipality_code",
                values="n_cases",
                aggfunc="sum",
                fill_value=0,
            )

        # Reindex to complete date range
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )
        pivot_total = pivot_total.reindex(date_range, fill_value=0)

        # Rename municipality_code to region_id for compatibility
        pivot_total.columns.name = REGION_COORD
        pivot_total.index.name = TEMPORAL_COORD

        # Convert to xarray
        cases_da = xr.DataArray(
            pivot_total.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot_total.index,
                REGION_COORD: pivot_total.columns.astype(str),
            },
        )

        # Build result dataset
        result_ds = xr.Dataset({"cases": cases_da})

        if by_test_type:
            # Reindex and add test-type-specific variables
            pivot_pcr = pivot_pcr.reindex(date_range, fill_value=0)
            pivot_tar = pivot_tar.reindex(date_range, fill_value=0)

            pivot_pcr.columns.name = REGION_COORD
            pivot_tar.columns.name = REGION_COORD

            cases_pcr = xr.DataArray(
                pivot_pcr.values,
                dims=[TEMPORAL_COORD, REGION_COORD],
                coords={
                    TEMPORAL_COORD: pivot_pcr.index,
                    REGION_COORD: pivot_pcr.columns.astype(str),
                },
            )

            cases_tar = xr.DataArray(
                pivot_tar.values,
                dims=[TEMPORAL_COORD, REGION_COORD],
                coords={
                    TEMPORAL_COORD: pivot_tar.index,
                    REGION_COORD: pivot_tar.columns.astype(str),
                },
            )

            result_ds["cases_pcr"] = cases_pcr
            result_ds["cases_tar"] = cases_tar

        print(
            f"  Processed cases: {cases_da.sizes[TEMPORAL_COORD]} dates x {cases_da.sizes[REGION_COORD]} regions"
        )
        print(f"  Total cases: {int(cases_da.sum())}")

        return result_ds
