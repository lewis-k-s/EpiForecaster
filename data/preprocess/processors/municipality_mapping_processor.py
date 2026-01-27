"""
Processor for municipality to comarca mapping data.

This module handles the loading and processing of the municipality-to-comarca
mapping table for spatial alignment across data sources.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class MunicipalityMappingProcessor:
    """
    Loads and processes municipality to comarca mapping data.

    The mapping is essential for:
    - Converting comarca-level data (deaths) to municipality-level
    - Validating spatial coverage across data sources
    - Creating population-weighted allocations
    """

    MUNICIPALITY_MAPPING_FILE = "mpiscatalunya.csv"

    # File has 4-line header before the actual column header
    HEADER_ROWS = 4

    def __init__(self, data_dir: str | Path):
        """
        Initialize the mapping processor.

        Args:
            data_dir: Directory containing mpiscatalunya.csv
        """
        self.data_dir = Path(data_dir)

    def load_mapping(self) -> pd.DataFrame:
        """
        Load municipality to comarca mapping from CSV.

        Returns:
            DataFrame with columns:
            - municipality_code: str (5-digit code with leading zeros)
            - municipality_name: str
            - comarca_code: str (2-digit code with leading zeros)
            - comarca_name: str
        """
        mapping_file = self.data_dir / self.MUNICIPALITY_MAPPING_FILE

        if not mapping_file.exists():
            raise FileNotFoundError(
                f"Municipality mapping file not found: {mapping_file}"
            )

        df = pd.read_csv(  # type: ignore[call-arg]
            mapping_file,
            skiprows=self.HEADER_ROWS,
            usecols=[0, 1, 2, 3],  # Codi, Nom, Codi comarca, Nom comarca
            names=[
                "municipality_code",
                "municipality_name",
                "comarca_code",
                "comarca_name",
            ],
            dtype={"municipality_code": str, "comarca_code": str},
        )

        # Clean up any trailing commas in data
        df["municipality_code"] = df["municipality_code"].str.strip()
        df["comarca_code"] = df["comarca_code"].str.strip()

        # Validate no missing codes
        if df["municipality_code"].isna().any():
            raise ValueError("Missing municipality codes in mapping data")
        if df["comarca_code"].isna().any():
            raise ValueError("Missing comarca codes in mapping data")

        print(f"  Loaded {len(df)} municipality -> comarca mappings")
        print(f"  Unique comarcas: {df['comarca_code'].nunique()}")

        return df

    def get_comarca_for_municipalities(
        self, municipality_codes: list[str] | np.ndarray
    ) -> dict[str, str]:
        """
        Get comarca code for each municipality code.

        Args:
            municipality_codes: List of municipality codes

        Returns:
            Dictionary mapping municipality_code -> comarca_code
        """
        mapping_df = self.load_mapping()

        # Create lookup dictionary
        lookup = dict(zip(mapping_df["municipality_code"], mapping_df["comarca_code"]))

        # Check for missing municipalities
        missing = set(municipality_codes) - set(lookup.keys())
        if missing:
            print(f"  Warning: {len(missing)} municipalities not in mapping")

        return lookup
