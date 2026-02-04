"""
Utility functions for data preprocessing.

This module provides common helper functions used across different
processors for loading and handling CSV files with proper data types.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv_with_string_ids(
    filepath: Path,
    string_cols: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load CSV with all ID columns forced to string type to preserve leading zeros.

    This is critical for municipality codes and other geographic identifiers
    that use leading zeros (e.g., "08001" becomes 8001 if parsed as integer).

    Args:
        filepath: Path to CSV file
        string_cols: List of column names that MUST be strings. If None, uses
            default patterns to detect ID columns automatically. When using the
            'names' parameter in kwargs, use string_cols to specify the target
            column names (after renaming).
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        DataFrame with string columns properly typed as strings

    Example:
        >>> df = load_csv_with_string_ids(Path("data.csv"))
        >>> assert df["municipality_code"].dtype == object  # strings are object dtype
        >>> assert df["municipality_code"].str.startswith("0").any()
    """
    # Handle 'names' parameter: if provided, string_cols should refer to the renamed columns
    names_arg = kwargs.get("names")
    if names_arg is not None and string_cols is not None:
        # When names is provided, directly use string_cols for dtype specification
        dtype_dict: dict[str, type] = {col: str for col in string_cols}
    else:
        # Read file header to get actual columns
        header = pd.read_csv(filepath, nrows=0, **{k: v for k, v in kwargs.items() if k not in ['names']})
        actual_cols = set(header.columns)

        # Build dtype dict for columns that exist in the file
        dtype_dict: dict[str, type] = {}

        if string_cols is None:
            # Auto-detect ID columns by pattern matching
            for col in actual_cols:
                col_lower = col.lower()
                # Common patterns for ID columns that should be strings
                if any(id_col in col_lower for id_col in [
                    "codi", "code", "id", "abs", "regio", "muni", "comarca"
                ]):
                    dtype_dict[col] = str
        else:
            # Use explicitly provided string columns
            for col in string_cols:
                if col in actual_cols:
                    dtype_dict[col] = str

    # Load with explicit dtypes
    # type: ignore[arg-type]  # pandas dtype has complex type signature
    df = pd.read_csv(filepath, dtype=dtype_dict or None, **kwargs)

    return df
