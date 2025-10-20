"""
Wastewater Preprocessing Pipeline

Advanced preprocessing functions for wastewater biomarker data handling
SARS-CoV-2 variants, duplicate removal, resampling, and flow calculations.

This module implements the preprocessing pipeline for wastewater surveillance data:
1. Filter data by date (starting from August 15th, 2020)
2. Remove duplicates
3. Select best optional variant (N2 or IP4)
4. Resample to consistent daily frequency with interpolation
5. Calculate total COVID flow
"""

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_data_by_date(
    ww_detections: pd.DataFrame,
    start_date: str = "2020-08-15",
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Filter data by date, removing older data starting from August 15th, 2020.

    Args:
        ww_detections: DataFrame with wastewater detection data
        start_date: Start date in YYYY-MM-DD format
        date_column: Name of the date column

    Returns:
        Filtered DataFrame
    """
    start_datetime = pd.to_datetime(start_date)
    initial_count = len(ww_detections)

    # Ensure date column is datetime
    if date_column in ww_detections.columns:
        ww_detections = ww_detections.copy()
        ww_detections[date_column] = pd.to_datetime(ww_detections[date_column])

        # Filter data
        filtered_data = ww_detections[ww_detections[date_column] >= start_datetime]

        logger.info(
            f"Date filtering: {initial_count} → {len(filtered_data)} records "
            f"(removed {initial_count - len(filtered_data)} records before {start_date})"
        )

        return filtered_data
    else:
        logger.warning(f"Date column '{date_column}' not found in data")
        return ww_detections


def aggregate_duplicates(series: pd.Series) -> Union[float, str]:
    """
    Aggregate duplicate measurements for the same date and EDAR.
    For numeric values, takes the mean. For non-numeric, uses single non-missing value.

    Args:
        series: Pandas Series with potentially duplicate values

    Returns:
        Aggregated value
    """
    # Drop NaN values
    non_null = series.dropna()

    if len(non_null) == 0:
        return np.nan
    elif len(non_null) == 1:
        return non_null.iloc[0]
    else:
        # Check if numeric
        if pd.api.types.is_numeric_dtype(non_null):
            return non_null.mean()
        else:
            # For non-numeric, return the first value
            unique_vals = non_null.unique()
            if len(unique_vals) == 1:
                return unique_vals[0]
            else:
                # Multiple different non-numeric values, return first
                logger.warning(
                    f"Multiple different non-numeric values found: {unique_vals}"
                )
                return non_null.iloc[0]


def remove_ww_duplicates(ww_detections: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates from the ww_detections dataframe.

    Identifies and aggregates duplicate measurements for the same date and EDAR.
    For numeric values, it takes the mean, otherwise it uses the single non-missing
    value if one exists.

    Args:
        ww_detections: DataFrame with wastewater detection data

    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(ww_detections)

    # Identify duplicated rows based on 'edar_id' and 'date'
    # First, need to map the column names to standard format
    edar_col = "depuradora"  # Based on the CSV structure we saw
    date_col = "date"

    if edar_col not in ww_detections.columns or date_col not in ww_detections.columns:
        logger.error(f"Required columns not found. Expected: {edar_col}, {date_col}")
        return ww_detections

    duplicated_mask = ww_detections.duplicated(subset=[edar_col, date_col], keep=False)
    duplicates = ww_detections[duplicated_mask]
    non_duplicates = ww_detections[~duplicated_mask]

    if len(duplicates) > 0:
        logger.info(
            f"Found {len(duplicates)} duplicate records across "
            f"{duplicates.groupby([edar_col, date_col]).size().sum()} groups"
        )

        # Aggregate duplicates
        deduped = (
            duplicates.groupby([edar_col, date_col])
            .agg(aggregate_duplicates)
            .reset_index()
        )

        # Combine deduplicated and non-duplicate data
        ww_detections = pd.concat([deduped, non_duplicates], ignore_index=True)

        logger.info(
            f"Duplicate removal: {initial_count} → {len(ww_detections)} records"
        )
    else:
        logger.info("No duplicates found in the data")

    return ww_detections


def consecutive_missing_values_table(
    ww_detections: pd.DataFrame, group_col: str, variant_cols: list[str]
) -> pd.DataFrame:
    """
    Analyze consecutive missing values for each group and variant.

    Args:
        ww_detections: DataFrame with wastewater detection data
        group_col: Column to group by (e.g., 'edar_id')
        variant_cols: List of variant columns to analyze

    Returns:
        DataFrame with missing value analysis and best variant selection
    """
    results = []

    for group_id in ww_detections[group_col].unique():
        group_data = ww_detections[ww_detections[group_col] == group_id].copy()
        group_data = group_data.sort_values("date")

        variant_scores = {}

        for variant in variant_cols:
            if variant in group_data.columns:
                # Convert to numeric, handling string values like "148"
                variant_data = pd.to_numeric(group_data[variant], errors="coerce")

                # Count consecutive missing values
                is_missing = variant_data.isna()

                if len(is_missing) == 0:
                    consecutive_missing = 0
                    total_missing = 0
                else:
                    # Calculate consecutive missing sequences
                    consecutive_groups = (is_missing != is_missing.shift()).cumsum()
                    consecutive_lengths = is_missing.groupby(consecutive_groups).sum()
                    consecutive_missing = consecutive_lengths[
                        consecutive_lengths > 0
                    ].sum()
                    total_missing = is_missing.sum()

                variant_scores[variant] = {
                    "consecutive_missing": consecutive_missing,
                    "total_missing": total_missing,
                    "missing_rate": total_missing / len(variant_data)
                    if len(variant_data) > 0
                    else 1.0,
                }
            else:
                # Variant not present in data
                variant_scores[variant] = {
                    "consecutive_missing": float("inf"),
                    "total_missing": float("inf"),
                    "missing_rate": 1.0,
                }

        # Choose best variant (least consecutive missing values, then least total missing)
        best_variant = min(
            variant_cols,
            key=lambda v: (
                variant_scores[v]["consecutive_missing"],
                variant_scores[v]["total_missing"],
            ),
        )

        result = {
            group_col: group_id,
            "best_variant_name": best_variant,
        }

        # Add variant scores to result
        for variant in variant_cols:
            result[f"{variant}_consecutive_missing"] = variant_scores[variant][
                "consecutive_missing"
            ]
            result[f"{variant}_total_missing"] = variant_scores[variant][
                "total_missing"
            ]
            result[f"{variant}_missing_rate"] = variant_scores[variant]["missing_rate"]

        results.append(result)

    return pd.DataFrame(results)


def choose_best_optional_variant(ww_detections: pd.DataFrame) -> pd.DataFrame:
    """
    Choose the best optional variant (N2 or IP4) with the least missing values.

    N2, IP4 have many missing consecutive values. We choose the variant
    with the least missing values for each edar.

    Args:
        ww_detections: DataFrame with wastewater detection data

    Returns:
        DataFrame with best_variant and best_variant_name columns added
    """
    optional_variants = ["N2(CG/L)", "IP4(CG/L)"]
    edar_col = "depuradora"  # Based on CSV structure

    # Analyze missing values for variant selection
    missing_table = consecutive_missing_values_table(
        ww_detections, edar_col, optional_variants
    )

    # Merge with original data
    ww_detections = ww_detections.merge(
        missing_table[[edar_col, "best_variant_name"]], on=edar_col, how="left"
    )

    # Create best_variant column by selecting the appropriate variant for each row
    def select_best_variant(row):
        variant_name = row["best_variant_name"]
        if pd.notna(variant_name) and variant_name in row.index:
            return row[variant_name]
        else:
            # Fallback to first available variant
            for variant in optional_variants:
                if variant in row.index and pd.notna(row[variant]):
                    return row[variant]
            return np.nan

    ww_detections["best_variant"] = ww_detections.apply(select_best_variant, axis=1)

    logger.info(
        f"Selected best variants: {missing_table['best_variant_name'].value_counts().to_dict()}"
    )

    return ww_detections


def ww_groups_resample_impute_missing(ww_detections: pd.DataFrame) -> pd.DataFrame:
    """
    Resample wastewater data to consistent daily frequency and impute missing values.

    Wastewater measures are not reported at a consistent frequency.
    This ensures consistency within each group (edar_id) by resampling to daily
    frequency and using linear interpolation to fill gaps.

    Args:
        ww_detections: DataFrame with wastewater detection data

    Returns:
        DataFrame resampled to daily frequency with interpolated values
    """
    edar_col = "depuradora"
    date_col = "date"

    # Columns to resample - biomarker measurements and flow data
    value_columns = ["N1(CG/L)", "best_variant", "Cabal últimes 24h(m3)"]

    # Filter to only columns that exist
    available_columns = [col for col in value_columns if col in ww_detections.columns]

    if not available_columns:
        logger.warning("No value columns found for resampling")
        return ww_detections

    logger.info(f"Resampling columns: {available_columns}")

    # Perform resampling and interpolation
    resampled_data = (
        ww_detections.set_index(date_col)
        .groupby(edar_col)[available_columns]
        .resample("D")  # Daily frequency
        .mean()  # Aggregate multiple measurements per day
        .interpolate(
            method="linear"
        )  # Linear interpolation for time series (MultiIndex compatible)
        .reset_index()
    )

    # Log resampling statistics
    original_points = len(ww_detections)
    resampled_points = len(resampled_data)

    logger.info(f"Resampling: {original_points} → {resampled_points} data points")

    # Calculate interpolation statistics
    for col in available_columns:
        if col in resampled_data.columns:
            original_missing = ww_detections[col].isna().sum()
            resampled_missing = resampled_data[col].isna().sum()
            logger.info(
                f"  {col}: {original_missing} → {resampled_missing} missing values"
            )

    return resampled_data


def sum_variant_measures(ww_detections: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total COVID-19 flow by summing variant measures and multiplying by water flow.

    Returns the total covid volume through each edar at each measurement date.
    We always take only two of the three measures (N1, N2, IP4) into account,
    choosing the variant with the least consecutive missing values for each edar.

    Args:
        ww_detections: DataFrame with wastewater detection data (must have best_variant column)

    Returns:
        DataFrame with total_covid_flow column added
    """
    if "best_variant" not in ww_detections.columns:
        raise ValueError(
            "determine best optional variant first and impute missing first!"
        )

    n1_col = "N1(CG/L)"
    flow_col = "Cabal últimes 24h(m3)"

    # Check required columns
    missing_cols = []
    for col in [n1_col, flow_col]:
        if col not in ww_detections.columns:
            missing_cols.append(col)

    if missing_cols:
        logger.error(f"Missing required columns for flow calculation: {missing_cols}")
        return ww_detections

    # Calculate variant sum: N1 + best_variant
    ww_detections = ww_detections.copy()

    # Convert to numeric, handling any non-numeric values
    ww_detections[n1_col] = pd.to_numeric(ww_detections[n1_col], errors="coerce")
    ww_detections["best_variant"] = pd.to_numeric(
        ww_detections["best_variant"], errors="coerce"
    )
    ww_detections[flow_col] = pd.to_numeric(ww_detections[flow_col], errors="coerce")

    # Sum variants (N1 + best_variant)
    variants_sum = ww_detections[n1_col].fillna(0) + ww_detections[
        "best_variant"
    ].fillna(0)

    # Total flow reported in m3/day, convert to litres (*1000, but we multiply by 0.001 in original)
    # This seems like the original formula converts m3 to some other unit
    h2o_daily_flow_litres = ww_detections[flow_col] * 0.001

    # Calculate total COVID flow
    ww_detections["total_covid_flow"] = h2o_daily_flow_litres * variants_sum

    # Log statistics
    valid_flows = ww_detections["total_covid_flow"].notna().sum()
    total_records = len(ww_detections)
    avg_flow = ww_detections["total_covid_flow"].mean()

    logger.info(
        f"Total COVID flow calculation: {valid_flows}/{total_records} valid calculations"
    )
    logger.info(f"Average total COVID flow: {avg_flow:.2f}")

    return ww_detections


def preprocess_wastewater_pipeline(
    ww_detections: pd.DataFrame,
    start_date: str = "2020-08-15",
    enable_date_filter: bool = True,
    enable_duplicate_removal: bool = True,
    enable_variant_selection: bool = True,
    enable_resampling: bool = True,
    enable_flow_calculation: bool = True,
) -> pd.DataFrame:
    """
    Complete wastewater preprocessing pipeline.

    Applies all preprocessing steps in sequence:
    1. Filter by date
    2. Remove duplicates
    3. Select best variant
    4. Resample and interpolate
    5. Calculate total flow

    Args:
        ww_detections: Raw wastewater detection DataFrame
        start_date: Start date for filtering (YYYY-MM-DD)
        enable_date_filter: Enable date filtering step
        enable_duplicate_removal: Enable duplicate removal step
        enable_variant_selection: Enable variant selection step
        enable_resampling: Enable resampling and interpolation step
        enable_flow_calculation: Enable flow calculation step

    Returns:
        Processed DataFrame
    """
    logger.info("Starting wastewater preprocessing pipeline")
    original_count = len(ww_detections)

    # Step 1: Filter by date
    if enable_date_filter:
        ww_detections = filter_data_by_date(ww_detections, start_date)

    # Step 2: Remove duplicates
    if enable_duplicate_removal:
        ww_detections = remove_ww_duplicates(ww_detections)

    # Step 3: Choose best variant
    if enable_variant_selection:
        ww_detections = choose_best_optional_variant(ww_detections)

    # Step 4: Resample and impute missing values
    if enable_resampling:
        ww_detections = ww_groups_resample_impute_missing(ww_detections)

    # Step 5: Calculate total COVID flow
    if enable_flow_calculation and enable_variant_selection:
        ww_detections = sum_variant_measures(ww_detections)

    final_count = len(ww_detections)
    logger.info(
        f"Preprocessing pipeline completed: {original_count} → {final_count} records"
    )

    return ww_detections
