"""
Temporal utilities for date parsing and dataset index conversion.

This module provides utilities for converting date strings to dataset indices
and computing temporal split boundaries for train/val/test splits.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from data.preprocess.config import TEMPORAL_COORD


def parse_date_string(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format to datetime.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        datetime object representing the parsed date.

    Raises:
        ValueError: If the date string is not in YYYY-MM-DD format.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date format: {date_str}. Expected YYYY-MM-DD format."
        ) from exc


def date_to_index(dataset: xr.Dataset, target_date: datetime) -> int:
    """Convert a datetime to the index of the matching date in the dataset.

    Args:
        dataset: xarray Dataset with a temporal coordinate.
        target_date: datetime to find in the dataset.

    Returns:
        Integer index of the date in the dataset's temporal coordinate.

    Raises:
        ValueError: If the date is not found in the dataset.
    """
    time_coord = dataset[TEMPORAL_COORD].values

    # Convert to pandas for flexible date comparison
    if isinstance(time_coord[0], np.datetime64):
        # Already numpy datetime64 - compare directly
        target_np64 = np.datetime64(target_date)
        matches = np.where(time_coord == target_np64)[0]
    else:
        # Convert to pandas datetime for comparison
        time_index = pd.DatetimeIndex(time_coord)
        target_pd = pd.Timestamp(target_date)
        matches = np.where(time_index == target_pd)[0]

    if len(matches) == 0:
        available_dates = pd.DatetimeIndex(time_coord)
        raise ValueError(
            f"Date {target_date.strftime('%Y-%m-%d')} not found in dataset. "
            f"Available date range: {available_dates.min().strftime('%Y-%m-%d')} "
            f"to {available_dates.max().strftime('%Y-%m-%d')}"
        )

    return int(matches[0])


def get_temporal_boundaries(
    dataset: xr.Dataset,
    train_end_date: str,
    val_end_date: str,
    test_end_date: str | None = None,
) -> tuple[int, int, int, int]:
    """Compute temporal split boundaries as dataset indices.

    The splits are defined as:
    - Train: [0, train_end)
    - Val: [train_end, val_end)
    - Test: [val_end, test_end) or [val_end, end) if test_end is None

    Args:
        dataset: xarray Dataset with a temporal coordinate.
        train_end_date: Train split end date (YYYY-MM-DD). Exclusive.
        val_end_date: Validation split end date (YYYY-MM-DD). Exclusive.
        test_end_date: Optional test split end date. If None, uses end of dataset.

    Returns:
        Tuple of (train_start, train_end, val_end, test_end) indices.
        train_start is always 0.

    Raises:
        ValueError: If dates are not found or in wrong order.
    """
    train_end_dt = parse_date_string(train_end_date)
    val_end_dt = parse_date_string(val_end_date)
    test_end_dt = parse_date_string(test_end_date) if test_end_date else None

    # Convert dates to indices
    train_end_idx = date_to_index(dataset, train_end_dt)
    val_end_idx = date_to_index(dataset, val_end_dt)

    if test_end_dt is not None:
        test_end_idx = date_to_index(dataset, test_end_dt)
    else:
        test_end_idx = len(dataset[TEMPORAL_COORD])

    # Validate ordering
    if not (0 < train_end_idx < val_end_idx < test_end_idx):
        raise ValueError(
            f"Invalid temporal boundary ordering: "
            f"train_end={train_end_idx}, val_end={val_end_idx}, test_end={test_end_idx}. "
            f"Must satisfy: 0 < train_end < val_end < test_end"
        )

    return 0, train_end_idx, val_end_idx, test_end_idx


def validate_temporal_range(
    time_range: tuple[int, int],
    history_length: int,
    forecast_horizon: int,
    total_time_steps: int,
) -> None:
    """Validate that a temporal range can accommodate the required windows.

    Args:
        time_range: (start_idx, end_idx) tuple, end_idx is exclusive.
        history_length: Required history window size (L).
        forecast_horizon: Required forecast horizon (H).
        total_time_steps: Total number of time steps in dataset.

    Raises:
        ValueError: If the range is too small or out of bounds.
    """
    start_idx, end_idx = time_range
    window_size = history_length + forecast_horizon

    if start_idx < 0 or end_idx > total_time_steps:
        raise ValueError(
            f"Time range [{start_idx}, {end_idx}) out of bounds "
            f"for dataset with {total_time_steps} time steps"
        )

    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid time range: start_idx ({start_idx}) >= end_idx ({end_idx})"
        )

    available_steps = end_idx - start_idx
    if available_steps < window_size:
        raise ValueError(
            f"Time range [{start_idx}, {end_idx}) has {available_steps} steps, "
            f"but requires at least {window_size} steps "
            f"(L={history_length} + H={forecast_horizon})"
        )


def format_date_range(dataset: xr.Dataset, time_range: tuple[int, int]) -> str:
    """Format a time range as human-readable date strings.

    Args:
        dataset: xarray Dataset with temporal coordinate.
        time_range: (start_idx, end_idx) tuple, end_idx is exclusive.

    Returns:
        String like "2020-01-01 to 2020-12-31".
    """
    time_coord = dataset[TEMPORAL_COORD].values
    start_date = pd.Timestamp(time_coord[time_range[0]])
    # For end_idx, we show the last INCLUDED index
    end_date = pd.Timestamp(time_coord[time_range[1] - 1])

    return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
