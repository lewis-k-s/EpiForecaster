"""Shared plotting utilities for dataviz and plotting modules."""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - Styling Configuration
# =============================================================================


class Colors:
    """Named color constants for consistent theming."""

    CASES = "#1f77b4"  # blue
    BIOMARKER = "#ff7f0e"  # orange
    MOBILITY = "#2ca02c"  # green
    GLOBAL_MEAN = "black"
    GLOBAL_MEDIAN = "tab:blue"


class FigureSizes:
    """Common figure size presets."""

    TIME_SERIES = (14, 6)
    MULTI_PANEL = (15, 10)


class Style:
    """Default style settings."""

    DPI = 200
    GRID_ALPHA = 0.3


# =============================================================================
# FIGURE SETUP UTILITIES
# =============================================================================


def save_figure(
    fig: Figure,
    path: str | Path,
    dpi: int = Style.DPI,
    bbox_inches: str = "tight",
    log_msg: str | None = None,
) -> None:
    """Save figure with standardized options and logging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    if log_msg:
        logger.info(f"{log_msg} to {path}")


# =============================================================================
# AXIS FORMATTING UTILITIES
# =============================================================================


def format_date_axis(
    ax: Axes,
    minticks: int = 5,
    maxticks: int = 8,
) -> None:
    """Apply concise date formatting to x-axis."""
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def add_grid(
    ax: Axes,
    axis: Literal["both", "x", "y"] = "both",
    alpha: float = Style.GRID_ALPHA,
) -> None:
    """Add grid to specified axis."""
    ax.grid(True, alpha=alpha, axis=axis)


# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================


def robust_bounds(
    values: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
    positive_only: bool = False,
) -> tuple[float, float] | None:
    """Compute robust percentile bounds.

    Returns None if no finite values remain after filtering.
    """
    finite = values[np.isfinite(values)]
    if positive_only:
        finite = finite[finite > 0]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [lower, upper])
    return float(low), float(high)


def ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 3D, adding trailing dimension if needed."""
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    return arr


# =============================================================================
# WINDOWING UTILITIES
# =============================================================================


def compute_valid_window_mask(
    cases: np.ndarray,
    history_length: int,
    horizon: int,
    window_stride: int,
    missing_permit: float,
) -> tuple[np.ndarray, int]:
    """Compute valid window starts and mask for missing data.

    Returns:
        valid_starts: Array of valid start indices
        n_windows: Total number of windows
    """
    n_time = cases.shape[0]
    n_windows = (n_time - history_length - horizon) // window_stride + 1

    valid_starts = []
    for w in range(n_windows):
        start = w * window_stride
        window_end = start + history_length + horizon
        window_data = cases[start:window_end]

        if np.isnan(window_data).mean() <= missing_permit:
            valid_starts.append(start)

    return np.array(valid_starts), n_windows


def compute_consecutive_missing(
    data: np.ndarray,
    axis: int = 0,
) -> int | np.ndarray:
    """Compute max consecutive missing values along axis.

    Returns int for 1D input, ndarray for 2D+ input.
    """

    def max_consecutive_nans(arr):
        mask = np.isnan(arr)
        max_count = 0
        current_count = 0
        for val in mask:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    if data.ndim == 1:
        return max_consecutive_nans(data)
    return np.apply_along_axis(max_consecutive_nans, axis, data)
