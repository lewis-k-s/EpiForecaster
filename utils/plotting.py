"""Shared plotting utilities for dataviz and plotting modules."""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - Styling Configuration
# =============================================================================


class Colors:
    """Named color constants for consistent theming."""

    CASES = "#4C78A8"
    BIOMARKER = "#2A9D8F"
    MOBILITY = "#2A9D8F"
    GLOBAL_MEAN = "#17324D"
    GLOBAL_MEDIAN = "#4C78A8"


class Palette:
    """Shared categorical colors for manuscript and report plots."""

    TARGET = {
        "wastewater": "#2A9D8F",
        "ww": "#2A9D8F",
        "biomarkers": "#2A9D8F",
        "cases": "#4C78A8",
        "hospitalizations": "#F2A541",
        "hosp": "#F2A541",
        "deaths": "#C44E52",
    }
    TARGET_LABELS = {
        "wastewater": "Wastewater",
        "cases": "Cases",
        "hospitalizations": "Hosp.",
        "deaths": "Deaths",
    }
    BIOMARKER_VARIANT = {
        "N1": "#4C78A8",
        "N2": "#E76F51",
        "IP4": "#7A5195",
    }
    SPLIT = {
        "train": "#4C78A8",
        "val": "#F2A541",
        "test": "#7A5195",
    }
    BASELINE_SERIES = {
        "Our model": "#17324D",
        "exp smoothing": "#7FB3A9",
        "last observed": "#F2A541",
        "sarima": "#C44E52",
        "sarimax": "#8C564B",
        "var": "#7A5195",
        "varmax": "#BCBD22",
    }
    SUPPORT = {
        "no_source": "#6C757D",
        "supported": "#2A9D8F",
        "edar": "#C44E52",
        "focus": "#F2A541",
        "background": "#F7F7F2",
        "boundary": "#C9CEC9",
    }
    NEUTRAL = {
        "dark": "#17324D",
        "text": "#263238",
        "muted": "#5F6366",
        "grid": "#DDE2E1",
        "missing": "#D9D9D9",
    }
    METRIC = {
        "mae": "#4C78A8",
        "rmse": "#F2A541",
    }
    FORECAST = {
        "observed": "#4C78A8",
        "forecast": "#E76F51",
        "overlay": "#C44E52",
        "boundary": "#17324D",
        "history_span": "#D9D9D9",
    }

    @classmethod
    def target(cls, name: str, fallback: str = "#4C78A8") -> str:
        """Return the canonical color for a surveillance target name."""
        return cls.TARGET.get(str(name).lower(), fallback)

    @classmethod
    def target_palette(cls, names: list[str] | tuple[str, ...]) -> list[str]:
        """Return canonical target colors in the requested order."""
        return [cls.target(name) for name in names]


class Colormaps:
    """Shared colormaps for continuous manuscript quantities."""

    WASTEWATER = LinearSegmentedColormap.from_list(
        "epiforecaster_wastewater",
        ["#F1FAFA", "#BFE7E0", "#6CC6B8", "#2A9D8F", "#1F5F77", "#17324D"],
    )
    SUPPORT = WASTEWATER
    AGE = LinearSegmentedColormap.from_list(
        "epiforecaster_age",
        ["#FFF8EF", "#FDD9A0", "#F2A541", "#E76F51", "#9E2F3A"],
    )
    ERROR_DELTA = "RdYlGn_r"
    R2_DELTA = "RdYlGn"


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
    input_window_length: int,
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
    n_windows = (n_time - input_window_length - horizon) // window_stride + 1

    valid_starts = []
    for w in range(n_windows):
        start = w * window_stride
        window_end = start + input_window_length + horizon
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
