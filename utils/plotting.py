"""Shared plotting utilities for dataviz and plotting modules."""

import logging
from pathlib import Path
from typing import Any, Literal

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
    RESIDUAL_POS = "blue"
    RESIDUAL_NEG = "red"


class FigureSizes:
    """Common figure size presets."""

    TIME_SERIES = (14, 6)
    DISTRIBUTION = (10, 6)
    MULTI_PANEL = (15, 10)
    SINGLE = (12, 8)


class Style:
    """Default style settings."""

    DPI = 200
    ALPHA_INDIVIDUAL = 0.35
    ALPHA_FILL = 0.6
    GRID_ALPHA = 0.3


# =============================================================================
# FIGURE SETUP UTILITIES
# =============================================================================


def setup_figure(
    figsize: tuple[float, float] = FigureSizes.TIME_SERIES,
    style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
) -> tuple[Figure, Axes]:
    """Create a figure with consistent styling."""
    import seaborn as sns

    sns.set_theme(style=style)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


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


def add_reference_line(
    ax: Axes,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    color: str = "red",
    linestyle: str = "--",
    alpha: float = 0.5,
) -> None:
    """Add horizontal or vertical reference line at zero."""
    if orientation == "horizontal":
        ax.axhline(y=0, color=color, linestyle=linestyle, alpha=alpha)
    else:
        ax.axvline(x=0, color=color, linestyle=linestyle, alpha=alpha)


# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================


def normalize_min_max(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Min-max normalization to [0, 1] range."""
    computed_vmin = float(values.min()) if vmin is None else vmin
    computed_vmax = float(values.max()) if vmax is None else vmax
    return (values - computed_vmin) / (computed_vmax - computed_vmin)


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


# =============================================================================
# ANNOTATION UTILITIES
# =============================================================================


def add_stats_textbox(
    ax: Axes,
    stats: dict[str, float | str],
    position: str = "upper left",
    **kwargs,
) -> None:
    """Add statistics textbox to axes."""
    positions = {
        "upper left": (0.02, 0.98),
        "upper right": (0.98, 0.98),
        "lower left": (0.02, 0.02),
        "lower right": (0.98, 0.02),
    }
    x, y = positions.get(position, (0.02, 0.98))

    text = "\n".join(f"{k}: {v}" for k, v in stats.items())
    bbox = kwargs.pop("bbox", {"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top" if "upper" in position else "bottom",
        ha="left" if "left" in position else "right",
        bbox=bbox,
        **kwargs,
    )


# =============================================================================
# OPTIONAL DEPENDENCY HANDLING
# =============================================================================


def import_optional(name: str) -> tuple[bool, Any | None]:
    """Safely import optional dependencies."""
    try:
        module = __import__(name)
        return True, module
    except ImportError:
        logger.warning(f"Optional dependency {name} not available")
        return False, None
