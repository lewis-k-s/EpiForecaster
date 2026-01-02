"""
Neighborhood vs Global Trend Regression Analysis

Computes sliding window regression comparing neighborhood-level case trends against
global trends, both scaled by population. This analysis reveals how closely local
mobility neighborhoods follow global epidemic patterns.

For each window and target node:
1. Compute per-capita cases for the neighborhood (incoming mobility neighbors)
2. Compute population-weighted global per-capita cases
3. Fit linear regression: global_trend ~ neighborhood_trend
4. Analyze slope, R², and residuals across time and space
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).parent.parent))

from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import EpiForecasterConfig

logger = logging.getLogger(__name__)


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (time, region, feature), adding trailing dim if needed."""
    if arr.ndim == 2:
        return arr[..., None]
    return arr


def resolve_mobility_array(mobility_da: xr.DataArray | xr.Dataset) -> np.ndarray:
    """Return mobility array in (time, origin, destination) order."""
    if isinstance(mobility_da, xr.Dataset):
        if "mobility" not in mobility_da:
            raise ValueError("Mobility dataset missing 'mobility' variable")
        mobility_da = mobility_da["mobility"]

    dims = list(mobility_da.dims)
    time_dim = TEMPORAL_COORD if TEMPORAL_COORD in dims else None
    if time_dim is None:
        raise ValueError(f"Mobility data missing {TEMPORAL_COORD} dimension")

    if "origin" in dims and "destination" in dims:
        ordered = mobility_da.transpose(time_dim, "origin", "destination")
        return ordered.values

    if dims.count(REGION_COORD) == 2:
        ordered = mobility_da.transpose(time_dim, REGION_COORD, REGION_COORD)
        return ordered.values

    raise ValueError(
        "Mobility data must include ('origin','destination') or two "
        f"'{REGION_COORD}' dims"
    )


def compute_valid_window_mask(
    cases_da: xr.DataArray,
    history_length: int,
    horizon: int,
    window_stride: int,
    missing_permit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute window starts and validity mask per node.

    Returns:
        starts: 1D array of window start indices
        valid_mask: 2D boolean array (num_windows, num_nodes)
    """
    if TEMPORAL_COORD not in cases_da.dims or REGION_COORD not in cases_da.dims:
        raise ValueError(
            f"Cases data must include {TEMPORAL_COORD} and {REGION_COORD} dims"
        )

    other_dims = [d for d in cases_da.dims if d not in (TEMPORAL_COORD, REGION_COORD)]
    cases_da = cases_da.transpose(TEMPORAL_COORD, REGION_COORD, *other_dims)
    cases_np = _ensure_3d(cases_da.values)
    if cases_np.ndim != 3:
        raise ValueError(f"Expected cases array with 2 or 3 dims, got {cases_np.shape}")

    T = cases_np.shape[0]
    seg = history_length + horizon
    if T < seg:
        starts = np.array([], dtype=np.int64)
        valid_mask = np.zeros((0, cases_np.shape[1]), dtype=bool)
        return starts, valid_mask

    valid = np.isfinite(cases_np).all(axis=2)
    valid_int = valid.astype(np.int32)

    cumsum = np.concatenate(
        [
            np.zeros((1, valid_int.shape[1]), dtype=np.int32),
            np.cumsum(valid_int, axis=0),
        ],
        axis=0,
    )

    history_counts = cumsum[history_length:] - cumsum[:-history_length]
    target_counts = cumsum[history_length + horizon :] - cumsum[history_length:-horizon]

    starts = np.arange(0, T - seg + 1, window_stride, dtype=np.int64)
    history_counts = history_counts[starts]
    target_counts = target_counts[starts]

    history_threshold = max(0, history_length - missing_permit)
    history_ok = history_counts >= history_threshold
    target_ok = target_counts >= horizon
    valid_mask = history_ok & target_ok

    return starts, valid_mask


def split_nodes(
    num_nodes: int, val_split: float, test_split: float, seed: int = 42
) -> tuple[list[int], list[int], list[int]]:
    """Split nodes into train/val/test lists using a fixed RNG seed."""
    train_split = 1 - val_split - test_split
    all_nodes = np.arange(num_nodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_nodes)

    n_train = int(len(all_nodes) * train_split)
    n_val = int(len(all_nodes) * val_split)

    train_nodes = all_nodes[:n_train]
    val_nodes = all_nodes[n_train : n_train + n_val]
    test_nodes = all_nodes[n_train + n_val :]

    return list(train_nodes), list(val_nodes), list(test_nodes)


def compute_window_regression(
    cases_window: np.ndarray,
    population: np.ndarray,
    target_node: int,
    neighbors: np.ndarray,
    window_size: int,
) -> dict[str, Any]:
    """Compute regression between neighborhood and global trends for one window.

    Args:
        cases_window: Cases array of shape (time, nodes, 1)
        population: Population array of shape (nodes,)
        target_node: Index of target node
        neighbors: Boolean mask of neighbors for target node
        window_size: Length of window

    Returns:
        Dictionary with regression statistics
    """
    if not np.any(neighbors):
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "n_neighbors": 0,
        }

    # Compute per-capita cases (cases per 100k)
    cases = cases_window[:, :, 0]  # (time, nodes)
    per_capita = (cases / population[None, :]) * 100000.0

    # Neighborhood trend: mean per-capita cases across neighbors
    # Only consider neighbors with valid population
    valid_pop = population > 0
    valid_neighbors = neighbors & valid_pop

    if not np.any(valid_neighbors):
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "n_neighbors": int(neighbors.sum()),
        }

    neighborhood_vals = per_capita[:, valid_neighbors]
    # Compute mean, handling cases where all neighbors have NaN at a timestep
    neighborhood_trend = np.full(per_capita.shape[0], np.nan)
    for t in range(per_capita.shape[0]):
        t_vals = neighborhood_vals[t, :]
        if np.any(np.isfinite(t_vals)):
            neighborhood_trend[t] = np.nanmean(t_vals)

    # Global trend: population-weighted mean per-capita cases
    # Only consider regions with valid population AND valid cases data at each timestep
    global_trend = np.full(per_capita.shape[0], np.nan)
    for t in range(per_capita.shape[0]):
        t_vals = per_capita[t, valid_pop]
        t_weights = population[valid_pop]
        valid_mask = np.isfinite(t_vals) & (t_weights > 0)
        if np.any(valid_mask):
            t_weights_valid = t_weights[valid_mask]
            t_vals_valid = t_vals[valid_mask]
            weights_valid = t_weights_valid / t_weights_valid.sum()
            global_trend[t] = np.sum(t_vals_valid * weights_valid)

    # Remove NaN values
    valid_mask = np.isfinite(neighborhood_trend) & np.isfinite(global_trend)
    if valid_mask.sum() < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "n_neighbors": int(valid_neighbors.sum()),
        }

    X = neighborhood_trend[valid_mask, None]
    y = global_trend[valid_mask]

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Compute statistics
    y_pred = model.predict(X)
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (rss / tss) if tss > 0 else np.nan

    # Standard error of slope
    n = len(y)
    x_mean = X.mean()
    s_x = np.sum((X - x_mean) ** 2)
    sse = rss / (n - 2) if n > 2 else 0
    std_err = np.sqrt(sse / s_x) if s_x > 0 else np.nan

    # P-value for slope (t-test)
    t_stat = (
        model.coef_[0] / std_err if std_err > 0 and not np.isnan(std_err) else np.nan
    )
    p_value = (
        2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))
        if not np.isnan(t_stat) and n > 2
        else np.nan
    )

    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_neighbors": int(neighbors.sum()),
    }


def run_regression_analysis(
    cases_da: xr.DataArray,
    mobility: np.ndarray,
    population: np.ndarray,
    starts: np.ndarray,
    valid_mask: np.ndarray,
    target_nodes: list[int],
    window_len: int,
    mobility_threshold: float,
    include_self: bool,
) -> pd.DataFrame:
    """Run regression analysis for all windows and target nodes."""
    results = []

    for w_idx, start in enumerate(starts):
        end = start + window_len
        if end > cases_da.shape[0]:
            break

        cases_window = cases_da.isel({TEMPORAL_COORD: slice(start, end)}).values
        if cases_window.ndim == 2:
            cases_window = cases_window[:, :, None]

        # Use mobility from middle of window
        t_mob = start + window_len // 2
        if t_mob >= mobility.shape[0]:
            continue

        for target_idx in target_nodes:
            if not valid_mask[w_idx, target_idx]:
                continue

            # Get neighbors based on incoming mobility
            inflow = mobility[t_mob, :, target_idx]
            neighbors = inflow >= mobility_threshold

            if not include_self:
                neighbors[target_idx] = False

            # Compute regression
            reg_stats = compute_window_regression(
                cases_window,
                population,
                target_idx,
                neighbors,
                window_len,
            )

            results.append(
                {
                    "window_start": start,
                    "window_end": end,
                    "window_idx": w_idx,
                    "target_node": target_idx,
                    **reg_stats,
                }
            )

    return pd.DataFrame(results)


def plot_scatter_samples(
    df: pd.DataFrame,
    cases_da: xr.DataArray,
    population: np.ndarray,
    mobility: np.ndarray,
    mobility_threshold: float,
    n_samples: int = 6,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot scatter plots of neighborhood vs global for sample windows."""
    valid_df = df.dropna(subset=["slope", "r2", "window_start", "window_end"])

    if len(valid_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid regression results", ha="center", va="center")
        return fig

    # Sample random (window, node) pairs
    sample_indices = np.random.choice(
        len(valid_df), min(n_samples, len(valid_df)), replace=False
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        if i >= len(axes):
            break

        row = valid_df.iloc[idx]
        start, end = int(row["window_start"]), int(row["window_end"])
        target_node = int(row["target_node"])

        cases_window = cases_da.isel({TEMPORAL_COORD: slice(start, end)}).values
        if cases_window.ndim == 2:
            cases_window = cases_window[:, :, None]

        t_mob = start + (end - start) // 2
        inflow = mobility[t_mob, :, target_node]
        neighbors = inflow >= mobility_threshold
        neighbors[target_node] = True

        cases = cases_window[:, :, 0]
        per_capita = (cases / population[None, :]) * 100000.0

        neighborhood_vals = per_capita[:, neighbors]
        neighborhood_trend = np.nanmean(neighborhood_vals, axis=1)

        weights = population / population.sum()
        global_trend = np.sum(per_capita * weights[None, :], axis=1)

        valid_mask = np.isfinite(neighborhood_trend) & np.isfinite(global_trend)
        x = neighborhood_trend[valid_mask]
        y = global_trend[valid_mask]

        ax = axes[i]
        ax.scatter(x, y, alpha=0.6, s=50)

        # Add regression line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2)

        ax.set_xlabel("Neighborhood per-capita cases")
        ax.set_ylabel("Global per-capita cases")
        ax.set_title(
            f"Window {start}-{end}, Node {target_node}\n"
            f"Slope={row['slope']:.3f}, R²={row['r2']:.3f}, N={row['n_neighbors']}"
        )
        ax.grid(True, alpha=0.3)

    plt.suptitle("Neighborhood vs Global Trends: Sample Windows", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info("Saved scatter samples to %s", output_path)

    return fig


def plot_slope_timeseries(
    df: pd.DataFrame, output_path: Path | None = None
) -> plt.Figure:
    """Plot time series of regression slopes aggregated across nodes."""
    valid_df = df.dropna(subset=["slope"])

    if len(valid_df) == 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No valid regression results", ha="center", va="center")
        return fig

    # Aggregate statistics per window
    agg_df = (
        valid_df.groupby("window_start")
        .agg(
            {
                "slope": ["mean", "median", "std", "min", "max"],
                "r2": "mean",
                "n_neighbors": "mean",
            }
        )
        .round(3)
    )

    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot slope statistics
    ax = axes[0]
    ax.plot(agg_df.index, agg_df["slope_mean"], "o-", label="Mean", linewidth=2)
    ax.plot(agg_df.index, agg_df["slope_median"], "s-", label="Median", linewidth=2)
    ax.fill_between(
        agg_df.index,
        agg_df["slope_mean"] - agg_df["slope_std"],
        agg_df["slope_mean"] + agg_df["slope_std"],
        alpha=0.3,
        label="±1 std",
    )
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Perfect tracking")
    ax.set_xlabel("Window start")
    ax.set_ylabel("Regression slope")
    ax.set_title("Neighborhood vs Global Regression Slopes Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot R²
    ax = axes[1]
    ax.plot(agg_df.index, agg_df["r2_mean"], "o-", color="green", linewidth=2)
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="R² = 0.5")
    ax.axhline(y=0.8, color="blue", linestyle="--", alpha=0.5, label="R² = 0.8")
    ax.set_xlabel("Window start")
    ax.set_ylabel("Mean R²")
    ax.set_title("Regression Fit Quality Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info("Saved slope timeseries to %s", output_path)

    return fig


def plot_slope_distribution(
    df: pd.DataFrame, output_path: Path | None = None
) -> plt.Figure:
    """Plot distribution of regression slopes across all nodes and windows."""
    valid_df = df.dropna(subset=["slope"])

    if len(valid_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid regression results", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram of slopes
    ax = axes[0, 0]
    ax.hist(valid_df["slope"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=1.0, color="r", linestyle="--", linewidth=2, label="Perfect tracking")
    ax.axvline(
        x=valid_df["slope"].median(),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median ({valid_df['slope'].median():.3f})",
    )
    ax.set_xlabel("Regression slope")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Regression Slopes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram of R²
    ax = axes[0, 1]
    ax.hist(valid_df["r2"], bins=50, edgecolor="black", alpha=0.7, color="green")
    ax.axvline(x=0.5, color="orange", linestyle="--", linewidth=2, label="R² = 0.5")
    ax.axvline(
        x=valid_df["r2"].median(),
        color="b",
        linestyle="--",
        linewidth=2,
        label=f"Median ({valid_df['r2'].median():.3f})",
    )
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of R² Values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot by node (sample if too many)
    ax = axes[1, 0]
    if valid_df["target_node"].nunique() > 20:
        sampled_nodes = valid_df["target_node"].unique()[:20]
        plot_df = valid_df[valid_df["target_node"].isin(sampled_nodes)]
    else:
        plot_df = valid_df

    plot_df.boxplot(column="slope", by="target_node", ax=ax)
    ax.set_xlabel("Target node")
    ax.set_ylabel("Slope")
    ax.set_title("Slope Distribution by Node (sampled)")
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
    plt.suptitle("")

    # Scatter: slope vs R²
    ax = axes[1, 1]
    sc = ax.scatter(
        valid_df["slope"],
        valid_df["r2"],
        c=valid_df["n_neighbors"],
        cmap="viridis",
        alpha=0.6,
        s=30,
    )
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5)
    ax.axvline(x=1.0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Regression slope")
    ax.set_ylabel("R²")
    ax.set_title("Slope vs R² (color = # neighbors)")
    plt.colorbar(sc, ax=ax, label="Number of neighbors")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info("Saved slope distribution to %s", output_path)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neighborhood vs global trend regression analysis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training config path",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Target node split to analyze.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=14,
        help="Stride for sliding windows (default: 14 days)",
    )
    parser.add_argument(
        "--mobility-threshold",
        type=float,
        default=None,
        help="Minimum incoming flow for neighbors (default: config value).",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include target node in neighborhood.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/neighborhood_global_regression"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and dataset
    config = EpiForecasterConfig.from_file(str(args.config))
    dataset = EpiDataset.load_canonical_dataset(Path(config.data.dataset_path))

    logger.info("Loaded dataset: %s", dataset)

    # Get data arrays
    cases_da = dataset["cases"]
    mobility = resolve_mobility_array(dataset["mobility"])
    population = dataset["population"].values

    # Compute window parameters
    history_len = int(config.model.history_length)
    horizon = int(config.model.forecast_horizon)
    window_len = history_len + horizon
    window_stride = int(args.window_stride)
    missing_permit = int(config.model.missing_permit)

    # Compute valid windows
    starts, valid_mask = compute_valid_window_mask(
        cases_da, history_len, horizon, window_stride, missing_permit
    )

    logger.info(
        "Found %d valid windows (L=%d, H=%d, stride=%d)",
        len(starts),
        history_len,
        horizon,
        window_stride,
    )

    if len(starts) == 0:
        logger.error("No valid windows found. Check config parameters.")
        return

    # Determine target nodes
    num_nodes = dataset[REGION_COORD].size
    train_nodes, val_nodes, test_nodes = split_nodes(
        num_nodes, config.training.val_split, config.training.test_split
    )

    if args.split == "train":
        target_nodes = train_nodes
    elif args.split == "val":
        target_nodes = val_nodes
    elif args.split == "test":
        target_nodes = test_nodes
    else:
        target_nodes = list(range(num_nodes))

    logger.info(
        "Analyzing %d target nodes from %s split", len(target_nodes), args.split
    )

    # Get mobility threshold
    mobility_threshold = (
        float(args.mobility_threshold)
        if args.mobility_threshold is not None
        else float(config.data.mobility_threshold)
    )

    # Run regression analysis
    logger.info("Running regression analysis...")
    results_df = run_regression_analysis(
        cases_da,
        mobility,
        population,
        starts,
        valid_mask,
        target_nodes,
        window_len,
        mobility_threshold,
        args.include_self,
    )

    logger.info("Computed %d regression results", len(results_df))

    if len(results_df) == 0:
        logger.error("No regression results computed. Check data and parameters.")
        return

    # Save results
    results_path = output_dir / "neighborhood_global_regression_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved results to %s", results_path)

    # Generate plots
    logger.info("Generating visualizations...")

    plot_scatter_samples(
        results_df,
        cases_da,
        population,
        mobility,
        mobility_threshold,
        n_samples=6,
        output_path=output_dir / "neighborhood_global_scatter.png",
    )

    plot_slope_timeseries(
        results_df, output_path=output_dir / "regression_slopes_timeseries.png"
    )

    plot_slope_distribution(
        results_df, output_path=output_dir / "regression_slopes_distribution.png"
    )

    # Print summary statistics
    valid_df = results_df.dropna(subset=["slope"])
    logger.info("=" * 60)
    logger.info("Regression Analysis Summary")
    logger.info("=" * 60)
    logger.info("Total regressions: %d", len(valid_df))
    logger.info(
        "Slope statistics: mean=%.3f, median=%.3f, std=%.3f",
        valid_df["slope"].mean(),
        valid_df["slope"].median(),
        valid_df["slope"].std(),
    )
    logger.info(
        "R² statistics: mean=%.3f, median=%.3f",
        valid_df["r2"].mean(),
        valid_df["r2"].median(),
    )
    logger.info(
        "Neighbors: mean=%.1f, median=%.1f",
        valid_df["n_neighbors"].mean(),
        valid_df["n_neighbors"].median(),
    )

    # Fraction of significant regressions
    significant = (valid_df["p_value"] < 0.05).sum() / len(valid_df)
    logger.info("Fraction significant (p<0.05): %.1f%%", significant * 100)

    # Fraction with good fit
    good_fit = (valid_df["r2"] > 0.5).sum() / len(valid_df)
    logger.info("Fraction with R² > 0.5: %.1f%%", good_fit * 100)

    logger.info("Done! Plots saved to %s", output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
