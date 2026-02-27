"""Visualization functions for ablation study results.

Functions to create plots from aggregated ablation metrics CSV files.

Usage:
    from dataviz.ablation_plots import plot_ablation_comparison
    plot_ablation_comparison("outputs/ablation_analysis/ablation_metrics_aggregated.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set default plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10


TARGET_ORDER = ["wastewater", "cases", "hospitalizations", "deaths"]
TARGET_LABELS = {
    "wastewater": "Wastewater",
    "cases": "Cases",
    "hospitalizations": "Hosp.",
    "deaths": "Deaths",
}
METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "sMAPE",
    "r2": "R²",
}


def _prepare_model_order(
    df: pd.DataFrame, baseline_name: str = "baseline"
) -> list[str]:
    """Prepare model order with baseline first, then alphabetical."""
    models = sorted(df["model"].unique())
    if baseline_name in models:
        models.remove(baseline_name)
        models.insert(0, baseline_name)
    return models


def _prepare_target_order(df: pd.DataFrame) -> list[str]:
    """Prepare target order following epidemiological significance."""
    targets = []
    for t in TARGET_ORDER:
        if t in df["target"].values:
            targets.append(t)
    # Add any other targets not in our predefined order
    for t in sorted(df["target"].unique()):
        if t not in targets:
            targets.append(t)
    return targets


def plot_ablation_comparison(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    metrics: list[str] | None = None,
    baseline_name: str = "baseline",
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> dict[str, Path]:
    """Create grouped horizontal bar charts comparing ablations across metrics.

    Args:
        csv_path: Path to ablation_metrics_aggregated.csv
        output_dir: Directory to save plots (default: same as csv_path parent)
        metrics: List of metrics to plot (default: ["mae", "rmse", "smape", "r2"])
        baseline_name: Name of baseline ablation
        figsize: Figure size (width, height)
        show: Whether to display plots interactively

    Returns:
        Dictionary mapping metric names to saved figure paths
    """
    csv_path = Path(csv_path)

    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    if metrics is None:
        metrics = ["mae", "rmse", "smape", "r2"]

    # Prepare ordering
    model_order = _prepare_model_order(df, baseline_name)
    target_order = _prepare_target_order(df)

    saved_paths = {}

    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            logger.warning(f"Metric {metric} not found in data")
            continue

        # Create pivot for plotting
        pivot_mean = df.pivot_table(
            index="model",
            columns="target",
            values=mean_col,
        ).reindex(model_order)[target_order]

        pivot_std = df.pivot_table(
            index="model",
            columns="target",
            values=std_col,
        ).reindex(model_order)[target_order]

        # Create plot
        n_ablations = len(model_order)
        n_targets = len(target_order)

        if figsize is None:
            figsize = (10, max(6, n_ablations * 0.4))

        fig, ax = plt.subplots(figsize=figsize)

        # Plot grouped horizontal bars
        y_positions = np.arange(n_ablations)
        bar_height = 0.15

        colors = sns.color_palette("husl", n_targets)

        for i, target in enumerate(target_order):
            means = pivot_mean[target].values
            stds = pivot_std[target].values

            offset = (i - (n_targets - 1) / 2) * (bar_height + 0.01)

            ax.barh(
                y_positions + offset,
                means,
                height=bar_height,
                xerr=stds,
                label=TARGET_LABELS.get(target, target),
                color=colors[i],
                alpha=0.8,
                capsize=3,
            )

        # Styling
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_order)
        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"Ablation Study: {METRIC_LABELS.get(metric, metric)} by Target")
        ax.legend(title="Target", loc="lower right")
        ax.grid(axis="x", alpha=0.3)

        # Add vertical reference lines at baseline values for each target
        if baseline_name in model_order:
            baseline_row = pivot_mean.loc[baseline_name]
            for i, target in enumerate(target_order):
                baseline_val = baseline_row[target]
                ax.axvline(
                    x=baseline_val,
                    ymin=0,
                    ymax=1,
                    color=colors[i],
                    linestyle="--",
                    alpha=0.4,
                    linewidth=1.5,
                )

        plt.tight_layout()

        # Save
        metric_slug = metric.replace("_median", "").replace("_", "-")
        output_path = output_dir / f"ablation_comparison_{metric_slug}.png"
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
        saved_paths[metric] = output_path

        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved_paths


def plot_ablation_deltas_heatmap(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    metric: str = "mae",
    baseline_name: str = "baseline",
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> Path:
    """Create heatmap showing percentage change from baseline.

    Args:
        csv_path: Path to ablation_metrics_deltas.csv
        output_dir: Directory to save plot
        metric: Metric to visualize (default: "mae")
        baseline_name: Name of baseline ablation (not shown, used for ordering)
        figsize: Figure size (width, height)
        show: Whether to display plot interactively

    Returns:
        Path to saved figure
    """
    csv_path = Path(csv_path)

    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    delta_col = f"{metric}_delta_pct"
    if delta_col not in df.columns:
        logger.warning(f"Delta column {delta_col} not found in data")
        # Try alternative naming
        alt_col = f"{metric.replace('_median', '')}_delta_pct"
        if alt_col in df.columns:
            delta_col = alt_col
        else:
            raise ValueError(f"Could not find delta column for {metric}")

    # Prepare ordering
    target_order = _prepare_target_order(df)
    model_order = sorted(df["model"].unique())

    # Create pivot
    pivot = df.pivot_table(
        index="model",
        columns="target",
        values=delta_col,
    ).reindex(model_order)[target_order]

    # Create plot
    if figsize is None:
        figsize = (max(8, len(target_order) * 1.5), max(6, len(model_order) * 0.5))

    fig, ax = plt.subplots(figsize=figsize)

    # Use diverging colormap centered at 0
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    vmin = -vmax

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",  # Red = worse, Green = better
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "% change from baseline"},
        ax=ax,
    )

    ax.set_title(f"Ablation Impact: % Change in {METRIC_LABELS.get(metric, metric)}")
    ax.set_xlabel("Target")
    ax.set_ylabel("Ablation")

    # Improve target labels
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in target_order])

    plt.tight_layout()

    # Save
    metric_slug = metric.replace("_median", "").replace("_", "-")
    output_path = output_dir / f"ablation_delta_heatmap_{metric_slug}.png"
    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"Saved heatmap to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_ablation_summary_grid(
    aggregated_csv: str | Path,
    deltas_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    baseline_name: str = "baseline",
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> Path:
    """Create comprehensive summary grid with all metrics.

    Creates a 2x2 grid showing MAE, RMSE, sMAPE, and R² comparisons.

    Args:
        aggregated_csv: Path to ablation_metrics_aggregated.csv
        deltas_csv: Path to ablation_metrics_deltas.csv (optional)
        output_dir: Directory to save plot
        baseline_name: Name of baseline ablation
        figsize: Figure size (width, height)
        show: Whether to display plot interactively

    Returns:
        Path to saved figure
    """
    aggregated_csv = Path(aggregated_csv)

    if output_dir is None:
        output_dir = aggregated_csv.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(aggregated_csv)

    # Prepare ordering
    model_order = _prepare_model_order(df, baseline_name)
    target_order = _prepare_target_order(df)

    # Create 2x2 grid
    metrics = ["mae", "rmse", "smape", "r2"]

    if figsize is None:
        figsize = (16, max(10, len(model_order) * 0.3))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            ax.text(0.5, 0.5, f"{metric} not available", ha="center", va="center")
            ax.set_title(METRIC_LABELS.get(metric, metric))
            continue

        # Create pivot
        pivot_mean = df.pivot_table(
            index="model",
            columns="target",
            values=mean_col,
        ).reindex(model_order)[target_order]

        pivot_std = df.pivot_table(
            index="model",
            columns="target",
            values=std_col,
        ).reindex(model_order)[target_order]

        # Plot
        n_ablations = len(model_order)
        n_targets = len(target_order)
        y_positions = np.arange(n_ablations)
        bar_height = 0.18

        colors = sns.color_palette("husl", n_targets)

        for i, target in enumerate(target_order):
            means = pivot_mean[target].values
            stds = pivot_std[target].values

            offset = (i - (n_targets - 1) / 2) * (bar_height + 0.01)

            ax.barh(
                y_positions + offset,
                means,
                height=bar_height,
                xerr=stds,
                label=TARGET_LABELS.get(target, target) if idx == 0 else "",
                color=colors[i],
                alpha=0.8,
                capsize=2,
            )

        # Styling
        ax.set_yticks(y_positions)
        if idx % 2 == 0:  # Left column
            ax.set_yticklabels(model_order)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric), fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add vertical reference lines at baseline values for each target
        if baseline_name in model_order:
            baseline_row = pivot_mean.loc[baseline_name]
            for i, target in enumerate(target_order):
                baseline_val = baseline_row[target]
                ax.axvline(
                    x=baseline_val,
                    ymin=0,
                    ymax=1,
                    color=colors[i],
                    linestyle="--",
                    alpha=0.4,
                    linewidth=1.5,
                )

    # Add legend to first subplot
    axes[0].legend(title="Target", loc="lower right", fontsize=8)

    # Overall title
    fig.suptitle(
        "Ablation Study: Comprehensive Metric Comparison",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    output_path = output_dir / "ablation_summary_grid.png"
    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"Saved summary grid to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


# Cross-head impact heatmap labels
CROSS_HEAD_LABELS = {
    "ww": "Wastewater",
    "cases": "Cases",
    "hosp": "Hosp.",
    "deaths": "Deaths",
}


def plot_cross_head_impact_heatmap(
    mean_csv: str | Path,
    std_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> Path:
    """Create heatmap showing cross-head impact from ablation study.

    Shows how ablating each observation head (rows) affects losses on
    all other heads (columns). Values are percentage change from baseline.

    Args:
        mean_csv: Path to cross_head_mean_matrix.csv
        std_csv: Path to cross_head_std_matrix.csv (optional, for annotations)
        output_dir: Directory to save plot (default: same as mean_csv parent)
        figsize: Figure size (width, height)
        show: Whether to display plot interactively

    Returns:
        Path to saved figure
    """
    mean_csv = Path(mean_csv)

    if output_dir is None:
        output_dir = mean_csv.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mean matrix
    mean_df = pd.read_csv(mean_csv, index_col=0)

    # Load std matrix if available
    std_df = None
    if std_csv is not None:
        std_path = Path(std_csv)
        if std_path.exists():
            std_df = pd.read_csv(std_path, index_col=0)

    # Create plot
    if figsize is None:
        figsize = (8, 6)

    fig, ax = plt.subplots(figsize=figsize)

    # Determine color scale (symmetric around 0)
    vmax = max(abs(mean_df.min().min()), abs(mean_df.max().max()))
    if np.isnan(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax

    # Create annotations with std if available
    if std_df is not None:
        annotations = mean_df.copy().astype(str)
        for i in annotations.index:
            for c in annotations.columns:
                mean_val = mean_df.loc[i, c]
                std_val = std_df.loc[i, c]
                if pd.notna(mean_val) and pd.notna(std_val):
                    annotations.loc[i, c] = f"{mean_val:+.1f}\n±{std_val:.1f}"
                elif pd.notna(mean_val):
                    annotations.loc[i, c] = f"{mean_val:+.1f}"
                else:
                    annotations.loc[i, c] = ""
        annot_fmt = ""
    else:
        annotations = True
        annot_fmt = ".1f"

    # Plot heatmap
    sns.heatmap(
        mean_df,
        annot=annotations,
        fmt=annot_fmt,
        cmap="RdBu_r",  # Red = worse (positive delta), Blue = better (negative delta)
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "% change in loss\n(ablated - baseline) / baseline × 100"},
        ax=ax,
        square=True,
        linewidths=0.5,
    )

    # Labels
    ax.set_xlabel("Measured Head (Loss)", fontweight="bold")
    ax.set_ylabel("Ablated Head (Removed)", fontweight="bold")
    ax.set_title(
        "Cross-Head Impact Analysis\nEffect of Ablating Each Head on Other Head Losses",
        fontweight="bold",
        pad=20,
    )

    # Improve tick labels
    x_labels = [CROSS_HEAD_LABELS.get(col, col) for col in mean_df.columns]
    y_labels = [CROSS_HEAD_LABELS.get(idx, idx) for idx in mean_df.index]
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels, rotation=0)

    # Add diagonal mask or annotation for N/A (same head)
    for i, idx in enumerate(mean_df.index):
        for j, col in enumerate(mean_df.columns):
            if idx == col:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1, fill=True, facecolor="lightgray", edgecolor="gray"
                    )
                )
                ax.text(
                    j + 0.5, i + 0.5, "N/A", ha="center", va="center", fontweight="bold"
                )

    plt.tight_layout()

    # Save
    output_path = output_dir / "cross_head_impact_heatmap.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    logger.info(f"Saved cross-head impact heatmap to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main():
    """CLI entry point for generating ablation plots."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ablation study visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m dataviz.ablation_plots outputs/ablation_analysis/ablation_metrics_aggregated.csv
    python -m dataviz.ablation_plots --aggregated metrics.csv --deltas deltas.csv --output-dir figures/
        """,
    )
    parser.add_argument(
        "aggregated_csv",
        type=Path,
        help="Path to ablation_metrics_aggregated.csv",
    )
    parser.add_argument(
        "--deltas-csv",
        type=Path,
        default=None,
        help="Path to ablation_metrics_deltas.csv (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as input)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline",
        help="Name of baseline ablation (default: baseline)",
    )
    parser.add_argument(
        "--cross-head-mean-csv",
        type=Path,
        default=None,
        help="Path to cross_head_mean_matrix.csv for cross-head impact heatmap (optional)",
    )
    parser.add_argument(
        "--cross-head-std-csv",
        type=Path,
        default=None,
        help="Path to cross_head_std_matrix.csv for cross-head impact heatmap (optional)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Generate plots
    logger.info("Generating ablation comparison plots...")

    # Individual metric plots
    plot_ablation_comparison(
        args.aggregated_csv,
        output_dir=args.output_dir,
        baseline_name=args.baseline,
        show=args.show,
    )

    # Summary grid
    plot_ablation_summary_grid(
        args.aggregated_csv,
        deltas_csv=args.deltas_csv,
        output_dir=args.output_dir,
        baseline_name=args.baseline,
        show=args.show,
    )

    # Heatmap if deltas available
    if args.deltas_csv and args.deltas_csv.exists():
        logger.info("Generating delta heatmap...")
        plot_ablation_deltas_heatmap(
            args.deltas_csv,
            output_dir=args.output_dir,
            metric="mae",
            baseline_name=args.baseline,
            show=args.show,
        )

    # Cross-head impact heatmap
    if args.cross_head_mean_csv and args.cross_head_mean_csv.exists():
        logger.info("Generating cross-head impact heatmap...")
        plot_cross_head_impact_heatmap(
            args.cross_head_mean_csv,
            std_csv=args.cross_head_std_csv,
            output_dir=args.output_dir,
            show=args.show,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
