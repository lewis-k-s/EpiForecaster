"""
Visualization utilities for epidemiological forecasting results.

This module provides plotting functions for time series forecasts, spatial results,
and model evaluation metrics.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Set style
plt.style.use("default")
sns.set_palette("husl")


def create_results_table(
    metrics: dict[str, float], region_ids: Optional[list[str]] = None
) -> pd.DataFrame:
    """Create a formatted results table for CLI display."""
    results_data = []

    # Overall metrics
    for metric_name, value in metrics.items():
        results_data.append(
            {
                "Metric": metric_name.upper(),
                "Value": f"{value:.6f}",
                "Description": get_metric_description(metric_name),
            }
        )

    return pd.DataFrame(results_data)


def get_metric_description(metric_name: str) -> str:
    """Get description for evaluation metrics."""
    descriptions = {
        "mse": "Mean Squared Error",
        "mae": "Mean Absolute Error",
        "mape": "Mean Absolute Percentage Error (%)",
        "rmse": "Root Mean Squared Error",
        "r2": "R-squared (coefficient of determination)",
    }
    return descriptions.get(metric_name.lower(), metric_name)


def print_results_table(
    metrics: dict[str, float], region_ids: Optional[list[str]] = None
):
    """Print a formatted results table to console."""
    print("\n" + "=" * 60)
    print("               FORECASTING RESULTS")
    print("=" * 60)

    # Print overall metrics
    print(f"{'Metric':<8} {'Value':<12} {'Description'}")
    print("-" * 60)

    for metric_name, value in metrics.items():
        description = get_metric_description(metric_name)
        print(f"{metric_name.upper():<8} {value:<12.6f} {description}")

    print("=" * 60)

    if region_ids:
        print(f"Total geographic zones: {len(region_ids)}")
        print(f"Sample zones: {', '.join(region_ids[:5])}")
        if len(region_ids) > 5:
            print(f"... and {len(region_ids) - 5} more")

    print()


def plot_forecast_time_series(
    predictions: np.ndarray,
    targets: np.ndarray,
    forecast_horizon: int = 7,
    output_dir: str = "outputs/",
    region_sample: int = 5,
) -> None:
    """
    Create time series plots comparing predictions vs targets.

    Args:
        predictions: Model predictions [n_samples, n_regions, forecast_horizon]
        targets: Ground truth targets [n_samples, n_regions, forecast_horizon]
        forecast_horizon: Number of forecast days
        output_dir: Directory to save plots
        region_sample: Number of regions to plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Select sample regions to plot
    n_regions = predictions.shape[1] if len(predictions.shape) > 1 else 1
    sample_regions = min(region_sample, n_regions)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(sample_regions):
        ax = axes[i] if sample_regions > 1 else axes[0]

        # Get data for this region
        if len(predictions.shape) > 1:
            pred_region = (
                predictions[:, i, :]
                if len(predictions.shape) == 3
                else predictions[i, :]
            )
            target_region = (
                targets[:, i, :] if len(targets.shape) == 3 else targets[i, :]
            )
        else:
            pred_region = predictions
            target_region = targets

        # Create time axis
        days = list(range(1, forecast_horizon + 1))

        if len(pred_region.shape) == 2:
            # Multiple samples - plot mean and confidence intervals
            pred_mean = np.mean(pred_region, axis=0)
            pred_std = np.std(pred_region, axis=0)
            target_mean = np.mean(target_region, axis=0)

            ax.plot(days, pred_mean, "b-", label="Predicted", linewidth=2)
            ax.fill_between(
                days,
                pred_mean - pred_std,
                pred_mean + pred_std,
                alpha=0.3,
                color="blue",
                label="Prediction std",
            )
            ax.plot(days, target_mean, "r-", label="Actual", linewidth=2)
        else:
            # Single sample
            ax.plot(days, pred_region, "b-", label="Predicted", linewidth=2)
            ax.plot(days, target_region, "r-", label="Actual", linewidth=2)

        ax.set_xlabel("Forecast Day")
        ax.set_ylabel("Value")
        ax.set_title(f"Region {i + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(sample_regions, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path / "forecast_time_series.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Time series forecast plot saved to {output_path / 'forecast_time_series.png'}"
    )


def plot_residual_analysis(
    predictions: np.ndarray, targets: np.ndarray, output_dir: str = "outputs/"
) -> None:
    """
    Create residual analysis plots.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Flatten arrays for residual analysis
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    residuals = target_flat - pred_flat

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Predictions
    axes[0, 0].scatter(pred_flat, residuals, alpha=0.5, s=1)
    axes[0, 0].axhline(y=0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("Predicted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Predicted Values")
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=50, density=True, alpha=0.7)
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Distribution of Residuals")
    axes[0, 1].grid(True, alpha=0.3)

    # QQ plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normal Distribution)")
    axes[1, 0].grid(True, alpha=0.3)

    # Predicted vs Actual scatter
    axes[1, 1].scatter(target_flat, pred_flat, alpha=0.5, s=1)
    axes[1, 1].plot(
        [target_flat.min(), target_flat.max()],
        [target_flat.min(), target_flat.max()],
        "r--",
        lw=2,
    )
    axes[1, 1].set_xlabel("Actual Values")
    axes[1, 1].set_ylabel("Predicted Values")
    axes[1, 1].set_title("Predicted vs Actual")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "residual_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Residual analysis plot saved to {output_path / 'residual_analysis.png'}"
    )


def plot_metrics_summary(
    metrics: dict[str, float], output_dir: str = "outputs/"
) -> None:
    """
    Create a summary bar plot of evaluation metrics.

    Args:
        metrics: Dictionary of metric names and values
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create normalized metrics for visualization
    metric_names = []
    metric_values = []

    for name, value in metrics.items():
        if name.lower() != "r2":  # R² can be negative, handle separately
            metric_names.append(name.upper())
            metric_values.append(value)

    if not metric_names:
        logger.warning("No metrics available for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot of metrics
    bars = ax1.bar(metric_names, metric_values)
    ax1.set_ylabel("Metric Value")
    ax1.set_title("Model Evaluation Metrics")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    ax1.grid(True, alpha=0.3, axis="y")

    # R² separately if available
    if "r2" in metrics:
        ax2.bar(["R²"], [metrics["r2"]], color="green" if metrics["r2"] > 0 else "red")
        ax2.set_ylabel("R² Value")
        ax2.set_title("R² Score")
        ax2.set_ylim(-1, 1)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax2.text(
            0, metrics["r2"] + 0.05, f"{metrics['r2']:.4f}", ha="center", va="bottom"
        )
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "R² not available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path / "metrics_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Metrics summary plot saved to {output_path / 'metrics_summary.png'}")


def plot_training_history(
    training_history: dict[str, list],
    output_dir: str = "outputs/",
) -> None:
    """
    Create training history plots showing loss curves and learning rate.

    Args:
        training_history: Dictionary containing training history metrics
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    epochs = training_history["epochs"]
    train_losses = training_history["train_losses"]
    val_losses = training_history["val_losses"]
    learning_rates = training_history["learning_rates"]

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot losses on left y-axis
    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    line1 = ax1.plot(
        epochs, train_losses, color="blue", label="Train Loss", linewidth=2
    )
    line2 = ax1.plot(epochs, val_losses, color="red", label="Val Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Add best model marker
    if training_history.get("best_epoch") is not None:
        best_epoch = training_history["best_epoch"]
        best_val_loss = val_losses[epochs.index(best_epoch)]
        ax1.scatter(
            best_epoch,
            best_val_loss,
            color="green",
            s=100,
            zorder=5,
            marker="*",
            label=f"Best Model (Epoch {best_epoch})",
        )

    # Add early stopping marker
    if training_history.get("early_stopping_epoch") is not None:
        es_epoch = training_history["early_stopping_epoch"]
        val_losses[epochs.index(es_epoch)]
        ax1.axvline(
            x=es_epoch,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"Early Stop (Epoch {es_epoch})",
        )

    # Plot learning rate on right y-axis
    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Learning Rate", color=color)
    line3 = ax2.plot(
        epochs,
        learning_rates,
        color="green",
        linestyle=":",
        label="Learning Rate",
        linewidth=2,
        alpha=0.8,
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_yscale("log")  # Log scale for learning rate

    # Combine legends
    lines = line1 + line2 + line3
    if training_history.get("best_epoch") is not None:
        lines.append(ax1.collections[0])  # Best model marker
    if training_history.get("early_stopping_epoch") is not None:
        lines.append(ax1.lines[-1])  # Early stopping line

    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", bbox_to_anchor=(1, 1))

    plt.title("Training History: Loss Curves and Learning Rate")
    plt.tight_layout()
    plt.savefig(output_path / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Training history plot saved to {output_path / 'training_history.png'}"
    )


def plot_prediction_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    forecast_horizon: int = 7,
    output_dir: str = "outputs/",
) -> None:
    """
    Create scatter plots for each forecast horizon showing predictions vs targets.

    Args:
        predictions: Model predictions [n_samples, n_regions, forecast_horizon] or [n_samples, forecast_horizon]
        targets: Ground truth targets [n_samples, n_regions, forecast_horizon] or [n_samples, forecast_horizon]
        forecast_horizon: Number of forecast days
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Handle different input shapes
    if len(predictions.shape) == 3:  # [n_samples, n_regions, forecast_horizon]
        pred_reshaped = predictions.reshape(
            -1, forecast_horizon
        )  # [n_samples*n_regions, forecast_horizon]
        target_reshaped = targets.reshape(-1, forecast_horizon)
    else:  # [n_samples, forecast_horizon]
        pred_reshaped = predictions
        target_reshaped = targets

    # Create subplots for each forecast horizon
    n_cols = min(4, forecast_horizon)
    n_rows = (forecast_horizon + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if forecast_horizon == 1:
        axes = np.array([[axes]])  # Make it 2D
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Color palette for different horizons
    colors = plt.cm.viridis(np.linspace(0, 1, forecast_horizon))

    for h in range(forecast_horizon):
        row = h // n_cols
        col = h % n_cols
        ax = axes[row][col]

        pred_h = pred_reshaped[:, h]
        target_h = target_reshaped[:, h]

        # Create scatter plot
        ax.scatter(target_h, pred_h, alpha=0.6, s=10, color=colors[h])

        # Perfect prediction line
        min_val = min(target_h.min(), pred_h.min())
        max_val = max(target_h.max(), pred_h.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            alpha=0.8,
            label="Perfect Prediction",
        )

        # Calculate R²
        r2 = r2_score(target_h, pred_h)

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Day t+{h + 1} (R² = {r2:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Equal aspect ratio
        ax.set_aspect("equal", adjustable="box")

    # Hide unused subplots
    for h in range(forecast_horizon, n_rows * n_cols):
        row = h // n_cols
        col = h % n_cols
        axes[row][col].set_visible(False)

    plt.suptitle("Prediction vs Actual Scatter Plots by Forecast Horizon", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "prediction_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Prediction scatter plot saved to {output_path / 'prediction_scatter.png'}"
    )


def plot_attention_alignment(
    edar_attention_mask: Optional[np.ndarray],
    learned_attention_weights: Optional[np.ndarray],
    output_dir: str = "outputs/",
) -> None:
    """
    Create attention alignment visualization comparing EDAR mask vs learned attention.

    Args:
        edar_attention_mask: Ground truth EDAR attention mask [n_municipalities, n_edars]
        learned_attention_weights: Learned temporal attention weights [n_municipalities, sequence_length] or similar
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if edar_attention_mask is None and learned_attention_weights is None:
        logger.warning("No attention data available for alignment plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: EDAR attention mask (ground truth)
    if edar_attention_mask is not None:
        im1 = axes[0].imshow(edar_attention_mask, cmap="Blues", aspect="auto")
        axes[0].set_title("EDAR Attention Mask\n(Ground Truth Contributions)")
        axes[0].set_xlabel("EDAR Plants")
        axes[0].set_ylabel("Municipalities")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Add sparsity information
        sparsity = (edar_attention_mask == 0).mean()
        axes[0].text(
            0.02,
            0.98,
            f"Sparsity: {sparsity:.2%}",
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    else:
        axes[0].text(
            0.5,
            0.5,
            "EDAR Mask\nNot Available",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[0].set_xticks([])
        axes[0].set_yticks([])

    # Plot 2: Learned attention weights
    if learned_attention_weights is not None:
        im2 = axes[1].imshow(learned_attention_weights, cmap="Reds", aspect="auto")
        axes[1].set_title("Learned Temporal Attention\n(Model Weights)")
        axes[1].set_xlabel("Time Steps / Features")
        axes[1].set_ylabel("Municipalities / Nodes")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Add statistics
        mean_attention = learned_attention_weights.mean()
        std_attention = learned_attention_weights.std()
        axes[1].text(
            0.02,
            0.98,
            f"Mean: {mean_attention:.3f}\nStd: {std_attention:.3f}",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Learned Attention\nNot Available",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    # Plot 3: Alignment analysis (correlation if both available)
    if edar_attention_mask is not None and learned_attention_weights is not None:
        # Try to align dimensions for correlation analysis
        if edar_attention_mask.shape[0] == learned_attention_weights.shape[0]:
            # Calculate municipality-level alignment
            municipality_edar_mean = edar_attention_mask.mean(axis=1)
            municipality_learned_mean = learned_attention_weights.mean(axis=1)

            axes[2].scatter(
                municipality_edar_mean, municipality_learned_mean, alpha=0.6
            )
            axes[2].set_xlabel("EDAR Mask Strength (avg)")
            axes[2].set_ylabel("Learned Attention (avg)")
            axes[2].set_title("Municipality-Level Alignment")
            axes[2].grid(True, alpha=0.3)

            # Add correlation coefficient
            corr = np.corrcoef(municipality_edar_mean, municipality_learned_mean)[0, 1]
            axes[2].text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}",
                transform=axes[2].transAxes,
                va="top",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

            # Add trend line
            z = np.polyfit(municipality_edar_mean, municipality_learned_mean, 1)
            p = np.poly1d(z)
            x_line = np.linspace(
                municipality_edar_mean.min(), municipality_edar_mean.max(), 100
            )
            axes[2].plot(x_line, p(x_line), "r--", alpha=0.8)
        else:
            axes[2].text(
                0.5,
                0.5,
                f"Dimension Mismatch\nEDAR: {edar_attention_mask.shape}\nLearned: {learned_attention_weights.shape}",
                transform=axes[2].transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[2].set_xticks([])
            axes[2].set_yticks([])
    else:
        axes[2].text(
            0.5,
            0.5,
            "Alignment Analysis\nRequires Both Masks",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[2].set_xticks([])
        axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path / "attention_alignment.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Attention alignment plot saved to {output_path / 'attention_alignment.png'}"
    )


def plot_residual_choropleth(
    predictions: np.ndarray,
    targets: np.ndarray,
    region_coords: Optional[np.ndarray] = None,
    region_ids: Optional[list[str]] = None,
    output_dir: str = "outputs/",
) -> None:
    """
    Create choropleth map of residuals for the latest forecast week.

    Args:
        predictions: Model predictions [n_samples, n_regions, forecast_horizon]
        targets: Ground truth targets [n_samples, n_regions, forecast_horizon]
        region_coords: Coordinates for regions [n_regions, 2] (lat, lon)
        region_ids: List of region identifiers
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if region_coords is None:
        logger.warning(
            "No coordinates provided, creating simplified residual visualization"
        )

        # Calculate latest week residuals (last sample, all regions, last forecast day)
        if len(predictions.shape) == 3:
            latest_pred = predictions[-1, :, -1]  # Last sample, all regions, last day
            latest_target = targets[-1, :, -1]
        elif len(predictions.shape) == 2:
            latest_pred = predictions[-1, :]  # Last sample, all forecast days
            latest_target = targets[-1, :]
        else:
            latest_pred = predictions[-1]  # Last sample, single value
            latest_target = targets[-1]

        residuals = latest_target - latest_pred

        # Ensure residuals is an array
        if np.isscalar(residuals):
            residuals = np.array([residuals])
        elif len(residuals.shape) == 0:
            residuals = residuals.reshape(1)

        # Create bar plot as fallback
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot of residuals
        region_indices = range(len(residuals))
        colors = ["red" if r < 0 else "blue" for r in residuals]
        ax1.bar(region_indices, residuals, color=colors, alpha=0.7)
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax1.set_xlabel("Region Index")
        ax1.set_ylabel("Residual (Actual - Predicted)")
        ax1.set_title("Latest Week Residuals by Region")
        ax1.grid(True, alpha=0.3)

        # Add color legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", alpha=0.7, label="Over-prediction"),
            Patch(facecolor="red", alpha=0.7, label="Under-prediction"),
        ]
        ax1.legend(handles=legend_elements)

        # Histogram of residuals
        ax2.hist(residuals, bins=20, alpha=0.7, color="gray", edgecolor="black")
        ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Residual Value")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Latest Week Residuals")
        ax2.grid(True, alpha=0.3)

        # Add statistics
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        ax2.text(
            0.05,
            0.95,
            f"Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    else:
        # Create actual choropleth with coordinates
        logger.info("Creating choropleth map with provided coordinates")

        # Calculate latest week residuals
        if len(predictions.shape) == 3:
            latest_pred = predictions[-1, :, -1]
            latest_target = targets[-1, :, -1]
        elif len(predictions.shape) == 2:
            latest_pred = predictions[-1, :]
            latest_target = targets[-1, :]
        else:
            latest_pred = predictions[-1]
            latest_target = targets[-1]

        residuals = latest_target - latest_pred

        # Ensure residuals is an array
        if np.isscalar(residuals):
            residuals = np.array([residuals])
        elif len(residuals.shape) == 0:
            residuals = residuals.reshape(1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Scatter plot with color-coded residuals
        scatter = ax.scatter(
            region_coords[:, 1],
            region_coords[:, 0],
            c=residuals,
            cmap="RdBu_r",
            s=60,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Latest Week Residuals - Geographic Distribution")

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Residual (Actual - Predicted)")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        ax.text(
            0.02,
            0.98,
            f"Mean Residual: {mean_residual:.3f}\nStd Residual: {std_residual:.3f}\nRegions: {len(residuals)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    plt.tight_layout()
    plt.savefig(output_path / "residual_choropleth.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Residual choropleth plot saved to {output_path / 'residual_choropleth.png'}"
    )


def plot_seen_unseen_performance(
    predictions: np.ndarray,
    targets: np.ndarray,
    train_regions: Optional[list[int]] = None,
    test_regions: Optional[list[int]] = None,
    region_ids: Optional[list[str]] = None,
    output_dir: str = "outputs/",
) -> None:
    """
    Create performance comparison between seen (training) and unseen (test) regions.

    Args:
        predictions: Model predictions [n_samples, n_regions, forecast_horizon]
        targets: Ground truth targets [n_samples, n_regions, forecast_horizon]
        train_regions: Indices of regions seen during training
        test_regions: Indices of regions not seen during training
        region_ids: List of region identifiers
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # If no split provided, use time-based split as proxy
    if train_regions is None or test_regions is None:
        logger.warning("No train/test region split provided, using time-based analysis")

        # Split samples temporally (first 70% as "seen", last 30% as "unseen")
        n_samples = predictions.shape[0]
        split_point = int(0.7 * n_samples)

        seen_pred = predictions[:split_point]
        seen_target = targets[:split_point]
        unseen_pred = predictions[split_point:]
        unseen_target = targets[split_point:]

        performance_type = "Temporal"
        seen_label = f"Early Period (n={len(seen_pred)})"
        unseen_label = f"Late Period (n={len(unseen_pred)})"

    else:
        # Use region-based split
        seen_pred = predictions[:, train_regions, :]
        seen_target = targets[:, train_regions, :]
        unseen_pred = predictions[:, test_regions, :]
        unseen_target = targets[:, test_regions, :]

        performance_type = "Regional"
        seen_label = f"Training Regions (n={len(train_regions)})"
        unseen_label = f"Test Regions (n={len(test_regions)})"

    # Calculate metrics for each group

    # Flatten predictions for metric calculation
    seen_pred_flat = seen_pred.flatten()
    seen_target_flat = seen_target.flatten()
    unseen_pred_flat = unseen_pred.flatten()
    unseen_target_flat = unseen_target.flatten()

    # Calculate metrics
    metrics_seen = {
        "MSE": mean_squared_error(seen_target_flat, seen_pred_flat),
        "MAE": mean_absolute_error(seen_target_flat, seen_pred_flat),
        "RMSE": np.sqrt(mean_squared_error(seen_target_flat, seen_pred_flat)),
        "R²": r2_score(seen_target_flat, seen_pred_flat),
    }

    metrics_unseen = {
        "MSE": mean_squared_error(unseen_target_flat, unseen_pred_flat),
        "MAE": mean_absolute_error(unseen_target_flat, unseen_pred_flat),
        "RMSE": np.sqrt(mean_squared_error(unseen_target_flat, unseen_pred_flat)),
        "R²": r2_score(unseen_target_flat, unseen_pred_flat),
    }

    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Metric comparison bar chart
    metrics_names = list(metrics_seen.keys())
    seen_values = list(metrics_seen.values())
    unseen_values = list(metrics_unseen.values())

    x = np.arange(len(metrics_names))
    width = 0.35

    ax1.bar(
        x - width / 2, seen_values, width, label=seen_label, alpha=0.8, color="blue"
    )
    ax1.bar(
        x + width / 2, unseen_values, width, label=unseen_label, alpha=0.8, color="red"
    )

    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Values")
    ax1.set_title(f"{performance_type} Performance Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (seen_val, unseen_val) in enumerate(zip(seen_values, unseen_values)):
        ax1.text(
            i - width / 2,
            seen_val + max(seen_values) * 0.01,
            f"{seen_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax1.text(
            i + width / 2,
            unseen_val + max(unseen_values) * 0.01,
            f"{unseen_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Residual distribution comparison
    seen_residuals = seen_target_flat - seen_pred_flat
    unseen_residuals = unseen_target_flat - unseen_pred_flat

    ax2.hist(
        seen_residuals, bins=30, alpha=0.7, label=seen_label, color="blue", density=True
    )
    ax2.hist(
        unseen_residuals,
        bins=30,
        alpha=0.7,
        label=unseen_label,
        color="red",
        density=True,
    )
    ax2.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distributions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction scatter for seen data
    sample_size = min(1000, len(seen_pred_flat))
    indices = np.random.choice(len(seen_pred_flat), sample_size, replace=False)
    ax3.scatter(
        seen_target_flat[indices],
        seen_pred_flat[indices],
        alpha=0.6,
        s=10,
        color="blue",
    )

    min_val = min(seen_target_flat[indices].min(), seen_pred_flat[indices].min())
    max_val = max(seen_target_flat[indices].max(), seen_pred_flat[indices].max())
    ax3.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, alpha=0.8)

    ax3.set_xlabel("Actual Values")
    ax3.set_ylabel("Predicted Values")
    ax3.set_title(f"{seen_label} - Predictions vs Actual")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Prediction scatter for unseen data
    sample_size = min(1000, len(unseen_pred_flat))
    indices = np.random.choice(len(unseen_pred_flat), sample_size, replace=False)
    ax4.scatter(
        unseen_target_flat[indices],
        unseen_pred_flat[indices],
        alpha=0.6,
        s=10,
        color="red",
    )

    min_val = min(unseen_target_flat[indices].min(), unseen_pred_flat[indices].min())
    max_val = max(unseen_target_flat[indices].max(), unseen_pred_flat[indices].max())
    ax4.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, alpha=0.8)

    ax4.set_xlabel("Actual Values")
    ax4.set_ylabel("Predicted Values")
    ax4.set_title(f"{unseen_label} - Predictions vs Actual")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path / "seen_unseen_performance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info(
        f"Seen vs unseen performance plot saved to {output_path / 'seen_unseen_performance.png'}"
    )

    # Print summary statistics
    print(f"\n{performance_type} Performance Summary:")
    print("=" * 50)
    print(f"{seen_label:25} | {unseen_label}")
    print("-" * 50)
    for metric in metrics_names:
        print(
            f"{metric:10}: {metrics_seen[metric]:8.4f} | {metrics_unseen[metric]:8.4f}"
        )


def generate_all_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: dict[str, float],
    forecast_horizon: int = 7,
    output_dir: str = "outputs/",
    region_ids: Optional[list[str]] = None,
    training_history: Optional[dict[str, list]] = None,
    edar_attention_mask: Optional[np.ndarray] = None,
    learned_attention_weights: Optional[np.ndarray] = None,
    region_coords: Optional[np.ndarray] = None,
    train_regions: Optional[list[int]] = None,
    test_regions: Optional[list[int]] = None,
) -> None:
    """
    Generate all visualization plots and tables.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        metrics: Evaluation metrics dictionary
        forecast_horizon: Number of forecast days
        output_dir: Directory to save plots
        region_ids: List of region identifiers
        training_history: Training history metrics for loss/LR plots
        edar_attention_mask: Ground truth EDAR attention mask
        learned_attention_weights: Learned attention weights from model
        region_coords: Geographic coordinates for regions
        train_regions: Indices of training regions for seen/unseen analysis
        test_regions: Indices of test regions for seen/unseen analysis
    """
    logger.info("Generating comprehensive visualization plots...")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Generate original plots
    plot_forecast_time_series(predictions, targets, forecast_horizon, output_dir)
    plot_residual_analysis(predictions, targets, output_dir)
    plot_metrics_summary(metrics, output_dir)

    # Generate new enhanced plots
    if training_history is not None:
        plot_training_history(training_history, output_dir)

    plot_prediction_scatter(predictions, targets, forecast_horizon, output_dir)
    plot_attention_alignment(edar_attention_mask, learned_attention_weights, output_dir)
    plot_residual_choropleth(
        predictions, targets, region_coords, region_ids, output_dir
    )
    plot_seen_unseen_performance(
        predictions, targets, train_regions, test_regions, region_ids, output_dir
    )

    # Print results table
    print_results_table(metrics, region_ids)

    # Save results table to CSV
    results_df = create_results_table(metrics, region_ids)
    results_df.to_csv(Path(output_dir) / "results_table.csv", index=False)

    logger.info(f"All plots and results saved to {output_dir}")
    logger.info("Generated plots:")
    logger.info("  - forecast_time_series.png (original time series)")
    logger.info("  - residual_analysis.png (original residual analysis)")
    logger.info("  - metrics_summary.png (original metrics summary)")
    logger.info("  - training_history.png (NEW: loss curves + learning rate)")
    logger.info("  - prediction_scatter.png (NEW: multi-horizon scatter plots)")
    logger.info("  - attention_alignment.png (NEW: EDAR vs learned attention)")
    logger.info("  - residual_choropleth.png (NEW: spatial residual analysis)")
    logger.info("  - seen_unseen_performance.png (NEW: performance segmentation)")
