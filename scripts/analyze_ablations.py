#!/usr/bin/env python
"""Aggregate ablation study results from training outputs.

Scans outputs/training/mn5_ablation_* directories and aggregates test metrics
across runs for each ablation. Outputs CSV files with mean/std statistics
and percentage changes from baseline.

Usage:
    python scripts/analyze_ablations.py [--output-dir OUTPUT_DIR]

Outputs:
    - ablation_metrics_aggregated.csv: Mean and std per ablation
    - ablation_metrics_deltas.csv: Percentage change from baseline
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_ablation_name(dir_name: str) -> str | None:
    """Extract ablation name from directory name.

    Args:
        dir_name: Directory name like 'mn5_ablation_baseline' or 'mn5_ablation_no_sir_loss'

    Returns:
        Ablation name or None if not an ablation directory
    """
    match = re.match(r"mn5_ablation_(.+)", dir_name)
    return match.group(1) if match else None


def collect_ablation_runs(
    training_dir: Path,
) -> dict[str, list[Path]]:
    """Collect all run directories grouped by ablation name.

    Args:
        training_dir: Path to outputs/training directory

    Returns:
        Dictionary mapping ablation name to list of run directories
    """
    ablation_runs: dict[str, list[Path]] = {}

    if not training_dir.exists():
        logger.warning(f"Training directory not found: {training_dir}")
        return ablation_runs

    for item in training_dir.iterdir():
        if not item.is_dir():
            continue

        ablation_name = parse_ablation_name(item.name)
        if ablation_name is None:
            continue

        # Find all run subdirectories (job IDs)
        runs = [run_dir for run_dir in item.iterdir() if run_dir.is_dir()]

        if runs:
            ablation_runs[ablation_name] = runs
            logger.info(f"Found {len(runs)} runs for ablation: {ablation_name}")

    return ablation_runs


def load_test_metrics(run_dir: Path) -> pd.DataFrame | None:
    """Load test metrics CSV from a run directory.

    Args:
        run_dir: Path to run directory containing test_main_model_aggregate_metrics.csv

    Returns:
        DataFrame with metrics or None if file not found
    """
    metrics_file = run_dir / "test_main_model_aggregate_metrics.csv"

    if not metrics_file.exists():
        logger.warning(f"Metrics file not found: {metrics_file}")
        return None

    try:
        df = pd.read_csv(metrics_file)
        return df
    except Exception as e:
        logger.error(f"Failed to load {metrics_file}: {e}")
        return None


def aggregate_ablation_metrics(
    ablation_runs: dict[str, list[Path]],
) -> pd.DataFrame:
    """Aggregate metrics across all runs for each ablation.

    Args:
        ablation_runs: Dictionary mapping ablation name to list of run directories

    Returns:
        DataFrame with aggregated metrics (mean ± std)
    """
    all_records = []

    for ablation_name, runs in sorted(ablation_runs.items()):
        metrics_list = []

        for run_dir in runs:
            df = load_test_metrics(run_dir)
            if df is not None:
                metrics_list.append(df)

        if not metrics_list:
            logger.warning(f"No valid metrics found for ablation: {ablation_name}")
            continue

        # Combine all runs for this ablation
        combined = pd.concat(metrics_list, ignore_index=True)

        # Compute statistics per target
        for target in combined["target"].unique():
            target_df = combined[combined["target"] == target]

            record: dict[str, Any] = {
                "ablation": ablation_name,
                "target": target,
                "n_runs": len(metrics_list),
            }

            # Aggregate each metric
            for metric in ["mae_median", "rmse_median", "smape_median", "r2_median"]:
                if metric in target_df.columns:
                    values = target_df[metric].values
                    record[f"{metric}_mean"] = np.mean(values)
                    record[f"{metric}_std"] = (
                        np.std(values, ddof=1) if len(values) > 1 else 0.0
                    )

            all_records.append(record)

    return pd.DataFrame(all_records)


def compute_baseline_deltas(
    df: pd.DataFrame,
    baseline_name: str = "baseline",
) -> pd.DataFrame:
    """Compute percentage change from baseline for each ablation.

    Args:
        df: Aggregated metrics DataFrame
        baseline_name: Name of baseline ablation

    Returns:
        DataFrame with delta percentages
    """
    if baseline_name not in df["ablation"].values:
        logger.warning(f"Baseline '{baseline_name}' not found in data")
        return pd.DataFrame()

    baseline_df = df[df["ablation"] == baseline_name].copy()

    delta_records = []

    for ablation in df["ablation"].unique():
        if ablation == baseline_name:
            continue

        ablation_df = df[df["ablation"] == ablation]

        for target in ablation_df["target"].unique():
            target_row = ablation_df[ablation_df["target"] == target].iloc[0]
            baseline_row = baseline_df[baseline_df["target"] == target]

            if baseline_row.empty:
                continue

            baseline_row = baseline_row.iloc[0]

            record = {
                "ablation": ablation,
                "target": target,
            }

            # Compute % change for each metric
            for metric in ["mae_median", "rmse_median", "smape_median"]:
                mean_col = f"{metric}_mean"
                if mean_col in target_row and mean_col in baseline_row:
                    baseline_val = baseline_row[mean_col]
                    ablation_val = target_row[mean_col]
                    if baseline_val != 0:
                        delta = ((ablation_val - baseline_val) / baseline_val) * 100
                        record[f"{metric}_delta_pct"] = delta

            # R2 is different: higher is better, so invert the logic
            if "r2_median_mean" in target_row and "r2_median_mean" in baseline_row:
                baseline_r2 = baseline_row["r2_median_mean"]
                ablation_r2 = target_row["r2_median_mean"]
                if abs(baseline_r2) > 0.01:  # Avoid division by near-zero
                    delta = ((ablation_r2 - baseline_r2) / abs(baseline_r2)) * 100
                    record["r2_median_delta_pct"] = delta

            delta_records.append(record)

    return pd.DataFrame(delta_records)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ablation study results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/analyze_ablations.py
    python scripts/analyze_ablations.py --output-dir outputs/ablation_analysis
        """,
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("outputs/training"),
        help="Directory containing training outputs (default: outputs/training)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/ablation_analysis"),
        help="Output directory for aggregated results (default: outputs/reports/ablation_analysis)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline",
        help="Name of baseline ablation (default: baseline)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect ablation runs
    logger.info(f"Scanning {args.training_dir} for ablation results...")
    ablation_runs = collect_ablation_runs(args.training_dir)

    if not ablation_runs:
        logger.error("No ablation runs found")
        return 1

    logger.info(
        f"Found {len(ablation_runs)} ablations with {sum(len(runs) for runs in ablation_runs.values())} total runs"
    )

    # Aggregate metrics
    logger.info("Aggregating metrics...")
    aggregated_df = aggregate_ablation_metrics(ablation_runs)

    if aggregated_df.empty:
        logger.error("No metrics could be aggregated")
        return 1

    # Save aggregated metrics
    aggregated_path = args.output_dir / "ablation_metrics_aggregated.csv"
    aggregated_df.to_csv(aggregated_path, index=False)
    logger.info(f"Saved aggregated metrics to {aggregated_path}")

    # Compute deltas
    logger.info("Computing deltas from baseline...")
    deltas_df = compute_baseline_deltas(aggregated_df, baseline_name=args.baseline)

    if not deltas_df.empty:
        deltas_path = args.output_dir / "ablation_metrics_deltas.csv"
        deltas_df.to_csv(deltas_path, index=False)
        logger.info(f"Saved delta metrics to {deltas_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(
        f"\nAggregated metrics for {len(aggregated_df['ablation'].unique())} ablations"
    )
    print(f"Targets: {', '.join(sorted(aggregated_df['target'].unique()))}")
    print("\nMetrics: MAE, RMSE, sMAPE, R²")
    print(f"\nOutput directory: {args.output_dir}")
    print("\nFiles generated:")
    print(f"  - {aggregated_path.name}")
    if not deltas_df.empty:
        print(f"  - {deltas_path.name}")

    # Show sample of data
    print("\n" + "-" * 80)
    print("Sample of aggregated metrics (MAE means):")
    print("-" * 80)
    pivot = aggregated_df.pivot_table(
        index="ablation",
        columns="target",
        values="mae_median_mean",
    )
    print(pivot.to_string())

    if not deltas_df.empty:
        print("\n" + "-" * 80)
        print("Sample of deltas from baseline (MAE % change):")
        print("-" * 80)
        pivot_delta = deltas_df.pivot_table(
            index="ablation",
            columns="target",
            values="mae_median_delta_pct",
        )
        print(pivot_delta.to_string())

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
