#!/usr/bin/env python
"""Cross-head impact analysis for ablation studies.

Computes how ablating each observation head affects losses on all other heads,
using pairwise subtraction against baseline (matched by seed).

This creates a matrix showing cross-head dependencies:
- Rows: Which head is ablated (no_ww, no_cases, no_hosp, no_deaths)
- Columns: Which head's loss is measured (ww, cases, hosp, deaths)
- Values: Percentage change in loss (Δ = (ablated - baseline) / baseline * 100)

Usage:
    python scripts/analyze_cross_head_impact.py \
        --training-dir outputs/training \
        --output-dir outputs/reports/cross_head_analysis \
        --campaign-id ablation_12345
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EXPERIMENT_PREFIX_NEW = "mn5_ablation__"
EXPERIMENT_REGEX_LEGACY = re.compile(r"^mn5_ablation_(.+)$")

# Heads to analyze (excluding SIR)
HEADS = ["ww", "cases", "hosp", "deaths"]

# Column mapping: head name -> joint loss CSV column
HEAD_TO_COLUMN = {
    "ww": "joint_loss_ww_median",
    "cases": "joint_loss_cases_median",
    "hosp": "joint_loss_hosp_median",
    "deaths": "joint_loss_deaths_median",
}

# Ablations that zero out specific heads (from run_ablations.sbatch)
ABLATION_TO_HEAD = {
    "no_ww_loss": "ww",
    "no_cases_loss": "cases",
    "no_hosp_loss": "hosp",
    "no_deaths_loss": "deaths",
}

FINGERPRINT_EXCLUDE_PATHS = {
    ("training", "seed"),
    ("output", "wandb_group"),
    ("output", "wandb_tags"),
    ("output", "experiment_name"),
}


@dataclass(frozen=True)
class CrossHeadRun:
    """Represents a single training run with metadata."""

    ablation: str
    campaign_id: str | None
    experiment_dir: Path
    run_dir: Path
    seed: int | None = None


def parse_experiment_name(experiment_name: str) -> tuple[str | None, str] | None:
    """Parse experiment names for new and legacy ablation naming conventions."""
    if experiment_name.startswith(EXPERIMENT_PREFIX_NEW):
        suffix = experiment_name[len(EXPERIMENT_PREFIX_NEW) :]
        parts = suffix.split("__", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return parts[0], parts[1]

    legacy_match = EXPERIMENT_REGEX_LEGACY.match(experiment_name)
    if legacy_match:
        return None, legacy_match.group(1)

    return None


def extract_seed_from_config(run_dir: Path) -> int | None:
    """Extract seed from run config.yaml."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None

    try:
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            seed = parsed.get("training", {}).get("seed")
            if seed is not None:
                return int(seed)
    except Exception:
        pass

    return None


def extract_seed_from_run_dir(run_dir: Path) -> int | None:
    """Try to extract seed from run directory name or model_id file."""
    # Try model_id file first
    model_id_path = run_dir / "model_id.txt"
    if model_id_path.exists():
        try:
            content = model_id_path.read_text().strip()
            # Look for seed pattern: _s{number}
            match = re.search(r"_s(\d+)$", content)
            if match:
                return int(match.group(1))
        except Exception:
            pass

    # Try run directory name
    # Pattern: {timestamp}_seed{number} or similar
    match = re.search(r"[Ss]eed(\d+)", run_dir.name)
    if match:
        return int(match.group(1))

    return None


def get_run_seed(run_dir: Path) -> int | None:
    """Get seed for a run, trying multiple methods."""
    # Try config first (most reliable)
    seed = extract_seed_from_config(run_dir)
    if seed is not None:
        return seed

    # Try directory/filename extraction
    return extract_seed_from_run_dir(run_dir)


def load_joint_loss_metrics(
    run_dir: Path, split: str = "test"
) -> dict[str, float] | None:
    """Load per-head losses from joint loss aggregate CSV."""
    metrics_file = run_dir / f"{split}_main_model_joint_loss_aggregate.csv"
    if not metrics_file.exists():
        logger.debug(f"Joint loss file not found: {metrics_file}")
        return None

    try:
        df = pd.read_csv(metrics_file)
        if df.empty:
            return None

        # Extract first (and should be only) row
        row = df.iloc[0]
        metrics = {}
        for head, col in HEAD_TO_COLUMN.items():
            if col in row:
                val = row[col]
                if pd.notna(val) and np.isfinite(val):
                    metrics[head] = float(val)
                else:
                    metrics[head] = None
            else:
                metrics[head] = None
        return metrics
    except Exception as exc:
        logger.warning(f"Failed to load metrics from {metrics_file}: {exc}")
        return None


def collect_cross_head_runs(
    training_dir: Path,
    campaign_id: str | None,
    relevant_ablations: list[str] | None = None,
) -> dict[str, list[CrossHeadRun]]:
    """Collect baseline and head-ablation runs.

    Returns dict mapping ablation name to list of CrossHeadRun objects.
    """
    if relevant_ablations is None:
        relevant_ablations = ["baseline"] + list(ABLATION_TO_HEAD.keys())

    ablation_runs: dict[str, list[CrossHeadRun]] = {
        abl: [] for abl in relevant_ablations
    }

    if not training_dir.exists():
        logger.warning(f"Training directory not found: {training_dir}")
        return ablation_runs

    pattern = EXPERIMENT_PREFIX_NEW + "*"
    for experiment_dir in sorted(training_dir.glob(pattern)):
        if not experiment_dir.is_dir():
            continue

        parsed = parse_experiment_name(experiment_dir.name)
        if parsed is None:
            continue

        exp_campaign_id, ablation = parsed

        if campaign_id is not None and exp_campaign_id != campaign_id:
            continue

        if ablation not in relevant_ablations:
            continue

        run_dirs = [run_dir for run_dir in experiment_dir.iterdir() if run_dir.is_dir()]
        if not run_dirs:
            continue

        for run_dir in run_dirs:
            seed = get_run_seed(run_dir)
            ablation_runs[ablation].append(
                CrossHeadRun(
                    ablation=ablation,
                    campaign_id=exp_campaign_id,
                    experiment_dir=experiment_dir,
                    run_dir=run_dir,
                    seed=seed,
                )
            )

    return ablation_runs


def compute_pairwise_deltas(
    baseline_runs: list[CrossHeadRun],
    ablation_runs: list[CrossHeadRun],
    split: str = "test",
) -> pd.DataFrame | None:
    """Compute pairwise deltas between ablation and baseline runs matched by seed.

    Returns DataFrame with columns:
    - ablated_head: Which head was ablated
    - measured_head: Which head's loss was measured
    - baseline_loss: Baseline loss value
    - ablation_loss: Ablation loss value
    - delta_abs: Absolute difference (ablation - baseline)
    - delta_pct: Percentage difference ((ablation - baseline) / baseline * 100)
    - seed: The seed used for matching
    """
    # Index baseline runs by seed
    baseline_by_seed: dict[int, CrossHeadRun] = {}
    for run in baseline_runs:
        if run.seed is not None:
            baseline_by_seed[run.seed] = run

    if not baseline_by_seed:
        logger.warning("No baseline runs with extractable seeds found")
        return None

    records = []

    for abl_run in ablation_runs:
        if abl_run.seed is None:
            logger.debug(f"Skipping ablation run without seed: {abl_run.run_dir}")
            continue

        # Find matching baseline
        baseline_run = baseline_by_seed.get(abl_run.seed)
        if baseline_run is None:
            logger.debug(f"No matching baseline for seed {abl_run.seed}")
            continue

        # Load metrics
        baseline_metrics = load_joint_loss_metrics(baseline_run.run_dir, split)
        ablation_metrics = load_joint_loss_metrics(abl_run.run_dir, split)

        if baseline_metrics is None or ablation_metrics is None:
            continue

        # Compute deltas for each measured head
        for measured_head in HEADS:
            baseline_val = baseline_metrics.get(measured_head)
            ablation_val = ablation_metrics.get(measured_head)

            if baseline_val is None or ablation_val is None:
                continue

            if baseline_val == 0:
                delta_pct = np.nan
            else:
                delta_pct = ((ablation_val - baseline_val) / baseline_val) * 100

            records.append(
                {
                    "ablated_head": ABLATION_TO_HEAD.get(
                        abl_run.ablation, abl_run.ablation
                    ),
                    "measured_head": measured_head,
                    "baseline_loss": baseline_val,
                    "ablation_loss": ablation_val,
                    "delta_abs": ablation_val - baseline_val,
                    "delta_pct": delta_pct,
                    "seed": abl_run.seed,
                }
            )

    if not records:
        return None

    return pd.DataFrame(records)


def aggregate_cross_head_matrix(
    pairwise_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate pairwise deltas into mean, std, and count matrices.

    Returns three DataFrames (ablated_head × measured_head):
    - mean_matrix: Mean percentage change
    - std_matrix: Standard deviation of percentage change
    - count_matrix: Number of pairwise comparisons
    """
    # Pivot to get mean, std, count
    mean_matrix = pairwise_df.pivot_table(
        index="ablated_head",
        columns="measured_head",
        values="delta_pct",
        aggfunc="mean",
    )

    std_matrix = pairwise_df.pivot_table(
        index="ablated_head",
        columns="measured_head",
        values="delta_pct",
        aggfunc="std",
    )

    count_matrix = pairwise_df.pivot_table(
        index="ablated_head",
        columns="measured_head",
        values="delta_pct",
        aggfunc="count",
    )

    # Reorder to standard head order
    ablated_order = [h for h in HEADS if h in mean_matrix.index]
    measured_order = [h for h in HEADS if h in mean_matrix.columns]

    mean_matrix = mean_matrix.reindex(index=ablated_order, columns=measured_order)
    std_matrix = std_matrix.reindex(index=ablated_order, columns=measured_order)
    count_matrix = count_matrix.reindex(index=ablated_order, columns=measured_order)

    return mean_matrix, std_matrix, count_matrix


def compute_cross_head_impact(
    training_dir: Path,
    campaign_id: str | None,
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Main entry point: compute cross-head impact matrix.

    Returns (pairwise_df, mean_matrix, (std_matrix, count_matrix)) or None if no data.
    """
    ablation_runs = collect_cross_head_runs(training_dir, campaign_id)

    baseline_runs = ablation_runs.get("baseline", [])
    if not baseline_runs:
        logger.error("No baseline runs found")
        return None

    logger.info(f"Found {len(baseline_runs)} baseline runs")

    all_pairwise = []

    for ablation_name, head in ABLATION_TO_HEAD.items():
        abl_runs = ablation_runs.get(ablation_name, [])
        if not abl_runs:
            logger.warning(f"No runs found for ablation: {ablation_name}")
            continue

        logger.info(f"Processing {ablation_name}: {len(abl_runs)} runs")

        pairwise_df = compute_pairwise_deltas(baseline_runs, abl_runs, split)
        if pairwise_df is not None and not pairwise_df.empty:
            all_pairwise.append(pairwise_df)

    if not all_pairwise:
        logger.error("No pairwise comparisons could be computed")
        return None

    combined_pairwise = pd.concat(all_pairwise, ignore_index=True)
    mean_matrix, std_matrix, count_matrix = aggregate_cross_head_matrix(
        combined_pairwise
    )

    return combined_pairwise, mean_matrix, (std_matrix, count_matrix)


def format_impact_matrix(
    mean_matrix: pd.DataFrame,
    std_matrix: pd.DataFrame,
    count_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Create formatted matrix with mean ± std for display."""
    # Initialize with object dtype to allow string values
    formatted = pd.DataFrame(
        index=mean_matrix.index, columns=mean_matrix.columns, dtype=object
    )

    for idx in formatted.index:
        for col in formatted.columns:
            mean_val = mean_matrix.loc[idx, col]
            std_val = std_matrix.loc[idx, col]
            count_val = count_matrix.loc[idx, col]

            if pd.notna(mean_val):
                formatted.loc[idx, col] = (
                    f"{mean_val:+.1f}±{std_val:.1f} (n={int(count_val)})"
                )
            else:
                formatted.loc[idx, col] = "N/A"

    return formatted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze cross-head impact in ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze specific campaign
    python scripts/analyze_cross_head_impact.py --campaign-id ablation_12345

    # Use validation split instead of test
    python scripts/analyze_cross_head_impact.py --campaign-id ablation_12345 --split val

    # Custom output directory
    python scripts/analyze_cross_head_impact.py --campaign-id ablation_12345 \\
        --output-dir reports/cross_head
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
        default=Path("outputs/reports/cross_head_analysis"),
        help="Output directory for results (default: outputs/reports/cross_head_analysis)",
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Campaign ID to analyze (default: all campaigns)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Which split to analyze (default: test)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Analyzing cross-head impact for campaign: {args.campaign_id or 'ALL'}"
    )
    logger.info(f"Split: {args.split}")

    result = compute_cross_head_impact(
        args.training_dir,
        args.campaign_id,
        args.split,
    )

    if result is None:
        logger.error("Failed to compute cross-head impact")
        return 1

    pairwise_df, mean_matrix, (std_matrix, count_matrix) = result

    # Save pairwise data
    pairwise_path = args.output_dir / "cross_head_pairwise.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    logger.info(f"Saved pairwise data to {pairwise_path}")

    # Save matrices
    mean_matrix.to_csv(args.output_dir / "cross_head_mean_matrix.csv")
    std_matrix.to_csv(args.output_dir / "cross_head_std_matrix.csv")
    count_matrix.to_csv(args.output_dir / "cross_head_count_matrix.csv")

    # Save formatted version
    formatted = format_impact_matrix(mean_matrix, std_matrix, count_matrix)
    formatted.to_csv(args.output_dir / "cross_head_formatted.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("CROSS-HEAD IMPACT ANALYSIS")
    print("=" * 80)
    print(f"\nCampaign: {args.campaign_id or 'ALL'}")
    print(f"Split: {args.split}")
    print(f"Total pairwise comparisons: {len(pairwise_df)}")

    print("\n" + "-" * 80)
    print("Mean Percentage Change Matrix (%)")
    print("(Rows: Ablated Head, Columns: Measured Head)")
    print("-" * 80)
    print(mean_matrix.to_string())

    print("\n" + "-" * 80)
    print("Formatted (Mean ± Std, n=count)")
    print("-" * 80)
    print(formatted.to_string())

    print("\n" + "=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
