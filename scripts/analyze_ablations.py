#!/usr/bin/env python
"""Aggregate ablation study results from training outputs.

Scans outputs/training experiment directories, aggregates test metrics across runs
for each ablation, and computes deltas vs baseline (absolute and percentage).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EXPERIMENT_PREFIX_NEW = "mn5_ablation__"
EXPERIMENT_REGEX_LEGACY = re.compile(r"^mn5_ablation_(.+)$")
METRICS = ["mae_median", "rmse_median", "smape_median", "r2_median"]
FINGERPRINT_EXCLUDE_PATHS = {
    ("training", "seed"),
    ("output", "wandb_group"),
    ("output", "wandb_tags"),
    ("output", "experiment_name"),
}


@dataclass(frozen=True)
class AblationRun:
    ablation: str
    campaign_id: str | None
    experiment_dir: Path
    run_dir: Path


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


def _nested_get(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = obj
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _nested_pop(obj: dict[str, Any], path: tuple[str, ...]) -> None:
    cursor: Any = obj
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            return
        cursor = cursor[key]
    if isinstance(cursor, dict):
        cursor.pop(path[-1], None)


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load run config from config.yaml."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Missing config file for run: {run_dir}")

    try:
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - parsing exceptions are data dependent
        raise ValueError(f"Failed to parse config file {config_path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Config file is not a mapping: {config_path}")
    return parsed


def build_config_fingerprint(config: dict[str, Any]) -> str:
    """Build a deterministic comparability fingerprint from run config."""
    normalized = copy.deepcopy(config)
    for path in FINGERPRINT_EXCLUDE_PATHS:
        _nested_pop(normalized, path)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def collect_ablation_runs(
    training_dir: Path,
    experiment_pattern: str,
    campaign_id: str | None,
) -> tuple[dict[str, list[AblationRun]], set[str]]:
    """Collect all run directories grouped by ablation name."""
    ablation_runs: dict[str, list[AblationRun]] = {}
    campaigns_found: set[str] = set()

    if not training_dir.exists():
        logger.warning(f"Training directory not found: {training_dir}")
        return ablation_runs, campaigns_found

    for experiment_dir in sorted(training_dir.glob(experiment_pattern)):
        if not experiment_dir.is_dir():
            continue

        parsed = parse_experiment_name(experiment_dir.name)
        if parsed is None:
            continue
        experiment_campaign_id, ablation = parsed

        if campaign_id is not None and experiment_campaign_id != campaign_id:
            continue

        if experiment_campaign_id is not None:
            campaigns_found.add(experiment_campaign_id)

        run_dirs = sorted(
            run_dir for run_dir in experiment_dir.iterdir() if run_dir.is_dir()
        )
        if not run_dirs:
            continue

        existing = ablation_runs.setdefault(ablation, [])
        for run_dir in run_dirs:
            existing.append(
                AblationRun(
                    ablation=ablation,
                    campaign_id=experiment_campaign_id,
                    experiment_dir=experiment_dir,
                    run_dir=run_dir,
                )
            )

    return ablation_runs, campaigns_found


def validate_ablation_run_consistency(
    ablation: str,
    runs: list[AblationRun],
    assert_consistent_config: bool,
) -> None:
    """Validate that runs in a single ablation are comparable."""
    if not runs or not assert_consistent_config:
        return

    epochs_by_run: dict[str, str] = {}
    split_by_run: dict[str, str] = {}
    fingerprint_by_run: dict[str, str] = {}

    for run in runs:
        run_key = f"{run.experiment_dir.name}/{run.run_dir.name}"
        config = load_run_config(run.run_dir)

        epochs_by_run[run_key] = str(_nested_get(config, ("training", "epochs")))
        split_by_run[run_key] = str(_nested_get(config, ("training", "split_strategy")))
        fingerprint_by_run[run_key] = build_config_fingerprint(config)

    unique_epochs = set(epochs_by_run.values())
    unique_split_strategies = set(split_by_run.values())
    unique_fingerprints = set(fingerprint_by_run.values())

    errors: list[str] = []
    if len(unique_epochs) > 1:
        details = ", ".join(f"{k}: {v}" for k, v in sorted(epochs_by_run.items()))
        errors.append(f"training.epochs mismatch -> {details}")
    if len(unique_split_strategies) > 1:
        details = ", ".join(f"{k}: {v}" for k, v in sorted(split_by_run.items()))
        errors.append(f"training.split_strategy mismatch -> {details}")
    if len(unique_fingerprints) > 1:
        errors.append(
            "config fingerprint mismatch after excluding "
            "training.seed/output.wandb_group/output.wandb_tags/output.experiment_name"
        )

    if errors:
        message = "\n".join(f"- {item}" for item in errors)
        raise ValueError(
            f"Inconsistent run config for ablation '{ablation}':\n{message}"
        )


def load_test_metrics(run_dir: Path) -> pd.DataFrame | None:
    """Load test metrics CSV from a run directory."""
    metrics_file = run_dir / "test_main_model_aggregate_metrics.csv"
    if not metrics_file.exists():
        logger.warning(f"Metrics file not found: {metrics_file}")
        return None

    try:
        return pd.read_csv(metrics_file)
    except Exception as exc:  # pragma: no cover - depends on filesystem state
        logger.error(f"Failed to load {metrics_file}: {exc}")
        return None


def aggregate_ablation_metrics(
    ablation_runs: dict[str, list[AblationRun]],
    assert_consistent_config: bool,
) -> pd.DataFrame:
    """Aggregate metrics across all runs for each ablation."""
    all_records: list[dict[str, Any]] = []

    for ablation_name, runs in sorted(ablation_runs.items()):
        validate_ablation_run_consistency(
            ablation_name,
            runs,
            assert_consistent_config=assert_consistent_config,
        )

        metrics_list: list[pd.DataFrame] = []
        for run in runs:
            df = load_test_metrics(run.run_dir)
            if df is not None:
                metrics_list.append(df)

        if not metrics_list:
            logger.warning(f"No valid metrics found for ablation: {ablation_name}")
            continue

        combined = pd.concat(metrics_list, ignore_index=True)
        for target in combined["target"].unique():
            target_df = combined[combined["target"] == target]
            record: dict[str, Any] = {
                "model": ablation_name,
                "target": target,
                "folds": len(metrics_list),
            }

            for metric in METRICS:
                if metric not in target_df.columns:
                    continue
                values = target_df[metric].values
                # Strip "_median" suffix for cleaner column names (e.g., mae_mean instead of mae_mean)
                clean_metric = metric.replace("_median", "")
                record[f"{clean_metric}_mean"] = np.mean(values)
                record[f"{clean_metric}_std"] = (
                    np.std(values, ddof=1) if len(values) > 1 else 0.0
                )

            all_records.append(record)

    return pd.DataFrame(all_records)


def compute_baseline_deltas(
    df: pd.DataFrame,
    baseline_name: str = "baseline",
) -> pd.DataFrame:
    """Compute absolute and percentage change from baseline for each ablation."""
    if baseline_name not in df["model"].values:
        logger.warning(f"Baseline '{baseline_name}' not found in data")
        return pd.DataFrame()

    baseline_df = df[df["model"] == baseline_name].copy()
    delta_records: list[dict[str, Any]] = []

    for ablation in df["model"].unique():
        if ablation == baseline_name:
            continue

        ablation_df = df[df["model"] == ablation]
        for target in ablation_df["target"].unique():
            target_row = ablation_df[ablation_df["target"] == target].iloc[0]
            baseline_row = baseline_df[baseline_df["target"] == target]
            if baseline_row.empty:
                continue
            baseline_row = baseline_row.iloc[0]

            record: dict[str, Any] = {"model": ablation, "target": target}

            for metric in METRICS:
                clean_metric = metric.replace("_median", "")
                mean_col = f"{clean_metric}_mean"
                if mean_col not in target_row or mean_col not in baseline_row:
                    continue

                baseline_val = baseline_row[mean_col]
                ablation_val = target_row[mean_col]
                record[f"{clean_metric}_delta_abs"] = ablation_val - baseline_val

                if clean_metric == "r2":
                    if abs(baseline_val) > 0.01:
                        record[f"{clean_metric}_delta_pct"] = (
                            (ablation_val - baseline_val) / abs(baseline_val)
                        ) * 100
                elif baseline_val != 0:
                    record[f"{clean_metric}_delta_pct"] = (
                        (ablation_val - baseline_val) / baseline_val
                    ) * 100

            delta_records.append(record)

    return pd.DataFrame(delta_records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate ablation study results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--campaign-id",
        type=str,
        default=None,
        help="Only analyze runs from this campaign ID (default: all matching pattern)",
    )
    parser.add_argument(
        "--experiment-pattern",
        type=str,
        default="mn5_ablation__*__*",
        help="Glob pattern for ablation experiment directories",
    )
    parser.add_argument(
        "--assert-consistent-config",
        dest="assert_consistent_config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert run config consistency per ablation (default: enabled)",
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
        "Scanning %s for pattern '%s' (campaign_id=%s)",
        args.training_dir,
        args.experiment_pattern,
        args.campaign_id,
    )
    ablation_runs, campaigns_found = collect_ablation_runs(
        args.training_dir,
        experiment_pattern=args.experiment_pattern,
        campaign_id=args.campaign_id,
    )

    if not ablation_runs:
        logger.error("No ablation runs found")
        return 1

    logger.info(
        "Found %d ablations with %d total runs",
        len(ablation_runs),
        sum(len(runs) for runs in ablation_runs.values()),
    )
    if campaigns_found:
        logger.info("Campaigns found: %s", ", ".join(sorted(campaigns_found)))

    logger.info("Aggregating metrics...")
    aggregated_df = aggregate_ablation_metrics(
        ablation_runs,
        assert_consistent_config=args.assert_consistent_config,
    )
    if aggregated_df.empty:
        logger.error("No metrics could be aggregated")
        return 1

    aggregated_path = args.output_dir / "ablation_metrics_aggregated.csv"
    aggregated_df.to_csv(aggregated_path, index=False)
    logger.info(f"Saved aggregated metrics to {aggregated_path}")

    logger.info("Computing deltas from baseline...")
    deltas_df = compute_baseline_deltas(aggregated_df, baseline_name=args.baseline)
    if not deltas_df.empty:
        deltas_path = args.output_dir / "ablation_metrics_deltas.csv"
        deltas_df.to_csv(deltas_path, index=False)
        logger.info(f"Saved delta metrics to {deltas_path}")

    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"\nAggregated metrics for {len(aggregated_df['model'].unique())} ablations")
    print(f"Targets: {', '.join(sorted(aggregated_df['target'].unique()))}")
    print("\nMetrics: MAE, RMSE, sMAPE, R²")
    print(f"\nOutput directory: {args.output_dir}")
    print("\nFiles generated:")
    print(f"  - {aggregated_path.name}")
    if not deltas_df.empty:
        print(f"  - {deltas_path.name}")

    print("\n" + "-" * 80)
    print("Sample of aggregated metrics (MAE means):")
    print("-" * 80)
    pivot = aggregated_df.pivot_table(
        index="model",
        columns="target",
        values="mae_mean",
    )
    print(pivot.to_string())

    if not deltas_df.empty:
        print("\n" + "-" * 80)
        print("Sample of deltas from baseline (MAE % change):")
        print("-" * 80)
        pivot_delta = deltas_df.pivot_table(
            index="model",
            columns="target",
            values="mae_delta_pct",
        )
        print(pivot_delta.to_string())

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
