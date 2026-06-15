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
    ("training", "crossval_fold_index"),
    ("output", "wandb_group"),
    ("output", "wandb_tags"),
    ("output", "experiment_name"),
}
CONTEXT_PATHS: dict[str, tuple[str, ...]] = {
    "model_type_mobility": ("model", "type", "mobility"),
    "model_type_regions": ("model", "type", "regions"),
    "model_mobility_embedding_dim": ("model", "mobility_embedding_dim"),
    "model_graph_adjacency_source": ("model", "graph_adjacency_source"),
    "model_max_neighbors": ("model", "max_neighbors"),
    "model_gnn_depth": ("model", "gnn_depth"),
    "model_gnn_hidden_dim": ("model", "gnn_hidden_dim"),
    "model_gnn_module": ("model", "gnn_module"),
    "model_input_window_length": ("model", "input_window_length"),
    "model_forecast_horizon": ("model", "forecast_horizon"),
    "data_dataset_path": ("data", "dataset_path"),
    "data_mobility_threshold": ("data", "mobility_threshold"),
    "data_mobility_lags": ("data", "mobility_lags"),
    "training_epochs": ("training", "epochs"),
    "training_split_strategy": ("training", "split_strategy"),
    "training_node_split_strategy": ("training", "node_split_strategy"),
    "training_crossval_enabled": ("training", "crossval_enabled"),
    "training_crossval_fold_index": ("training", "crossval_fold_index"),
    "output_experiment_name": ("output", "experiment_name"),
    "output_wandb_group": ("output", "wandb_group"),
}


@dataclass(frozen=True)
class AblationRun:
    ablation: str
    campaign_id: str | None
    experiment_dir: Path
    run_dir: Path
    seed: int | None


def _run_label(run: AblationRun) -> str:
    return f"{run.experiment_dir.name}/{run.run_dir.name}"


def _has_test_metrics(run: AblationRun) -> bool:
    return (run.run_dir / "test_main_model_aggregate_metrics.csv").exists()


def _parse_csv_set(raw: str | None, *, cast_type: type = str) -> set[Any] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return set()
    return {cast_type(value) for value in values}


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


def get_run_seed(run_dir: Path) -> int | None:
    """Extract training seed from run config if available."""
    try:
        config = load_run_config(run_dir)
    except ValueError:
        return None

    seed = _nested_get(config, ("training", "seed"))
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


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
            seed = get_run_seed(run_dir)
            existing.append(
                AblationRun(
                    ablation=ablation,
                    campaign_id=experiment_campaign_id,
                    experiment_dir=experiment_dir,
                    run_dir=run_dir,
                    seed=seed,
                )
            )

    return ablation_runs, campaigns_found


def build_seed_coverage_report(
    ablation_runs: dict[str, list[AblationRun]],
) -> pd.DataFrame:
    """Build per-ablation/per-seed run coverage diagnostics."""
    records: list[dict[str, Any]] = []
    for ablation, runs in sorted(ablation_runs.items()):
        seeds = sorted({run.seed for run in runs if run.seed is not None})
        report_seeds: list[int | None] = []
        if any(run.seed is None for run in runs):
            report_seeds.append(None)
        report_seeds.extend(seeds)

        if not report_seeds:
            records.append(
                {
                    "model": ablation,
                    "seed": None,
                    "run_count": len(runs),
                    "metrics_count": sum(_has_test_metrics(run) for run in runs),
                    "runs": ";".join(_run_label(run) for run in runs),
                    "metric_runs": ";".join(
                        _run_label(run) for run in runs if _has_test_metrics(run)
                    ),
                }
            )
            continue

        for seed in report_seeds:
            seed_runs = [run for run in runs if run.seed == seed]
            metric_runs = [run for run in seed_runs if _has_test_metrics(run)]
            records.append(
                {
                    "model": ablation,
                    "seed": seed,
                    "run_count": len(seed_runs),
                    "metrics_count": len(metric_runs),
                    "runs": ";".join(_run_label(run) for run in seed_runs),
                    "metric_runs": ";".join(_run_label(run) for run in metric_runs),
                }
            )
    return pd.DataFrame(records)


def _context_value(config: dict[str, Any], key: str) -> Any:
    value = _nested_get(config, CONTEXT_PATHS[key])
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _classify_neighborhood_aggregation(
    config: dict[str, Any],
) -> tuple[bool, str, str]:
    mobility_enabled = bool(_nested_get(config, ("model", "type", "mobility")))
    adjacency_source = str(
        _nested_get(config, ("model", "graph_adjacency_source")) or "unknown"
    )

    if not mobility_enabled:
        return (
            False,
            "disabled",
            "Neighborhood aggregation disabled; mobility branch is ablated.",
        )
    if adjacency_source == "mobility":
        return (
            True,
            "dynamic_mobility",
            "Dynamic origin-destination mobility matrix neighborhoods.",
        )
    if adjacency_source == "spatial_knn":
        return (
            True,
            "static_spatial_knn",
            "Static centroid-distance KNN neighborhoods.",
        )
    if adjacency_source == "spatial_queen":
        return (
            True,
            "static_spatial_queen",
            "Static queen-contiguity neighborhoods.",
        )
    return (
        True,
        adjacency_source,
        f"Neighborhood aggregation source: {adjacency_source}",
    )


def build_run_context_report(
    ablation_runs: dict[str, list[AblationRun]],
) -> pd.DataFrame:
    """Build per-run context for neighborhood aggregation comparisons."""
    records: list[dict[str, Any]] = []
    for ablation, runs in sorted(ablation_runs.items()):
        for run in sorted(runs, key=lambda item: (item.seed is None, item.seed or -1)):
            try:
                config = load_run_config(run.run_dir)
            except ValueError as exc:
                logger.warning("Skipping context for %s: %s", _run_label(run), exc)
                continue

            enabled, method, description = _classify_neighborhood_aggregation(config)
            record: dict[str, Any] = {
                "campaign_id": run.campaign_id,
                "model": ablation,
                "seed": run.seed,
                "run": _run_label(run),
                "has_test_metrics": _has_test_metrics(run),
                "neighborhood_aggregation_enabled": enabled,
                "neighborhood_aggregation_method": method,
                "neighborhood_aggregation_description": description,
            }
            for key in CONTEXT_PATHS:
                record[key] = _context_value(config, key)
            records.append(record)

    return pd.DataFrame(records)


def build_neighborhood_context_report(run_context_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the run context at the ablation/method level."""
    if run_context_df.empty:
        return pd.DataFrame()

    context_cols = [
        "model",
        "neighborhood_aggregation_enabled",
        "neighborhood_aggregation_method",
        "neighborhood_aggregation_description",
        "model_type_mobility",
        "model_type_regions",
        "model_graph_adjacency_source",
        "model_max_neighbors",
        "model_gnn_depth",
        "model_gnn_hidden_dim",
        "model_gnn_module",
        "model_mobility_embedding_dim",
        "data_mobility_threshold",
        "data_mobility_lags",
        "model_input_window_length",
        "model_forecast_horizon",
        "training_epochs",
        "training_split_strategy",
        "training_node_split_strategy",
        "training_crossval_enabled",
        "data_dataset_path",
        "output_wandb_group",
    ]
    records: list[dict[str, Any]] = []
    for values, group in run_context_df.groupby(context_cols, dropna=False, sort=True):
        record = dict(zip(context_cols, values, strict=True))
        seeds = sorted(seed for seed in group["seed"].dropna().astype(int).unique())
        record["seeds"] = ",".join(str(seed) for seed in seeds)
        record["run_count"] = int(len(group))
        record["metric_run_count"] = int(group["has_test_metrics"].sum())
        record["runs"] = ";".join(group["run"].astype(str))
        records.append(record)

    return pd.DataFrame(records)


def validate_seed_coverage(
    ablation_runs: dict[str, list[AblationRun]],
    baseline_name: str,
    *,
    included_ablations: set[str] | None = None,
    expected_seeds: set[int] | None = None,
) -> None:
    """Assert a complete one-run-per-seed block across selected ablations."""
    if included_ablations is None:
        included_ablations = set(ablation_runs)
    else:
        included_ablations = set(included_ablations)
    included_ablations.add(baseline_name)

    missing_ablations = sorted(
        ablation for ablation in included_ablations if ablation not in ablation_runs
    )
    errors: list[str] = []
    if missing_ablations:
        errors.append(f"missing ablations: {', '.join(missing_ablations)}")

    metrics_by_ablation: dict[str, dict[int, list[AblationRun]]] = {}
    for ablation in sorted(included_ablations):
        runs = ablation_runs.get(ablation, [])
        by_seed: dict[int, list[AblationRun]] = {}
        unseeded_metric_runs = [
            run for run in runs if run.seed is None and _has_test_metrics(run)
        ]
        if unseeded_metric_runs:
            errors.append(
                f"{ablation}: metric-bearing runs with missing seed "
                f"{[_run_label(run) for run in unseeded_metric_runs]}"
            )
        for run in runs:
            if run.seed is None or not _has_test_metrics(run):
                continue
            by_seed.setdefault(run.seed, []).append(run)
        metrics_by_ablation[ablation] = by_seed

    if expected_seeds is None:
        expected_seeds = set().union(
            *(set(seed_runs) for seed_runs in metrics_by_ablation.values())
        )
    expected_seeds = set(expected_seeds)

    if not expected_seeds:
        errors.append("no expected seeds resolved from metric-bearing runs")

    for ablation in sorted(included_ablations):
        by_seed = metrics_by_ablation.get(ablation, {})
        present_seeds = set(by_seed)
        missing_seeds = sorted(expected_seeds - present_seeds)
        extra_seeds = sorted(present_seeds - expected_seeds)
        duplicate_seeds = sorted(
            seed for seed, seed_runs in by_seed.items() if len(seed_runs) > 1
        )

        if missing_seeds:
            errors.append(f"{ablation}: missing metric-bearing seeds {missing_seeds}")
        if extra_seeds:
            errors.append(f"{ablation}: unexpected metric-bearing seeds {extra_seeds}")
        if duplicate_seeds:
            duplicate_details = ", ".join(
                f"{seed} -> {[_run_label(run) for run in by_seed[seed]]}"
                for seed in duplicate_seeds
            )
            errors.append(
                f"{ablation}: duplicate metric-bearing seeds {duplicate_details}"
            )

    if errors:
        message = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Seed coverage validation failed:\n{message}")


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


def compute_seed_matched_baseline_deltas(
    ablation_runs: dict[str, list[AblationRun]],
    baseline_name: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute baseline deltas using only seed-matched run pairs.

    Returns:
        pairwise_df: one record per (ablation run, baseline run, target, seed)
        summary_df: aggregated mean/std deltas per (ablation, target)
    """
    baseline_runs = ablation_runs.get(baseline_name, [])
    if not baseline_runs:
        logger.warning(
            "Cannot compute seed-matched deltas: baseline '%s' not found",
            baseline_name,
        )
        return pd.DataFrame(), pd.DataFrame()

    baseline_by_seed: dict[int, AblationRun] = {}
    for run in baseline_runs:
        if run.seed is None:
            continue
        if run.seed in baseline_by_seed:
            logger.warning(
                "Multiple baseline runs for seed=%s; keeping %s/%s",
                run.seed,
                baseline_by_seed[run.seed].experiment_dir.name,
                baseline_by_seed[run.seed].run_dir.name,
            )
            continue
        baseline_by_seed[run.seed] = run

    if not baseline_by_seed:
        logger.warning("No baseline runs with extractable seed found")
        return pd.DataFrame(), pd.DataFrame()

    pairwise_records: list[dict[str, Any]] = []
    for ablation, runs in sorted(ablation_runs.items()):
        if ablation == baseline_name:
            continue

        for run in runs:
            if run.seed is None:
                continue
            baseline_run = baseline_by_seed.get(run.seed)
            if baseline_run is None:
                continue

            baseline_df = load_test_metrics(baseline_run.run_dir)
            ablation_df = load_test_metrics(run.run_dir)
            if baseline_df is None or ablation_df is None:
                continue

            common_targets = sorted(
                set(baseline_df["target"]).intersection(set(ablation_df["target"]))
            )
            for target in common_targets:
                baseline_row_df = baseline_df[baseline_df["target"] == target]
                ablation_row_df = ablation_df[ablation_df["target"] == target]
                if baseline_row_df.empty or ablation_row_df.empty:
                    continue

                baseline_row = baseline_row_df.iloc[0]
                ablation_row = ablation_row_df.iloc[0]

                record: dict[str, Any] = {
                    "model": ablation,
                    "target": target,
                    "seed": run.seed,
                    "baseline_run": f"{baseline_run.experiment_dir.name}/{baseline_run.run_dir.name}",
                    "ablation_run": f"{run.experiment_dir.name}/{run.run_dir.name}",
                }

                for metric in METRICS:
                    if metric not in baseline_row or metric not in ablation_row:
                        continue

                    baseline_val = baseline_row[metric]
                    ablation_val = ablation_row[metric]
                    if pd.isna(baseline_val) or pd.isna(ablation_val):
                        continue

                    clean_metric = metric.replace("_median", "")
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

                pairwise_records.append(record)

    pairwise_df = pd.DataFrame(pairwise_records)
    if pairwise_df.empty:
        return pairwise_df, pd.DataFrame()

    summary_records: list[dict[str, Any]] = []
    grouped = pairwise_df.groupby(["model", "target"], sort=True)
    for (model, target), group in grouped:
        record: dict[str, Any] = {
            "model": model,
            "target": target,
            "paired_seeds": int(group["seed"].nunique()),
            "pairs": int(len(group)),
        }

        for metric in METRICS:
            clean_metric = metric.replace("_median", "")
            for suffix in ["delta_abs", "delta_pct"]:
                col = f"{clean_metric}_{suffix}"
                if col not in group:
                    continue
                values = group[col].dropna()
                if values.empty:
                    continue
                record[f"{clean_metric}_{suffix}_mean"] = float(values.mean())
                record[f"{clean_metric}_{suffix}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )

        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    return pairwise_df, summary_df


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
        "--include-ablations",
        type=str,
        default=None,
        help=(
            "Comma-separated candidate ablations to include in the analysis. "
            "Baseline is always included. Defaults to all discovered ablations."
        ),
    )
    parser.add_argument(
        "--assert-consistent-config",
        dest="assert_consistent_config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert run config consistency per ablation (default: enabled)",
    )
    parser.add_argument(
        "--seed-coverage",
        choices=("off", "warn", "strict"),
        default="warn",
        help=(
            "Seed coverage policy for metric-bearing runs. 'strict' fails before "
            "aggregation when selected ablations do not have exactly one run per "
            "expected seed (default: warn)."
        ),
    )
    parser.add_argument(
        "--seed-coverage-ablations",
        type=str,
        default=None,
        help=(
            "Comma-separated ablations to include in seed coverage validation. "
            "Baseline is always included. Defaults to all discovered ablations."
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated expected seeds for seed coverage validation "
            "(for example: 42,43,44,45,46). Defaults to the union of selected "
            "metric-bearing seeds."
        ),
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

    # Determine default output directory based on campaign_id
    if (
        args.output_dir == Path("outputs/reports/ablation_analysis")
        and args.campaign_id
    ):
        args.output_dir = Path("outputs/reports") / f"ablation_cv_{args.campaign_id}"

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

    include_ablations = _parse_csv_set(args.include_ablations)
    if include_ablations is not None:
        include_ablations.add(args.baseline)
        excluded_ablations = sorted(set(ablation_runs) - include_ablations)
        ablation_runs = {
            ablation: runs
            for ablation, runs in ablation_runs.items()
            if ablation in include_ablations
        }
        logger.info(
            "Filtered analysis to %d ablations: %s",
            len(ablation_runs),
            ", ".join(sorted(ablation_runs)),
        )
        if excluded_ablations:
            logger.info(
                "Excluded ablations from focused analysis: %s",
                ", ".join(excluded_ablations),
            )

    seed_coverage_df = build_seed_coverage_report(ablation_runs)
    seed_coverage_path = args.output_dir / "seed_coverage_report.csv"
    seed_coverage_df.to_csv(seed_coverage_path, index=False)
    logger.info("Saved seed coverage report to %s", seed_coverage_path)

    run_context_df = build_run_context_report(ablation_runs)
    run_context_path = args.output_dir / "ablation_run_context.csv"
    run_context_df.to_csv(run_context_path, index=False)
    logger.info("Saved ablation run context to %s", run_context_path)

    neighborhood_context_df = build_neighborhood_context_report(run_context_df)
    neighborhood_context_path = args.output_dir / "neighborhood_aggregation_context.csv"
    if not neighborhood_context_df.empty:
        neighborhood_context_df.to_csv(neighborhood_context_path, index=False)
        logger.info(
            "Saved neighborhood aggregation context to %s",
            neighborhood_context_path,
        )

    if args.seed_coverage != "off":
        try:
            validate_seed_coverage(
                ablation_runs,
                baseline_name=args.baseline,
                included_ablations=_parse_csv_set(args.seed_coverage_ablations),
                expected_seeds=_parse_csv_set(args.expected_seeds, cast_type=int),
            )
            logger.info("Seed coverage validation passed")
        except ValueError as exc:
            if args.seed_coverage == "strict":
                logger.error("%s", exc)
                return 1
            logger.warning("%s", exc)

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

    logger.info("Computing seed-matched deltas from baseline...")
    seed_pairwise_df, seed_matched_df = compute_seed_matched_baseline_deltas(
        ablation_runs,
        baseline_name=args.baseline,
    )
    if not seed_pairwise_df.empty:
        seed_pairwise_path = (
            args.output_dir / "ablation_metrics_deltas_seed_pairwise.csv"
        )
        seed_pairwise_df.to_csv(seed_pairwise_path, index=False)
        logger.info(f"Saved seed-paired deltas to {seed_pairwise_path}")

    if not seed_matched_df.empty:
        seed_matched_path = args.output_dir / "ablation_metrics_deltas_seed_matched.csv"
        seed_matched_df.to_csv(seed_matched_path, index=False)
        logger.info(f"Saved seed-matched summary deltas to {seed_matched_path}")

    # Cross-head impact analysis
    logger.info("Computing cross-head impact analysis...")
    try:
        from scripts.analysis.analyze_cross_head_impact import compute_cross_head_impact

        cross_head_result = compute_cross_head_impact(
            args.training_dir,
            args.campaign_id,
            split="test",
        )

        if cross_head_result is not None:
            pairwise_df, mean_matrix, (std_matrix, count_matrix) = cross_head_result

            # Save cross-head matrices
            cross_head_dir = args.output_dir / "cross_head"
            cross_head_dir.mkdir(exist_ok=True)

            pairwise_path = cross_head_dir / "cross_head_pairwise.csv"
            pairwise_df.to_csv(pairwise_path, index=False)

            mean_matrix.to_csv(cross_head_dir / "cross_head_mean_matrix.csv")
            std_matrix.to_csv(cross_head_dir / "cross_head_std_matrix.csv")
            count_matrix.to_csv(cross_head_dir / "cross_head_count_matrix.csv")

            logger.info(f"Saved cross-head analysis to {cross_head_dir}")

            cross_head_success = True
        else:
            logger.warning("Cross-head impact analysis returned no data")
            cross_head_success = False
    except Exception as exc:
        logger.warning(f"Cross-head impact analysis failed: {exc}")
        cross_head_success = False

    # Granular Comparison Suite
    if args.campaign_id:
        logger.info("Running granular comparison suite...")
        try:
            from dataviz.granular_comparison import compare_ablation_suite

            candidate_ablations = [
                m for m in aggregated_df["model"].unique() if m != args.baseline
            ]

            granular_dir = args.output_dir / "granular_comparison"
            compare_ablation_suite(
                training_dir=args.training_dir,
                campaign_id=args.campaign_id,
                baseline_ablation=args.baseline,
                candidate_ablations=candidate_ablations,
                output_dir=granular_dir,
                split="test",
            )
            logger.info(f"Saved granular comparison suite to {granular_dir}")
        except Exception as exc:
            logger.warning(f"Granular comparison suite failed: {exc}")

    logger.info("Generating ablation plots...")
    try:
        from dataviz.ablation_plots import (
            plot_ablation_comparison,
            plot_ablation_deltas_heatmap,
            plot_ablation_summary_grid,
            plot_cross_head_impact_heatmap,
            plot_head_ablation_heatmap,
            plot_mobility_ablation_heatmap,
            plot_neighborhood_aggregation_heatmap,
            plot_seed_matched_delta_diagnostics,
        )

        plot_ablation_comparison(
            aggregated_path,
            output_dir=args.output_dir,
            baseline_name=args.baseline,
        )
        plot_ablation_summary_grid(
            aggregated_path,
            deltas_csv=deltas_path if not deltas_df.empty else None,
            output_dir=args.output_dir,
            baseline_name=args.baseline,
        )
        heatmap_df = seed_matched_df if not seed_matched_df.empty else deltas_df
        heatmap_csv = (
            seed_matched_path
            if not seed_matched_df.empty and seed_matched_path.exists()
            else deltas_path
        )
        if not heatmap_df.empty:
            for metric in ["mae", "rmse", "smape", "r2"]:
                try:
                    plot_ablation_deltas_heatmap(
                        heatmap_csv,
                        output_dir=args.output_dir,
                        metric=metric,
                        baseline_name=args.baseline,
                    )
                except ValueError as exc:
                    logger.warning("Skipping %s delta heatmap: %s", metric, exc)
            plot_neighborhood_aggregation_heatmap(
                heatmap_csv,
                output_dir=args.output_dir,
                metric="mae",
                baseline_name=args.baseline,
            )
            plot_mobility_ablation_heatmap(
                heatmap_csv,
                output_dir=args.output_dir,
                metric="mae",
                baseline_name=args.baseline,
            )
            try:
                plot_head_ablation_heatmap(
                    heatmap_csv,
                    output_dir=args.output_dir,
                    metric="mae",
                    baseline_name=args.baseline,
                )
            except ValueError as exc:
                logger.warning("Skipping head ablation heatmap: %s", exc)
        if not seed_matched_df.empty and not seed_pairwise_df.empty:
            plot_seed_matched_delta_diagnostics(
                seed_matched_path,
                seed_pairwise_path,
                output_dir=args.output_dir / "seed_matched_diagnostics",
                baseline_name=args.baseline,
            )
        if cross_head_success:
            plot_cross_head_impact_heatmap(
                cross_head_dir / "cross_head_mean_matrix.csv",
                std_csv=cross_head_dir / "cross_head_std_matrix.csv",
                output_dir=cross_head_dir,
            )
    except Exception as exc:
        logger.warning(f"Ablation plot generation failed: {exc}")

    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"\nAggregated metrics for {len(aggregated_df['model'].unique())} ablations")
    print(f"Targets: {', '.join(sorted(aggregated_df['target'].unique()))}")
    print("\nMetrics: MAE, RMSE, sMAPE, R²")
    print(f"\nOutput directory: {args.output_dir}")
    print("\nFiles generated:")
    print(f"  - {aggregated_path.name}")
    print(f"  - {seed_coverage_path.name}")
    print(f"  - {run_context_path.name}")
    if not neighborhood_context_df.empty:
        print(f"  - {neighborhood_context_path.name}")
    if not deltas_df.empty:
        print(f"  - {deltas_path.name}")
    if not seed_pairwise_df.empty:
        print(f"  - {seed_pairwise_path.name}")
    if not seed_matched_df.empty:
        print(f"  - {seed_matched_path.name}")
    if cross_head_success:
        print("  - cross_head/")
        print("    - cross_head_mean_matrix.csv")
        print("    - cross_head_pairwise.csv")

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

    if not seed_matched_df.empty:
        print("\n" + "-" * 80)
        print("Seed-matched deltas from baseline (MAE % change, mean across pairs):")
        print("-" * 80)
        pivot_seed = seed_matched_df.pivot_table(
            index="model",
            columns="target",
            values="mae_delta_pct_mean",
        )
        print(pivot_seed.to_string())

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
