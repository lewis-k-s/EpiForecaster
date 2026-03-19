#!/usr/bin/env python3
"""Aggregate repeated-seed cross-validation runs for one campaign."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dataviz.granular_crossval import CrossvalGranularRun, analyze_crossval_granular
from scripts.analyze_ablations import (
    build_config_fingerprint,
    get_run_seed,
    load_run_config,
)

logger = logging.getLogger(__name__)

EXPERIMENT_PREFIX = "crossval__"


@dataclass(frozen=True)
class CrossvalRun:
    campaign_id: str
    experiment_dir: Path
    run_dir: Path
    seed: int | None
    fold: int


def parse_crossval_experiment_name(experiment_name: str) -> str | None:
    if not experiment_name.startswith(EXPERIMENT_PREFIX):
        return None
    campaign_id = experiment_name[len(EXPERIMENT_PREFIX) :]
    return campaign_id or None


def collect_crossval_runs(
    training_dir: Path,
    campaign_id: str,
) -> tuple[list[CrossvalRun], set[str]]:
    runs: list[CrossvalRun] = []
    campaigns_found: set[str] = set()

    if not training_dir.exists():
        logger.warning("Training directory not found: %s", training_dir)
        return runs, campaigns_found

    experiment_dir = training_dir / f"{EXPERIMENT_PREFIX}{campaign_id}"
    if not experiment_dir.exists():
        return runs, campaigns_found

    parsed_campaign = parse_crossval_experiment_name(experiment_dir.name)
    if parsed_campaign is None or parsed_campaign != campaign_id:
        return runs, campaigns_found
    campaigns_found.add(parsed_campaign)

    unsorted_runs: list[tuple[Path, int | None]] = []
    for run_dir in sorted(d for d in experiment_dir.iterdir() if d.is_dir()):
        seed = get_run_seed(run_dir)
        unsorted_runs.append((run_dir, seed))

    unsorted_runs.sort(
        key=lambda item: (
            item[1] is None,
            item[1] if item[1] is not None else 10**18,
            item[0].name,
        )
    )

    for fold, (run_dir, seed) in enumerate(unsorted_runs):
        runs.append(
            CrossvalRun(
                campaign_id=campaign_id,
                experiment_dir=experiment_dir,
                run_dir=run_dir,
                seed=seed,
                fold=fold,
            )
        )

    return runs, campaigns_found


def validate_crossval_run_consistency(
    runs: list[CrossvalRun],
    *,
    assert_consistent_config: bool,
) -> tuple[dict[str, Any], str]:
    if not runs:
        raise ValueError("No cross-val runs found")

    reference_config: dict[str, Any] | None = None
    reference_fingerprint: str | None = None
    epochs_by_run: dict[str, str] = {}
    split_by_run: dict[str, str] = {}
    fingerprint_by_run: dict[str, str] = {}

    for run in runs:
        run_key = f"{run.experiment_dir.name}/{run.run_dir.name}"
        config = load_run_config(run.run_dir)
        fingerprint = build_config_fingerprint(config)
        epochs_by_run[run_key] = str(config.get("training", {}).get("epochs"))
        split_by_run[run_key] = str(config.get("training", {}).get("split_strategy"))
        fingerprint_by_run[run_key] = fingerprint
        if reference_config is None:
            reference_config = config
            reference_fingerprint = fingerprint

    assert reference_config is not None
    assert reference_fingerprint is not None

    if not assert_consistent_config:
        return reference_config, reference_fingerprint

    errors: list[str] = []
    if len(set(epochs_by_run.values())) > 1:
        details = ", ".join(f"{k}: {v}" for k, v in sorted(epochs_by_run.items()))
        errors.append(f"training.epochs mismatch -> {details}")
    if len(set(split_by_run.values())) > 1:
        details = ", ".join(f"{k}: {v}" for k, v in sorted(split_by_run.items()))
        errors.append(f"training.split_strategy mismatch -> {details}")
    if len(set(fingerprint_by_run.values())) > 1:
        errors.append(
            "config fingerprint mismatch after excluding "
            "training.seed/output.wandb_group/output.wandb_tags/output.experiment_name"
        )

    if errors:
        message = "\n".join(f"- {item}" for item in errors)
        raise ValueError(f"Inconsistent cross-val run config:\n{message}")

    return reference_config, reference_fingerprint


def _std_or_zero(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _load_target_metrics(run_dir: Path, split: str) -> pd.DataFrame:
    target_path = run_dir / f"{split}_main_model_aggregate_metrics.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target metrics CSV: {target_path}")
    return pd.read_csv(target_path)


def _load_joint_metrics(run_dir: Path, split: str) -> pd.DataFrame:
    joint_path = run_dir / f"{split}_main_model_joint_loss_aggregate.csv"
    if not joint_path.exists():
        raise FileNotFoundError(f"Missing joint metrics CSV: {joint_path}")
    return pd.read_csv(joint_path)


def build_target_fold_metrics(runs: list[CrossvalRun], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = _load_target_metrics(run.run_dir, split)
        for row in df.itertuples(index=False):
            rows.append(
                {
                    "target": row.target,
                    "fold": run.fold,
                    "seed": run.seed,
                    "run_dir": str(run.run_dir),
                    "mae": row.mae_median,
                    "rmse": row.rmse_median,
                    "smape": row.smape_median,
                    "r2": row.r2_median,
                    "observed_count": row.observed_count_median,
                }
            )
    return pd.DataFrame(rows)


def aggregate_target_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, group in fold_df.groupby("target", sort=True):
        rows.append(
            {
                "target": target,
                "folds": int(group["fold"].nunique()),
                "mae_mean": float(group["mae"].mean()),
                "mae_std": _std_or_zero(group["mae"]),
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_std": _std_or_zero(group["rmse"]),
                "smape_mean": float(group["smape"].mean()),
                "smape_std": _std_or_zero(group["smape"]),
                "r2_mean": float(group["r2"].mean()),
                "r2_std": _std_or_zero(group["r2"]),
                "observed_count_mean": float(group["observed_count"].mean()),
                "observed_count_std": _std_or_zero(group["observed_count"]),
            }
        )
    return pd.DataFrame(rows)


def build_joint_fold_metrics(runs: list[CrossvalRun], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        df = _load_joint_metrics(run.run_dir, split)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        record = {
            "fold": run.fold,
            "seed": run.seed,
            "run_dir": str(run.run_dir),
        }
        for key, value in row.items():
            if key in {"model", "folds"}:
                continue
            if key.endswith("_median"):
                record[key.removesuffix("_median")] = value
        rows.append(record)
    return pd.DataFrame(rows)


def aggregate_joint_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame(columns=["folds"])
    metric_cols = [col for col in fold_df.columns if col not in {"fold", "seed", "run_dir"}]
    record: dict[str, Any] = {"folds": int(fold_df["fold"].nunique())}
    for col in metric_cols:
        record[f"{col}_mean"] = float(fold_df[col].mean())
        record[f"{col}_std"] = _std_or_zero(fold_df[col])
    return pd.DataFrame([record])


def analyze_crossval_campaign(
    *,
    training_dir: Path,
    campaign_id: str,
    split: str,
    output_dir: Path,
    assert_consistent_config: bool,
) -> dict[str, Any]:
    split_key = split.lower()
    if split_key not in {"val", "test"}:
        raise ValueError("split must be either 'val' or 'test'")

    runs, campaigns_found = collect_crossval_runs(training_dir, campaign_id)
    if not runs:
        raise ValueError(f"No cross-val runs found for campaign '{campaign_id}'")
    if campaign_id not in campaigns_found:
        raise ValueError(f"Campaign '{campaign_id}' not found under {training_dir}")

    reference_config, fingerprint = validate_crossval_run_consistency(
        runs,
        assert_consistent_config=assert_consistent_config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    target_fold_df = build_target_fold_metrics(runs, split_key)
    target_aggregate_df = aggregate_target_fold_metrics(target_fold_df)
    target_fold_path = output_dir / f"{split_key}_crossval_fold_metrics.csv"
    target_aggregate_path = output_dir / f"{split_key}_crossval_aggregate_metrics.csv"
    target_fold_df.to_csv(target_fold_path, index=False)
    target_aggregate_df.to_csv(target_aggregate_path, index=False)

    joint_fold_df = build_joint_fold_metrics(runs, split_key)
    joint_aggregate_df = aggregate_joint_fold_metrics(joint_fold_df)
    joint_fold_path = output_dir / f"{split_key}_crossval_joint_loss_fold.csv"
    joint_aggregate_path = output_dir / f"{split_key}_crossval_joint_loss_aggregate.csv"
    joint_fold_df.to_csv(joint_fold_path, index=False)
    joint_aggregate_df.to_csv(joint_aggregate_path, index=False)

    granular_runs = [
        CrossvalGranularRun(
            fold=run.fold,
            seed=run.seed,
            run_dir=run.run_dir,
            granular_csv_path=run.run_dir / f"{split_key}_granular_metrics.csv",
        )
        for run in runs
    ]
    granular_artifacts = analyze_crossval_granular(
        runs=granular_runs,
        split=split_key,
        output_dir=output_dir,
    )

    metadata = {
        "campaign_id": campaign_id,
        "config_fingerprint": fingerprint,
        "config_path": str(runs[0].run_dir / "config.yaml"),
        "experiment_dir": str(runs[0].experiment_dir),
        "fold_count": len(runs),
        "run_dirs": [str(run.run_dir) for run in runs],
        "seeds": [run.seed for run in runs],
        "split": split_key,
        "training_dir": str(training_dir),
        "training_split_strategy": reference_config.get("training", {}).get(
            "split_strategy"
        ),
    }
    metadata_path = output_dir / "crossval_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return {
        "target_fold_metrics": target_fold_path,
        "target_aggregate_metrics": target_aggregate_path,
        "joint_fold_metrics": joint_fold_path,
        "joint_aggregate_metrics": joint_aggregate_path,
        "crossval_metadata": metadata_path,
        "granular": granular_artifacts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated-seed cross-validation runs for one campaign."
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("outputs/training"),
    )
    parser.add_argument(
        "--campaign-id",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--assert-consistent-config",
        dest="assert_consistent_config",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("outputs/reports/crossval") / args.campaign_id / args.split
    )
    logger.info(
        "Starting cross-val analysis: campaign_id=%s split=%s training_dir=%s "
        "output_dir=%s assert_consistent_config=%s verbose=%s",
        args.campaign_id,
        args.split,
        args.training_dir,
        output_dir,
        args.assert_consistent_config,
        args.verbose,
    )

    artifacts = analyze_crossval_campaign(
        training_dir=args.training_dir,
        campaign_id=args.campaign_id,
        split=args.split,
        output_dir=output_dir,
        assert_consistent_config=args.assert_consistent_config,
    )
    logger.info("Cross-val analysis complete: %s", output_dir)
    for name, artifact in artifacts.items():
        logger.info("%s: %s", name, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
