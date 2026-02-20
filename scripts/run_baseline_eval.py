#!/usr/bin/env python3
"""Run baseline evaluation in production-style environments.

This wrapper is intentionally thin:
- Loads a training config
- Applies optional data-path overrides
- Runs baseline evaluation
- Prints produced artifacts
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.baseline_eval import run_tiered_baseline_evaluation
from models.configs import EpiForecasterConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EpiForecaster baseline evaluation (rolling-origin folds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/production_only/train_epifor_mn5_full.yaml"),
        help="Path to training config used as evaluation baseline config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where baseline artifacts will be written.",
    )
    parser.add_argument(
        "--models",
        choices=["tiered"],
        default="tiered",
        help="Baseline model family.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--rolling-folds",
        type=int,
        default=5,
        help="Number of expanding rolling-origin folds.",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=7,
        help="Seasonal period for seasonal-naive / SARIMA family.",
    )
    parser.add_argument(
        "--disable-sparsity-bins",
        action="store_true",
        help="Disable sparsity-decile stratified reporting.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional override for data.dataset_path (useful in production mounts).",
    )
    parser.add_argument(
        "--region2vec-path",
        type=Path,
        default=None,
        help="Optional override for data.region2vec_path.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional override for data.run_id.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("run_baseline_eval")

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if args.rolling_folds <= 0:
        raise ValueError("--rolling-folds must be > 0")
    if args.seasonal_period <= 0:
        raise ValueError("--seasonal-period must be > 0")

    cfg = EpiForecasterConfig.load(args.config)
    if args.dataset_path is not None:
        cfg.data.dataset_path = str(args.dataset_path)
    if args.region2vec_path is not None:
        cfg.data.region2vec_path = str(args.region2vec_path)
    if args.run_id is not None:
        cfg.data.run_id = args.run_id

    logger.info("Running baseline eval")
    logger.info("  config: %s", args.config)
    logger.info("  split: %s", args.split)
    logger.info("  rolling_folds: %d", args.rolling_folds)
    logger.info("  output_dir: %s", args.output_dir)
    logger.info("  dataset_path: %s", cfg.data.dataset_path)
    logger.info("  run_id: %s", cfg.data.run_id)

    artifacts = run_tiered_baseline_evaluation(
        config=cfg,
        config_path=str(args.config),
        output_dir=args.output_dir,
        split=args.split,
        rolling_folds=args.rolling_folds,
        seasonal_period=args.seasonal_period,
        include_sparsity_bins=not args.disable_sparsity_bins,
    )

    print("Baseline evaluation completed.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
