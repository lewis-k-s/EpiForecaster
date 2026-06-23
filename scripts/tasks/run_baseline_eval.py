#!/usr/bin/env python3
"""Run same-slice baseline evaluation in production-style environments.

This wrapper is intentionally thin:
- Loads a training config (also used as the evaluation config)
- Applies optional data-path overrides
- Runs same-slice baseline evaluation
- Prints produced artifacts

All baselines are evaluated on the same target slices/masks as the neural model
(``same_eval_slice`` comparison scope) and fall back to last-observed on fit
failure so every method is scored on common support (see
``baseline_metadata.json`` and ``primary_success_rate_mean`` in the aggregate
metrics for the disclosed fallback rate).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.baseline_eval import run_baseline_evaluation
from models.configs import EpiForecasterConfig

# Reported/default comparison: stable univariate references plus covariate-aware
# SARIMAX/VARMAX. Covariate-free VAR remains available only when explicitly
# requested through the lower-level evaluation API.
_DEFAULT_MODELS = [
    "exp_smoothing",
    "last_observed",
    "sarima",
    "sarimax",
    "varmax",
]
_ALL_ALIAS = "all"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EpiForecaster same-slice baseline evaluation.",
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
        nargs="+",
        default=["sarima"],
        help=(
            "Baseline model families to run (e.g. sarima sarimax varmax "
            "exp_smoothing last_observed, or 'all')."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=7,
        help="Seasonal period for seasonal-naive / SARIMA-family baselines.",
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


def _resolve_models(models: list[str]) -> list[str]:
    if any(m.lower() == _ALL_ALIAS for m in models):
        return list(_DEFAULT_MODELS)
    return models


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("run_baseline_eval")

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if args.seasonal_period <= 0:
        raise ValueError("--seasonal-period must be > 0")

    cfg = EpiForecasterConfig.load(args.config)
    if args.dataset_path is not None:
        cfg.data.dataset_path = str(args.dataset_path)
    if args.region2vec_path is not None:
        cfg.data.region2vec_path = str(args.region2vec_path)
    if args.run_id is not None:
        cfg.data.run_id = args.run_id

    selected_models = _resolve_models(args.models)

    logger.info("Running baseline eval")
    logger.info("  config: %s", args.config)
    logger.info("  models: %s", ",".join(selected_models))
    logger.info("  split: %s", args.split)
    logger.info("  output_dir: %s", args.output_dir)
    logger.info("  dataset_path: %s", cfg.data.dataset_path)
    logger.info("  run_id: %s", cfg.data.run_id)

    artifacts = run_baseline_evaluation(
        config=cfg,
        config_path=str(args.config),
        output_dir=args.output_dir,
        models=selected_models,
        split=args.split,
        seasonal_period=args.seasonal_period,
    )

    print("Baseline evaluation completed.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
