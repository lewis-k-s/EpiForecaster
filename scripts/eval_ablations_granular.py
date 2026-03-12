#!/usr/bin/env python3
"""Re-evaluate ablation checkpoints with granular per-example output.

NOTE: Run on the same cluster where ablations were trained (e.g., MN5).
The checkpoints depend on the production dataset which is not available locally.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.eval_loop import eval_checkpoint

logger = logging.getLogger(__name__)

# Default overrides for MN5 ablations trained before explicit temporal_covariates config
DEFAULT_OVERRIDES = [
    "model.include_day_of_week=true",
    "model.include_holidays=true",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Granular eval for ablation checkpoints")
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("outputs/training"),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="mn5_ablation__*__*",
        help="Glob pattern for ablation experiment dirs",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        help="Additional config overrides (can repeat)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.training_dir.exists():
        logger.error("Training dir not found: %s", args.training_dir)
        return 1

    # Find all ablation experiments
    experiments = sorted(args.training_dir.glob(args.pattern))
    if not experiments:
        logger.error("No experiments matching pattern: %s", args.pattern)
        return 1

    # Merge default overrides with user overrides
    overrides = list(DEFAULT_OVERRIDES)
    if args.overrides:
        overrides.extend(args.overrides)

    logger.info("Found %d experiments", len(experiments))
    logger.info("Using overrides: %s", overrides)

    for exp_dir in experiments:
        for run_dir in sorted(exp_dir.glob("*")):
            if not run_dir.is_dir():
                continue
            checkpoint = run_dir / "checkpoints" / "best_model.pt"
            if not checkpoint.exists():
                logger.warning("No checkpoint: %s", checkpoint)
                continue

            granular_csv = run_dir / f"{args.split}_granular_metrics.csv"
            if granular_csv.exists():
                logger.info("Skipping (exists): %s", granular_csv)
                continue

            logger.info("Evaluating: %s", checkpoint)
            try:
                result = eval_checkpoint(
                    checkpoint_path=checkpoint,
                    split=args.split,
                    device=args.device,
                    granular_csv_path=granular_csv,
                    overrides=overrides,
                )
                logger.info(
                    "Done: %s | loss=%.4f | mae=%.4f",
                    run_dir.name,
                    result["eval_loss"],
                    result["eval_metrics"].get("mae", float("nan")),
                )
            except Exception:
                logger.exception("Failed: %s", checkpoint)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
