#!/usr/bin/env python3
"""Explicit post-train granular evaluation for cross-val runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from evaluation.aggregate_export import write_main_model_aggregate_csvs
from evaluation.eval_loop import eval_checkpoint

logger = logging.getLogger(__name__)


def _iter_run_dirs(*, run_dir: Path | None, experiment_dir: Path | None) -> list[Path]:
    if run_dir is not None:
        return [run_dir]
    assert experiment_dir is not None
    return sorted(path for path in experiment_dir.iterdir() if path.is_dir())


def _has_prefetch_override(overrides: list[str] | None) -> bool:
    if not overrides:
        return False
    return any(override.startswith("training.prefetch_factor=") for override in overrides)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate cross-val checkpoints with granular output."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--run-dir", type=Path, default=None)
    target_group.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["val", "test"],
        default=["val", "test"],
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--memory-log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--disable-prefetch",
        action="store_true",
        help="Disable DataLoader/CUDA prefetching. This script does this by default unless training.prefetch_factor is explicitly overridden.",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override pin-memory for the eval loader. Defaults to disabled for this script.",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        help="Additional config overrides (can repeat)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run eval even if granular CSV already exists.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_dirs = _iter_run_dirs(run_dir=args.run_dir, experiment_dir=args.experiment_dir)
    if not run_dirs:
        logger.error("No run directories found")
        return 1
    overrides = list(args.overrides) if args.overrides else []
    effective_num_workers = 0 if args.num_workers is None else args.num_workers
    effective_pin_memory = False if args.pin_memory is None else args.pin_memory
    effective_prefetch_factor = None
    if args.disable_prefetch or not _has_prefetch_override(overrides):
        effective_prefetch_factor = 0

    logger.info(
        "Starting crossval granular eval: runs=%d splits=%s device=%s batch_size=%s "
        "log_every=%d memory_log_every=%d num_workers=%d pin_memory=%s prefetch_factor=%s overrides=%s",
        len(run_dirs),
        args.splits,
        args.device,
        args.batch_size,
        args.log_every,
        args.memory_log_every,
        effective_num_workers,
        effective_pin_memory,
        effective_prefetch_factor if effective_prefetch_factor is not None else "config",
        overrides,
    )

    for current_run_dir in run_dirs:
        checkpoint = current_run_dir / "checkpoints" / "best_model.pt"
        if not checkpoint.exists():
            logger.warning("Skipping missing checkpoint: %s", checkpoint)
            continue

        for split in args.splits:
            granular_csv = current_run_dir / f"{split}_granular_metrics.csv"
            if granular_csv.exists() and not args.force:
                logger.info("Skipping existing granular CSV: %s", granular_csv)
                continue

            logger.info(
                "Run startup: run_dir=%s split=%s checkpoint=%s granular_csv=%s device=%s "
                "batch_size=%s num_workers=%d pin_memory=%s prefetch_factor=%s",
                current_run_dir,
                split,
                checkpoint,
                granular_csv,
                args.device,
                args.batch_size,
                effective_num_workers,
                effective_pin_memory,
                effective_prefetch_factor if effective_prefetch_factor is not None else "config",
            )
            started_at = time.perf_counter()
            logger.info(
                "Evaluating run=%s split=%s checkpoint=%s",
                current_run_dir.name,
                split,
                checkpoint,
            )
            try:
                result = eval_checkpoint(
                    checkpoint_path=checkpoint,
                    split=split,
                    device=args.device,
                    granular_csv_path=granular_csv,
                    overrides=overrides,
                    batch_size=args.batch_size,
                    num_workers=effective_num_workers,
                    pin_memory=effective_pin_memory,
                    prefetch_factor=effective_prefetch_factor,
                    log_every=args.log_every,
                )
            except Exception:
                elapsed_s = time.perf_counter() - started_at
                logger.exception(
                    "Failed run=%s split=%s after %.1fs checkpoint=%s granular_csv=%s",
                    current_run_dir.name,
                    split,
                    elapsed_s,
                    checkpoint,
                    granular_csv,
                )
                raise
            write_main_model_aggregate_csvs(
                run_dir=current_run_dir,
                split=split,
                eval_metrics=result["eval_metrics"],
            )
            elapsed_s = time.perf_counter() - started_at
            logger.info(
                "Completed run=%s split=%s in %.1fs | loss=%.4f mae=%.4f",
                current_run_dir.name,
                split,
                elapsed_s,
                result["eval_loss"],
                result["eval_metrics"].get("mae", float("nan")),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
