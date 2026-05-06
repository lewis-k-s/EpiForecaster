#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the real-data EpiForecaster dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from evaluation.aggregate_export import write_main_model_aggregate_csvs
from evaluation.eval_loop import eval_checkpoint
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the real dataset without training."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--real-dataset-path",
        default="data/processed/real_with_id.zarr",
        help="Real-data Zarr dataset path to evaluate against.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["val", "test"],
        default=["val", "test"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prefetch-factor", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--granular",
        action="store_true",
        help="Write per-example granular CSVs in addition to aggregate metrics.",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        default=[],
        help="Additional dotted config override. Can be repeated.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"data.dataset_path={args.real_dataset_path}",
        "training.curriculum.enabled=false",
        "output.write_granular_eval=false",
        *args.overrides,
    ]

    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "output_dir": output_dir,
        "real_dataset_path": args.real_dataset_path,
        "splits": args.splits,
        "overrides": overrides,
        "results": {},
    }

    for split in args.splits:
        logger.info(
            "Evaluating checkpoint on real data: checkpoint=%s split=%s output_dir=%s",
            args.checkpoint,
            split,
            output_dir,
        )
        result = eval_checkpoint(
            checkpoint_path=args.checkpoint,
            split=split,
            device=args.device,
            overrides=overrides,
            node_metrics_csv_path=output_dir / f"{split}_node_metrics.csv",
            per_head_node_metrics_csv_path=output_dir
            / f"{split}_node_metrics_per_head.csv",
            granular_csv_path=output_dir / f"{split}_granular_metrics.csv"
            if args.granular
            else None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            log_every=args.log_every,
        )
        aggregate_paths = write_main_model_aggregate_csvs(
            run_dir=output_dir,
            split=split,
            eval_metrics=result["eval_metrics"],
        )
        summary["results"][split] = {
            "eval_loss": result["eval_loss"],
            "eval_metrics": result["eval_metrics"],
            "aggregate_paths": aggregate_paths,
        }
        logger.info(
            "Completed real-data checkpoint eval: split=%s loss=%.6f",
            split,
            float(result["eval_loss"]),
        )

    summary_path = output_dir / "real_eval_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Wrote real-data eval summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
