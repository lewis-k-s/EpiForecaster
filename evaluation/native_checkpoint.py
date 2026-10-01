"""Evaluate a saved native checkpoint without starting a training run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from models.configs import EpiForecasterConfig
from evaluation.run_report_evidence import export_native_report_examples, report_context
from training.native_trainer import NativeRuntime, forecaster_config_to_dict


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def evaluate_native_checkpoint(
    *,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
    overrides: list[str],
    granular: bool,
    summary_path: Path,
) -> dict[str, Any]:
    """Rebuild the saved native split and evaluate its selected model state."""
    if split not in {"val", "test"}:
        raise ValueError("Native checkpoint evaluation requires split 'val' or 'test'")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "forecaster_config" not in checkpoint:
        raise ValueError("Checkpoint is not a native forecaster checkpoint")
    config = EpiForecasterConfig.from_dict(checkpoint["config"])
    if overrides:
        config = EpiForecasterConfig.apply_overrides(config, overrides)
    config.output.granular_eval_format = "parquet"
    if checkpoint.get("split_asset") is not None:
        asset_dir = output_dir / ".inputs/native_source_sets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        (asset_dir / "manifest.json").write_bytes(checkpoint["split_asset"]["manifest"])
        (asset_dir / "assignments.parquet").write_bytes(
            checkpoint["split_asset"]["assignments"]
        )
        config.native.fold_manifest = str(asset_dir / "manifest.json")
    trainer = NativeRuntime(config, model_id=checkpoint["model_id"])
    if (
        forecaster_config_to_dict(trainer.forecaster_config)
        != checkpoint["forecaster_config"]
    ):
        raise ValueError(
            "Native evaluation model configuration differs from checkpoint"
        )
    if trainer.split_provenance != checkpoint["split_provenance"]:
        raise ValueError(
            "Native evaluation inputs differ from checkpoint split provenance"
        )
    trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = trainer.val_loader if split == "val" else trainer.test_loader
    if loader is None:
        raise ValueError(f"Native checkpoint has no {split} split")
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_loss, metrics = trainer._evaluate(
        loader=loader,
        split_name=split.capitalize(),
        granular_output_path=(
            output_dir / f"{split}_granular.parquet" if granular else None
        ),
        aggregate_output_dir=output_dir,
    )
    summary = _json_value(
        {
            "split": split,
            "checkpoint": str(checkpoint_path),
            "objective_loss": objective_loss,
            "primary_metric": "joint_wis",
            "joint_wis": metrics["joint_wis"],
            "metrics": metrics,
            "report_context": {
                **report_context(config),
                "selected_epoch": checkpoint["epoch"] + 1,
                "selected_step": checkpoint["global_step"],
            },
        }
    )
    if granular:
        export_native_report_examples(config, output_dir, split)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
