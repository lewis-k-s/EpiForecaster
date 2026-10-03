"""Read canonical run evidence and compute operational telemetry tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.analysis.render_run_report import sha256


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_run(
    run_root: Path,
    *,
    model_root: Path | None = None,
    evaluation_dir: Path | None = None,
) -> dict:
    """Check completion and declared files once, before loading plot data.

    Missing files and recorded incomplete states produce a small status report.
    Malformed schemas and programming errors remain normal stage failures.
    """
    config = yaml.safe_load((run_root / "config.yaml").read_text())
    training = config["training"]
    model_root = (
        model_root
        or run_root
        / "outputs/training"
        / training["experiment_name"]
        / training["model_id"]
    )
    evaluation_dir = evaluation_dir or run_root / "outputs/evaluation"
    files = {
        "run_manifest": run_root / "manifest.yaml",
        "training_manifest": model_root / "training_manifest.json",
        "dynamics": model_root / "training_dynamics_manifest.json",
        "evaluation_manifest": evaluation_dir / "evaluation_manifest.json",
        "val_summary": evaluation_dir / "val_summary.json",
        "test_summary": evaluation_dir / "test_summary.json",
    }
    errors = [
        {"code": "MISSING_ARTIFACT", "artifact": str(path), "recorded_value": "absent"}
        for path in files.values()
        if not path.is_file()
    ]
    dynamics = read_json(files["dynamics"]) if files["dynamics"].is_file() else {}
    summary = dynamics.get("summary", {})
    train_rows = summary.get("train_epoch_trace", [])
    history = summary.get("training_history", {})
    completed = len(train_rows) if train_rows else len(history.get("train_loss", []))
    config_path = model_root / "config.json"
    if config_path.is_file():
        model_config = read_json(config_path)
    elif (model_root / "config.yaml").is_file():
        config_path = model_root / "config.yaml"
        model_config = yaml.safe_load(config_path.read_text())
    else:
        model_config = {}
    expected = model_config.get("training", {}).get(
        "epochs", summary.get("total_epochs")
    )
    facts = {
        "run_id": config["run"]["id"],
        "training_status": dynamics.get("status", "unavailable"),
        "completed_train_epochs": completed,
        "expected_epochs": expected,
        "epoch_budget_reached": completed >= expected if expected is not None else None,
        "global_step": summary.get("global_step", dynamics.get("last_step")),
    }
    if dynamics and dynamics["status"] != "complete":
        errors.append(
            {
                "code": "TRAINING_NOT_COMPLETE",
                "artifact": str(files["dynamics"]),
                "recorded_value": dynamics["status"],
            }
        )
    inputs = [run_root / "config.yaml", *[p for p in files.values() if p.is_file()]]
    if model_config:
        inputs.append(config_path)
    provenance = model_root / "training_provenance.json"
    if provenance.is_file():
        inputs.append(provenance)
    if files["evaluation_manifest"].is_file():
        evaluation = read_json(files["evaluation_manifest"])
        for item in evaluation["artifacts"]:
            path = evaluation_dir / item["path"]
            if not path.is_file():
                if item["required"]:
                    errors.append(
                        {
                            "code": "MISSING_ARTIFACT",
                            "artifact": str(path),
                            "recorded_value": "absent",
                        }
                    )
            elif sha256(path) != item["sha256"]:
                errors.append(
                    {
                        "code": "HASH_MISMATCH",
                        "artifact": str(path),
                        "recorded_value": item["sha256"],
                    }
                )
            else:
                inputs.append(path)
        if files["training_manifest"].is_file():
            recorded = read_json(files["training_manifest"])["checkpoint_sha256"]
            if evaluation["checkpoint_sha256"] != recorded:
                errors.append(
                    {
                        "code": "CHECKPOINT_MISMATCH",
                        "artifact": str(files["evaluation_manifest"]),
                        "recorded_value": evaluation["checkpoint_sha256"],
                    }
                )
    if dynamics:
        path = model_root / dynamics["artifact"]
        if not path.is_file():
            errors.append(
                {
                    "code": "MISSING_ARTIFACT",
                    "artifact": str(path),
                    "recorded_value": "absent",
                }
            )
        elif sha256(path) != dynamics["artifact_sha256"]:
            errors.append(
                {
                    "code": "HASH_MISMATCH",
                    "artifact": str(path),
                    "recorded_value": dynamics["artifact_sha256"],
                }
            )
        else:
            inputs.append(path)
    return {
        "facts": facts,
        "errors": errors,
        "inputs": sorted(set(inputs)),
        "config": config,
        "model_config": model_config,
        "summary": summary,
        "model_root": model_root,
        "evaluation_dir": evaluation_dir,
    }


def gpu_tables(
    frame: pd.DataFrame, identity: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize each device collection; unavailable counters remain missing."""
    summaries, bands = [], []
    for uuid, samples in frame.groupby("gpu_uuid", sort=True):
        keys = {**identity, "gpu_uuid": uuid}
        for field in (
            "gpu_utilization_percent",
            "memory_controller_utilization_percent",
            "memory_used_mib",
            "memory_total_mib",
            "power_watts",
            "temperature_celsius",
        ):
            values = pd.to_numeric(samples[field], errors="coerce")
            summaries.append(
                {
                    **keys,
                    "field": field,
                    "valid_samples": int(values.notna().sum()),
                    "missing_samples": int(values.isna().sum()),
                    "sample_mean": values.mean(),
                    "median": values.median(),
                    "p95": values.quantile(0.95),
                    "max": values.max(),
                }
            )
        activity = pd.to_numeric(
            samples.gpu_utilization_percent, errors="coerce"
        ).dropna()
        for label, mask in (
            ("0%", activity == 0),
            ("1–9%", (activity > 0) & (activity < 10)),
            ("10–24%", (activity >= 10) & (activity < 25)),
            ("25–100%", activity >= 25),
        ):
            count = int(mask.sum())
            bands.append(
                {
                    **keys,
                    "activity_band": label,
                    "samples": count,
                    "sample_percent": count / len(activity) * 100
                    if len(activity)
                    else None,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(bands)


def epoch_tables(summary: dict) -> dict[str, pd.DataFrame]:
    """Present saved epoch values; do not reconstruct scores from step samples."""
    if "train_epoch_trace" in summary:
        frames = {
            split: pd.DataFrame(summary[key])
            for split, key in (
                ("train", "train_epoch_trace"),
                ("val", "validation_epoch_trace"),
                ("test", "test_epoch_trace"),
            )
        }
        # Read immutable historical native records without presenting their
        # training-objective diagnostic as an evaluation score.
        train = frames["train"]
        if "train_objective" not in train and "joint_loss" in train:
            frames["train"] = train.rename(columns={"joint_loss": "train_objective"})
        for split in ("val", "test"):
            frame = frames[split].drop(columns=["joint_loss"], errors="ignore")
            if f"{split}_wis" not in frame and "selection_metric" in frame:
                frame[f"{split}_wis"] = frame.selection_score.where(
                    frame.selection_metric == "mean_target_wis"
                )
            frames[split] = frame
        return frames
    history = summary.get("training_history", {})
    return {
        split: pd.DataFrame(
            {
                "epoch": range(1, len(history.get(key, [])) + 1),
                "joint_loss": history.get(key, []),
            }
        )
        for split, key in (("train", "train_loss"), ("val", "val_loss"))
    }
