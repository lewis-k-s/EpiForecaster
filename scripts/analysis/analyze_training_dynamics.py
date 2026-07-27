"""Analyze canonical EpiForecaster training-dynamics artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analysis.skill_output import SkillOutputBuilder, print_output
from scripts.analysis.query_run_provenance import (
    collect_training_provenance,
    provenance_table,
)
from training.training_dynamics import (
    DYNAMICS_ARTIFACT_NAME,
    DYNAMICS_MANIFEST_NAME,
    DYNAMICS_SCHEMA_VERSION,
    DYNAMICS_TERMINAL_STATUSES,
    load_training_dynamics_identity,
    sha256_file,
)

MIN_ANALYSIS_ROWS = 20
COMPONENT_FIELDS = (
    "gradnorm_sird_physics",
    "gradnorm_backbone_encoder",
    "gradnorm_mobility_gnn",
    "gradnorm_other",
    "gradnorm_obs_ww",
    "gradnorm_obs_hosp",
    "gradnorm_obs_cases",
    "gradnorm_obs_deaths",
)
TIMING_FIELDS = (
    "time_batch_s",
    "time_dataload_s",
    "time_model_step_s",
    "train_step_sps",
)
TRACE_GLOB = "*.pt.trace.json"
TOP_PROFILE_OPERATIONS = 15
GRADNORM_TASKS = ("ww", "hosp", "cases", "deaths")


def _resolve_input(path: Path) -> tuple[Path, dict[str, Any] | None]:
    path = path.resolve()
    if path.is_dir():
        manifest_path = path / DYNAMICS_MANIFEST_NAME
        artifact_path = path / DYNAMICS_ARTIFACT_NAME
        if manifest_path.is_file():
            path = manifest_path
        elif artifact_path.is_file():
            return artifact_path, None
        else:
            raise FileNotFoundError(
                f"No {DYNAMICS_MANIFEST_NAME} or {DYNAMICS_ARTIFACT_NAME} in {path}"
            )

    if path.name == DYNAMICS_MANIFEST_NAME or path.suffix == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != DYNAMICS_SCHEMA_VERSION:
            raise ValueError("Unsupported training dynamics manifest schema")
        artifact = path.parent / str(manifest["artifact"])
        if (
            artifact.parent.resolve() != path.parent.resolve()
            or not artifact.is_file()
            or artifact.is_symlink()
        ):
            raise FileNotFoundError(f"Training dynamics artifact is missing: {artifact}")
        if manifest.get("status") in DYNAMICS_TERMINAL_STATUSES:
            load_training_dynamics_identity(path.parent)
        return artifact, manifest
    if not path.is_file():
        raise FileNotFoundError(f"Training dynamics input is missing: {path}")
    return path, None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL record at line {line_number}: {exc}"
                ) from exc
            if row.get("schema_version") != DYNAMICS_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported row schema at line {line_number}: "
                    f"{row.get('schema_version')!r}"
                )
            rows.append(row)

    steps = [int(row["step"]) for row in rows]
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("Training dynamics steps are not strictly increasing")
    return rows


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [
        float(value)
        for row in rows
        if (value := row.get(key)) is not None and math.isfinite(float(value))
    ]
    return np.asarray(values, dtype=float)


def _series_summary(values: np.ndarray) -> dict[str, float | int] | None:
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _rolling_median(values: np.ndarray, width: int) -> list[float]:
    half_width = width // 2
    return [
        float(
            np.median(
                values[max(0, index - half_width) : min(values.size, index + half_width + 1)]
            )
        )
        for index in range(values.size)
    ]


def _loss_analysis(
    values: np.ndarray,
    steps: list[int],
    epochs: list[int],
) -> dict[str, Any]:
    if values.size < MIN_ANALYSIS_ROWS:
        return {
            "status": "insufficient_data",
            "count": int(values.size),
            "minimum_required": MIN_ANALYSIS_ROWS,
            "series": [
                {"step": step, "epoch": epoch, "loss": float(value)}
                for step, epoch, value in zip(steps, epochs, values, strict=True)
            ],
        }

    width = max(1, math.ceil(values.size * 0.2))
    first_median = float(np.median(values[:width]))
    last_median = float(np.median(values[-width:]))
    relative_change = (
        (last_median - first_median) / abs(first_median)
        if first_median != 0
        else None
    )
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        robust_scores = np.where(values == median, 0.0, np.inf)
    else:
        robust_scores = 0.67448975 * np.abs(values - median) / mad
    spike_indexes = np.flatnonzero(robust_scores > 6.0)
    spike_steps = [steps[int(index)] for index in spike_indexes]
    rolling_width = min(51, max(5, math.ceil(values.size * 0.05)))
    if rolling_width % 2 == 0:
        rolling_width += 1
    rolling = np.asarray(_rolling_median(values, rolling_width), dtype=float)
    step_values = np.asarray(steps, dtype=float)
    positive = rolling > 0
    log_slope = (
        _linear_slope(step_values[positive], np.log(rolling[positive]))
        if positive.sum() >= 2
        else None
    )
    recent_steps = step_values[-width:]
    recent_values = rolling[-width:]
    recent_slope = _linear_slope(recent_steps, recent_values)
    recent_relative_slope_per_1000 = (
        recent_slope * 1000.0 / abs(last_median)
        if recent_slope is not None and last_median != 0
        else None
    )
    best_index = int(np.argmin(values))
    epoch_summaries = []
    for epoch in sorted(set(epochs)):
        epoch_values = values[np.asarray(epochs) == epoch]
        epoch_steps = np.asarray(steps)[np.asarray(epochs) == epoch]
        epoch_summaries.append(
            {
                "epoch": epoch,
                "count": int(epoch_values.size),
                "first_step": int(epoch_steps[0]),
                "last_step": int(epoch_steps[-1]),
                "median": float(np.median(epoch_values)),
                "p25": float(np.percentile(epoch_values, 25)),
                "p75": float(np.percentile(epoch_values, 75)),
                "min": float(np.min(epoch_values)),
            }
        )

    return {
        "status": "analyzed",
        "count": int(values.size),
        "first_window_median": first_median,
        "last_window_median": last_median,
        "relative_change": relative_change,
        "robust_spike_count": int(spike_indexes.size),
        "robust_spike_fraction": float(spike_indexes.size / values.size),
        "robust_spike_steps": spike_steps,
        "spike_threshold_mad": 6.0,
        "best_sampled_loss": float(values[best_index]),
        "best_sampled_step": int(steps[best_index]),
        "global_median": median,
        "global_mad": mad,
        "relative_mad": mad / abs(median) if median != 0 else None,
        "rolling_window_observations": rolling_width,
        "log_loss_slope_per_1000_steps": (
            log_slope * 1000.0 if log_slope is not None else None
        ),
        "recent_relative_slope_per_1000_steps": recent_relative_slope_per_1000,
        "epoch_summaries": epoch_summaries,
        "series": [
            {
                "step": step,
                "epoch": epoch,
                "loss": float(value),
                "rolling_median": float(rolling_value),
                "robust_spike": bool(index in spike_indexes),
            }
            for index, (step, epoch, value, rolling_value) in enumerate(
                zip(steps, epochs, values, rolling, strict=True)
            )
        ],
    }


def _clipping_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (
            int(row["step"]),
            float(total),
            float(threshold),
            float(row.get("gradient_clip_factor", min(1.0, float(threshold) / float(total)))),
        )
        for row in rows
        if (total := row.get("gradnorm_total_preclip")) is not None
        and (threshold := row.get("gradient_clip_threshold")) is not None
        and math.isfinite(float(total))
        and math.isfinite(float(threshold))
        and float(total) > 0
    ]
    if not pairs:
        return {"status": "not_sampled", "count": 0}
    clipped_pairs = [pair for pair in pairs if pair[1] > pair[2]]
    clipped_factors = np.asarray([pair[3] for pair in clipped_pairs], dtype=float)
    return {
        "status": "analyzed",
        "count": len(pairs),
        "clipped_count": len(clipped_pairs),
        "clipped_fraction": len(clipped_pairs) / len(pairs),
        "clipped_steps": [pair[0] for pair in clipped_pairs],
        "median_clip_factor_when_clipped": (
            float(np.median(clipped_factors)) if clipped_factors.size else None
        ),
        "strongest_clip_factor": (
            float(np.min(clipped_factors)) if clipped_factors.size else None
        ),
        "series": [
            {
                "step": step,
                "gradnorm_total_preclip": total,
                "gradient_clip_threshold": threshold,
                "gradient_clip_factor": factor,
                "clipped": total > threshold,
            }
            for step, total, threshold, factor in pairs
        ],
    }


def _component_analysis(
    rows: list[dict[str, Any]],
    *,
    mobility_enabled: bool | None,
) -> dict[str, Any]:
    contributions: dict[str, list[float]] = defaultdict(list)
    sampled: dict[str, int] = defaultdict(int)
    for row in rows:
        total = row.get("gradnorm_total_preclip")
        if total is None or not math.isfinite(float(total)) or float(total) <= 0:
            continue
        for field in COMPONENT_FIELDS:
            value = row.get(field)
            if value is None or not math.isfinite(float(value)):
                continue
            sampled[field] += 1
            contributions[field].append((float(value) / float(total)) ** 2 * 100.0)

    result: dict[str, Any] = {}
    for field in COMPONENT_FIELDS:
        if field == "gradnorm_mobility_gnn" and mobility_enabled is False:
            result[field] = {"status": "not_applicable"}
        elif not contributions[field]:
            result[field] = {"status": "not_sampled", "count": sampled[field]}
        else:
            values = np.asarray(contributions[field], dtype=float)
            component_rows = [
                row
                for row in rows
                if row.get(field) is not None
                and row.get("gradnorm_total_preclip") is not None
                and math.isfinite(float(row[field]))
                and math.isfinite(float(row["gradnorm_total_preclip"]))
                and float(row["gradnorm_total_preclip"]) > 0
            ]
            norms = np.asarray([float(row[field]) for row in component_rows], dtype=float)
            component_steps = np.asarray(
                [int(row["step"]) for row in component_rows],
                dtype=float,
            )
            positive = norms > 0
            log_slope = (
                _linear_slope(component_steps[positive], np.log(norms[positive]))
                if positive.sum() >= 2
                else None
            )
            result[field] = {
                "status": "analyzed",
                "count": int(values.size),
                "median_norm": float(np.median(norms)),
                "p95_norm": float(np.percentile(norms, 95)),
                "zero_fraction": float(np.mean(norms == 0)),
                "log_norm_slope_per_1000_steps": (
                    log_slope * 1000.0 if log_slope is not None else None
                ),
                "median_squared_norm_share_pct": float(np.median(values)),
                "p95_squared_norm_share_pct": float(np.percentile(values, 95)),
                "series": [
                    {
                        "step": int(row["step"]),
                        "norm": float(row[field]),
                        "squared_norm_share_pct": (
                            float(row[field]) / float(row["gradnorm_total_preclip"])
                        )
                        ** 2
                        * 100.0,
                    }
                    for row in component_rows
                ],
            }
    return result


def _format_profile_operations(
    durations: dict[str, float],
    counts: dict[str, int],
) -> list[dict[str, float | int | str]]:
    return [
        {
            "name": name,
            "count": counts[name],
            "total_duration_us": duration,
            "total_duration_ms": duration / 1000.0,
            "mean_duration_us": duration / counts[name],
        }
        for name, duration in sorted(
            durations.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:TOP_PROFILE_OPERATIONS]
    ]


def _analyze_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise ValueError(f"No traceEvents found in profiler trace: {path}")

    duration_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and isinstance(event.get("ts"), int | float)
        and isinstance(event.get("dur"), int | float)
        and float(event["dur"]) >= 0
    ]
    if not duration_events:
        raise ValueError(f"No complete duration events in profiler trace: {path}")

    category_durations: defaultdict[str, float] = defaultdict(float)
    category_counts: defaultdict[str, int] = defaultdict(int)
    cpu_durations: defaultdict[str, float] = defaultdict(float)
    cpu_counts: defaultdict[str, int] = defaultdict(int)
    kernel_durations: defaultdict[str, float] = defaultdict(float)
    kernel_counts: defaultdict[str, int] = defaultdict(int)
    synchronization_duration = 0.0
    synchronization_count = 0
    memcpy_duration = 0.0
    memcpy_count = 0
    compilation_duration = 0.0
    compilation_count = 0
    profiler_steps = []

    for event in duration_events:
        category = str(event.get("cat", "unknown"))
        name = str(event.get("name", "unknown"))
        duration = float(event["dur"])
        category_durations[category] += duration
        category_counts[category] += 1
        if category == "cpu_op":
            cpu_durations[name] += duration
            cpu_counts[name] += 1
        elif category == "kernel":
            kernel_durations[name] += duration
            kernel_counts[name] += 1

        normalized_name = name.lower()
        if "synchronize" in normalized_name:
            synchronization_duration += duration
            synchronization_count += 1
        if "memcpy" in normalized_name or "memset" in normalized_name:
            memcpy_duration += duration
            memcpy_count += 1
        if (
            "compile" in normalized_name
            or "compiledfxgraph" in normalized_name
            or "inductor" in normalized_name
        ):
            compilation_duration += duration
            compilation_count += 1
        if name.startswith("ProfilerStep#") and category == "user_annotation":
            profiler_steps.append(
                {
                    "name": name,
                    "start_us": float(event["ts"]),
                    "duration_us": duration,
                }
            )

    start = min(float(event["ts"]) for event in duration_events)
    end = max(float(event["ts"]) + float(event["dur"]) for event in duration_events)
    return {
        "artifact": str(path.resolve()),
        "artifact_sha256": sha256_file(path),
        "trace_duration_ms": (end - start) / 1000.0,
        "duration_event_count": len(duration_events),
        "profiled_steps": profiler_steps,
        "categories": {
            category: {
                "count": category_counts[category],
                "summed_inclusive_duration_ms": duration / 1000.0,
            }
            for category, duration in sorted(category_durations.items())
        },
        "top_cpu_operations": _format_profile_operations(cpu_durations, cpu_counts),
        "top_cuda_kernels": _format_profile_operations(
            kernel_durations,
            kernel_counts,
        ),
        "synchronization": {
            "count": synchronization_count,
            "summed_duration_ms": synchronization_duration / 1000.0,
        },
        "memory_copy_or_set": {
            "count": memcpy_count,
            "summed_duration_ms": memcpy_duration / 1000.0,
        },
        "compilation": {
            "count": compilation_count,
            "summed_duration_ms": compilation_duration / 1000.0,
        },
        "duration_semantics": "inclusive_nested",
    }


def _profile_analysis(artifact: Path) -> dict[str, Any]:
    trace_paths = sorted(artifact.parent.glob(TRACE_GLOB))
    if not trace_paths:
        return {
            "status": "not_available",
            "trace_count": 0,
            "missing_reason": "no_trace_artifacts",
        }
    traces = [_analyze_trace(path) for path in trace_paths]
    return {
        "status": "analyzed",
        "trace_count": len(traces),
        "profiled_step_count": sum(len(trace["profiled_steps"]) for trace in traces),
        "traces": traces,
        "coverage": "configured_active_windows",
        "duration_semantics": "inclusive_nested",
    }


def _validation_analysis(manifest: dict[str, Any] | None) -> dict[str, Any]:
    summary = manifest.get("summary", {}) if manifest is not None else {}
    history = summary.get("training_history", {})
    validation = history.get("val_loss", [])
    training = history.get("train_loss", [])
    selection = history.get(
        "val_selection_score",
        history.get("val_mae", []),
    )
    selection_metrics = history.get("val_selection_metric", [])
    default_selection_metric = summary.get("best_val_selection_metric")
    if not validation:
        return {
            "status": "not_available",
            "missing_reason": "not_persisted",
        }
    series = [
        {
            "epoch": index,
            "train_loss": training[index] if index < len(training) else None,
            "val_joint_loss": value,
            "val_selection_score": (
                selection[index] if index < len(selection) else None
            ),
            "val_selection_metric": (
                selection_metrics[index]
                if index < len(selection_metrics)
                else default_selection_metric
            ),
            "learning_rate": (
                history.get("learning_rate", [])[index]
                if index < len(history.get("learning_rate", []))
                else None
            ),
            "epoch_time_s": (
                history.get("epoch_times", [])[index]
                if index < len(history.get("epoch_times", []))
                else None
            ),
        }
        for index, value in enumerate(validation)
    ]
    finite_validation = [
        (row["epoch"], float(row["val_joint_loss"]))
        for row in series
        if row["val_joint_loss"] is not None
        and math.isfinite(float(row["val_joint_loss"]))
    ]
    if not finite_validation:
        return {
            "status": "not_available",
            "missing_reason": "no_finite_values",
        }
    best_epoch, best_loss = min(finite_validation, key=lambda item: item[1])
    return {
        "status": "analyzed",
        "count": len(finite_validation),
        "best_epoch": best_epoch,
        "best_val_joint_loss": best_loss,
        "last_val_joint_loss": finite_validation[-1][1],
        "selection_metric": summary.get("best_val_selection_metric"),
        "best_selection_score": summary.get("best_val_selection_score"),
        "series": series,
    }


def _adaptive_gradnorm_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    controller_rows = [
        row
        for row in rows
        if any(row.get(f"gradnorm_w_{task}") is not None for task in GRADNORM_TASKS)
    ]
    if not controller_rows:
        return {
            "status": "not_available",
            "missing_reason": "not_persisted_or_disabled",
        }

    series = []
    update_count = 0
    for row in controller_rows:
        values: dict[str, Any] = {"step": int(row["step"])}
        active_weights = []
        for task in GRADNORM_TASKS:
            for metric in (
                "w",
                "active",
                "task_loss",
                "task_norm",
                "target_norm",
                "rate",
            ):
                key = f"gradnorm_{metric}_{task}"
                if row.get(key) is not None:
                    values[f"{metric}_{task}"] = float(row[key])
            if values.get(f"active_{task}", 1.0) > 0.5 and f"w_{task}" in values:
                active_weights.append(values[f"w_{task}"])
        if row.get("gradnorm_sidecar_ran") is not None:
            values["sidecar_ran"] = float(row["gradnorm_sidecar_ran"])
            update_count += int(float(row["gradnorm_sidecar_ran"]) > 0.5)
        if row.get("gradnorm_L_grad") is not None:
            values["controller_loss"] = float(row["gradnorm_L_grad"])
        if active_weights:
            values["active_weight_spread"] = max(active_weights) / max(
                min(active_weights),
                1.0e-12,
            )
        series.append(values)

    tasks = {}
    for task in GRADNORM_TASKS:
        weights = np.asarray(
            [row[f"w_{task}"] for row in series if f"w_{task}" in row],
            dtype=float,
        )
        tasks[task] = (
            {
                "status": "analyzed",
                "count": int(weights.size),
                "median_weight": float(np.median(weights)),
                "min_weight": float(np.min(weights)),
                "max_weight": float(np.max(weights)),
            }
            if weights.size
            else {"status": "not_sampled", "count": 0}
        )
    target_evidence = any(
        any(f"target_norm_{task}" in row for task in GRADNORM_TASKS)
        for row in series
    )
    return {
        "status": "analyzed" if target_evidence else "partial",
        "count": len(series),
        "sidecar_update_count": update_count,
        "tasks": tasks,
        "series": series,
        "target_evidence": target_evidence,
    }


def _gradient_snapshot_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot_rows = [
        row for row in rows if row.get("grad_snapshot_global_norm") is not None
    ]
    if not snapshot_rows:
        return {
            "status": "not_available",
            "missing_reason": "disabled_or_not_sampled",
        }
    unexpected_zero_steps = {
        task: [
            int(row["step"])
            for row in snapshot_rows
            if float(row.get(f"grad_snapshot_head_{task}_unexpected_zero", 0)) > 0.5
        ]
        for task in GRADNORM_TASKS
    }
    return {
        "status": "analyzed",
        "count": len(snapshot_rows),
        "steps": [int(row["step"]) for row in snapshot_rows],
        "non_finite_layer_observations": sum(
            int(row.get("grad_snapshot_non_finite_layers", 0))
            for row in snapshot_rows
        ),
        "max_vanishing_layers": max(
            int(row.get("grad_snapshot_vanishing_layers", 0))
            for row in snapshot_rows
        ),
        "max_exploding_layers": max(
            int(row.get("grad_snapshot_exploding_layers", 0))
            for row in snapshot_rows
        ),
        "unexpected_zero_steps": unexpected_zero_steps,
    }


def _association(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(loss), math.log1p(max(0.0, float(gradnorm))))
        for row in rows
        if (loss := row.get("loss_train_step")) is not None
        and (gradnorm := row.get("gradnorm_total_preclip")) is not None
        and math.isfinite(float(loss))
        and math.isfinite(float(gradnorm))
    ]
    if len(pairs) < MIN_ANALYSIS_ROWS:
        return {
            "status": "insufficient_data",
            "count": len(pairs),
            "minimum_required": MIN_ANALYSIS_ROWS,
        }
    loss_values = np.asarray([pair[0] for pair in pairs], dtype=float)
    grad_values = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.std(loss_values) == 0 or np.std(grad_values) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(loss_values, grad_values)[0, 1])
    return {
        "status": "analyzed",
        "count": len(pairs),
        "pearson_loss_vs_log1p_preclip_gradnorm": correlation,
        "interpretation": "association_only",
    }


def analyze_training_dynamics(
    path: Path,
    *,
    run_manifest_path: Path | None = None,
) -> dict[str, Any]:
    artifact, manifest = _resolve_input(path)
    rows = _load_rows(artifact)
    mobility_enabled = None
    if manifest is not None:
        mobility_enabled = manifest.get("features", {}).get("mobility")

    nonfinite_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for field in row.get("nonfinite_fields", []):
            nonfinite_counts[str(field)] += 1

    loss_rows = [
        row
        for row in rows
        if (value := row.get("loss_train_step")) is not None
        and math.isfinite(float(value))
    ]
    loss_values = np.asarray(
        [float(row["loss_train_step"]) for row in loss_rows],
        dtype=float,
    )
    loss = _loss_analysis(
        loss_values,
        [int(row["step"]) for row in loss_rows],
        [int(row["epoch"]) for row in loss_rows],
    )
    findings: list[dict[str, Any]] = []
    manifest_status = manifest.get("status") if manifest else "unmanifested"
    if manifest is not None:
        expected_range = (
            [int(rows[0]["step"]), int(rows[-1]["step"])] if rows else None
        )
        observed_range = (
            [manifest.get("first_step"), manifest.get("last_step")]
            if manifest.get("first_step") is not None
            else None
        )
        if manifest.get("row_count") != len(rows):
            raise ValueError("Manifest row count does not match the JSONL artifact")
        if observed_range != expected_range:
            raise ValueError("Manifest step range does not match the JSONL artifact")
        if manifest_status != "complete":
            findings.append(
                {
                    "code": "partial_lifecycle",
                    "severity": "warning",
                    "evidence": {
                        "manifest_status": manifest_status,
                        "error": manifest.get("error"),
                    },
                }
            )
    if nonfinite_counts:
        findings.append(
            {
                "code": "nonfinite_observation",
                "severity": "error",
                "evidence": {"nonfinite_fields": dict(sorted(nonfinite_counts.items()))},
            }
        )
    if loss.get("status") == "analyzed" and loss["robust_spike_count"] > 0:
        findings.append(
            {
                "code": "robust_loss_spikes",
                "severity": "warning",
                "evidence": {
                    "robust_spike_count": loss["robust_spike_count"],
                    "steps": loss["robust_spike_steps"],
                    "threshold_mad": loss["spike_threshold_mad"],
                },
            }
        )

    clipping = _clipping_analysis(rows)
    components = _component_analysis(
        rows,
        mobility_enabled=mobility_enabled,
    )
    profiler = _profile_analysis(artifact)
    validation = _validation_analysis(manifest)
    adaptive_gradnorm = _adaptive_gradnorm_analysis(rows)
    gradient_snapshots = _gradient_snapshot_analysis(rows)
    timing = {
        field: summary
        for field in TIMING_FIELDS
        if (summary := _series_summary(_finite_values(rows, field))) is not None
    }
    timing["series"] = [
        {
            "step": int(row["step"]),
            **{
                field: float(row[field])
                for field in TIMING_FIELDS
                if row.get(field) is not None and math.isfinite(float(row[field]))
            },
        }
        for row in rows
    ]
    if loss.get("status") == "analyzed" and clipping.get("status") == "analyzed":
        coincident_steps = sorted(
            set(loss["robust_spike_steps"]) & set(clipping["clipped_steps"])
        )
        if coincident_steps:
            findings.append(
                {
                    "code": "loss_spike_clipping_coincidence",
                    "severity": "warning",
                    "evidence": {"coincident_steps": coincident_steps},
                }
            )
    if profiler.get("status") == "analyzed":
        compilation_ms = sum(
            trace["compilation"]["summed_duration_ms"]
            for trace in profiler["traces"]
        )
        if compilation_ms > 0:
            findings.append(
                {
                    "code": "profile_compilation_activity",
                    "severity": "info",
                    "evidence": {
                        "summed_inclusive_compilation_duration_ms": compilation_ms
                    },
                }
            )
    steps = [int(row["step"]) for row in rows]
    progress_frequency = (
        int(manifest["progress_log_frequency"]) if manifest is not None else None
    )
    cadence_mismatches = (
        [
            [left, right]
            for left, right in zip(steps, steps[1:])
            if (right - left) % progress_frequency != 0
        ]
        if progress_frequency
        else []
    )
    identity = {
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "manifest_status": manifest_status,
        "manifest": manifest,
    }
    provenance = collect_training_provenance(
        artifact.parent,
        run_manifest_path=run_manifest_path,
    )
    return {
        "analysis_status": (
            "analyzed"
            if loss_values.size >= MIN_ANALYSIS_ROWS
            else "insufficient_data"
        ),
        "identity": identity,
        "provenance": provenance,
        "data_quality": {
            "row_count": len(rows),
            "step_range": (
                [int(rows[0]["step"]), int(rows[-1]["step"])] if rows else None
            ),
            "cadence": {
                "progress_log_frequency": progress_frequency,
                "irregular_step_pairs": cadence_mismatches,
                "status": "ok" if not cadence_mismatches else "irregular",
            },
            "nonfinite_fields": dict(sorted(nonfinite_counts.items())),
        },
        "loss": loss,
        "clipping": clipping,
        "components": components,
        "adaptive_gradnorm": adaptive_gradnorm,
        "gradient_snapshots": gradient_snapshots,
        "validation": validation,
        "loss_gradient_association": _association(rows),
        "timing": timing,
        "profiler": profiler,
        "evidence_coverage": {
            "training_loss": "available" if loss_rows else "not_available",
            "validation_history": validation["status"],
            "gradient_norm": (
                "available"
                if any(row.get("gradnorm_total_preclip") is not None for row in rows)
                else "not_sampled"
            ),
            "component_gradient_norm": (
                "available"
                if any(
                    any(row.get(field) is not None for field in COMPONENT_FIELDS)
                    for row in rows
                )
                else "not_sampled"
            ),
            "adaptive_gradnorm_controller": (
                adaptive_gradnorm["status"]
            ),
            "gradient_snapshots": gradient_snapshots["status"],
            "profiler": (
                "available"
                if any(artifact.parent.glob(TRACE_GLOB))
                else "not_available"
            ),
            "provenance": provenance["status"],
        },
        "findings": findings,
    }


def _write_loss_figure(data: dict[str, Any], output_path: Path) -> bool:
    loss = data["loss"]
    series = loss.get("series", [])
    if not series:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in series]
    values = [row["loss"] for row in series]
    rolling = [row.get("rolling_median") for row in series]
    spike_rows = [row for row in series if row.get("robust_spike")]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(steps, values, color="#5b8ff9", alpha=0.35, label="Sampled train loss")
    if all(value is not None for value in rolling):
        axis.plot(steps, rolling, color="#1d39c4", linewidth=2, label="Rolling median")
    if spike_rows:
        axis.scatter(
            [row["step"] for row in spike_rows],
            [row["loss"] for row in spike_rows],
            color="#cf1322",
            marker="x",
            s=55,
            label="Robust spike",
            zorder=3,
        )
    validation = data["validation"]
    if validation["status"] == "analyzed":
        epoch_last_steps = {
            row["epoch"]: row["last_step"] for row in loss.get("epoch_summaries", [])
        }
        validation_points = [
            (epoch_last_steps[row["epoch"]], row["val_joint_loss"])
            for row in validation["series"]
            if row["epoch"] in epoch_last_steps and row["val_joint_loss"] is not None
        ]
        if validation_points:
            axis.plot(
                [point[0] for point in validation_points],
                [point[1] for point in validation_points],
                color="#d46b08",
                marker="o",
                linewidth=1.5,
                label="Validation joint loss",
            )
    axis.set(
        title="Sampled training loss",
        xlabel="Optimizer step",
        ylabel="Training loss",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _write_gradient_figure(data: dict[str, Any], output_path: Path) -> bool:
    series = data["clipping"].get("series", [])
    if not series:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    steps = [row["step"] for row in series]
    norms = [row["gradnorm_total_preclip"] for row in series]
    thresholds = [row["gradient_clip_threshold"] for row in series]
    axis.plot(steps, norms, color="#722ed1", label="Pre-clip gradient norm")
    axis.plot(
        steps,
        thresholds,
        color="#cf1322",
        linestyle="--",
        label="Clip threshold",
    )
    clipped = [row for row in series if row["clipped"]]
    if clipped:
        axis.scatter(
            [row["step"] for row in clipped],
            [row["gradnorm_total_preclip"] for row in clipped],
            color="#cf1322",
            s=30,
            label="Clipped sample",
            zorder=3,
        )
    axis.set(
        title="Gradient norm and clipping",
        xlabel="Optimizer step",
        ylabel="Gradient norm",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _write_component_figure(data: dict[str, Any], output_path: Path) -> bool:
    analyzed = {
        name.removeprefix("gradnorm_"): component
        for name, component in data["components"].items()
        if component.get("status") == "analyzed" and component.get("series")
    }
    if not analyzed:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2
    rows = math.ceil(len(analyzed) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, max(3.0 * rows, 4.5)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, (name, component) in zip(
        axes.flat,
        analyzed.items(),
        strict=False,
    ):
        series = component["series"]
        steps = [row["step"] for row in series]
        shares = np.asarray(
            [row["squared_norm_share_pct"] for row in series],
            dtype=float,
        )
        rolling = _rolling_median(shares, min(5, len(shares)))
        axis.plot(steps, shares, color="#91caff", alpha=0.5, linewidth=1)
        axis.plot(steps, rolling, color="#0958d9", linewidth=1.8)
        axis.set(
            title=name,
            ylim=(-2, 102),
        )
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(analyzed) :]:
        axis.set_visible(False)
    figure.suptitle("Component squared gradient-norm shares")
    figure.supxlabel("Optimizer step")
    figure.supylabel("Squared-norm share (%)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _write_timing_figure(data: dict[str, Any], output_path: Path) -> bool:
    series = data["timing"].get("series", [])
    fields = ("time_dataload_s", "time_model_step_s", "time_batch_s")
    if not series or not any(any(field in row for field in fields) for row in series):
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(10, 8),
        sharex=True,
    )
    for axis, field in zip(axes, fields, strict=True):
        points = [(row["step"], row[field]) for row in series if field in row]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] * 1000.0 for point in points],
                color="#1677ff",
                alpha=0.8,
            )
        axis.set_ylabel("ms")
        axis.set_title(field.removeprefix("time_").removesuffix("_s"))
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Optimizer step")
    figure.suptitle("Sampled training timing")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _write_adaptive_gradnorm_figure(
    data: dict[str, Any],
    output_path: Path,
) -> bool:
    controller = data["adaptive_gradnorm"]
    if controller.get("status") not in {"analyzed", "partial"}:
        return False
    series = controller.get("series", [])
    if not series:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    for task in GRADNORM_TASKS:
        points = [
            (row["step"], row[f"w_{task}"])
            for row in series
            if f"w_{task}" in row
        ]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                label=task,
            )
    axis.set(
        title="Adaptive GradNorm task weights",
        xlabel="Optimizer step",
        ylabel="Task weight",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _write_profiler_figure(data: dict[str, Any], output_path: Path) -> bool:
    profiler = data["profiler"]
    if profiler.get("status") != "analyzed":
        return False
    durations: defaultdict[str, float] = defaultdict(float)
    for trace in profiler["traces"]:
        for operation in trace["top_cuda_kernels"]:
            durations[str(operation["name"])] += float(operation["total_duration_ms"])
    if not durations:
        for trace in profiler["traces"]:
            for operation in trace["top_cpu_operations"]:
                durations[str(operation["name"])] += float(
                    operation["total_duration_ms"]
                )
    if not durations:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = sorted(durations.items(), key=lambda item: item[1], reverse=True)[:10]
    top.reverse()
    labels = [
        name if len(name) <= 64 else f"{name[:34]}…{name[-25:]}"
        for name, _ in top
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(
        labels,
        [duration for _, duration in top],
        color="#08979c",
    )
    axis.set(
        title="Profiler operation hotspots (inclusive sampled time)",
        xlabel="Summed duration (ms)",
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, dict | list)
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return True


def _tabular_data(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    component_series = [
        {
            "component": component_name,
            **row,
        }
        for component_name, component in data["components"].items()
        if component.get("status") == "analyzed"
        for row in component.get("series", [])
    ]
    component_summary = [
        {
            "component": component_name,
            **{
                key: value
                for key, value in component.items()
                if key != "series"
            },
        }
        for component_name, component in data["components"].items()
    ]
    timing_summary = [
        {"metric": metric, **summary}
        for metric, summary in data["timing"].items()
        if metric != "series"
    ]
    profiler_steps = []
    profiler_operations = []
    profiler_summary = []
    for trace_index, trace in enumerate(data["profiler"].get("traces", [])):
        profiler_summary.append(
            {
                "trace_index": trace_index,
                "artifact": trace["artifact"],
                "artifact_sha256": trace["artifact_sha256"],
                "trace_duration_ms": trace["trace_duration_ms"],
                "duration_event_count": trace["duration_event_count"],
                "synchronization_count": trace["synchronization"]["count"],
                "synchronization_duration_ms": trace["synchronization"][
                    "summed_duration_ms"
                ],
                "memory_copy_or_set_count": trace["memory_copy_or_set"]["count"],
                "memory_copy_or_set_duration_ms": trace["memory_copy_or_set"][
                    "summed_duration_ms"
                ],
                "compilation_count": trace["compilation"]["count"],
                "compilation_duration_ms": trace["compilation"][
                    "summed_duration_ms"
                ],
                "duration_semantics": trace["duration_semantics"],
            }
        )
        profiler_steps.extend(
            {"trace_index": trace_index, **row}
            for row in trace["profiled_steps"]
        )
        for activity, key in (
            ("cpu", "top_cpu_operations"),
            ("cuda", "top_cuda_kernels"),
        ):
            profiler_operations.extend(
                {
                    "trace_index": trace_index,
                    "activity": activity,
                    **row,
                }
                for row in trace[key]
            )

    tables = {
        "evidence_coverage": [
            {"evidence": name, "status": status}
            for name, status in data["evidence_coverage"].items()
        ],
        "loss_series": data["loss"].get("series", []),
        "loss_epochs": data["loss"].get("epoch_summaries", []),
        "validation_series": data["validation"].get("series", []),
        "clipping_series": data["clipping"].get("series", []),
        "component_summary": component_summary,
        "component_series": component_series,
        "adaptive_gradnorm_tasks": [
            {"task": task, **summary}
            for task, summary in data["adaptive_gradnorm"].get("tasks", {}).items()
        ],
        "adaptive_gradnorm_series": data["adaptive_gradnorm"].get("series", []),
        "gradient_snapshots": (
            [
                {
                    key: value
                    for key, value in data["gradient_snapshots"].items()
                    if key != "unexpected_zero_steps"
                }
                | {
                    "unexpected_zero_steps": data["gradient_snapshots"].get(
                        "unexpected_zero_steps",
                        {},
                    )
                }
            ]
            if data["gradient_snapshots"]["status"] == "analyzed"
            else []
        ),
        "timing_summary": timing_summary,
        "timing_series": data["timing"].get("series", []),
        "profiler_summary": profiler_summary,
        "profiler_steps": profiler_steps,
        "profiler_operations": profiler_operations,
        "signals": [
            {
                "code": finding["code"],
                "severity": finding["severity"],
                "evidence": finding["evidence"],
            }
            for finding in data["findings"]
        ],
        "provenance": provenance_table(data["provenance"]),
    }
    return {name: rows for name, rows in tables.items() if rows}


def write_analysis_bundle(data: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Analysis output directory is not empty; refuse to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            data["provenance"],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    tables_dir = output_dir / "tables"
    tables_dir.mkdir()
    table_paths = {}
    for name, rows in _tabular_data(data).items():
        table_path = tables_dir / f"{name}.csv"
        if _write_csv(table_path, rows):
            table_paths[name] = str(table_path.relative_to(output_dir))

    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    figure_paths = {}
    previous_mpl_config = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix=".matplotlib-cache-",
    ) as mpl_config:
        os.environ["MPLCONFIGDIR"] = mpl_config
        try:
            for name, writer in (
                ("loss_curve", _write_loss_figure),
                ("gradient_clipping", _write_gradient_figure),
                ("gradient_components", _write_component_figure),
                ("adaptive_gradnorm", _write_adaptive_gradnorm_figure),
                ("timing", _write_timing_figure),
                ("profiler_hotspots", _write_profiler_figure),
            ):
                figure_path = figures_dir / f"{name}.png"
                if writer(data, figure_path):
                    figure_paths[name] = str(figure_path.relative_to(output_dir))
        finally:
            if previous_mpl_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_mpl_config

    manifest = {
        "schema_version": 2,
        "analysis": analysis_path.name,
        "provenance": provenance_path.name,
        "source": {
            "artifact": data["identity"]["artifact"],
            "artifact_sha256": data["identity"]["artifact_sha256"],
            "manifest_status": data["identity"]["manifest_status"],
        },
        "tables": table_paths,
        "figures": figure_paths,
    }
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "root": str(output_dir),
        "manifest": str(manifest_path),
        "analysis": str(analysis_path),
        "provenance": str(provenance_path),
        "tables": {
            name: str(output_dir / relative_path)
            for name, relative_path in table_paths.items()
        },
        "figures": {
            name: str(output_dir / relative_path)
            for name, relative_path in figure_paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze canonical EpiForecaster training dynamics"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Model/trial directory, dynamics manifest, or JSONL artifact",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Write analysis JSON, CSV tables, plots, and a bundle manifest to a "
            "new or empty directory"
        ),
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Explicit immutable Snakemake run manifest.yaml",
    )
    args = parser.parse_args()
    builder = SkillOutputBuilder(
        "training-dynamics-analyze",
        input_path=str(args.path),
    )
    try:
        data = analyze_training_dynamics(
            args.path,
            run_manifest_path=args.run_manifest,
        )
        bundle = (
            write_analysis_bundle(data, args.output_dir)
            if args.output_dir is not None
            else None
        )
        output_data = (
            {
                "analysis_status": data["analysis_status"],
                "identity": data["identity"],
                "evidence_coverage": data["evidence_coverage"],
                "bundle": bundle,
            }
            if bundle is not None
            else data
        )
        print_output(
            builder.success(output_data),
            indent=None if args.compact else 2,
        )
    except Exception as exc:
        print_output(
            builder.error(type(exc).__name__, str(exc)),
            indent=None if args.compact else 2,
        )


if __name__ == "__main__":
    main()
