"""Helpers for training-loop logging and gradient norm diagnostics."""

from __future__ import annotations

import statistics
from typing import Any

import torch

from utils.console import (
    format_horizon_status_lines,
    format_joint_loss_components_status,
)
from utils.log_keys import (
    JOINT_LOSS_COMPONENTS,
    build_curriculum_metric_key,
    build_eval_metric_key,
    build_horizon_metric_key,
    build_loss_key,
)


def build_train_step_log_data(
    *,
    lr: float,
    grad_norm: torch.Tensor,
    batch_time_s: float,
    data_time_s: float,
    model_step_time_s: float,
    epoch: int,
    component_gradnorm_log_data: dict[str, float],
    gradnorm_step_log_data: dict[str, torch.Tensor],
    gradient_snapshot_log_data: dict[str, float | int],
) -> dict[str, float | torch.Tensor]:
    """Build per-step logging payload before progress-only metrics are added."""
    log_data: dict[str, float | torch.Tensor] = {
        build_eval_metric_key("learning_rate", "step"): lr,
        "gradnorm_clipped_total": grad_norm,
        "time_batch_s": batch_time_s,
        "time_dataload_s": data_time_s,
        "time_step_s": batch_time_s,
        "time_model_step_s": model_step_time_s,
        "epoch": epoch,
    }
    log_data.update(component_gradnorm_log_data)
    log_data.update(gradnorm_step_log_data)
    log_data.update(gradient_snapshot_log_data)
    return log_data


def get_wandb_step_payload(
    *,
    log_this_step: bool,
    log_data: dict[str, float | int | torch.Tensor],
    component_gradnorm_log_data: dict[str, float],
    gradient_snapshot_log_data: dict[str, float | int],
) -> dict[str, float | int | torch.Tensor] | None:
    """Select the exact payload that should be sent to wandb for this step."""
    if log_this_step:
        return log_data
    sparse_payload: dict[str, float | int | torch.Tensor] = {}
    sparse_payload.update(component_gradnorm_log_data)
    sparse_payload.update(gradient_snapshot_log_data)
    if sparse_payload:
        return sparse_payload
    return None


def add_joint_loss_metrics(
    *,
    log_data: dict[str, Any],
    split_prefix: str,
    metrics: dict[str, Any],
) -> None:
    """Append optional joint-loss metrics (raw + weighted) to epoch payload."""
    for component in JOINT_LOSS_COMPONENTS:
        raw_key = build_loss_key(component=component)
        weighted_key = build_loss_key(component=component, weighted=True)
        if raw_key in metrics:
            log_data[build_loss_key(split=split_prefix, component=component)] = metrics[
                raw_key
            ]
        if weighted_key in metrics:
            log_data[
                build_loss_key(split=split_prefix, component=component, weighted=True)
            ] = metrics[weighted_key]


def compute_horizon_metric_series(
    *,
    aggregation: str,
    mae_per_h: list[float],
    rmse_per_h: list[float],
) -> list[tuple[str, float, float]]:
    """Compute horizon metric triplets as (label, mae, rmse)."""
    if aggregation == "weekly" and len(mae_per_h) > 0:
        horizon_metrics: list[tuple[str, float, float]] = []
        week_num = 1
        for start_idx in range(0, len(mae_per_h), 7):
            end_idx = min(start_idx + 7, len(mae_per_h))
            week_mae_values = mae_per_h[start_idx:end_idx]
            week_rmse_values = rmse_per_h[start_idx:end_idx]
            horizon_metrics.append(
                (
                    f"w{week_num}",
                    float(statistics.median(week_mae_values)),
                    float(statistics.median(week_rmse_values)),
                )
            )
            week_num += 1
        return horizon_metrics

    return [
        (f"h{idx + 1}", float(mae_h), float(rmse_h))
        for idx, (mae_h, rmse_h) in enumerate(zip(mae_per_h, rmse_per_h, strict=False))
    ]


def add_horizon_metrics_to_log_data(
    *,
    log_data: dict[str, Any],
    split_prefix: str,
    horizon_metrics: list[tuple[str, float, float]],
) -> None:
    """Append per-horizon metrics to an epoch logging payload."""
    for label, mae, rmse in horizon_metrics:
        log_data[build_horizon_metric_key("mae", split_prefix, label)] = mae
        log_data[build_horizon_metric_key("rmse", split_prefix, label)] = rmse


def add_curriculum_metrics(
    *,
    log_data: dict[str, Any],
    curriculum_sampler: Any | None,
    key_suffix: str,
    include_synth_ratio: bool,
) -> None:
    """Append curriculum metrics when sampler state is available."""
    if curriculum_sampler is None or not hasattr(curriculum_sampler, "state"):
        return

    log_data[build_curriculum_metric_key("sparsity", key_suffix)] = (
        curriculum_sampler.state.max_sparsity or 0.0
    )
    if include_synth_ratio:
        log_data[build_curriculum_metric_key("synth_ratio", key_suffix)] = (
            curriculum_sampler.state.synth_ratio
        )


def build_epoch_logging_bundle(
    *,
    split_name: str,
    loss: float,
    metrics: dict[str, Any],
    epoch: int,
    aggregation: str,
    curriculum_sampler: Any | None,
) -> tuple[dict[str, Any], list[str]]:
    """Build epoch payload and associated console status lines."""
    prefix = split_name.capitalize()
    prefix_lower = prefix.lower()

    log_data: dict[str, Any] = {
        "epoch": epoch,
        build_loss_key(split=prefix_lower): loss,
        build_eval_metric_key("mae", prefix_lower): metrics["mae"],
        build_eval_metric_key("rmse", prefix_lower): metrics["rmse"],
        build_eval_metric_key("smape", prefix_lower): metrics["smape"],
        build_eval_metric_key("r2", prefix_lower): metrics["r2"],
    }

    add_joint_loss_metrics(
        log_data=log_data,
        split_prefix=prefix_lower,
        metrics=metrics,
    )

    horizon_metrics = compute_horizon_metric_series(
        aggregation=aggregation,
        mae_per_h=metrics.get("mae_per_h", []),
        rmse_per_h=metrics.get("rmse_per_h", []),
    )
    add_horizon_metrics_to_log_data(
        log_data=log_data,
        split_prefix=prefix_lower,
        horizon_metrics=horizon_metrics,
    )
    add_curriculum_metrics(
        log_data=log_data,
        curriculum_sampler=curriculum_sampler,
        key_suffix="epoch",
        include_synth_ratio=True,
    )

    status_lines = [
        (
            f"{prefix} loss: {loss:.4g} | MAE: {metrics['mae']:.4g} | "
            f"RMSE: {metrics['rmse']:.4g} | sMAPE: {metrics['smape']:.4g} | "
            f"R2: {metrics['r2']:.4g}"
        )
    ]
    components_str = format_joint_loss_components_status(metrics)
    if components_str is not None:
        status_lines.append(f"{prefix} loss components: {components_str}")
    status_lines.extend(
        format_horizon_status_lines(
            prefix=prefix,
            horizon_metrics=horizon_metrics,
        )
    )

    return log_data, status_lines
