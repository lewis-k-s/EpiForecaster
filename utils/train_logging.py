"""Helpers for training-loop logging and gradient norm diagnostics."""

from __future__ import annotations

import statistics
from typing import Any

import torch

JOINT_LOSS_COMPONENTS: tuple[str, ...] = ("ww", "hosp", "cases", "deaths", "sir")


def should_log_gradnorm_components(step: int, frequency: int) -> bool:
    """Return whether component gradnorm diagnostics should run for this step."""
    return frequency > 0 and (step % frequency == 0 or step == 0)


def format_component_gradnorm_status(
    step: int, component_gradnorm_log_data: dict[str, float]
) -> str:
    """Format debug status line for component gradient norms."""
    return (
        f"Grad norms @ step {step}: "
        f"Total={component_gradnorm_log_data.get('gradnorm_total_preclip', 0.0):.4f} | "
        f"SIRD={component_gradnorm_log_data.get('gradnorm_sird_physics', 0.0):.4f} | "
        f"Enc={component_gradnorm_log_data.get('gradnorm_backbone_encoder', 0.0):.4f} | "
        f"GNN={component_gradnorm_log_data.get('gradnorm_mobility_gnn', 0.0):.4f} | "
        f"Obs={component_gradnorm_log_data.get('gradnorm_observation_heads', 0.0):.4f} | "
        f"Other={component_gradnorm_log_data.get('gradnorm_other', 0.0):.4f}"
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
) -> dict[str, float | torch.Tensor]:
    """Build per-step logging payload before progress-only metrics are added."""
    log_data: dict[str, float | torch.Tensor] = {
        "learning_rate_step": lr,
        "gradnorm_clipped_total": grad_norm,
        "time_batch_s": batch_time_s,
        "time_dataload_s": data_time_s,
        "time_step_s": batch_time_s,
        "time_model_step_s": model_step_time_s,
        "epoch": epoch,
    }
    log_data.update(component_gradnorm_log_data)
    log_data.update(gradnorm_step_log_data)
    return log_data


def format_train_progress_status(
    *,
    epoch: int,
    step: int,
    loss_value: float,
    lr: float,
    grad_norm: torch.Tensor,
    samples_per_s: float,
    gradnorm_status_suffix: str,
) -> str:
    """Format the console status line emitted at progress log cadence."""
    return (
        f"Epoch {epoch} | Step {step} | "
        f"Loss: {loss_value:.4g} | Lr: {lr:.2e} | "
        f"Grad: {float(grad_norm):.3f} | SPS: {samples_per_s:7.1f}"
        f"{gradnorm_status_suffix}"
    )


def get_wandb_step_payload(
    *,
    log_this_step: bool,
    log_data: dict[str, float | torch.Tensor],
    component_gradnorm_log_data: dict[str, float],
) -> dict[str, float | torch.Tensor] | None:
    """Select the exact payload that should be sent to wandb for this step."""
    if log_this_step:
        return log_data
    if component_gradnorm_log_data:
        return component_gradnorm_log_data
    return None


def add_joint_loss_metrics(
    *,
    log_data: dict[str, Any],
    split_prefix: str,
    metrics: dict[str, Any],
) -> None:
    """Append optional joint-loss metrics (raw + weighted) to epoch payload."""
    for component in JOINT_LOSS_COMPONENTS:
        raw_key = f"loss_{component}"
        objective_raw_key = f"{raw_key}_raw"
        weighted_key = f"{raw_key}_weighted"
        if raw_key in metrics:
            log_data[f"loss_{split_prefix}_{component}"] = metrics[raw_key]
        if objective_raw_key in metrics:
            log_data[f"loss_{split_prefix}_{component}_raw"] = metrics[objective_raw_key]
        if weighted_key in metrics:
            log_data[f"loss_{split_prefix}_{component}_weighted"] = metrics[
                weighted_key
            ]


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
        log_data[f"mae_{split_prefix}_{label}"] = mae
        log_data[f"rmse_{split_prefix}_{label}"] = rmse


def format_joint_loss_components_status(metrics: dict[str, Any]) -> str | None:
    """Format optional detailed component-loss status text for console logs."""
    required = (
        "loss_ww",
        "loss_hosp",
        "loss_sir",
        "loss_ww_weighted",
        "loss_hosp_weighted",
        "loss_sir_weighted",
    )
    if not all(key in metrics for key in required):
        return None

    components = [
        f"WW={metrics['loss_ww']:.4g} (w={metrics['loss_ww_weighted']:.4g})",
        f"Hosp={metrics['loss_hosp']:.4g} (w={metrics['loss_hosp_weighted']:.4g})",
        f"SIR={metrics['loss_sir']:.4g} (w={metrics['loss_sir_weighted']:.4g})",
    ]
    if "loss_cases" in metrics:
        components.append(
            f"Cases={metrics['loss_cases']:.4g} "
            f"(w={metrics.get('loss_cases_weighted', 0):.4g})"
        )
    if "loss_deaths" in metrics:
        components.append(
            f"Deaths={metrics['loss_deaths']:.4g} "
            f"(w={metrics.get('loss_deaths_weighted', 0):.4g})"
        )
    return " | ".join(components)


def format_horizon_status_lines(
    *,
    prefix: str,
    horizon_metrics: list[tuple[str, float, float]],
) -> list[str]:
    """Format per-horizon status lines for console output."""
    return [
        f"{prefix} MAE_{label}: {mae:.6f} | RMSE_{label}: {rmse:.6f}"
        for label, mae, rmse in horizon_metrics
    ]


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

    log_data[f"train_sparsity_{key_suffix}"] = (
        curriculum_sampler.state.max_sparsity or 0.0
    )
    if include_synth_ratio:
        log_data[f"train_synth_ratio_{key_suffix}"] = curriculum_sampler.state.synth_ratio


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
        f"loss_{prefix_lower}": loss,
        f"mae_{prefix_lower}": metrics["mae"],
        f"rmse_{prefix_lower}": metrics["rmse"],
        f"smape_{prefix_lower}": metrics["smape"],
        f"r2_{prefix_lower}": metrics["r2"],
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
    if horizon_metrics:
        mixed_mae = float(statistics.mean(mae for _label, mae, _rmse in horizon_metrics))
        mixed_rmse = float(
            statistics.mean(rmse for _label, _mae, rmse in horizon_metrics)
        )
        log_data[f"mae_{prefix_lower}_mixed_horizon"] = mixed_mae
        log_data[f"rmse_{prefix_lower}_mixed_horizon"] = mixed_rmse
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
    if horizon_metrics:
        status_lines.append(
            f"{prefix} MAE_mixed: {mixed_mae:.6f} | RMSE_mixed: {mixed_rmse:.6f}"
        )

    return log_data, status_lines


def compute_gradient_norms_and_clip(
    *,
    grad_norm_groups: dict[str, list[torch.nn.Parameter]],
    model: torch.nn.Module,
    device: torch.device,
    step: int,
    frequency: int,
    clip_value: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute component grad norms and apply clipping in one pass."""
    should_log = should_log_gradnorm_components(step, frequency)

    gnn_sq_sum = torch.tensor(0.0, device=device)
    ww_sq_sum = torch.tensor(0.0, device=device)
    hosp_sq_sum = torch.tensor(0.0, device=device)
    cases_sq_sum = torch.tensor(0.0, device=device)
    deaths_sq_sum = torch.tensor(0.0, device=device)
    sird_sq_sum = torch.tensor(0.0, device=device)
    encoder_sq_sum = torch.tensor(0.0, device=device)
    other_sq_sum = torch.tensor(0.0, device=device)

    all_grads: list[torch.Tensor] = []

    for group_name, params in grad_norm_groups.items():
        for param in params:
            if param.grad is None:
                continue

            grad = param.grad.detach()
            all_grads.append(grad)
            sq_norm = grad.pow(2).sum()

            if should_log:
                if group_name == "mobility_gnn":
                    gnn_sq_sum += sq_norm
                elif group_name == "ww_head":
                    ww_sq_sum += sq_norm
                elif group_name == "hosp_head":
                    hosp_sq_sum += sq_norm
                elif group_name == "cases_head":
                    cases_sq_sum += sq_norm
                elif group_name == "deaths_head":
                    deaths_sq_sum += sq_norm
                elif group_name == "sird":
                    sird_sq_sum += sq_norm
                elif group_name == "backbone":
                    encoder_sq_sum += sq_norm
                else:
                    other_sq_sum += sq_norm

    if all_grads:
        global_norm = torch.linalg.vector_norm(
            torch.stack([g.pow(2).sum() for g in all_grads]).sum().sqrt()
        )
    else:
        global_norm = torch.tensor(0.0, device=device)

    if global_norm > clip_value:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value, foreach=True)

    norms_dict: dict[str, float] = {}
    if should_log:
        obs_heads_sq_sum = ww_sq_sum + hosp_sq_sum + cases_sq_sum + deaths_sq_sum
        component_sq_sums = torch.stack(
            [
                sird_sq_sum,
                encoder_sq_sum,
                gnn_sq_sum,
                obs_heads_sq_sum,
                other_sq_sum,
            ]
        )
        total_sq_sum = component_sq_sums.sum()
        per_head_sq_sums = torch.stack([ww_sq_sum, hosp_sq_sum, cases_sq_sum, deaths_sq_sum])
        all_sq_sums = torch.cat([total_sq_sum.unsqueeze(0), component_sq_sums, per_head_sq_sums])
        all_norms = all_sq_sums.sqrt().cpu().numpy()

        norms_dict = {
            "gradnorm_total_preclip": float(all_norms[0]),
            "gradnorm_sird_physics": float(all_norms[1]),
            "gradnorm_backbone_encoder": float(all_norms[2]),
            "gradnorm_mobility_gnn": float(all_norms[3]),
            "gradnorm_observation_heads": float(all_norms[4]),
            "gradnorm_other": float(all_norms[5]),
            "gradnorm_obs_ww": float(all_norms[6]),
            "gradnorm_obs_hosp": float(all_norms[7]),
            "gradnorm_obs_cases": float(all_norms[8]),
            "gradnorm_obs_deaths": float(all_norms[9]),
            "gradnorm_backbone": float(all_norms[1] + all_norms[2]),
        }

    return global_norm, norms_dict
