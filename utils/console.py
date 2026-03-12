"""Console formatting utilities for training progress."""

from __future__ import annotations

from typing import Any

import torch


def format_component_gradnorm_status(
    step: int, component_gradnorm_log_data: dict[str, float]
) -> str:
    """Format debug status line for component gradient norms."""
    return (
        f"Grad norms @ step {step}: "
        f"SIRD={component_gradnorm_log_data.get('gradnorm_sird_physics', 0.0):.4f} | "
        f"Enc={component_gradnorm_log_data.get('gradnorm_backbone_encoder', 0.0):.4f} | "
        f"GNN={component_gradnorm_log_data.get('gradnorm_mobility_gnn', 0.0):.4f} | "
        f"Other={component_gradnorm_log_data.get('gradnorm_other', 0.0):.4f}"
    )


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
