"""Helpers for GradNorm-specific logging and payload formatting."""

from __future__ import annotations

import time

import torch


def init_gradnorm_sidecar_log_data(_device: torch.device) -> dict[str, torch.Tensor]:
    """Create an empty sidecar payload (sidecar metrics go to console only, not W&B)."""
    return {}


def mark_gradnorm_sidecar_complete(
    log_data: dict[str, torch.Tensor],
    *,
    started_at: float,
    device: torch.device,
    gradnorm_loss: torch.Tensor | None = None,
) -> None:
    """Mark sidecar execution and populate timing (+ optional GradNorm loss)."""
    log_data["gradnorm_sidecar_ran"] = torch.tensor(1.0, device=device)
    if gradnorm_loss is not None:
        log_data["gradnorm_L_grad"] = gradnorm_loss.detach()
    log_data["time_gradnorm_sidecar_s"] = torch.tensor(
        time.time() - started_at,
        device=device,
    )


def format_gradnorm_controller_status(
    *,
    gradnorm_enabled: bool,
    gradnorm_step_log_data: dict[str, torch.Tensor],
    cached_weights: torch.Tensor,
    last_active_mask: torch.Tensor,
    task_names: tuple[str, ...],
) -> tuple[str, dict[str, float]]:
    """Build a concise, interpretable GradNorm controller progress summary."""
    if not gradnorm_enabled:
        return "", {}

    def _to_float(value: torch.Tensor | float | int | None) -> float:
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.detach())
        return float(value)

    tasks = list(task_names)
    weights: list[float] = []
    for idx, task in enumerate(tasks):
        value = gradnorm_step_log_data.get(f"gradnorm_w_{task}")
        if value is None:
            value = cached_weights[idx]
        weights.append(_to_float(value))

    active_mask = last_active_mask.detach().to(dtype=torch.bool).cpu()
    active_indices = [
        idx for idx, is_active in enumerate(active_mask.tolist()) if is_active
    ]
    if not active_indices:
        return "", {}

    active_weights = [weights[idx] for idx in active_indices]
    max_weight = max(active_weights)
    min_weight = max(min(active_weights), 1.0e-12)
    spread = max_weight / min_weight
    dominant_local_idx = active_indices[active_weights.index(max_weight)]
    dominant_task = tasks[dominant_local_idx]
    sidecar_ran = _to_float(gradnorm_step_log_data.get("gradnorm_sidecar_ran"))
    gradnorm_l_grad = _to_float(gradnorm_step_log_data.get("gradnorm_L_grad"))

    metrics: dict[str, float] = {}
    summary = (
        f"GN sidecar={int(round(sidecar_ran))} "
        f"Lg={gradnorm_l_grad:.3g} dom={dominant_task} spread={spread:.2f} "
        f"w=[ww:{weights[0]:.3f},hosp:{weights[1]:.3f},"
        f"cases:{weights[2]:.3f},deaths:{weights[3]:.3f}]"
    )
    return summary, metrics


def did_gradnorm_sidecar_run(
    gradnorm_step_log_data: dict[str, torch.Tensor],
) -> bool:
    """Return True when this step executed the GradNorm sidecar update."""
    value = gradnorm_step_log_data.get("gradnorm_sidecar_ran")
    if value is None:
        return False
    return bool(float(value.detach()) > 0.5)
