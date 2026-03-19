"""Gradient computation and clipping utilities for training."""

from __future__ import annotations

import torch

from utils.log_keys import OBSERVATION_HEADS, build_gradnorm_obs_key


def should_log_gradnorm_components(step: int, frequency: int) -> bool:
    """Return whether component gradnorm diagnostics should run for this step."""
    return frequency > 0 and (step % frequency == 0 or step == 0)


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
    head_sq_sums = {
        head: torch.tensor(0.0, device=device) for head in OBSERVATION_HEADS
    }
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
                elif group_name.startswith("observation_head_"):
                    head_name = group_name.removeprefix("observation_head_")
                    if head_name in head_sq_sums:
                        head_sq_sums[head_name] += sq_norm
                    else:
                        other_sq_sum += sq_norm
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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_value, foreach=True
        )

    norms_dict: dict[str, float] = {}
    if should_log:
        norms_dict = {
            "gradnorm_sird_physics": float(sird_sq_sum.sqrt().item()),
            "gradnorm_backbone_encoder": float(encoder_sq_sum.sqrt().item()),
            "gradnorm_mobility_gnn": float(gnn_sq_sum.sqrt().item()),
            "gradnorm_other": float(other_sq_sum.sqrt().item()),
            build_gradnorm_obs_key("ww"): float(head_sq_sums["ww"].sqrt().item()),
            build_gradnorm_obs_key("hosp"): float(head_sq_sums["hosp"].sqrt().item()),
            build_gradnorm_obs_key("cases"): float(
                head_sq_sums["cases"].sqrt().item()
            ),
            build_gradnorm_obs_key("deaths"): float(
                head_sq_sums["deaths"].sqrt().item()
            ),
        }

    return global_norm, norms_dict
