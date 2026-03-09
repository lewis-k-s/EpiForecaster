"""Gradient computation and clipping utilities for training."""

from __future__ import annotations

import torch


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
