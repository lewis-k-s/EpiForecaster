from __future__ import annotations

import torch

_EPSILON = 1.0e-12
_LATENT_TASKS = ("s", "i", "h", "r", "death_flow")


def should_probe_pathway_gradients(step: int, frequency: int) -> bool:
    return frequency > 0 and step % frequency == 0


def _gradient_summary(
    *,
    gradient: torch.Tensor | None,
    activation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if gradient is None:
        zero = activation.new_zeros((), dtype=torch.float32)
        return {
            "norm": zero,
            "rms": zero,
            "grad_activation_rms": zero,
            "grad_activation": activation.new_zeros(
                activation.shape,
                dtype=torch.float32,
            ),
        }

    gradient_f32 = torch.nan_to_num(
        gradient.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    activation_f32 = torch.nan_to_num(
        activation.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    grad_activation = gradient_f32 * activation_f32
    return {
        "norm": torch.linalg.vector_norm(gradient_f32),
        "rms": torch.sqrt(gradient_f32.square().mean()),
        "grad_activation_rms": torch.sqrt(grad_activation.square().mean()),
        "grad_activation": grad_activation,
    }


def compute_pathway_gradient_metrics(
    *,
    head_name: str,
    loss: torch.Tensor,
    mechanistic_source: torch.Tensor,
    residual_input: torch.Tensor,
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
    """Measure observation-loss routing at mechanistic and residual intermediates."""
    grad_inputs: list[torch.Tensor] = []
    input_names: list[str] = []
    for name, value in (
        ("source", mechanistic_source),
        ("residual", residual_input),
    ):
        if value.requires_grad:
            grad_inputs.append(value)
            input_names.append(name)

    gradients_by_name: dict[str, torch.Tensor | None] = {
        "source": None,
        "residual": None,
    }
    if loss.requires_grad and grad_inputs:
        gradients = torch.autograd.grad(
            loss,
            grad_inputs,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        gradients_by_name.update(dict(zip(input_names, gradients, strict=True)))

    source = _gradient_summary(
        gradient=gradients_by_name["source"],
        activation=mechanistic_source,
    )
    residual = _gradient_summary(
        gradient=gradients_by_name["residual"],
        activation=residual_input,
    )
    source_ga = source["grad_activation"].reshape(-1)
    residual_ga = residual["grad_activation"].reshape(-1)
    if source_ga.shape == residual_ga.shape:
        cosine = torch.nn.functional.cosine_similarity(
            source_ga.unsqueeze(0),
            residual_ga.unsqueeze(0),
            dim=1,
            eps=_EPSILON,
        ).squeeze(0)
    else:
        cosine = loss.new_full((), float("nan"), dtype=torch.float32)

    source_norm = source["norm"]
    residual_norm = residual["norm"]
    source_ga_rms = source["grad_activation_rms"]
    residual_ga_rms = residual["grad_activation_rms"]
    prefix = f"pathgrad_{head_name}"
    return {
        f"{prefix}_mechanistic_source_grad_norm": source_norm.detach(),
        f"{prefix}_residual_input_grad_norm": residual_norm.detach(),
        f"{prefix}_mechanistic_source_grad_rms": source["rms"].detach(),
        f"{prefix}_residual_input_grad_rms": residual["rms"].detach(),
        f"{prefix}_mechanistic_source_grad_activation_rms": source_ga_rms.detach(),
        f"{prefix}_residual_input_grad_activation_rms": residual_ga_rms.detach(),
        f"{prefix}_residual_to_mechanistic_grad_norm_ratio": (
            residual_norm / source_norm.clamp_min(_EPSILON)
        ).detach(),
        f"{prefix}_residual_to_mechanistic_grad_activation_ratio": (
            residual_ga_rms / source_ga_rms.clamp_min(_EPSILON)
        ).detach(),
        f"{prefix}_residual_grad_activation_fraction": (
            residual_ga_rms / (source_ga_rms + residual_ga_rms).clamp_min(_EPSILON)
        ).detach(),
        f"{prefix}_path_grad_activation_cosine": cosine.detach(),
    }


def _parameter_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> list[torch.Tensor]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [
        torch.zeros_like(parameter, dtype=torch.float32)
        if gradient is None
        else torch.nan_to_num(gradient.float(), nan=0.0, posinf=0.0, neginf=0.0)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]


def _gradient_norm(gradients: list[torch.Tensor]) -> torch.Tensor:
    if not gradients:
        return torch.tensor(0.0)
    return torch.sqrt(sum(gradient.square().sum() for gradient in gradients))


def _gradient_cosine(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> torch.Tensor:
    if not left or not right:
        return torch.tensor(float("nan"))
    left_norm = _gradient_norm(left)
    right_norm = _gradient_norm(right)
    denominator = left_norm * right_norm
    if float(left_norm.detach()) <= _EPSILON or float(right_norm.detach()) <= _EPSILON:
        return denominator.new_full((), float("nan"))
    dot = sum((a * b).sum() for a, b in zip(left, right, strict=True))
    return dot / denominator


def compute_latent_parameter_gradient_metrics(
    *,
    components: dict[str, torch.Tensor],
    selected_parameters: list[torch.nn.Parameter],
    all_parameters: list[torch.nn.Parameter],
    latent_total_weight: float,
    clip_value: float,
    actual_update_global_norm_preclip: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure each latent objective on shared X0 and rate parameters."""
    if not selected_parameters or latent_total_weight <= 0:
        return {}

    total_gradients = _parameter_gradients(
        components["total"], all_parameters, retain_graph=True
    )
    total_global_norm = _gradient_norm(total_gradients)
    actual_update_global_norm_preclip = actual_update_global_norm_preclip.float().to(
        total_global_norm.device
    )
    clip_scale = torch.clamp(
        torch.as_tensor(
            clip_value,
            dtype=torch.float32,
            device=total_global_norm.device,
        )
        / actual_update_global_norm_preclip.clamp_min(_EPSILON),
        max=1.0,
    )
    obs_gradients = _parameter_gradients(
        components["obs_total"], selected_parameters, retain_graph=True
    )
    obs_norm = _gradient_norm(obs_gradients)
    metrics = {
        "latentgrad_sidecar_total_global_norm_preclip": total_global_norm.detach(),
        "latentgrad_actual_update_global_norm_preclip": (
            actual_update_global_norm_preclip.detach()
        ),
        "latentgrad_actual_update_global_norm_postclip": (
            actual_update_global_norm_preclip * clip_scale
        ).detach(),
        "latentgrad_actual_update_clip_scale": clip_scale.detach(),
        "latentgrad_obs_shared_grad_norm_preclip": obs_norm.detach(),
        "latentgrad_obs_shared_grad_norm_counterfactual_postclip": (
            obs_norm * clip_scale
        ).detach(),
    }

    task_gradients: dict[str, list[torch.Tensor]] = {}
    for task in _LATENT_TASKS:
        component_name = f"latent_{task}"
        weighted_loss = components[f"{component_name}_weighted"] * latent_total_weight
        gradients = _parameter_gradients(
            weighted_loss,
            selected_parameters,
            retain_graph=True,
        )
        task_gradients[task] = gradients
        norm = _gradient_norm(gradients)
        metrics.update(
            {
                f"latentgrad_{task}_loss_raw": components[
                    f"{component_name}_raw"
                ].detach(),
                f"latentgrad_{task}_loss_scaled": components[component_name].detach(),
                f"latentgrad_{task}_loss_total_weighted": weighted_loss.detach(),
                f"latentgrad_{task}_shared_grad_norm_preclip": norm.detach(),
                f"latentgrad_{task}_shared_grad_norm_counterfactual_postclip": (
                    norm * clip_scale
                ).detach(),
                f"latentgrad_{task}_shared_grad_cosine_obs": _gradient_cosine(
                    gradients, obs_gradients
                ).detach(),
            }
        )
    combined_latent_gradients = _parameter_gradients(
        components["sird_supervision_weighted"],
        selected_parameters,
        retain_graph=True,
    )
    combined_latent_norm = _gradient_norm(combined_latent_gradients)
    metrics.update(
        {
            "latentgrad_combined_shared_grad_norm_preclip": (
                combined_latent_norm.detach()
            ),
            "latentgrad_combined_shared_grad_norm_counterfactual_postclip": (
                combined_latent_norm * clip_scale
            ).detach(),
            "latentgrad_combined_to_obs_grad_norm_ratio": (
                combined_latent_norm / obs_norm.clamp_min(_EPSILON)
            ).detach(),
            "latentgrad_combined_squared_gradient_share": (
                combined_latent_norm.square()
                / (combined_latent_norm.square() + obs_norm.square()).clamp_min(
                    _EPSILON
                )
            ).detach(),
            "latentgrad_combined_shared_grad_cosine_obs": _gradient_cosine(
                combined_latent_gradients, obs_gradients
            ).detach(),
        }
    )
    for left_index, left in enumerate(_LATENT_TASKS):
        for right in _LATENT_TASKS[left_index + 1 :]:
            metrics[f"latentgrad_cosine_{left}_{right}"] = _gradient_cosine(
                task_gradients[left], task_gradients[right]
            ).detach()
    return metrics


def compute_latent_variation_metrics(
    *,
    model_outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    """Measure prediction and target variation for each latent task."""
    mapping = {
        "s": ("S_trajectory", "S_target"),
        "i": ("I_trajectory", "I_target"),
        "h": ("H_trajectory", "H_target"),
        "r": ("R_trajectory", "R_target"),
        "death_flow": ("death_flow", "death_flow_target"),
    }
    metrics: dict[str, torch.Tensor] = {}
    for task, (prediction_key, target_key) in mapping.items():
        target = targets.get(target_key)
        if target is None:
            continue
        prediction = model_outputs[prediction_key].float()
        target_f32 = target.float().to(prediction.device)
        finite = torch.isfinite(prediction) & torch.isfinite(target_f32)
        prediction_values = prediction[finite]
        target_values = target_f32[finite]
        if prediction_values.numel() == 0:
            continue
        prediction_std = prediction_values.std(correction=0)
        target_std = target_values.std(correction=0)
        metrics.update(
            {
                f"latentgrad_{task}_prediction_std": prediction_std.detach(),
                f"latentgrad_{task}_target_std": target_std.detach(),
                f"latentgrad_{task}_prediction_target_std_ratio": (
                    prediction_std / target_std.clamp_min(_EPSILON)
                ).detach(),
            }
        )
    return metrics
