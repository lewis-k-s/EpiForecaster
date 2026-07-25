from __future__ import annotations

import torch

_EPSILON = 1.0e-12


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
            residual_ga_rms
            / (source_ga_rms + residual_ga_rms).clamp_min(_EPSILON)
        ).detach(),
        f"{prefix}_path_grad_activation_cosine": cosine.detach(),
    }
