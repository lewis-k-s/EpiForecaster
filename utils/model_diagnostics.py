"""Lightweight model activation and normalization diagnostics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn


def should_capture_model_diagnostics(step: int, frequency: int) -> bool:
    """Return whether model diagnostics should run for this training step."""
    return frequency > 0 and (step % frequency == 0 or step == 0)


def _safe_tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    if values.numel() == 0:
        return {}

    finite = torch.isfinite(values)
    finite_ratio = float(finite.float().mean().item())
    if not bool(finite.any().item()):
        return {
            f"{prefix}/finite_ratio": finite_ratio,
            f"{prefix}/mean": float("nan"),
            f"{prefix}/std": float("nan"),
            f"{prefix}/rms": float("nan"),
            f"{prefix}/abs_max": float("nan"),
            f"{prefix}/near_zero_frac": float("nan"),
        }

    finite_values = values[finite]
    return {
        f"{prefix}/finite_ratio": finite_ratio,
        f"{prefix}/mean": float(finite_values.mean().item()),
        f"{prefix}/std": float(finite_values.std(unbiased=False).item()),
        f"{prefix}/rms": float(finite_values.pow(2).mean().sqrt().item()),
        f"{prefix}/abs_max": float(finite_values.abs().max().item()),
        f"{prefix}/near_zero_frac": float(
            (finite_values.abs() <= 1.0e-6).float().mean().item()
        ),
    }


def _feature_std_mean(tensor: torch.Tensor) -> float:
    values = tensor.detach().float()
    if values.ndim < 2 or values.numel() == 0:
        return float("nan")

    values = values.flatten(0, -2)
    finite_rows = torch.isfinite(values).all(dim=-1)
    if not bool(finite_rows.any().item()):
        return float("nan")

    feature_std = values[finite_rows].std(dim=-1, unbiased=False)
    return float(feature_std.mean().item())


class ModelDiagnosticsCapture:
    """Collect scalar diagnostics from selected forward hooks and parameters."""

    def __init__(self, *, include_mobility: bool = True):
        self.include_mobility = include_mobility
        self.activations: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _capture_hook(self, name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            if isinstance(output, tuple):
                output = output[0]
            if torch.is_tensor(output):
                self.activations[name] = output.detach()

        return hook

    def register(self, model: nn.Module) -> None:
        module_targets: list[tuple[str, nn.Module]] = []

        backbone = getattr(model, "backbone", None)
        if backbone is None and hasattr(model, "input_projection"):
            backbone = model
        if backbone is not None:
            input_projection = getattr(backbone, "input_projection", None)
            if isinstance(input_projection, nn.Module):
                module_targets.append(("backbone/input_projection", input_projection))

            encoder_layers = getattr(backbone, "encoder_layers", None)
            if encoder_layers is not None and len(encoder_layers) > 0:
                module_targets.append(("backbone/encoder_first", encoder_layers[0]))
                if len(encoder_layers) > 1:
                    module_targets.append(("backbone/encoder_last", encoder_layers[-1]))

            final_norm = getattr(backbone, "final_norm", None)
            if isinstance(final_norm, nn.Module):
                module_targets.append(("backbone/final_norm", final_norm))

            obs_context_projection = getattr(backbone, "obs_context_projection", None)
            if isinstance(obs_context_projection, nn.Module):
                module_targets.append(("backbone/obs_context", obs_context_projection))

        mobility_gnn = getattr(model, "mobility_gnn", None)
        if self.include_mobility and isinstance(mobility_gnn, nn.Module):
            module_targets.append(("mobility/output", mobility_gnn))

        for name, module in module_targets:
            self.handles.append(module.register_forward_hook(self._capture_hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def build_log_data(self, model: nn.Module) -> dict[str, float]:
        log_data: dict[str, float] = {}

        activation_rms: dict[str, float] = {}
        for name, tensor in sorted(self.activations.items()):
            prefix = f"model_diag/{name}"
            stats = _safe_tensor_stats(prefix, tensor)
            log_data.update(stats)
            rms = stats.get(f"{prefix}/rms")
            if rms is not None:
                activation_rms[name] = rms
            log_data[f"{prefix}/feature_std_mean"] = _feature_std_mean(tensor)

        first_rms = activation_rms.get("backbone/encoder_first")
        last_rms = activation_rms.get("backbone/encoder_last")
        if first_rms is not None and last_rms is not None and first_rms > 0:
            log_data["model_diag/backbone/encoder_last_to_first_rms"] = (
                last_rms / first_rms
            )

        log_data.update(_normalization_parameter_stats(model))
        log_data.update(_rezero_parameter_stats(model))
        return log_data


@contextmanager
def capture_model_diagnostics(
    model: nn.Module,
    *,
    include_mobility: bool = True,
) -> Iterator[ModelDiagnosticsCapture]:
    """Temporarily attach model diagnostics hooks during one forward pass."""
    capture = ModelDiagnosticsCapture(include_mobility=include_mobility)
    capture.register(model)
    try:
        yield capture
    finally:
        capture.remove()


def _normalization_parameter_stats(model: nn.Module) -> dict[str, float]:
    weights = []
    grad_norm_sq = 0.0
    grad_count = 0

    for module in model.modules():
        if not isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.BatchNorm1d)):
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        weight_values = weight.detach().float().flatten()
        if weight_values.numel() > 0:
            weights.append(weight_values)
        if weight.grad is not None:
            grad = weight.grad.detach().float()
            finite_grad = grad[torch.isfinite(grad)]
            if finite_grad.numel() > 0:
                grad_norm_sq += float(finite_grad.pow(2).sum().item())
                grad_count += 1

    if not weights:
        return {}

    merged = torch.cat(weights)
    log_data = _safe_tensor_stats("model_diag/norm_weight", merged)
    log_data["model_diag/norm_weight/grad_norm"] = grad_norm_sq**0.5
    log_data["model_diag/norm_weight/grad_param_count"] = float(grad_count)
    return log_data


def _rezero_parameter_stats(model: nn.Module) -> dict[str, float]:
    values = []
    grad_norm_sq = 0.0
    grad_count = 0

    for name, param in model.named_parameters():
        if not ("alpha_attn" in name or "alpha_ffn" in name):
            continue
        values.append(param.detach().float().flatten())
        if param.grad is not None:
            grad = param.grad.detach().float()
            finite_grad = grad[torch.isfinite(grad)]
            if finite_grad.numel() > 0:
                grad_norm_sq += float(finite_grad.pow(2).sum().item())
                grad_count += 1

    if not values:
        return {}

    merged = torch.cat(values)
    log_data = _safe_tensor_stats("model_diag/rezero_alpha", merged)
    log_data["model_diag/rezero_alpha/grad_norm"] = grad_norm_sq**0.5
    log_data["model_diag/rezero_alpha/grad_param_count"] = float(grad_count)
    return log_data
