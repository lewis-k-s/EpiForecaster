from __future__ import annotations

import torch

_DISTRIBUTION_PARAMETER_MARKERS = (
    ".log_distribution_scale",
    ".distribution_scale_proj.",
)


def initialize_missing_distribution_parameters(
    *,
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Initialize distribution parameters absent from legacy point checkpoints."""
    missing_keys = tuple(
        key
        for key in model_state_dict
        if key not in state_dict
        and any(marker in key for marker in _DISTRIBUTION_PARAMETER_MARKERS)
    )
    if not missing_keys:
        return state_dict, ()

    patched = dict(state_dict)
    for key in missing_keys:
        patched[key] = model_state_dict[key].detach().clone()
    return patched, missing_keys
