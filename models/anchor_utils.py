from __future__ import annotations

import torch


def reduce_variant_mask(mask: torch.Tensor) -> torch.Tensor:
    """Reduce per-variant masks with fixed OR semantics."""
    if mask.ndim < 1:
        raise ValueError("mask must have at least one dimension")
    if mask.shape[-1] == 0:
        raise ValueError("mask must include at least one variant channel")
    return (mask > 0.5).any(dim=-1).to(dtype=mask.dtype)


def resolve_last_valid_anchor(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the last valid value in a history window.

    Args:
        values: History values [B, T]
        mask: Binary observation mask [B, T]

    Returns:
        Tuple of:
            - anchor_value [B]: last valid value, or 0.0 when no observation exists
            - anchor_mask [B]: 1.0 when any valid observation exists, else 0.0
    """
    if values.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"values and mask must be rank-2 [B, T], got {values.shape} and {mask.shape}"
        )
    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must have matching shape, got {values.shape} and {mask.shape}"
        )

    valid = mask > 0.5
    time_steps = values.shape[1]
    time_index = torch.arange(time_steps, device=values.device, dtype=torch.long)
    expanded_time = time_index.unsqueeze(0).expand_as(mask)
    last_index = torch.where(
        valid,
        expanded_time,
        torch.full_like(expanded_time, -1),
    ).max(dim=1).values
    has_anchor = last_index >= 0

    safe_index = last_index.clamp_min(0)
    gathered = values.gather(dim=1, index=safe_index.unsqueeze(1)).squeeze(1)
    anchor_value = torch.where(has_anchor, gathered, torch.zeros_like(gathered))
    anchor_mask = has_anchor.to(dtype=values.dtype)
    return anchor_value, anchor_mask
