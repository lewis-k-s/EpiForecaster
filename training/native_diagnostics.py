"""Fixed native evaluation panels and disjoint pre-clip gradient groups."""

from collections.abc import Mapping, Sequence

import torch


def fixed_training_panel(
    head_source_items: Mapping[str, Mapping[tuple[str, str], Sequence[int]]],
    queries_per_source: int,
) -> list[int]:
    """Select evenly spaced eligible origins per head/source without RNG use.

    Input sequences must be in chronological order. The selected panel contains
    only training positions and gives each source the same maximum query budget.
    """
    selected = []
    for head in sorted(head_source_items):
        for source in sorted(head_source_items[head]):
            items = head_source_items[head][source]
            count = min(queries_per_source, len(items))
            if count == 1:
                selected.append(items[len(items) // 2])
            elif count > 1:
                selected.extend(
                    items[index * (len(items) - 1) // (count - 1)]
                    for index in range(count)
                )
    return selected


def native_gradient_group(name: str) -> str:
    """Assign each parameter to exactly one native diagnostic group."""
    if name.startswith("spatial_encoder."):
        return "gradnorm_native_spatial_encoder"
    if name.startswith(("input_adapter.", "base.backbone.")):
        return "gradnorm_backbone_encoder"
    if name.startswith("base.sir_rollforward."):
        return "gradnorm_sird_physics"
    if name.startswith(("wastewater_head.", "base.ww_head.")):
        return "gradnorm_obs_ww"
    for head in ("cases", "hosp", "deaths"):
        if name.startswith(f"base.{head}_head."):
            return f"gradnorm_obs_{head}"
    return "gradnorm_other"


@torch.no_grad()
def native_component_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    """Read pre-clip gradients without changing them; absent groups are omitted."""
    squared: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            squared.setdefault(native_gradient_group(name), []).append(
                parameter.grad.detach().float().square().sum()
            )
    return {
        group: float(torch.stack(values).sum().sqrt().item())
        for group, values in squared.items()
    }
