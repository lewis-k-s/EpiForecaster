"""Assemble dense mobility context features from prebuilt tables."""

from __future__ import annotations

import torch

from einops import repeat
from utils.precision_policy import MODEL_PARAM_DTYPE

_BIT_WEIGHTS = torch.tensor([1 << bit for bit in range(8)], dtype=torch.uint8)


def pack_mask_bits(mask: torch.Tensor) -> torch.Tensor:
    """Pack a bool mask along its last dim into uint8 bytes (8 entries each)."""
    pad = -mask.shape[-1] % 8
    if pad:
        mask = torch.nn.functional.pad(mask, (0, pad))
    groups = mask.reshape(*mask.shape[:-1], -1, 8).to(torch.uint8)
    return (groups * _BIT_WEIGHTS.to(mask.device)).sum(dim=-1, dtype=torch.uint8)


def unpack_mask_bits(packed: torch.Tensor, size: int) -> torch.Tensor:
    """Invert ``pack_mask_bits``: uint8 [..., ceil(size/8)] -> bool [..., size]."""
    weights = _BIT_WEIGHTS.to(packed.device)
    bits = (packed.unsqueeze(-1) & weights) != 0
    return bits.reshape(*packed.shape[:-1], -1)[..., :size]


def assemble_context_features(
    dataset: object,
    window_starts: torch.Tensor,
    target_positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Assemble [S*L, N_ctx, F] features in sample-major graph order.

    ``window_starts`` is [S]; ``target_positions`` is [S*L].
    """
    cache = getattr(dataset, "_context_device_cache", None)
    if cache is None:
        cache = {}
        setattr(dataset, "_context_device_cache", cache)
    if device not in cache:
        names = (
            "context_feature_clinical",
            "context_feature_bio",
            "context_feature_ww",
            "context_neigh_table",
            "context_khop_table",
            "context_bio_start",
            "context_target_slot_by_position",
        )
        cache[device] = {
            name: tensor.to(device, non_blocking=True)
            for name in names
            if (tensor := getattr(dataset, name, None)) is not None
        }
    tables = cache[device]
    window_starts = window_starts.to(device, dtype=torch.long).reshape(-1)
    target_positions = target_positions.to(device, dtype=torch.long).reshape(-1)
    target_slots = tables["context_target_slot_by_position"].index_select(
        0, target_positions
    )
    if int(target_slots.min()) < 0:
        raise ValueError(
            "Batch target is not part of the dataset's target list; "
            "context tables cannot be indexed for it."
        )

    # The neighbor and k-hop tables are bit-packed along the context axis.
    context_size = tables["context_target_slot_by_position"].numel()
    history_length = int(getattr(dataset, "config").model.input_window_length)
    # (S, 1) + (1, L) -> (S, L) absolute table indices per graph step,
    # flattened to (S * L,) in the same sample-major order as the outputs.
    time_grid = (
        window_starts.unsqueeze(-1)
        + torch.arange(history_length, device=device, dtype=torch.long).unsqueeze(0)
    ).reshape(-1)

    clinical = tables.get("context_feature_clinical")
    bio = tables.get("context_feature_bio")
    ww = tables.get("context_feature_ww")

    blocks: list[torch.Tensor] = []
    if clinical is not None:
        clinical_window = clinical[time_grid]
        # Pair each graph's timestep with its target slot in the
        # (time, target_slot, ctx_pos) table -> (S * L, N_ctx) mask.
        neighbor_mask = unpack_mask_bits(
            tables["context_neigh_table"][time_grid, target_slots], context_size
        )
        blocks.append(
            torch.where(
                neighbor_mask.unsqueeze(-1),
                clinical_window,
                torch.zeros_like(clinical_window),
            )
        )

    if ww is not None:
        bio_start = tables["context_bio_start"]
        window_end = window_starts + history_length - 1
        # (S, 1) window end vs (1, N_ctx) series starts -> (S, N_ctx) flag:
        # True where the node's biomarker series exists by the window end.
        has_data = (window_end.unsqueeze(-1) >= bio_start.unsqueeze(0)) & (
            bio_start.unsqueeze(0) >= 0
        )
        # Tile each sample's flag across its L steps to match time_grid rows.
        has_data = repeat(has_data, "s n -> (s l) n", l=history_length)
        bio_block = torch.cat(
            [
                bio[time_grid],
                has_data.to(torch.float16).unsqueeze(-1),
                ww[time_grid],
            ],
            dim=-1,
        )
        blocks.append(bio_block)

    features = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)

    k_hop = tables.get("context_khop_table")
    if k_hop is not None:
        # Same (time, target_slot) lookup -> (S * L, N_ctx) keep-mask.
        keep_mask = unpack_mask_bits(k_hop[time_grid, target_slots], context_size)
        features = torch.where(
            keep_mask.unsqueeze(-1),
            features,
            torch.zeros_like(features),
        )

    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.to(MODEL_PARAM_DTYPE)
