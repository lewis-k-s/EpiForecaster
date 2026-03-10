"""Node selection utilities for evaluation and plotting.

This module provides functions for selecting nodes based on loss metrics,
supporting various strategies like top-k, quartile, worst, best, and random sampling.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.epiforecaster import EpiForecaster

logger = logging.getLogger(__name__)

# Global seeded RNG for reproducibility across evaluation/plotting
_GLOBAL_RNG = np.random.default_rng(42)


def select_nodes_by_loss(
    *,
    node_mae: dict[int, float],
    strategy: str = "quartile",
    k: int = 5,
    samples_per_group: int = 4,
    rng: np.random.Generator | None = None,
) -> dict[str, list[int]]:
    """
    Select nodes by different loss-based strategies using in-memory node_mae.

    Args:
        node_mae: Dict mapping node_id -> average MAE
        strategy: "topk", "quartile", "worst", "best", "random"
        k: Number of nodes for topk/worst/best strategies
        samples_per_group: Number of nodes per group for quartile strategy (default 4)
        rng: Random generator for deterministic sampling (default: global seeded RNG)

    Returns:
        Dict mapping group name -> list of node_ids
        Examples:
            strategy="topk": {"Top-k": [1, 2, 3, 4, 5]}
            strategy="quartile": {"Q1 (Worst)": [...], "Q2 (Poor)": [...], ...}
            strategy="worst": {"Worst": [1, 2, 3, 4, 5]}
    """
    if rng is None:
        rng = _GLOBAL_RNG

    if not node_mae:
        logger.warning("[eval] No node MAE values available for selection")
        return {
            "Q1 (Worst)": [],
            "Q2 (Poor)": [],
            "Q3 (Average)": [],
            "Q4 (Best)": [],
        }

    if strategy == "random":
        all_nodes = list(node_mae.keys())
        k = min(k, len(all_nodes))
        selected = rng.choice(all_nodes, size=k, replace=False).tolist()
        return {"Random": selected}

    # Sort by MAE for other strategies
    sorted_nodes = sorted(node_mae.items(), key=lambda kv: (kv[1], kv[0]))

    if strategy == "topk":
        top_k = [node_id for node_id, _mae in sorted_nodes[:k]]
        return {"Top-k": top_k}

    elif strategy == "best":
        top_k = [node_id for node_id, _mae in sorted_nodes[:k]]
        return {"Best": top_k}

    elif strategy == "worst":
        bottom_k = [node_id for node_id, _mae in sorted_nodes[-k:]]
        return {"Worst": bottom_k}

    elif strategy == "quartile":
        maes = [mae for _node_id, mae in sorted_nodes]
        q1_cutoff = np.percentile(maes, 25)
        q2_cutoff = np.percentile(maes, 50)
        q3_cutoff = np.percentile(maes, 75)

        quartile_groups: dict[str, list[int]] = {
            "Q1 (Worst)": [],
            "Q2 (Poor)": [],
            "Q3 (Average)": [],
            "Q4 (Best)": [],
        }

        for node_id, mae in sorted_nodes:
            if mae <= q1_cutoff:
                quartile_groups["Q1 (Worst)"].append(node_id)
            elif mae <= q2_cutoff:
                quartile_groups["Q2 (Poor)"].append(node_id)
            elif mae <= q3_cutoff:
                quartile_groups["Q3 (Average)"].append(node_id)
            else:
                quartile_groups["Q4 (Best)"].append(node_id)

        # Sample from each quartile
        for quartile_name, nodes in quartile_groups.items():
            k = min(samples_per_group, len(nodes))
            quartile_groups[quartile_name] = (
                rng.choice(nodes, k, replace=False).tolist() if nodes else []
            )

        return quartile_groups

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def topk_target_nodes_by_mae(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    region_embeddings: torch.Tensor | None = None,
    k: int = 5,
) -> list[int]:
    """Compute top-k target node ids by average per-window MAE over the loader."""
    device = next(model.parameters()).device
    forward_model = cast(EpiForecaster, model)

    node_mae_sum: dict[int, torch.Tensor] = {}
    node_mae_count: dict[int, int] = {}

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            eval_iter = loader
            for batch in eval_iter:
                from utils.training_utils import inject_gpu_mobility

                inject_gpu_mobility(batch, eval_iter.dataset, device)

                model_outputs, targets_dict = forward_model.forward_batch(
                    batch_data=batch,
                    region_embeddings=region_embeddings,
                )
                predictions = model_outputs.get("pred_hosp")
                targets = targets_dict.get("hosp")
                mask = targets_dict.get("hosp_mask")
                if predictions is None or targets is None:
                    raise ValueError(
                        "topk_target_nodes_by_mae requires hospitalization targets "
                        "('HospTarget') to be present in the batch."
                    )
                if mask is None:
                    mask = torch.ones_like(targets)
                abs_diff = (predictions - targets).abs()
                valid_per_sample = mask.sum(dim=1) > 0
                per_sample_mae = (abs_diff * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1.0)
                target_nodes = batch.target_node
                for sample_mae, target_node, is_valid in zip(
                    per_sample_mae, target_nodes, valid_per_sample, strict=False
                ):
                    if not bool(is_valid):
                        continue
                    node_id = int(target_node)
                    if node_id not in node_mae_sum:
                        node_mae_sum[node_id] = torch.tensor(0.0, device=device)
                    node_mae_sum[node_id] += sample_mae.detach()
                    node_mae_count[node_id] = node_mae_count.get(node_id, 0) + 1
    finally:
        if model_was_training:
            model.train()

    if not node_mae_sum:
        return []

    node_mae = {
        node_id: (node_mae_sum[node_id] / max(1, node_mae_count[node_id])).item()
        for node_id in node_mae_sum
    }
    return [
        node_id
        for node_id, _mae in sorted(node_mae.items(), key=lambda kv: (kv[1], kv[0]))[:k]
    ]
