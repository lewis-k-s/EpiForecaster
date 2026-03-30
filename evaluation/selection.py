"""Node selection utilities for evaluation and plotting.

This module provides functions for selecting nodes based on loss metrics,
supporting various strategies like top-k, quartile, worst, best, and random sampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.epiforecaster import EpiForecaster
from utils.device import prepare_batch_for_device

logger = logging.getLogger(__name__)

# Global seeded RNG for reproducibility across evaluation/plotting
_GLOBAL_RNG = np.random.default_rng(42)
VALID_NODE_METRIC_TARGETS = (
    "hospitalizations",
    "wastewater",
    "cases",
    "deaths",
)
_QUARTILE_GROUP_NAMES = (
    "Q1 (Best MAE)",
    "Q2 (Good MAE)",
    "Q3 (Poor MAE)",
    "Q4 (Worst MAE)",
)


@dataclass(frozen=True)
class WindowSelectionSpec:
    node_id: int
    window_start: int
    score: float
    observed_targets: tuple[str, ...]
    observed_points: int


def select_nodes_by_loss(
    *,
    node_mae: dict[int, float],
    target_name: str = "hospitalizations",
    strategy: str = "quartile",
    k: int = 5,
    samples_per_group: int = 4,
    rng: np.random.Generator | None = None,
) -> dict[str, list[int]]:
    """
    Select nodes by different loss-based strategies using in-memory node_mae.

    Args:
        node_mae: Dict mapping node_id -> average MAE for the selected target
        target_name: Canonical target name used for the scores
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

    if target_name not in VALID_NODE_METRIC_TARGETS:
        raise ValueError(
            f"Unknown node metric target: {target_name!r}. "
            f"Expected one of {VALID_NODE_METRIC_TARGETS}."
        )

    if not node_mae:
        logger.warning(
            "[eval] No node MAE values available for selection for target=%s",
            target_name,
        )
        return {name: [] for name in _QUARTILE_GROUP_NAMES}

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
            name: [] for name in _QUARTILE_GROUP_NAMES
        }

        for node_id, mae in sorted_nodes:
            if mae <= q1_cutoff:
                quartile_groups["Q1 (Best MAE)"].append(node_id)
            elif mae <= q2_cutoff:
                quartile_groups["Q2 (Good MAE)"].append(node_id)
            elif mae <= q3_cutoff:
                quartile_groups["Q3 (Poor MAE)"].append(node_id)
            else:
                quartile_groups["Q4 (Worst MAE)"].append(node_id)

        # Sample from each quartile
        for quartile_name, nodes in quartile_groups.items():
            k = min(samples_per_group, len(nodes))
            quartile_groups[quartile_name] = (
                rng.choice(nodes, k, replace=False).tolist() if nodes else []
            )

        return quartile_groups

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def load_window_selection_specs_from_granular(
    *,
    granular_csv: Path,
    split: str | None = None,
) -> list[WindowSelectionSpec]:
    granular_df = pd.read_csv(granular_csv)
    if granular_df.empty:
        return []

    required_cols = {"target", "node_id", "window_start", "abs_error"}
    missing_cols = required_cols.difference(granular_df.columns)
    if missing_cols:
        raise ValueError(
            f"Granular CSV is missing required columns: {sorted(missing_cols)}"
        )

    if split is not None and "split" in granular_df.columns:
        granular_df = granular_df[
            granular_df["split"].astype(str).str.lower() == split.strip().lower()
        ].copy()
    if granular_df.empty:
        return []

    granular_df["node_id"] = pd.to_numeric(granular_df["node_id"], errors="coerce")
    granular_df["window_start"] = pd.to_numeric(
        granular_df["window_start"], errors="coerce"
    )
    granular_df["abs_error"] = pd.to_numeric(granular_df["abs_error"], errors="coerce")
    granular_df = granular_df.dropna(
        subset=["target", "node_id", "window_start", "abs_error"]
    ).copy()
    if granular_df.empty:
        return []

    per_target = (
        granular_df.groupby(["node_id", "window_start", "target"], dropna=False)
        .agg(
            mae=("abs_error", "mean"),
            observed_points=("abs_error", "size"),
        )
        .reset_index()
    )
    if per_target.empty:
        return []

    specs: list[WindowSelectionSpec] = []
    for (node_id, window_start), group in per_target.groupby(
        ["node_id", "window_start"], dropna=False
    ):
        scores = pd.to_numeric(group["mae"], errors="coerce").dropna()
        if scores.empty:
            continue
        observed_targets = tuple(sorted(group["target"].astype(str).tolist()))
        observed_points = int(pd.to_numeric(group["observed_points"]).sum())
        specs.append(
            WindowSelectionSpec(
                node_id=int(node_id),
                window_start=int(window_start),
                score=float(scores.mean()),
                observed_targets=observed_targets,
                observed_points=observed_points,
            )
        )
    return sorted(specs, key=lambda spec: (spec.score, spec.node_id, spec.window_start))


def select_windows_by_loss(
    *,
    window_specs: list[WindowSelectionSpec],
    samples_per_group: int = 4,
    rng: np.random.Generator | None = None,
) -> dict[str, list[WindowSelectionSpec]]:
    if rng is None:
        rng = _GLOBAL_RNG

    if not window_specs:
        logger.warning("[eval] No granular window scores available for selection")
        return {name: [] for name in _QUARTILE_GROUP_NAMES}

    scores = [spec.score for spec in window_specs]
    q1_cutoff = np.percentile(scores, 25)
    q2_cutoff = np.percentile(scores, 50)
    q3_cutoff = np.percentile(scores, 75)

    quartile_groups: dict[str, list[WindowSelectionSpec]] = {
        name: [] for name in _QUARTILE_GROUP_NAMES
    }
    for spec in window_specs:
        if spec.score <= q1_cutoff:
            quartile_groups["Q1 (Best MAE)"].append(spec)
        elif spec.score <= q2_cutoff:
            quartile_groups["Q2 (Good MAE)"].append(spec)
        elif spec.score <= q3_cutoff:
            quartile_groups["Q3 (Poor MAE)"].append(spec)
        else:
            quartile_groups["Q4 (Worst MAE)"].append(spec)

    sampled: dict[str, list[WindowSelectionSpec]] = {}
    for quartile_name, specs in quartile_groups.items():
        k = min(samples_per_group, len(specs))
        if not specs:
            sampled[quartile_name] = []
            continue
        indices = rng.choice(len(specs), size=k, replace=False).tolist()
        sampled[quartile_name] = [specs[idx] for idx in indices]
    return sampled


def topk_target_nodes_by_mae(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    region_embeddings: torch.Tensor | None = None,
    k: int = 5,
    target_name: str = "hospitalizations",
) -> list[int]:
    """Compute top-k target node ids by average per-window MAE over the loader."""
    target_to_prediction_key = {
        "hospitalizations": "pred_hosp",
        "wastewater": "pred_ww",
        "cases": "pred_cases",
        "deaths": "pred_deaths",
    }
    target_to_target_key = {
        "hospitalizations": "hosp",
        "wastewater": "ww",
        "cases": "cases",
        "deaths": "deaths",
    }
    target_to_mask_key = {
        "hospitalizations": "hosp_mask",
        "wastewater": "ww_mask",
        "cases": "cases_mask",
        "deaths": "deaths_mask",
    }
    if target_name not in VALID_NODE_METRIC_TARGETS:
        raise ValueError(
            f"Unknown node metric target: {target_name!r}. "
            f"Expected one of {VALID_NODE_METRIC_TARGETS}."
        )

    device = next(model.parameters()).device
    forward_model = cast(EpiForecaster, model)

    node_mae_sum: dict[int, float] = {}
    node_mae_count: dict[int, int] = {}

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            eval_iter = loader
            for batch in eval_iter:
                batch = prepare_batch_for_device(
                    batch,
                    dataset=getattr(loader, "dataset", None),
                    device=device,
                )

                model_outputs, targets_dict = forward_model.forward_batch(
                    batch_data=batch,
                    region_embeddings=region_embeddings,
                )
                predictions = model_outputs.get(target_to_prediction_key[target_name])
                targets = targets_dict.get(target_to_target_key[target_name])
                mask = targets_dict.get(target_to_mask_key[target_name])
                if predictions is None or targets is None:
                    raise ValueError(
                        "topk_target_nodes_by_mae requires targets for "
                        f"{target_name!r} to be present in the batch."
                    )
                if mask is None:
                    mask = torch.ones_like(targets)
                abs_diff = (predictions - targets).abs()
                valid_per_sample = mask.sum(dim=1) > 0
                per_sample_mae = (abs_diff * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1.0)
                target_nodes = batch.target_node
                valid_cpu = valid_per_sample.cpu().tolist()
                nodes_cpu = target_nodes.cpu().tolist()
                maes_cpu = per_sample_mae.detach().cpu().tolist()
                for is_valid, node_id, sample_mae in zip(valid_cpu, nodes_cpu, maes_cpu):
                    if not is_valid:
                        continue
                    if node_id not in node_mae_sum:
                        node_mae_sum[node_id] = 0.0
                    node_mae_sum[node_id] += sample_mae
                    node_mae_count[node_id] = node_mae_count.get(node_id, 0) + 1
    finally:
        if model_was_training:
            model.train()

    if not node_mae_sum:
        return []

    node_mae = {
        node_id: node_mae_sum[node_id] / max(1, node_mae_count[node_id])
        for node_id in node_mae_sum
    }
    return [
        node_id
        for node_id, _mae in sorted(node_mae.items(), key=lambda kv: (kv[1], kv[0]))[:k]
    ]
