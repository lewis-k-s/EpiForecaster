"""Evaluation helpers and pipelines for EpiForecaster checkpoints.

This module provides high-level evaluation functions that orchestrate model loading,
evaluation loops, and plotting. Core components have been extracted to submodules:
- loaders.py: Model and data loader construction
- selection.py: Node selection utilities
- eval_loop.py: Core evaluation loop implementation
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from data.epi_dataset import EpiDataset
from evaluation.eval_loop import _ensure_wandb_run, eval_checkpoint, evaluate_loader
from evaluation.loaders import build_loader_from_config, load_model_from_checkpoint
from evaluation.losses import get_loss_from_config
from evaluation.selection import select_nodes_by_loss, topk_target_nodes_by_mae
from plotting.forecast_plots import (
    DEFAULT_PLOT_TARGETS,
    collect_forecast_samples_for_target_nodes,
    generate_forecast_plots,
    make_forecast_figure,
    make_joint_forecast_figure,
)

logger = logging.getLogger(__name__)


def evaluate_checkpoint_topk_forecasts(
    *,
    checkpoint_path: Path,
    split: str = "val",
    k: int = 5,
    device: str = "auto",
    window: str = "last",
    output_path: Path | None = None,
    log_dir: Path | None = None,
    eval_csv_path: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    End-to-end: load checkpoint, compute top-k nodes, collect series, and (optionally) save figure.

    Returns a dict containing: model, config, loader, topk_nodes, samples, figure.
    """

    start_time = time.time()
    logger.info(f"[eval] Loading checkpoint: {checkpoint_path}")
    model, config, checkpoint = load_model_from_checkpoint(
        checkpoint_path, device=device
    )
    logger.info(
        f"[eval] Loaded model (params={sum(p.numel() for p in model.parameters()):,})"
    )
    logger.info(
        f"[eval] Building {split} loader from dataset: {config.data.dataset_path}"
    )
    loader, region_embeddings = build_loader_from_config(
        config, split=split, device=device, batch_size=batch_size
    )
    dataset = cast(EpiDataset, loader.dataset)
    logger.info(f"[eval] {split} samples: {len(dataset)}")
    logger.info(f"[eval] Scanning for top-k nodes by MAE (k={k})...")

    topk_nodes = topk_target_nodes_by_mae(
        model=model, loader=loader, region_embeddings=region_embeddings, k=k
    )
    logger.debug(f"[eval] Top-k scan done in {time.time() - start_time:.2f}s")
    logger.info("[eval] Collecting forecast samples for top-k nodes...")
    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=topk_nodes,
        model=model,
        loader=loader,
        window=window,
        context_pre=30,
        context_post=30,
    )

    fig = make_forecast_figure(
        samples=samples,
        input_window_length=int(config.model.input_window_length),
        forecast_horizon=int(config.model.forecast_horizon),
        context_pre=30,
        context_post=30,
    )
    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    eval_loss = float("nan")
    eval_metrics: dict[str, Any] = {}
    node_mae_dict: dict[int, float] = {}
    try:
        criterion = get_loss_from_config(
            config.training.loss,
            data_config=config.data,
            forecast_horizon=config.model.forecast_horizon,
        )
        eval_loss, eval_metrics, node_mae_dict = evaluate_loader(
            model=model,
            loader=loader,
            criterion=criterion,
            horizon=int(config.model.forecast_horizon),
            device=next(model.parameters()).device,
            region_embeddings=region_embeddings,
            split_name=split.capitalize(),
            output_csv_path=eval_csv_path,
        )
    except Exception as exc:  # pragma: no cover - evaluation best-effort
        logger.warning(f"[eval] Metrics evaluation failed: {exc}")

    if log_dir is not None or wandb.run is not None:
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"eval_{split}_{checkpoint_path.parent.parent.name}"
        _ensure_wandb_run(
            config=config, log_dir=log_dir, name=run_name, job_type="eval"
        )
        if wandb.run is not None:
            log_data: dict[str, Any] = {}
            if math.isfinite(eval_loss):
                log_data[f"loss_{split}"] = eval_loss
            for key in ("mae", "rmse", "smape", "r2"):
                if key in eval_metrics:
                    log_data[f"{key}_{split}"] = eval_metrics[key]
            if log_data:
                wandb.log(log_data, step=0)

    return {
        "checkpoint": checkpoint,
        "config": config,
        "model": model,
        "loader": loader,
        "topk_nodes": topk_nodes,
        "samples": samples,
        "figure": fig,
        "eval_loss": eval_loss,
        "eval_metrics": eval_metrics,
        "node_mae": node_mae_dict,
        "log_dir": log_dir,
    }


def plot_forecasts_from_csv(
    *,
    csv_path: Path,
    checkpoint_path: Path,
    samples_per_quartile: int = 2,
    window: str = "last",
    device: str = "auto",
    output_path: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Load evaluation CSV, sample nodes from quartiles, and generate forecast plots.

    Args:
        csv_path: Path to CSV with columns node_id, mae, num_samples
        checkpoint_path: Path to model checkpoint
        samples_per_quartile: Number of nodes to sample from each quartile (default 2)
        window: Which window to plot ('last' or 'random')
        device: Device to use for inference
        output_path: Optional path to save the figure

    Returns:
        Dict containing: figure, selected_nodes, quartile_groups, config
    """
    import csv as csv_lib
    import random

    logger.info(f"[plot] Loading evaluation CSV: {csv_path}")
    node_mae_list: list[tuple[int, float, int]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv_lib.DictReader(f)
        for row in reader:
            node_id = int(row["node_id"])
            mae = float(row["mae"])
            num_samples = int(row["num_samples"])
            node_mae_list.append((node_id, mae, num_samples))

    if not node_mae_list:
        logger.warning("[plot] No valid nodes found in CSV")
        return {
            "figure": None,
            "selected_nodes": [],
            "quartile_groups": {},
            "config": None,
        }

    node_mae_list.sort(key=lambda x: x[1])

    maes = [mae for _, mae, _ in node_mae_list]
    q1_cutoff = np.percentile(maes, 25)
    q2_cutoff = np.percentile(maes, 50)
    q3_cutoff = np.percentile(maes, 75)

    quartile_groups: dict[str, list[int]] = {
        "Q1 (Worst)": [],
        "Q2 (Poor)": [],
        "Q3 (Average)": [],
        "Q4 (Best)": [],
    }

    for node_id, mae, _num_samples in node_mae_list:
        if mae <= q1_cutoff:
            quartile_groups["Q1 (Worst)"].append(node_id)
        elif mae <= q2_cutoff:
            quartile_groups["Q2 (Poor)"].append(node_id)
        elif mae <= q3_cutoff:
            quartile_groups["Q3 (Average)"].append(node_id)
        else:
            quartile_groups["Q4 (Best)"].append(node_id)

    selected_nodes: list[int] = []

    for quartile_name, nodes in quartile_groups.items():
        available = len(nodes)
        k = min(samples_per_quartile, available)
        sampled = random.sample(nodes, k)
        quartile_groups[quartile_name] = sampled
        selected_nodes.extend(sampled)
        logger.info(
            f"[plot] {quartile_name}: sampled {k} nodes (available: {available})"
        )

    if not selected_nodes:
        logger.warning("[plot] No nodes selected for plotting")
        return {
            "figure": None,
            "selected_nodes": [],
            "quartile_groups": {},
            "config": None,
        }

    logger.info(f"[plot] Loading checkpoint: {checkpoint_path}")
    model, config, _checkpoint = load_model_from_checkpoint(
        checkpoint_path, device=device
    )

    loader, _region_embeddings = build_loader_from_config(
        config, split="val", device=device, batch_size=batch_size
    )
    logger.info(
        f"[plot] Collecting forecast samples for {len(selected_nodes)} nodes..."
    )
    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=selected_nodes,
        model=model,
        loader=loader,
        window=window,
        context_pre=30,
        context_post=30,
    )

    quartile_samples: dict[str, list[dict[str, Any]]] = {
        name: [] for name in quartile_groups.keys()
    }
    node_to_quartile: dict[int, str] = {}
    for quartile_name, nodes in quartile_groups.items():
        for node_id in nodes:
            node_to_quartile[node_id] = quartile_name

    for sample in samples:
        node_id = sample["node_id"]
        if node_id in node_to_quartile:
            quartile_samples[node_to_quartile[node_id]].append(sample)

    fig = make_forecast_figure(
        samples=quartile_samples,
        input_window_length=int(config.model.input_window_length),
        forecast_horizon=int(config.model.forecast_horizon),
        context_pre=30,
        context_post=30,
    )

    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"[plot] Saved figure to: {output_path}")

    return {
        "figure": fig,
        "selected_nodes": selected_nodes,
        "quartile_groups": quartile_groups,
        "samples": samples,
        "config": config,
    }


__all__ = [
    "eval_checkpoint",
    "evaluate_checkpoint_topk_forecasts",
    "evaluate_loader",
    "generate_forecast_plots",
    "plot_forecasts_from_csv",
    "load_model_from_checkpoint",
    "build_loader_from_config",
    "select_nodes_by_loss",
    "topk_target_nodes_by_mae",
]
