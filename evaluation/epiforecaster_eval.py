from __future__ import annotations

import logging
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.collate import collate_epidataset_batch
from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    """Resolve the torch device string using the same priority as training."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return resolved


def load_model_from_checkpoint(
    checkpoint_path: Path, *, device: str = "auto"
) -> tuple[torch.nn.Module, EpiForecasterConfig, dict[str, Any]]:
    """Load an EpiForecaster model + config from a saved trainer checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: EpiForecasterConfig = checkpoint["config"]

    model = EpiForecaster(
        variant_type=config.model.type,
        temporal_input_dim=config.model.cases_dim,
        biomarkers_dim=config.model.biomarkers_dim,
        region_embedding_dim=config.model.region_embedding_dim,
        mobility_embedding_dim=config.model.mobility_embedding_dim,
        gnn_depth=config.model.gnn_depth,
        sequence_length=config.model.history_length,
        forecast_horizon=config.model.forecast_horizon,
        use_population=config.model.use_population,
        population_dim=config.model.population_dim,
        device=resolve_device(device),
        gnn_module=config.model.gnn_module,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolve_device(device))
    return model, config, checkpoint


def split_nodes(config: EpiForecasterConfig) -> tuple[list[int], list[int], list[int]]:
    """Match the node holdout split logic used during training."""
    train_split = 1 - config.training.val_split - config.training.test_split
    aligned_dataset = EpiDataset.load_canonical_dataset(Path(config.data.dataset_path))
    N = aligned_dataset[REGION_COORD].size
    all_nodes = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(all_nodes)
    n_train = int(len(all_nodes) * train_split)
    n_val = int(len(all_nodes) * config.training.val_split)
    train_nodes = all_nodes[:n_train]
    val_nodes = all_nodes[n_train : n_train + n_val]
    test_nodes = all_nodes[n_train + n_val :]
    return list(train_nodes), list(val_nodes), list(test_nodes)


def build_loader_from_config(
    config: EpiForecasterConfig,
    *,
    split: str,
    batch_size: int | None = None,
    device: str = "auto",
) -> DataLoader:
    """Build a DataLoader for the given split from the checkpoint config."""
    train_nodes, val_nodes, test_nodes = split_nodes(config)
    split_key = split.lower()
    if split_key not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    if split_key == "val":
        dataset = EpiDataset(
            config=config,
            target_nodes=val_nodes,
            context_nodes=train_nodes + val_nodes,
        )
    else:
        dataset = EpiDataset(
            config=config,
            target_nodes=test_nodes,
            context_nodes=train_nodes + val_nodes,
        )

    # Platform-aware workers (macOS multiprocessing issues)
    if platform.system() == "Darwin":
        num_workers = 0
    else:
        avail_cores = (os.cpu_count() or 1) - 1
        cfg_workers = config.training.num_workers
        if cfg_workers == -1:
            num_workers = max(0, avail_cores)
        else:
            num_workers = min(max(0, avail_cores), cfg_workers)

    resolved_batch = batch_size or config.training.batch_size
    pin_memory = (
        bool(config.training.pin_memory) and resolve_device(device).type == "cuda"
    )
    return DataLoader(
        dataset,
        batch_size=resolved_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_epidataset_batch,
    )


def topk_target_nodes_by_mae(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    k: int = 5,
    use_tqdm: bool = False,
) -> list[int]:
    """Compute top-k target node ids by average per-window MAE over the loader."""
    device = next(model.parameters()).device
    dataset = loader.dataset
    region_embeddings = getattr(dataset, "region_embeddings", None)
    if region_embeddings is not None:
        region_embeddings = region_embeddings.to(device)

    node_mae_sum: dict[int, float] = {}
    node_mae_count: dict[int, int] = {}

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            eval_iter = loader
            if use_tqdm:
                eval_iter = tqdm(
                    loader,
                    desc="Top-k scan",
                    leave=False,
                    total=len(loader),
                    position=1,
                    dynamic_ncols=True,
                )
            for batch in eval_iter:
                predictions, targets = _forward_batch(
                    model=model,
                    batch_data=batch,
                    device=device,
                    region_embeddings=region_embeddings,
                )
                abs_diff = (predictions - targets).abs()
                per_sample_mae = abs_diff.mean(dim=1).detach().cpu()
                target_nodes = batch["TargetNode"].detach().cpu()
                for sample_mae, target_node in zip(
                    per_sample_mae, target_nodes, strict=False
                ):
                    node_id = int(target_node.item())
                    mae_val = float(sample_mae.item())
                    if not math.isfinite(mae_val):
                        continue
                    node_mae_sum[node_id] = node_mae_sum.get(node_id, 0.0) + mae_val
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


def evaluate_checkpoint_topk_forecasts(
    *,
    checkpoint_path: Path,
    split: str = "val",
    k: int = 5,
    device: str = "auto",
    window: str = "last",
    output_path: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """
    End-to-end: load checkpoint, compute top-k nodes, collect series, and (optionally) save figure.

    Returns a dict containing: model, config, loader, topk_nodes, samples, figure.
    """
    from plotting.forecast_plots import (
        collect_forecast_samples_for_target_nodes,
        make_forecast_figure,
    )

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
    loader = build_loader_from_config(config, split=split, device=device)
    logger.info(f"[eval] {split} samples: {len(loader.dataset)}")
    logger.info(f"[eval] Scanning for top-k nodes by MAE (k={k})...")

    topk_nodes = topk_target_nodes_by_mae(
        model=model, loader=loader, k=k, use_tqdm=logger.isEnabledFor(logging.INFO)
    )
    logger.debug(f"[eval] Top-k scan done in {time.time() - start_time:.2f}s")
    logger.info("[eval] Collecting forecast samples for top-k nodes...")
    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=topk_nodes,
        model=model,
        loader=loader,
        window=window,
    )

    fig = make_forecast_figure(
        samples=samples,
        history_length=int(config.model.history_length),
        forecast_horizon=int(config.model.forecast_horizon),
    )
    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    eval_loss = float("nan")
    eval_metrics: dict[str, Any] = {}
    try:
        criterion = nn.MSELoss()
        region_embeddings = getattr(loader.dataset, "region_embeddings", None)
        if region_embeddings is not None:
            region_embeddings = region_embeddings.to(next(model.parameters()).device)
        eval_loss, eval_metrics = evaluate_loader(
            model=model,
            loader=loader,
            criterion=criterion,
            horizon=int(config.model.forecast_horizon),
            device=next(model.parameters()).device,
            region_embeddings=region_embeddings,
            use_tqdm=logger.isEnabledFor(logging.INFO),
            split_name=split.capitalize(),
        )
    except Exception as exc:  # pragma: no cover - evaluation best-effort
        logger.warning(f"[eval] Metrics evaluation failed: {exc}")

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_text(
            "eval/summary", _format_eval_summary(eval_loss, eval_metrics), 0
        )
        if math.isfinite(eval_loss):
            writer.add_scalar(f"eval/{split}/loss", eval_loss, 0)
        for key in ("mae", "rmse", "smape", "r2"):
            if key in eval_metrics:
                writer.add_scalar(f"eval/{split}/{key}", eval_metrics[key], 0)
        writer.close()

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
        "log_dir": log_dir,
    }


def _format_eval_summary(loss: float, metrics: dict[str, Any]) -> str:
    def _fmt(value: float) -> str:
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.6f}"

    rows = [
        ("Loss", _fmt(loss)),
        ("MAE", _fmt(metrics.get("mae"))),
        ("RMSE", _fmt(metrics.get("rmse"))),
        ("sMAPE", _fmt(metrics.get("smape"))),
        ("R2", _fmt(metrics.get("r2"))),
    ]
    table = ["| Metric | Value |", "|---|---|"]
    for name, value in rows:
        table.append(f"| {name} | {value} |")
    return "\n".join(table)


def evaluate_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    horizon: int,
    device: torch.device,
    region_embeddings: torch.Tensor | None = None,
    use_tqdm: bool = False,
    split_name: str = "Eval",
    max_batches: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a loader and compute loss/metrics matching trainer behavior."""
    total_loss = 0.0
    mae_sum = 0.0
    mse_sum = 0.0
    smape_sum = 0.0

    total_count = 0
    target_mean_acc = 0.0
    target_m2 = 0.0

    per_h_mae_sum = torch.zeros(horizon)
    per_h_mse_sum = torch.zeros(horizon)

    num_batches = len(loader)
    eval_iter = loader
    log_every = 50
    if use_tqdm:
        eval_iter = tqdm(
            loader,
            desc=f"{split_name} batch progress",
            leave=False,
            total=num_batches,
            position=1,
            dynamic_ncols=True,
        )

    epsilon = 1e-6
    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_iter):
                if max_batches and batch_idx >= max_batches:
                    break
                if batch_idx % log_every == 0:
                    logger.debug(f"{split_name} evaluation: {batch_idx}/{num_batches}")

                batch_start_time = time.time()
                batch_start_time = time.time()
                predictions, targets, target_mean, target_scale = _forward_batch(
                    model=model,
                    batch_data=batch_data,
                    device=device,
                    region_embeddings=region_embeddings,
                )

                loss = criterion(predictions, targets)
                total_loss += loss.item()

                # Unscale for metrics reporting
                # target_mean/scale are (B,) -> unsqueeze to (B, 1) for broadcasting
                scale = target_scale.unsqueeze(-1)
                mean = target_mean.unsqueeze(-1)

                pred_unscaled = predictions * scale + mean
                targets_unscaled = targets * scale + mean

                diff = pred_unscaled - targets_unscaled
                abs_diff = diff.abs()
                mae_sum += abs_diff.sum().item()
                mse_sum += (diff**2).sum().item()
                smape_sum += (
                    (
                        2
                        * abs_diff
                        / (pred_unscaled.abs() + targets_unscaled.abs() + epsilon)
                    )
                    .sum()
                    .item()
                )

                flat_targets = targets_unscaled.detach().float().reshape(-1)
                batch_count = flat_targets.numel()
                batch_mean = flat_targets.mean().item()
                batch_m2 = ((flat_targets - batch_mean) ** 2).sum().item()

                delta = batch_mean - target_mean_acc
                new_count = total_count + batch_count
                target_mean_acc += delta * batch_count / new_count
                target_m2 += (
                    batch_m2 + (delta**2) * (total_count * batch_count) / new_count
                )
                total_count = new_count

                per_h_mae_sum += abs_diff.sum(dim=0).detach().cpu()
                per_h_mse_sum += (diff**2).sum(dim=0).detach().cpu()

                if use_tqdm:
                    bsz = int(batch_data["CaseNode"].shape[0])
                    batch_time_s = time.time() - batch_start_time
                    samples_per_s = (
                        (bsz / batch_time_s) if batch_time_s > 0 else float("inf")
                    )
                    eval_iter.set_postfix(
                        n=f"{batch_idx + 1}/{num_batches}",
                        sps=samples_per_s,
                    )
    finally:
        if model_was_training:
            model.train()

    mean_loss = total_loss / max(1, num_batches)
    mean_mae = mae_sum / max(1, total_count)
    mean_rmse = math.sqrt(mse_sum / max(1, total_count)) if total_count else 0.0
    mean_smape = smape_sum / max(1, total_count)

    ss_res = mse_sum
    ss_tot = target_m2
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    per_h_count = total_count / max(1, horizon)
    per_h_mae = (per_h_mae_sum / max(1, per_h_count)).tolist()
    per_h_rmse = (
        (per_h_mse_sum / max(1, per_h_count)).sqrt().tolist() if per_h_count else []
    )

    metrics = {
        "mae": mean_mae,
        "rmse": mean_rmse,
        "smape": mean_smape,
        "r2": r2,
        "mae_per_h": per_h_mae,
        "rmse_per_h": per_h_rmse,
    }

    logger.debug("EVAL COMPLETE")
    return mean_loss, metrics


def _forward_batch(
    *,
    model: torch.nn.Module,
    batch_data: dict[str, Any],
    device: torch.device,
    region_embeddings: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = batch_data["Target"].to(device)
    target_mean = batch_data["TargetMean"].to(device)
    target_scale = batch_data["TargetScale"].to(device)
    predictions = model.forward(
        cases_hist=batch_data["CaseNode"].to(device),
        biomarkers_hist=batch_data["BioNode"].to(device),
        mob_graphs=batch_data["MobBatch"],
        target_nodes=batch_data["TargetNode"].to(device),
        region_embeddings=region_embeddings,
        population=batch_data["Population"].to(device),
        target_mean=target_mean,
    )
    return predictions, targets, target_mean, target_scale
