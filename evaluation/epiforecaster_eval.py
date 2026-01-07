from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import zarr.errors

from data.collate import collate_epidataset_batch
from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD
from utils.normalization import unscale_forecasts
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster
from plotting.forecast_plots import (
    collect_forecast_samples_for_target_nodes,
    make_forecast_figure,
)

logger = logging.getLogger(__name__)

# Global seeded RNG for reproducibility across evaluation/plotting
_GLOBAL_RNG = np.random.default_rng(42)


class ForecastLoss(nn.Module):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class WrappedTorchLoss(ForecastLoss):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        _ = (target_mean, target_scale)
        return self.loss_fn(predictions, targets)


class SMAPELoss(ForecastLoss):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        pred_unscaled, targets_unscaled = unscale_forecasts(
            predictions, targets, target_mean, target_scale
        )
        numerator = 2 * (pred_unscaled - targets_unscaled).abs()
        denominator = pred_unscaled.abs() + targets_unscaled.abs() + self.epsilon
        return (numerator / denominator).mean()


def get_loss_function(name: str) -> ForecastLoss:
    name_lower = name.lower()
    if name_lower == "mse":
        return WrappedTorchLoss(nn.MSELoss())
    elif name_lower in ("mae", "l1"):
        return WrappedTorchLoss(nn.L1Loss())
    elif name_lower == "smape":
        return SMAPELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


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
    """Load an EpiForecaster model + config from a saved trainer checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load the model on

    Returns:
        Tuple of (model, config, checkpoint_dict)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint is missing required keys or has invalid config
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Validate required keys
    required_keys = ["model_state_dict", "config"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}. "
            f"This checkpoint may be from an incompatible version or corrupted."
        )

    config = checkpoint["config"]

    # Validate config type
    if not isinstance(config, EpiForecasterConfig):
        raise ValueError(
            f"Checkpoint config has invalid type: {type(config).__name__}. "
            f"Expected EpiForecasterConfig. "
            f"Please check that the checkpoint was created with a compatible version."
        )

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
        gnn_hidden_dim=config.model.gnn_hidden_dim,
        head_d_model=config.model.head_d_model,
        head_n_heads=config.model.head_n_heads,
        head_num_layers=config.model.head_num_layers,
        head_dropout=config.model.head_dropout,
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


def _suppress_zarr_warnings(worker_id: int) -> None:
    """Suppress zarr/numcodecs warnings in DataLoader worker processes."""
    import warnings

    warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)


def build_loader_from_config(
    config: EpiForecasterConfig,
    *,
    split: str,
    batch_size: int | None = None,
    device: str = "auto",
) -> tuple[DataLoader[EpiDataset], torch.Tensor | None]:
    """Build a DataLoader for the given split from the checkpoint config.

    Returns:
        Tuple of (DataLoader, region_embeddings). Region embeddings are pre-loaded
        to the target device to avoid repeated transfers during evaluation.
    """
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

    # Worker configuration - use val_workers for val/test splits
    avail_cores = (os.cpu_count() or 1) - 1
    cfg_workers = config.training.val_workers
    if cfg_workers == -1:
        num_workers = max(0, avail_cores)
    else:
        num_workers = min(max(0, avail_cores), cfg_workers)

    resolved_batch = batch_size or config.training.batch_size
    resolved_device = resolve_device(device)
    pin_memory = bool(config.training.pin_memory) and resolved_device.type == "cuda"

    # Pre-load region embeddings to device to avoid repeated transfers
    region_embeddings = getattr(dataset, "region_embeddings", None)
    if region_embeddings is not None:
        region_embeddings = region_embeddings.to(resolved_device)

    loader = DataLoader(
        dataset,
        batch_size=resolved_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_epidataset_batch,
        worker_init_fn=_suppress_zarr_warnings if num_workers > 0 else None,
    )
    return loader, region_embeddings


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
        node_mae: Dict mapping node_id → average MAE
        strategy: "topk", "quartile", "worst", "best", "random"
        k: Number of nodes for topk/worst/best strategies
        samples_per_group: Number of nodes per group for quartile strategy (default 4)
        rng: Random generator for deterministic sampling (default: global seeded RNG)

    Returns:
        Dict mapping group name → list of node_ids
        Examples:
            strategy="topk": {"Top-k": [1, 2, 3, 4, 5]}
            strategy="quartile": {"Q1 (Worst)": [...], "Q2 (Poor)": [...], ...}
            strategy="worst": {"Worst": [1, 2, 3, 4, 5]}
    """
    if rng is None:
        rng = _GLOBAL_RNG

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

    node_mae_sum: dict[int, torch.Tensor] = {}
    node_mae_count: dict[int, int] = {}

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            eval_iter = loader
            for batch in eval_iter:
                predictions, targets, _target_mean, _target_scale = _forward_batch(
                    model=model,
                    batch_data=batch,
                    device=device,
                    region_embeddings=region_embeddings,
                )
                abs_diff = (predictions - targets).abs()
                per_sample_mae = abs_diff.mean(dim=1)
                target_nodes = batch["TargetNode"]
                for sample_mae, target_node in zip(
                    per_sample_mae, target_nodes, strict=False
                ):
                    node_id = int(target_node.item())
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
        config, split=split, device=device
    )
    logger.info(f"[eval] {split} samples: {len(loader.dataset)}")
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
        history_length=int(config.model.history_length),
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
        criterion = get_loss_function("smape")
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
        "node_mae": node_mae_dict,
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
    criterion: ForecastLoss,
    horizon: int,
    device: torch.device,
    region_embeddings: torch.Tensor | None = None,
    split_name: str = "Eval",
    max_batches: int | None = None,
    output_csv_path: Path | None = None,
) -> tuple[float, dict[str, Any], dict[int, float]]:
    """Evaluate a loader and compute loss/metrics matching trainer behavior.

    Uses device-local metric accumulation to minimize CPU-GPU synchronization.
    """
    logger.info(f"{split_name} evaluation started...")
    # Device-local accumulators - avoid sync until end
    total_loss = torch.tensor(0.0, device=device)
    mae_sum = torch.tensor(0.0, device=device)
    mse_sum = torch.tensor(0.0, device=device)
    smape_sum = torch.tensor(0.0, device=device)

    total_count = 0
    # Welford's algorithm for variance - keep on device
    target_mean_acc = torch.tensor(0.0, device=device)
    target_m2 = torch.tensor(0.0, device=device)

    per_h_mae_sum = torch.zeros(horizon, device=device)
    per_h_mse_sum = torch.zeros(horizon, device=device)

    # For node-level MAE, accumulate in dict but defer item() calls
    node_mae_sum: dict[int, torch.Tensor] = {}
    node_mae_count: dict[int, int] = {}

    num_batches = len(loader)
    eval_iter = loader
    log_every = 10

    epsilon = 1e-6
    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_iter):
                if max_batches and batch_idx >= max_batches:
                    break
                if batch_idx % log_every == 0:
                    logger.info(f"{split_name} evaluation: {batch_idx}/{num_batches}")

                predictions, targets, target_mean, target_scale = _forward_batch(
                    model=model,
                    batch_data=batch_data,
                    device=device,
                    region_embeddings=region_embeddings,
                )

                loss = criterion(predictions, targets, target_mean, target_scale)
                total_loss += loss.detach()

                pred_unscaled, targets_unscaled = unscale_forecasts(
                    predictions, targets, target_mean, target_scale
                )

                diff = pred_unscaled - targets_unscaled
                abs_diff = diff.abs()
                mae_sum += abs_diff.sum()
                mse_sum += (diff**2).sum()
                smape_sum += (
                    2
                    * abs_diff
                    / (pred_unscaled.abs() + targets_unscaled.abs() + epsilon)
                ).sum()

                # Per-node MAE - keep tensors on device until end
                per_sample_mae = abs_diff.mean(dim=1)
                target_nodes = batch_data["TargetNode"]
                for sample_mae, target_node in zip(
                    per_sample_mae, target_nodes, strict=False
                ):
                    node_id = int(target_node.item())
                    if node_id not in node_mae_sum:
                        node_mae_sum[node_id] = torch.tensor(0.0, device=device)
                    node_mae_sum[node_id] += sample_mae.detach()
                    node_mae_count[node_id] = node_mae_count.get(node_id, 0) + 1

                # Welford's algorithm for variance (device-local)
                flat_targets = targets_unscaled.detach().float().reshape(-1)
                batch_count = flat_targets.numel()
                batch_mean = flat_targets.mean()
                batch_m2 = ((flat_targets - batch_mean) ** 2).sum()

                delta = batch_mean - target_mean_acc
                new_count = total_count + batch_count
                target_mean_acc += delta * batch_count / new_count
                target_m2 += (
                    batch_m2 + (delta**2) * (total_count * batch_count) / new_count
                )
                total_count = new_count

                per_h_mae_sum += abs_diff.sum(dim=0)
                per_h_mse_sum += (diff**2).sum(dim=0)

    finally:
        if model_was_training:
            model.train()

    # Final sync - transfer metrics to CPU once
    mean_loss = (total_loss / max(1, num_batches)).item()
    mean_mae = (mae_sum / max(1, total_count)).item()
    mean_rmse = (
        math.sqrt((mse_sum / max(1, total_count)).item()) if total_count else 0.0
    )
    mean_smape = (smape_sum / max(1, total_count)).item()

    ss_res = mse_sum.item()
    ss_tot = target_m2.item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Convert node MAE tensors to scalars
    node_mae = {
        node_id: (node_mae_sum[node_id] / max(1, node_mae_count[node_id])).item()
        for node_id in node_mae_sum
    }

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv as csv_lib

        with open(output_csv_path, "w", newline="") as f:
            writer = csv_lib.writer(f)
            writer.writerow(["node_id", "mae", "num_samples"])
            for node_id in sorted(node_mae.keys()):
                writer.writerow([node_id, node_mae[node_id], node_mae_count[node_id]])

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

    logger.info("EVAL COMPLETE")
    return mean_loss, metrics, node_mae


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
        cases_norm=batch_data["CaseNode"].to(device),
        cases_mean=batch_data["CaseMean"].to(device),
        cases_std=batch_data["CaseStd"].to(device),
        biomarkers_hist=batch_data["BioNode"].to(device),
        mob_graphs=batch_data["MobBatch"],
        target_nodes=batch_data["TargetNode"].to(device),
        region_embeddings=region_embeddings,
        population=batch_data["Population"].to(device),
    )
    return predictions, targets, target_mean, target_scale


def generate_forecast_plots(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    node_groups: dict[str, list[int]],
    window: str = "last",
    context_pre: int = 30,
    context_post: int = 30,
    output_path: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Generate forecast plots for given node groups (generic).

    Args:
        model: The trained model
        loader: Original DataLoader for data access
        node_groups: Dict mapping group name → list of node IDs
                     (could be quartiles, topk, worst, random, anything!)
        window: Which time window to plot ("last" or "random")
        context_pre: Days before forecast start
        context_post: Days after forecast end
        output_path: Optional path to save figure
        log_dir: Optional TensorBoard log directory

    Returns:
        Dict with figure, all_samples, selected_nodes, node_groups
    """
    # Flatten all nodes to collect samples once
    all_selected_nodes: list[int] = []
    for group_nodes in node_groups.values():
        all_selected_nodes.extend(group_nodes)

    if not all_selected_nodes:
        logger.warning("[plot] No nodes selected for plotting")
        return {
            "figure": None,
            "all_samples": [],
            "selected_nodes": [],
            "node_groups": {},
        }

    logger.info(
        f"[plot] Collecting forecast samples for {len(all_selected_nodes)} nodes..."
    )

    # Use existing function - it handles subset creation internally
    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=all_selected_nodes,
        model=model,
        loader=loader,
        window=window,
        context_pre=context_pre,
        context_post=context_post,
    )

    # Group samples by original group names
    node_to_group: dict[int, str] = {}
    for group_name, nodes in node_groups.items():
        for node_id in nodes:
            node_to_group[node_id] = group_name

    grouped_samples: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        node_id = sample["node_id"]
        if node_id in node_to_group:
            group_name = node_to_group[node_id]
            if group_name not in grouped_samples:
                grouped_samples[group_name] = []
            grouped_samples[group_name].append(sample)

    # Generate figure using existing generic function
    config = loader.dataset.config
    fig = make_forecast_figure(
        samples=grouped_samples,
        history_length=int(config.model.history_length),
        forecast_horizon=int(config.model.forecast_horizon),
        context_pre=context_pre,
        context_post=context_post,
    )

    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"[plot] Saved figure to: {output_path}")

    # Log to TensorBoard if provided
    if fig is not None and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_figure("forecasts", fig, 0)
        writer.close()

    return {
        "figure": fig,
        "all_samples": samples,
        "selected_nodes": all_selected_nodes,
        "node_groups": node_groups,
    }


def eval_checkpoint(
    *,
    checkpoint_path: Path,
    split: str = "val",
    device: str = "auto",
    log_dir: Path | None = None,
    overrides: list[str] | None = None,
    output_csv_path: Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate checkpoint - pure evaluation, no selection or plotting.

    Args:
        checkpoint_path: Path to checkpoint file
        split: Which split to evaluate ("val" or "test")
        device: Device to use for evaluation
        log_dir: Optional TensorBoard log directory
        overrides: Optional list of dotted-key config overrides (e.g., ["training.val_workers=4"])
        output_csv_path: Optional path to save node-level metrics CSV

    Returns:
        Dict with: checkpoint, config, model, loader, node_mae_dict,
                  eval_loss, eval_metrics
    """
    logger.info(f"[eval] Loading checkpoint: {checkpoint_path}")
    model, config, checkpoint = load_model_from_checkpoint(
        checkpoint_path, device=device
    )

    # Apply config overrides if provided
    if overrides:
        from models.configs import EpiForecasterConfig

        config = EpiForecasterConfig.apply_overrides(config, list(overrides))
        logger.info(f"[eval] Applied {len(overrides)} config overrides")
    logger.info(
        f"[eval] Loaded model (params={sum(p.numel() for p in model.parameters()):,})"
    )
    logger.info(
        f"[eval] Building {split} loader from dataset: {config.data.dataset_path}"
    )
    loader, region_embeddings = build_loader_from_config(
        config, split=split, device=device
    )
    logger.info(f"[eval] {split} samples: {len(loader.dataset)}")

    # Run evaluation - returns node_mae_dict as third value
    eval_loss = float("nan")
    eval_metrics: dict[str, Any] = {}
    node_mae_dict: dict[int, float] = {}
    try:
        criterion = get_loss_function("smape")
        eval_loss, eval_metrics, node_mae_dict = evaluate_loader(
            model=model,
            loader=loader,
            criterion=criterion,
            horizon=int(config.model.forecast_horizon),
            device=next(model.parameters()).device,
            region_embeddings=region_embeddings,
            split_name=split.capitalize(),
            output_csv_path=output_csv_path,
        )
    except Exception as exc:
        logger.warning(f"[eval] Metrics evaluation failed: {exc}")

    # Log to TensorBoard
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
        "node_mae": node_mae_dict,
        "eval_loss": eval_loss,
        "eval_metrics": eval_metrics,
    }


def plot_forecasts_from_csv(
    *,
    csv_path: Path,
    checkpoint_path: Path,
    samples_per_quartile: int = 2,
    window: str = "last",
    device: str = "auto",
    output_path: Path | None = None,
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

    for node_id, mae, num_samples in node_mae_list:
        if mae <= q1_cutoff:
            quartile_groups["Q1 (Worst)"].append(node_id)
        elif mae <= q2_cutoff:
            quartile_groups["Q2 (Poor)"].append(node_id)
        elif mae <= q3_cutoff:
            quartile_groups["Q3 (Average)"].append(node_id)
        else:
            quartile_groups["Q4 (Best)"].append(node_id)

    selected_nodes: list[int] = []
    import random

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
        config, split="val", device=device
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
        history_length=int(config.model.history_length),
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
