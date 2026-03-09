from __future__ import annotations

import logging
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
import zarr.errors

from data.epi_dataset import EpiDataset
from data.epi_batch import collate_epiforecaster_batch
from data.dataset_factory import build_datasets
from evaluation.losses import JointInferenceLoss, get_loss_from_config
from evaluation.metrics import TorchMaskedMetricAccumulator
from models.configs import EpiForecasterConfig
from utils.device import resolve_device
from utils.sparsity_logging import log_sparsity_loss_correlation
from utils.training_utils import drop_nowcast
from models.epiforecaster import EpiForecaster
from plotting.forecast_plots import (
    DEFAULT_PLOT_TARGETS,
    collect_forecast_samples_for_target_nodes,
    make_forecast_figure,
    make_joint_forecast_figure,
)

logger = logging.getLogger(__name__)

# Global seeded RNG for reproducibility across evaluation/plotting
_GLOBAL_RNG = np.random.default_rng(42)


def _ensure_wandb_run(
    *,
    config: EpiForecasterConfig | None,
    log_dir: Path | None,
    name: str,
    job_type: str,
) -> wandb.sdk.wandb_run.Run | None:
    if wandb.run is not None:
        return wandb.run
    if log_dir is None:
        return None
    project = config.output.wandb_project if config is not None else "epiforecaster"
    entity = config.output.wandb_entity if config is not None else None
    group = None
    mode = "online"
    if config is not None:
        group = config.output.wandb_group or config.output.experiment_name
        mode = config.output.wandb_mode
    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        dir=str(log_dir),
        config=config.to_dict() if config is not None else None,
        job_type=job_type,
        mode=mode,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "auto",
    overrides: list[str] | None = None,
) -> tuple[EpiForecaster, EpiForecasterConfig, dict[str, Any]]:
    """Load an EpiForecaster model + config from a saved trainer checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load the model on
        overrides: Optional list of dotted-key config overrides applied before model creation

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

    config_raw = checkpoint["config"]

    # Handle backwards compatibility: old checkpoints have pickled EpiForecasterConfig,
    # new checkpoints have plain dicts (robust to config class changes)
    if isinstance(config_raw, dict):
        # New format: plain dict (YAML-compatible)
        config = EpiForecasterConfig.from_dict(config_raw)
    elif isinstance(config_raw, EpiForecasterConfig):
        # Old format: pickled EpiForecasterConfig (for backwards compatibility)
        config = config_raw
    else:
        raise ValueError(
            f"Checkpoint config has invalid type: {type(config_raw).__name__}. "
            f"Expected EpiForecasterConfig or dict. "
            f"Please check that the checkpoint was created with a compatible version."
        )

    # Apply overrides BEFORE model creation (for architecture-affecting params)
    if overrides:
        config = EpiForecasterConfig.apply_overrides(config, overrides)
        logger.info(f"Applied {len(overrides)} config overrides before model creation")

    model = EpiForecaster(
        variant_type=config.model.type,
        temporal_input_dim=config.model.cases_dim,
        biomarkers_dim=config.model.biomarkers_dim,
        region_embedding_dim=config.model.region_embedding_dim,
        mobility_embedding_dim=config.model.mobility_embedding_dim,
        gnn_depth=config.model.gnn_depth,
        sequence_length=config.model.input_window_length,
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
        sir_physics=config.model.sir_physics,
        observation_heads=config.model.observation_heads,
        temporal_covariates_dim=config.model.temporal_covariates_dim,
    )
    # Strip _orig_mod. prefix from compiled model checkpoints
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(resolve_device(device))
    return model, config, checkpoint


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

    Uses the same factory as training to ensure fair evaluation:
    - Preprocessors are fitted on train split (same as training)
    - Applied consistently to val/test splits (same as training)
    - Train and unused eval splits are discarded after preprocessing

    Returns:
        Tuple of (DataLoader, region_embeddings). Region embeddings are pre-loaded
        to the target device to avoid repeated transfers during evaluation.
    """
    split_key = split.lower()
    if split_key not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    # Use factory to get all splits with fitted preprocessors (same as training)
    # This ensures fair evaluation - same preprocessing as training
    splits = build_datasets(config)

    # Select requested split, discard others
    dataset = splits.val if split_key == "val" else splits.test
    # splits.train and the other eval split are garbage collected here

    # Worker configuration mirrors training defaults:
    # - validation loader uses val_workers
    # - test loader uses test_workers
    avail_cores = (os.cpu_count() or 1) - 1
    cfg_workers = (
        config.training.val_workers
        if split_key == "val"
        else getattr(config.training, "test_workers", 0)
    )
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
        collate_fn=partial(
            collate_epiforecaster_batch,
            require_region_index=bool(config.model.type.regions),
        ),
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


def _format_eval_summary(loss: float, metrics: dict[str, Any]) -> str:
    def _fmt(value: float | None) -> str:
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
    criterion: JointInferenceLoss,
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
    hosp_metrics = TorchMaskedMetricAccumulator(device=device, horizon=horizon)
    ww_metrics = TorchMaskedMetricAccumulator(device=device, horizon=None)
    cases_metrics = TorchMaskedMetricAccumulator(device=device, horizon=None)
    deaths_metrics = TorchMaskedMetricAccumulator(device=device, horizon=None)
    loss_ww_sum = torch.tensor(0.0, device=device)
    loss_hosp_sum = torch.tensor(0.0, device=device)
    loss_cases_sum = torch.tensor(0.0, device=device)
    loss_deaths_sum = torch.tensor(0.0, device=device)
    loss_sir_sum = torch.tensor(0.0, device=device)
    loss_ww_weighted_sum = torch.tensor(0.0, device=device)
    loss_hosp_weighted_sum = torch.tensor(0.0, device=device)
    loss_cases_weighted_sum = torch.tensor(0.0, device=device)
    loss_deaths_weighted_sum = torch.tensor(0.0, device=device)
    loss_sir_weighted_sum = torch.tensor(0.0, device=device)

    # For node-level MAE, accumulate in dict but defer item() calls
    node_mae_sum: dict[int, torch.Tensor] = {}
    node_mae_count: dict[int, int] = {}

    num_batches = len(loader)
    eval_iter = loader
    log_every = 10

    model_was_training = model.training
    model.eval()
    forward_model = cast(EpiForecaster, model)
    try:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            for batch_idx, batch_data in enumerate(eval_iter):
                if max_batches and batch_idx >= max_batches:
                    break
                if batch_idx % log_every == 0:
                    logger.info(f"{split_name} evaluation: {batch_idx}/{num_batches}")

                from utils.training_utils import inject_gpu_mobility

                inject_gpu_mobility(batch_data, eval_iter.dataset, device)

                model_outputs, targets_dict = forward_model.forward_batch(
                    batch_data=batch_data,
                    region_embeddings=region_embeddings,
                    mask_cases=criterion.mask_input_cases,
                    mask_ww=criterion.mask_input_ww,
                    mask_hosp=criterion.mask_input_hosp,
                    mask_deaths=criterion.mask_input_deaths,
                )

                # Create sliced model outputs for metric computation
                sliced_model_outputs = {
                    k: drop_nowcast(v, horizon)
                    if k.startswith("pred_") and isinstance(v, torch.Tensor)
                    else v
                    for k, v in model_outputs.items()
                }

                # Compute loss with batch_data for continuity penalty
                components = criterion.compute_components(
                    model_outputs, targets_dict, batch_data
                )
                metric_supervision = criterion.compute_observation_supervision(
                    targets_dict,
                    device=device,
                )
                loss = components["total"]
                total_loss += loss.detach()
                loss_ww_sum += components["ww"].detach()
                loss_hosp_sum += components["hosp"].detach()
                loss_cases_sum += components["cases"].detach()
                loss_deaths_sum += components["deaths"].detach()
                loss_sir_sum += components["sir"].detach()
                if "continuity" in components:
                    pass  # Don't accumulate continuity loss in metrics
                loss_ww_weighted_sum += components["ww_weighted"].detach()
                loss_hosp_weighted_sum += components["hosp_weighted"].detach()
                loss_cases_weighted_sum += components["cases_weighted"].detach()
                loss_deaths_weighted_sum += components["deaths_weighted"].detach()
                loss_sir_weighted_sum += components["sir_weighted"].detach()

                # Log sparsity-loss correlation during evaluation (moved from training)
                if batch_idx % 10 == 0:
                    log_sparsity_loss_correlation(
                        batch=batch_data,
                        model_outputs=model_outputs,
                        targets=targets_dict,
                        wandb_run=None,
                        step=batch_idx,
                        epoch=0,
                    )

                pred_hosp = sliced_model_outputs.get("pred_hosp")
                hosp_targets = metric_supervision["hosp"]["target"]
                hosp_mask = targets_dict.get("hosp_mask")
                hosp_weights = metric_supervision["hosp"]["weights"]
                if (
                    pred_hosp is not None
                    and hosp_targets is not None
                    and hosp_weights is not None
                ):
                    _diff, abs_diff, weights = hosp_metrics.update(
                        predictions=pred_hosp,
                        targets=hosp_targets,
                        observed_mask=hosp_mask,
                        sample_weights=hosp_weights,
                    )
                    # Per-node MAE - keep tensors on device until end
                    valid_per_sample = weights.sum(dim=1) > 0
                    per_sample_mae = (abs_diff * weights).sum(dim=1) / weights.sum(
                        dim=1
                    ).clamp_min(1e-8)
                    target_nodes = batch_data.target_node
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

                pred_ww = sliced_model_outputs.get("pred_ww")
                ww_targets = metric_supervision["ww"]["target"]
                ww_mask = targets_dict.get("ww_mask")
                ww_weights = metric_supervision["ww"]["weights"]
                if (
                    pred_ww is not None
                    and ww_targets is not None
                    and ww_weights is not None
                ):
                    ww_metrics.update(
                        predictions=pred_ww,
                        targets=ww_targets,
                        observed_mask=ww_mask,
                        sample_weights=ww_weights,
                    )

                pred_cases = sliced_model_outputs.get("pred_cases")
                cases_targets = metric_supervision["cases"]["target"]
                cases_mask = targets_dict.get("cases_mask")
                cases_weights = metric_supervision["cases"]["weights"]
                if (
                    pred_cases is not None
                    and cases_targets is not None
                    and cases_weights is not None
                ):
                    cases_metrics.update(
                        predictions=pred_cases,
                        targets=cases_targets,
                        observed_mask=cases_mask,
                        sample_weights=cases_weights,
                    )

                pred_deaths = sliced_model_outputs.get("pred_deaths")
                deaths_targets = metric_supervision["deaths"]["target"]
                deaths_mask = targets_dict.get("deaths_mask")
                deaths_weights = metric_supervision["deaths"]["weights"]
                if (
                    pred_deaths is not None
                    and deaths_targets is not None
                    and deaths_weights is not None
                ):
                    deaths_metrics.update(
                        predictions=pred_deaths,
                        targets=deaths_targets,
                        observed_mask=deaths_mask,
                        sample_weights=deaths_weights,
                    )

    finally:
        if model_was_training:
            model.train()

    # Final sync - transfer metrics to CPU once
    mean_loss = (total_loss / max(1, num_batches)).item()
    hosp_summary = hosp_metrics.finalize()
    ww_summary = ww_metrics.finalize()
    cases_summary = cases_metrics.finalize()
    deaths_summary = deaths_metrics.finalize()

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

    metrics = {
        # Legacy primary metrics (hospitalizations)
        "mae": hosp_summary.mae,
        "rmse": hosp_summary.rmse,
        "smape": hosp_summary.smape,
        "r2": hosp_summary.r2,
        "mae_per_h": hosp_summary.mae_per_h,
        "rmse_per_h": hosp_summary.rmse_per_h,
        # Hospitalization metrics in log1p(per-100k) space
        "mae_hosp_log1p_per_100k": hosp_summary.mae,
        "rmse_hosp_log1p_per_100k": hosp_summary.rmse,
        "smape_hosp_log1p_per_100k": hosp_summary.smape,
        "r2_hosp_log1p_per_100k": hosp_summary.r2,
        "observed_count_hosp": hosp_summary.observed_count,
        "effective_count_hosp": hosp_summary.effective_count,
        # Wastewater metrics in log1p(per-100k) space
        "mae_ww_log1p_per_100k": ww_summary.mae,
        "rmse_ww_log1p_per_100k": ww_summary.rmse,
        "smape_ww_log1p_per_100k": ww_summary.smape,
        "r2_ww_log1p_per_100k": ww_summary.r2,
        "observed_count_ww": ww_summary.observed_count,
        "effective_count_ww": ww_summary.effective_count,
        # Cases metrics in log1p(per-100k) space
        "mae_cases_log1p_per_100k": cases_summary.mae,
        "rmse_cases_log1p_per_100k": cases_summary.rmse,
        "smape_cases_log1p_per_100k": cases_summary.smape,
        "r2_cases_log1p_per_100k": cases_summary.r2,
        "observed_count_cases": cases_summary.observed_count,
        "effective_count_cases": cases_summary.effective_count,
        # Deaths metrics in log1p(per-100k) space
        "mae_deaths_log1p_per_100k": deaths_summary.mae,
        "rmse_deaths_log1p_per_100k": deaths_summary.rmse,
        "smape_deaths_log1p_per_100k": deaths_summary.smape,
        "r2_deaths_log1p_per_100k": deaths_summary.r2,
        "observed_count_deaths": deaths_summary.observed_count,
        "effective_count_deaths": deaths_summary.effective_count,
        # Joint loss components (averaged per batch, same reduction as mean_loss)
        "loss_ww": (loss_ww_sum / max(1, num_batches)).item(),
        "loss_hosp": (loss_hosp_sum / max(1, num_batches)).item(),
        "loss_cases": (loss_cases_sum / max(1, num_batches)).item(),
        "loss_deaths": (loss_deaths_sum / max(1, num_batches)).item(),
        "loss_sir": (loss_sir_sum / max(1, num_batches)).item(),
        "loss_ww_weighted": (loss_ww_weighted_sum / max(1, num_batches)).item(),
        "loss_hosp_weighted": (loss_hosp_weighted_sum / max(1, num_batches)).item(),
        "loss_cases_weighted": (loss_cases_weighted_sum / max(1, num_batches)).item(),
        "loss_deaths_weighted": (loss_deaths_weighted_sum / max(1, num_batches)).item(),
        "loss_sir_weighted": (loss_sir_weighted_sum / max(1, num_batches)).item(),
    }

    logger.info("EVAL COMPLETE")
    return mean_loss, metrics, node_mae


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
    target_names: list[str] | None = None,
    wandb_prefix: str = "forecasts",
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
    log_dir: Optional W&B run directory for eval metrics

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
    resolved_targets = target_names or list(DEFAULT_PLOT_TARGETS)

    samples = collect_forecast_samples_for_target_nodes(
        target_node_ids=all_selected_nodes,
        model=model,
        loader=loader,
        window=window,
        context_pre=context_pre,
        context_post=context_post,
        target_names=resolved_targets,
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
    dataset = cast(EpiDataset, loader.dataset)
    config = dataset.config
    fig = make_joint_forecast_figure(
        samples=grouped_samples,
        input_window_length=int(config.model.input_window_length),
        forecast_horizon=int(config.model.forecast_horizon),
        context_pre=context_pre,
        context_post=context_post,
        target_names=resolved_targets,
    )

    if fig is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"[plot] Saved figure to: {output_path}")

    separate_figures: dict[str, Any] = {}
    for target_name in resolved_targets:
        target_fig = make_forecast_figure(
            samples=grouped_samples,
            input_window_length=int(config.model.input_window_length),
            forecast_horizon=int(config.model.forecast_horizon),
            context_pre=context_pre,
            context_post=context_post,
            target=target_name,
        )
        if target_fig is None:
            continue
        separate_figures[target_name] = target_fig
        if output_path is not None:
            target_output_path = output_path.with_name(
                f"{output_path.stem}_{target_name}{output_path.suffix}"
            )
            target_fig.savefig(target_output_path, dpi=200, bbox_inches="tight")
            logger.info(f"[plot] Saved figure to: {target_output_path}")

    if fig is not None and (log_dir is not None or wandb.run is not None):
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
        _ensure_wandb_run(
            config=dataset.config,
            log_dir=log_dir,
            name="forecast_plots",
            job_type="eval",
        )
        if wandb.run is not None:
            log_payload: dict[str, Any] = {}
            if fig is not None:
                log_payload[f"{wandb_prefix}/joint"] = wandb.Image(fig)
            for target_name, target_fig in separate_figures.items():
                log_payload[f"{wandb_prefix}/{target_name}"] = wandb.Image(target_fig)
            if log_payload:
                wandb.log(log_payload, step=0)

    return {
        "figure": fig,
        "joint_figure": fig,
        "separate_figures": separate_figures,
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
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate checkpoint - pure evaluation, no selection or plotting.

    Args:
        checkpoint_path: Path to checkpoint file
        split: Which split to evaluate ("val" or "test")
        device: Device to use for evaluation (overridden by training.device in overrides)
    log_dir: Optional W&B run directory for forecast plots
        overrides: Optional list of dotted-key config overrides (e.g., ["training.val_workers=4"])
        output_csv_path: Optional path to save node-level metrics CSV

    Returns:
        Dict with: checkpoint, config, model, loader, node_mae_dict,
                   eval_loss, eval_metrics
    """
    # Extract training.device from overrides if present
    resolved_device = device
    if overrides:
        for ov in overrides:
            if ov.startswith("training.device="):
                resolved_device = ov.split("=", 1)[1]
                break

    logger.info(f"[eval] Loading checkpoint: {checkpoint_path}")
    model, config, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        device=resolved_device,
        overrides=list(overrides) if overrides else None,
    )
    logger.info(
        f"[eval] Loaded model (params={sum(p.numel() for p in model.parameters()):,})"
    )
    logger.info(
        f"[eval] Building {split} loader from dataset: {config.data.dataset_path}"
    )
    loader, region_embeddings = build_loader_from_config(
        config, split=split, device=resolved_device, batch_size=batch_size
    )
    dataset = cast(EpiDataset, loader.dataset)
    logger.info(f"[eval] {split} samples: {len(dataset)}")

    # Run evaluation - returns node_mae_dict as third value
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
            output_csv_path=output_csv_path,
        )
    except Exception as exc:
        logger.warning(f"[eval] Metrics evaluation failed: {exc}")

    forecast_plot_result: dict[str, Any] | None = None
    if split.lower() == "test" and node_mae_dict:
        k = max(1, int(config.training.num_forecast_samples))
        worst_nodes = select_nodes_by_loss(
            node_mae=node_mae_dict, strategy="worst", k=k
        ).get("Worst", [])
        best_nodes = select_nodes_by_loss(
            node_mae=node_mae_dict, strategy="best", k=k
        ).get("Best", [])
        node_groups = {"Poorly-performing": worst_nodes, "Well-performing": best_nodes}

        if any(node_groups.values()):
            output_path = None
            if log_dir is not None:
                output_path = log_dir / f"{split}_forecasts_joint.png"
            forecast_plot_result = generate_forecast_plots(
                model=model,
                loader=loader,
                node_groups=node_groups,
                window="last",
                context_pre=30,
                context_post=30,
                output_path=output_path,
                log_dir=log_dir,
                target_names=list(DEFAULT_PLOT_TARGETS),
                wandb_prefix=f"forecasts_{split}",
            )
        else:
            logger.warning("[plot] Could not select test nodes for forecast plots")

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
        "node_mae": node_mae_dict,
        "eval_loss": eval_loss,
        "eval_metrics": eval_metrics,
        "forecast_plots": forecast_plot_result,
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
