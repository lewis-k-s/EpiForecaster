"""Evaluation loop implementation for EpiForecaster models.

This module provides the core evaluation loop that computes loss and metrics
over a DataLoader, matching the trainer's evaluation behavior.
"""

from __future__ import annotations

import csv as csv_lib
import logging
import math
from pathlib import Path
from typing import Any, cast

import torch
import wandb
from torch.utils.data import DataLoader

from data.epi_dataset import EpiDataset
from evaluation.loaders import build_loader_from_config, load_model_from_checkpoint
from evaluation.losses import JointInferenceLoss, get_loss_from_config
from evaluation.metrics import TorchMaskedMetricAccumulator
from evaluation.selection import select_nodes_by_loss
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster
from plotting.forecast_plots import DEFAULT_PLOT_TARGETS
from utils.sparsity_logging import log_sparsity_loss_correlation
from utils.training_utils import drop_nowcast, inject_gpu_mobility

logger = logging.getLogger(__name__)


def _format_eval_summary(loss: float, metrics: dict[str, Any]) -> str:
    """Format evaluation results as a markdown table."""

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

    Args:
        model: The EpiForecaster model to evaluate
        loader: DataLoader providing evaluation batches
        criterion: Loss function for computing evaluation loss
        horizon: Forecast horizon for dropping nowcast period
        device: Device for tensor operations
        region_embeddings: Optional pre-loaded region embeddings
        split_name: Name of the split for logging (e.g., "Val", "Test")
        max_batches: Optional limit on number of batches to evaluate
        output_csv_path: Optional path to save per-node MAE metrics as CSV

    Returns:
        Tuple of (mean_loss, metrics_dict, node_mae_dict) where:
        - mean_loss: Average loss per batch
        - metrics_dict: Dictionary of computed metrics (MAE, RMSE, sMAPE, R2, etc.)
        - node_mae_dict: Dictionary mapping node_id -> average MAE
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


def _ensure_wandb_run(
    *,
    config: EpiForecasterConfig | None,
    log_dir: Path | None,
    name: str,
    job_type: str,
) -> Any:
    """Ensure a W&B run exists, creating one if needed."""
    if wandb.run is not None:
        return wandb.run
    if log_dir is None:
        return None
    project = config.output.wandb_project if config is not None else "epiforecaster"
    entity = config.output.wandb_entity if config is not None else None
    group = None
    mode: Any = "online"
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
    # Lazy import to avoid circular dependency with epiforecaster_eval
    from evaluation.epiforecaster_eval import generate_forecast_plots

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
