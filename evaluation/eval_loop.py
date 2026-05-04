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
from evaluation.granular_export import (
    GRANULAR_SCHEMA_VERSION,
    GranularEvalWriter,
    write_metadata_sidecar,
    write_granular_metadata_sidecar,
)
from evaluation.loaders import build_loader_from_config, load_model_from_checkpoint
from evaluation.losses import JointInferenceLoss, get_loss_from_config
from evaluation.metrics import TorchMaskedMetricAccumulator
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster
from utils.device import (
    iter_device_ready_batches,
    prefetch_enabled,
    setup_device_streams,
)
from utils.log_keys import CORE_EVAL_METRICS, build_eval_metric_key, build_loss_key
from utils.sparsity_logging import log_sparsity_loss_correlation
from utils.training_utils import drop_nowcast

logger = logging.getLogger(__name__)

PER_HEAD_NODE_METRICS_SCHEMA_VERSION = "1"
PER_HEAD_NODE_METRICS_FIELDNAMES = [
    "target",
    "node_id",
    "region_id",
    "region_label",
    "population",
    "observed_count",
    "mae",
    "rmse",
]
_HEAD_CANONICAL_NAMES = {
    "hosp": "hospitalizations",
    "ww": "wastewater",
    "cases": "cases",
    "deaths": "deaths",
}


def _accumulate_per_head_node_metrics(
    *,
    target_name: str,
    target_nodes: torch.Tensor,
    diff: torch.Tensor,
    abs_diff: torch.Tensor,
    weights: torch.Tensor,
    accumulator: dict[str, dict[int, dict[str, float]]],
) -> None:
    canonical_target = _HEAD_CANONICAL_NAMES[target_name]
    target_stats = accumulator.setdefault(canonical_target, {})
    target_node_ids = target_nodes.detach().cpu().tolist()
    per_sample_abs_sum = (abs_diff * weights).sum(dim=1).detach().cpu().tolist()
    per_sample_sq_sum = ((diff**2) * weights).sum(dim=1).detach().cpu().tolist()
    per_sample_count = (weights > 0).sum(dim=1).detach().cpu().tolist()

    for node_id, abs_sum, sq_sum, observed_count in zip(
        target_node_ids,
        per_sample_abs_sum,
        per_sample_sq_sum,
        per_sample_count,
        strict=False,
    ):
        count_value = int(observed_count)
        if count_value <= 0:
            continue
        node_stats = target_stats.setdefault(
            int(node_id),
            {"abs_error_sum": 0.0, "sq_error_sum": 0.0, "observed_count": 0.0},
        )
        node_stats["abs_error_sum"] += float(abs_sum)
        node_stats["sq_error_sum"] += float(sq_sum)
        node_stats["observed_count"] += float(count_value)


def _write_per_head_node_metrics_csv(
    *,
    output_path: Path,
    per_head_node_metrics: dict[str, dict[int, dict[str, float]]],
    region_ids: list[str] | None,
    region_labels: list[str] | None,
    population_by_node: list[float] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_lib.DictWriter(f, fieldnames=PER_HEAD_NODE_METRICS_FIELDNAMES)
        writer.writeheader()
        for target in _HEAD_CANONICAL_NAMES.values():
            target_stats = per_head_node_metrics.get(target, {})
            for node_id in sorted(target_stats):
                stats = target_stats[node_id]
                observed_count = int(stats["observed_count"])
                if observed_count <= 0:
                    continue
                region_id = ""
                region_label = ""
                population = float("nan")
                if region_ids is not None and 0 <= node_id < len(region_ids):
                    region_id = str(region_ids[node_id])
                if region_labels is not None and 0 <= node_id < len(region_labels):
                    region_label = str(region_labels[node_id])
                if not region_label:
                    region_label = region_id
                if population_by_node is not None and 0 <= node_id < len(
                    population_by_node
                ):
                    population = float(population_by_node[node_id])

                writer.writerow(
                    {
                        "target": target,
                        "node_id": node_id,
                        "region_id": region_id,
                        "region_label": region_label,
                        "population": population,
                        "observed_count": observed_count,
                        "mae": stats["abs_error_sum"] / observed_count,
                        "rmse": math.sqrt(stats["sq_error_sum"] / observed_count),
                    }
                )


def _format_eval_summary(loss: float, metrics: dict[str, Any]) -> str:
    """Format evaluation results as a markdown table."""

    def _fmt(value: float | None) -> str:
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.6f}"

    rows = [
        ("Loss", _fmt(loss)),
        ("Joint Obs Loss", _fmt(metrics.get("mae"))),
        ("Hosp MAE", _fmt(metrics.get("mae_hosp_log1p_per_100k"))),
        ("WW MAE", _fmt(metrics.get("mae_ww_log1p_per_100k"))),
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
    node_metrics_csv_path: Path | None = None,
    per_head_node_metrics_csv_path: Path | None = None,
    granular_csv_path: Path | None = None,
    granular_metadata: dict[str, Any] | None = None,
    output_csv_path: Path | None = None,
    node_metrics_target: str = "hospitalizations",
    log_every: int = 10,
) -> tuple[float, dict[str, Any], dict[str, dict[int, float]]]:
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
        node_metrics_csv_path: Optional path to save per-node MAE metrics as CSV
        per_head_node_metrics_csv_path: Optional path to save per-head node metrics CSV
        granular_csv_path: Optional path to save granular per-example error rows as CSV
        granular_metadata: Optional metadata to write alongside the granular CSV
        output_csv_path: Deprecated alias for node_metrics_csv_path
        node_metrics_target: Canonical target name for the legacy node metrics CSV
        log_every: Progress logging cadence in batches

    Returns:
        Tuple of (mean_loss, metrics_dict, node_mae_dict) where:
        - mean_loss: Average loss per batch
        - metrics_dict: Dictionary of computed metrics
        - node_mae_dict: Dictionary mapping target name -> node_id -> average MAE
    """
    logger.info(f"{split_name} evaluation started...")
    if node_metrics_csv_path is None:
        node_metrics_csv_path = output_csv_path
    if node_metrics_target not in _HEAD_CANONICAL_NAMES.values():
        raise ValueError(
            f"Unknown node_metrics_target {node_metrics_target!r}; expected one of "
            f"{tuple(_HEAD_CANONICAL_NAMES.values())}."
        )

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
    loss_sird_supervision_sum = torch.tensor(0.0, device=device)
    loss_ww_weighted_sum = torch.tensor(0.0, device=device)
    loss_hosp_weighted_sum = torch.tensor(0.0, device=device)
    loss_cases_weighted_sum = torch.tensor(0.0, device=device)
    loss_deaths_weighted_sum = torch.tensor(0.0, device=device)
    loss_sird_supervision_weighted_sum = torch.tensor(0.0, device=device)
    loss_latent_s_sum = torch.tensor(0.0, device=device)
    loss_latent_i_sum = torch.tensor(0.0, device=device)
    loss_latent_r_sum = torch.tensor(0.0, device=device)
    loss_latent_d_sum = torch.tensor(0.0, device=device)

    per_head_node_metrics: dict[str, dict[int, dict[str, float]]] = {}

    num_batches = len(loader)
    eval_iter = loader
    effective_log_every = max(1, log_every)
    dataset = getattr(loader, "dataset", None)
    temporal_coords = getattr(dataset, "_temporal_coords", None)
    region_ids = getattr(dataset, "_region_ids", None)
    region_labels = getattr(dataset, "_region_labels", None)
    granular_writer = (
        GranularEvalWriter(path=granular_csv_path, split_name=split_name)
        if granular_csv_path is not None
        else None
    )

    model_was_training = model.training
    model.eval()
    forward_model = cast(EpiForecaster, model)
    dataset_config = getattr(dataset, "config", None)
    prefetch_factor = (
        dataset_config.training.prefetch_factor if dataset_config is not None else None
    )
    use_prefetch = prefetch_enabled(prefetch_factor)
    eval_iter = iter_device_ready_batches(
        loader,
        device=device,
        prefetch_factor=prefetch_factor,
        streams=setup_device_streams(device) if use_prefetch else None,
    )
    current_batch_idx: int | None = None
    current_stage = "init"
    processed_batches = 0
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
                current_batch_idx = batch_idx
                current_stage = "batch_fetched"
                should_log_batch = batch_idx % effective_log_every == 0
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )

                current_stage = "forward_batch"
                model_outputs, targets_dict = forward_model.forward_batch(
                    batch_data=batch_data,
                    region_embeddings=region_embeddings,
                    mask_cases=criterion.mask_input_cases,
                    mask_ww=criterion.mask_input_ww,
                    mask_hosp=criterion.mask_input_hosp,
                    mask_deaths=criterion.mask_input_deaths,
                )
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )

                # Create sliced model outputs for metric computation
                sliced_model_outputs = {
                    k: drop_nowcast(v, horizon)
                    if k.startswith("pred_") and isinstance(v, torch.Tensor)
                    else v
                    for k, v in model_outputs.items()
                }

                # Compute loss with batch_data for continuity penalty
                current_stage = "loss_components"
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
                loss_sird_supervision_sum += components["sird_supervision"].detach()
                loss_latent_s_sum += components["latent_s"].detach()
                loss_latent_i_sum += components["latent_i"].detach()
                loss_latent_r_sum += components["latent_r"].detach()
                loss_latent_d_sum += components["latent_d"].detach()
                if "continuity" in components:
                    pass  # Don't accumulate continuity loss in metrics
                loss_ww_weighted_sum += components["ww_weighted"].detach()
                loss_hosp_weighted_sum += components["hosp_weighted"].detach()
                loss_cases_weighted_sum += components["cases_weighted"].detach()
                loss_deaths_weighted_sum += components["deaths_weighted"].detach()
                loss_sird_supervision_weighted_sum += components[
                    "sird_supervision_weighted"
                ].detach()
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )

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
                    _unused_diff, abs_diff, weights = hosp_metrics.update(
                        predictions=pred_hosp,
                        targets=hosp_targets,
                        observed_mask=hosp_mask,
                        sample_weights=hosp_weights,
                    )
                    target_nodes = batch_data.target_node

                    _accumulate_per_head_node_metrics(
                        target_name="hosp",
                        target_nodes=target_nodes,
                        diff=_unused_diff,
                        abs_diff=abs_diff,
                        weights=weights,
                        accumulator=per_head_node_metrics,
                    )

                    if granular_writer is not None:
                        granular_writer.write_rows(
                            batch_data=batch_data,
                            target_name="hosp",
                            predictions=pred_hosp,
                            targets=hosp_targets,
                            weights=weights,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                        )
                    del (
                        _unused_diff,
                        abs_diff,
                        weights,
                        target_nodes,
                    )
                del pred_hosp, hosp_targets, hosp_mask, hosp_weights

                pred_ww = sliced_model_outputs.get("pred_ww")
                ww_targets = metric_supervision["ww"]["target"]
                ww_mask = targets_dict.get("ww_mask")
                ww_weights = metric_supervision["ww"]["weights"]
                if (
                    pred_ww is not None
                    and ww_targets is not None
                    and ww_weights is not None
                ):
                    ww_diff, ww_abs_diff, ww_effective_weights = ww_metrics.update(
                        predictions=pred_ww,
                        targets=ww_targets,
                        observed_mask=ww_mask,
                        sample_weights=ww_weights,
                    )
                    _accumulate_per_head_node_metrics(
                        target_name="ww",
                        target_nodes=batch_data.target_node,
                        diff=ww_diff,
                        abs_diff=ww_abs_diff,
                        weights=ww_effective_weights,
                        accumulator=per_head_node_metrics,
                    )
                    if granular_writer is not None:
                        granular_writer.write_rows(
                            batch_data=batch_data,
                            target_name="ww",
                            predictions=pred_ww,
                            targets=ww_targets,
                            weights=ww_weights,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                        )
                del pred_ww, ww_targets, ww_mask, ww_weights

                pred_cases = sliced_model_outputs.get("pred_cases")
                cases_targets = metric_supervision["cases"]["target"]
                cases_mask = targets_dict.get("cases_mask")
                cases_weights = metric_supervision["cases"]["weights"]
                if (
                    pred_cases is not None
                    and cases_targets is not None
                    and cases_weights is not None
                ):
                    cases_diff, cases_abs_diff, cases_effective_weights = (
                        cases_metrics.update(
                            predictions=pred_cases,
                            targets=cases_targets,
                            observed_mask=cases_mask,
                            sample_weights=cases_weights,
                        )
                    )
                    _accumulate_per_head_node_metrics(
                        target_name="cases",
                        target_nodes=batch_data.target_node,
                        diff=cases_diff,
                        abs_diff=cases_abs_diff,
                        weights=cases_effective_weights,
                        accumulator=per_head_node_metrics,
                    )
                    if granular_writer is not None:
                        granular_writer.write_rows(
                            batch_data=batch_data,
                            target_name="cases",
                            predictions=pred_cases,
                            targets=cases_targets,
                            weights=cases_weights,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                        )
                del pred_cases, cases_targets, cases_mask, cases_weights

                pred_deaths = sliced_model_outputs.get("pred_deaths")
                deaths_targets = metric_supervision["deaths"]["target"]
                deaths_mask = targets_dict.get("deaths_mask")
                deaths_weights = metric_supervision["deaths"]["weights"]
                if (
                    pred_deaths is not None
                    and deaths_targets is not None
                    and deaths_weights is not None
                ):
                    deaths_diff, deaths_abs_diff, deaths_effective_weights = (
                        deaths_metrics.update(
                            predictions=pred_deaths,
                            targets=deaths_targets,
                            observed_mask=deaths_mask,
                            sample_weights=deaths_weights,
                        )
                    )
                    _accumulate_per_head_node_metrics(
                        target_name="deaths",
                        target_nodes=batch_data.target_node,
                        diff=deaths_diff,
                        abs_diff=deaths_abs_diff,
                        weights=deaths_effective_weights,
                        accumulator=per_head_node_metrics,
                    )
                    if granular_writer is not None:
                        granular_writer.write_rows(
                            batch_data=batch_data,
                            target_name="deaths",
                            predictions=pred_deaths,
                            targets=deaths_targets,
                            weights=deaths_weights,
                            temporal_coords=temporal_coords,
                            region_ids=region_ids,
                            region_labels=region_labels,
                        )
                del pred_deaths, deaths_targets, deaths_mask, deaths_weights
                current_stage = "granular_write_complete"
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )
                processed_batches += 1
                del components, metric_supervision, model_outputs, targets_dict
                del sliced_model_outputs, batch_data, loss

    except Exception:
        logger.error(
            "[eval] Evaluation failed: split=%s batch=%s stage=%s",
            split_name,
            current_batch_idx if current_batch_idx is not None else "n/a",
            current_stage,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise
    finally:
        if granular_writer is not None:
            granular_writer.close()
        if model_was_training:
            model.train()

    # Final sync - transfer metrics to CPU once
    mean_loss = (total_loss / max(1, processed_batches)).item()
    hosp_summary = hosp_metrics.finalize()
    ww_summary = ww_metrics.finalize()
    cases_summary = cases_metrics.finalize()
    deaths_summary = deaths_metrics.finalize()

    node_mae_by_target: dict[str, dict[int, float]] = {}
    node_mae_count_by_target: dict[str, dict[int, int]] = {}
    for target_name, target_stats in per_head_node_metrics.items():
        node_mae_by_target[target_name] = {}
        node_mae_count_by_target[target_name] = {}
        for node_id, stats in target_stats.items():
            observed_count = int(stats["observed_count"])
            if observed_count <= 0:
                continue
            node_mae_by_target[target_name][node_id] = (
                stats["abs_error_sum"] / observed_count
            )
            node_mae_count_by_target[target_name][node_id] = observed_count

    if node_metrics_csv_path is not None:
        node_metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(node_metrics_csv_path, "w", newline="") as f:
            writer = csv_lib.writer(f)
            writer.writerow(["target", "node_id", "mae", "num_samples"])
            selected_node_mae = node_mae_by_target.get(node_metrics_target, {})
            selected_node_counts = node_mae_count_by_target.get(node_metrics_target, {})
            for node_id in sorted(selected_node_mae.keys()):
                writer.writerow(
                    [
                        node_metrics_target,
                        node_id,
                        selected_node_mae[node_id],
                        selected_node_counts[node_id],
                    ]
                )

    population_by_node: list[float] | None = None
    if dataset is not None and hasattr(dataset, "node_static_covariates"):
        population_tensor = dataset.node_static_covariates.get("Pop")
        if population_tensor is not None:
            population_by_node = population_tensor.detach().cpu().tolist()

    if per_head_node_metrics_csv_path is not None:
        _write_per_head_node_metrics_csv(
            output_path=per_head_node_metrics_csv_path,
            per_head_node_metrics=per_head_node_metrics,
            region_ids=region_ids,
            region_labels=region_labels,
            population_by_node=population_by_node,
        )
        per_head_metadata = {
            "dataset_path": getattr(dataset, "aligned_data_path", None),
            "run_id": getattr(dataset, "_run_id_value", None)
            if dataset is not None
            else None,
            "split": split_name.lower(),
        }
        if granular_metadata is not None:
            per_head_metadata.update(granular_metadata)
        write_metadata_sidecar(
            per_head_node_metrics_csv_path,
            per_head_metadata,
            schema_version=PER_HEAD_NODE_METRICS_SCHEMA_VERSION,
        )

    if granular_csv_path is not None:
        metadata = {
            "dataset_path": getattr(dataset, "aligned_data_path", None),
            "granular_csv_path": granular_csv_path,
            "observed_only": True,
            "region_name_source": getattr(dataset, "_region_name_source", None),
            "run_id": getattr(dataset, "run_id", None),
            "schema_version": GRANULAR_SCHEMA_VERSION,
            "split": split_name.lower(),
            "training_seed": getattr(getattr(dataset, "config", None), "training", None)
            and getattr(dataset.config.training, "seed", None),
            "node_split_strategy": getattr(
                getattr(dataset, "config", None), "training", None
            )
            and getattr(dataset.config.training, "node_split_strategy", None),
            "node_split_population_bins": getattr(
                getattr(dataset, "config", None), "training", None
            )
            and getattr(dataset.config.training, "node_split_population_bins", None),
            "val_split": getattr(getattr(dataset, "config", None), "training", None)
            and getattr(dataset.config.training, "val_split", None),
            "test_split": getattr(getattr(dataset, "config", None), "training", None)
            and getattr(dataset.config.training, "test_split", None),
        }
        if granular_metadata is not None:
            metadata.update(granular_metadata)
        write_granular_metadata_sidecar(
            granular_csv_path,
            metadata,
        )

    joint_obs_loss_total = float(
        (
            loss_ww_weighted_sum
            + loss_hosp_weighted_sum
            + loss_cases_weighted_sum
            + loss_deaths_weighted_sum
        )
        / max(1, processed_batches)
    )
    metrics = {
        # Generic primary metric: aliased joint observation loss
        "mae": joint_obs_loss_total,
        "joint_obs_loss_total": joint_obs_loss_total,
        # Explicit hospitalization metrics
        "mae_hosp_per_h": hosp_summary.mae_per_h,
        "rmse_hosp_per_h": hosp_summary.rmse_per_h,
        # Hospitalization metrics in log1p(per-100k) space
        "mae_hosp_log1p_per_100k": hosp_summary.mae,
        "rmse_hosp_log1p_per_100k": hosp_summary.rmse,
        "r2_hosp_log1p_per_100k": hosp_summary.r2,
        "observed_count_hosp": hosp_summary.observed_count,
        "effective_count_hosp": hosp_summary.effective_count,
        # Wastewater metrics in log1p(per-100k) space
        "mae_ww_log1p_per_100k": ww_summary.mae,
        "rmse_ww_log1p_per_100k": ww_summary.rmse,
        "r2_ww_log1p_per_100k": ww_summary.r2,
        "observed_count_ww": ww_summary.observed_count,
        "effective_count_ww": ww_summary.effective_count,
        # Cases metrics in log1p(per-100k) space
        "mae_cases_log1p_per_100k": cases_summary.mae,
        "rmse_cases_log1p_per_100k": cases_summary.rmse,
        "r2_cases_log1p_per_100k": cases_summary.r2,
        "observed_count_cases": cases_summary.observed_count,
        "effective_count_cases": cases_summary.effective_count,
        # Deaths metrics in log1p(per-100k) space
        "mae_deaths_log1p_per_100k": deaths_summary.mae,
        "rmse_deaths_log1p_per_100k": deaths_summary.rmse,
        "r2_deaths_log1p_per_100k": deaths_summary.r2,
        "observed_count_deaths": deaths_summary.observed_count,
        "effective_count_deaths": deaths_summary.effective_count,
        # Joint loss components (averaged per batch, same reduction as mean_loss)
        "loss_ww": (loss_ww_sum / max(1, processed_batches)).item(),
        "loss_hosp": (loss_hosp_sum / max(1, processed_batches)).item(),
        "loss_cases": (loss_cases_sum / max(1, processed_batches)).item(),
        "loss_deaths": (loss_deaths_sum / max(1, processed_batches)).item(),
        "loss_sird_supervision": (
            loss_sird_supervision_sum / max(1, processed_batches)
        ).item(),
        "loss_latent_s": (loss_latent_s_sum / max(1, processed_batches)).item(),
        "loss_latent_i": (loss_latent_i_sum / max(1, processed_batches)).item(),
        "loss_latent_r": (loss_latent_r_sum / max(1, processed_batches)).item(),
        "loss_latent_d": (loss_latent_d_sum / max(1, processed_batches)).item(),
        "loss_ww_weighted": (loss_ww_weighted_sum / max(1, processed_batches)).item(),
        "loss_hosp_weighted": (
            loss_hosp_weighted_sum / max(1, processed_batches)
        ).item(),
        "loss_cases_weighted": (
            loss_cases_weighted_sum / max(1, processed_batches)
        ).item(),
        "loss_deaths_weighted": (
            loss_deaths_weighted_sum / max(1, processed_batches)
        ).item(),
        "loss_sird_supervision_weighted": (
            loss_sird_supervision_weighted_sum / max(1, processed_batches)
        ).item(),
    }

    logger.info("EVAL COMPLETE")
    return mean_loss, metrics, node_mae_by_target


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
    node_metrics_csv_path: Path | None = None,
    per_head_node_metrics_csv_path: Path | None = None,
    granular_csv_path: Path | None = None,
    output_csv_path: Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    node_metrics_target: str = "hospitalizations",
    log_every: int = 10,
) -> dict[str, Any]:
    """
    Evaluate checkpoint - pure evaluation, no selection or plotting.

    Args:
        checkpoint_path: Path to checkpoint file
        split: Which split to evaluate ("val", "test", or "full")
        device: Device to use for evaluation (overridden by training.device in overrides)
        log_dir: Optional W&B run directory for forecast plots
        overrides: Optional list of dotted-key config overrides (e.g., ["training.val_workers=4"])
        node_metrics_csv_path: Optional path to save node-level metrics CSV
        per_head_node_metrics_csv_path: Optional path to save per-head node metrics CSV
        granular_csv_path: Optional path to save granular per-example error rows
        output_csv_path: Deprecated alias for node_metrics_csv_path
        batch_size: Optional batch size override
        num_workers: Optional DataLoader worker override
        pin_memory: Optional pin-memory override
        prefetch_factor: Optional prefetch override. Use 0 or None to disable.
        node_metrics_target: Canonical target name for node-level CSV export
        log_every: Progress logging cadence in batches

    Returns:
        Dict with: checkpoint, config, model, loader, node_mae_dict,
                   eval_loss, eval_metrics
    """
    # Extract training.device from overrides if present
    if node_metrics_csv_path is None:
        node_metrics_csv_path = output_csv_path

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
        "[eval] Building %s loader from dataset=%s resolved_device=%s batch_size=%s "
        "num_workers=%s prefetch_factor=%s pin_memory=%s",
        split,
        config.data.dataset_path,
        resolved_device,
        batch_size if batch_size is not None else config.training.batch_size,
        num_workers if num_workers is not None else "config",
        (
            prefetch_factor
            if prefetch_factor is not None
            else config.training.prefetch_factor
        ),
        pin_memory if pin_memory is not None else config.training.pin_memory,
    )
    loader, region_embeddings = build_loader_from_config(
        config,
        split=split,
        device=resolved_device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    dataset = cast(EpiDataset, loader.dataset)
    logger.info(
        "[eval] Environment: split=%s dataset=%s samples=%d batch_size=%s num_workers=%d "
        "prefetch_factor=%s pin_memory=%s device=%s",
        split,
        config.data.dataset_path,
        len(dataset),
        loader.batch_size,
        loader.num_workers,
        loader.prefetch_factor,
        loader.pin_memory,
        next(model.parameters()).device,
    )

    if granular_csv_path is None and config.output.write_granular_eval:
        granular_csv_path = (
            checkpoint_path.parent.parent
            / config.output.resolve_granular_eval_filename(split=split)
        )

    criterion = get_loss_from_config(
        config.training.loss,
        data_config=config.data,
        forecast_horizon=config.model.forecast_horizon,
    )
    try:
        eval_loss, eval_metrics, node_mae_dict = evaluate_loader(
            model=model,
            loader=loader,
            criterion=criterion,
            horizon=int(config.model.forecast_horizon),
            device=next(model.parameters()).device,
            region_embeddings=region_embeddings,
            split_name=split.capitalize(),
            node_metrics_csv_path=node_metrics_csv_path,
            per_head_node_metrics_csv_path=per_head_node_metrics_csv_path,
            granular_csv_path=granular_csv_path,
            node_metrics_target=node_metrics_target,
            log_every=log_every,
            granular_metadata={
                "batch_size": batch_size
                if batch_size is not None
                else loader.batch_size,
                "checkpoint_path": checkpoint_path,
                "config_path": checkpoint_path.parent.parent / "config.yaml",
                "forecast_horizon": int(config.model.forecast_horizon),
                "input_window_length": int(config.model.input_window_length),
                "max_batches": None,
                "model_experiment_name": config.output.experiment_name,
                "node_split_population_bins": int(
                    config.training.node_split_population_bins
                ),
                "node_split_strategy": config.training.node_split_strategy,
                "node_metrics_target": node_metrics_target,
                "overrides": list(overrides) if overrides else [],
                "run_dir": checkpoint_path.parent.parent,
                "split": split.lower(),
                "test_split": float(config.training.test_split),
                "training_seed": config.training.seed,
                "val_split": float(config.training.val_split),
                "window_stride": int(config.data.window_stride),
                "eval_log_every": int(log_every),
                "eval_num_workers": int(loader.num_workers),
                "eval_prefetch_factor": loader.prefetch_factor,
                "eval_pin_memory": bool(loader.pin_memory),
            },
        )
    except Exception:
        logger.exception("[eval] Metrics evaluation failed")
        raise

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
                log_data[build_loss_key(split=split)] = eval_loss
            for key in CORE_EVAL_METRICS:
                if key in eval_metrics:
                    log_data[build_eval_metric_key(key, split)] = eval_metrics[key]
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
    }
