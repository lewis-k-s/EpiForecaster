"""Evaluation loop implementation for EpiForecaster models.

This module provides the core evaluation loop that computes loss and metrics
over a DataLoader, matching the trainer's evaluation behavior.
"""

from __future__ import annotations

import csv as csv_lib
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import wandb
from torch.utils.data import DataLoader

from data.crossval_manifest import load_manifest_region_sets
from data.epi_dataset import EpiDataset
from data.latent_supervision import latent_scales_from_checkpoint
from evaluation.granular_export import (
    GRANULAR_SCHEMA_VERSION,
    GranularEvalWriter,
    infer_granular_eval_format,
    write_metadata_sidecar,
    write_granular_metadata_sidecar,
)
from evaluation.loaders import build_loader_from_config, load_model_from_checkpoint
from evaluation.latent_diagnostics import build_latent_diagnostic_protocols
from evaluation.latent_information import TorchLatentInformationAccumulator
from evaluation.losses import JointInferenceLoss, get_loss_from_config
from evaluation.pathway_metrics import (
    TorchLatentCoherenceAccumulator,
    TorchLatentPlausibilityAccumulator,
    TorchLatentProtocolAccumulator,
)
from evaluation.target_scoring import (
    CANONICAL_TARGET_NAMES,
    NodeStats,
    TargetScorer,
    build_target_forecast,
    region_set_members,
)
from evaluation.selection_scores import compute_validation_selection_score
from models.configs import EpiForecasterConfig
from models.epiforecaster import EpiForecaster
from models.observation_heads import DelayKernel, SheddingConvolution
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
_DEFAULT_RELEVANT_SOURCE_MAPPING = {
    "hosp": "hospitalization_flow",
    "ww": "I",
    "cases": "I",
    "deaths": "death_flow",
}
# Heads observed as weekly totals dated on the week start, by the daily latent
# flow that the total accumulates.
_WEEK_TOTAL_FLOW_SOURCES = {"hosp": "hospitalization_flow"}
_ALLOWED_RELEVANT_SOURCES = {
    "hosp": {"hospitalization_flow", "H"},
    "ww": {"I", "infection_flow"},
    "cases": {"I", "infection_flow"},
    "deaths": {"death_flow", "D"},
}

# Latent flow compared with each observed count series for implied ascertainment.
_ASCERTAINMENT_LATENT_FLOWS = {
    "cases": "infection_flow",
    "hosp": "hospitalization_flow",
    "deaths": "death_flow",
}


def _resolve_relevant_source_mapping(
    relevant_source_mapping: dict[str, str] | None,
) -> dict[str, str]:
    if relevant_source_mapping is None:
        return dict(_DEFAULT_RELEVANT_SOURCE_MAPPING)

    expected_heads = set(CANONICAL_TARGET_NAMES)
    actual_heads = set(relevant_source_mapping)
    if actual_heads != expected_heads:
        raise ValueError(
            "relevant_source_mapping must define exactly "
            f"{sorted(expected_heads)}, got {sorted(actual_heads)}"
        )

    resolved = dict(relevant_source_mapping)
    for head_name, source_name in resolved.items():
        allowed_sources = _ALLOWED_RELEVANT_SOURCES[head_name]
        if source_name not in allowed_sources:
            raise ValueError(
                f"Relevant source for {head_name!r} must be one of "
                f"{sorted(allowed_sources)}, got {source_name!r}"
            )
    return resolved


def _write_per_head_node_metrics_csv(
    *,
    output_path: Path,
    per_head_node_metrics: Mapping[str, NodeStats],
    region_ids: list[str] | None,
    region_labels: list[str] | None,
    population_by_node: list[float] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_lib.DictWriter(f, fieldnames=PER_HEAD_NODE_METRICS_FIELDNAMES)
        writer.writeheader()
        for target in CANONICAL_TARGET_NAMES.values():
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


def _add_region_set_metrics(
    *,
    metrics: dict[str, Any],
    region_set_metrics: dict[str, dict[str, Any]],
) -> None:
    """Add ``region_set_<name>_*`` probabilistic metrics and their WIS endpoint.

    ``region_set_metrics`` holds every target's probabilistic keys per set. The
    endpoint follows the checkpoint-selection rule: the equal-head mean of
    pooled per-head WIS, NaN when a head has no valid support in the set.
    """
    for name, set_metrics in region_set_metrics.items():
        try:
            mean_target_wis, _ = compute_validation_selection_score(
                set_metrics,
                observation_loss="wis",
                selection_metric="mean_target_wis",
                required_heads=tuple(CANONICAL_TARGET_NAMES),
            )
        except ValueError:
            mean_target_wis = float("nan")
        metrics.update(
            {f"region_set_{name}_{key}": value for key, value in set_metrics.items()}
        )
        metrics[f"region_set_{name}_mean_target_wis"] = mean_target_wis


class _LossLedger:
    """Device-local running means of named loss components over batches."""

    def __init__(self, names: Sequence[str], device: torch.device) -> None:
        self._sums = {name: torch.tensor(0.0, device=device) for name in names}
        self.batch_count = 0

    def update(self, components: Mapping[str, torch.Tensor]) -> None:
        for name, total in self._sums.items():
            total += components[name].detach()
        self.batch_count += 1

    def mean(self, *names: str, factor: float = 1.0) -> float:
        """Per-batch mean of the sum of ``names``, multiplied by ``factor``."""
        total = self._sums[names[0]]
        for name in names[1:]:
            total = total + self._sums[name]
        return (factor * total / max(1, self.batch_count)).item()


_OBSERVATION_LOSS_COMPONENTS = ("ww", "hosp", "cases", "deaths")
_LATENT_LOSS_TASKS = ("s", "i", "h", "r", "death_flow")
_REGULARIZER_COMPONENTS = (
    "history_reconstruction",
    "residual_penalty",
    "beta_smoothness",
)
# Metric key of each loss component that evaluation reports as a mean.
_LOSS_METRIC_COMPONENTS = {
    **{f"loss_{name}": name for name in _OBSERVATION_LOSS_COMPONENTS},
    "loss_sird_supervision": "sird_supervision",
    **{f"loss_latent_{task}": f"latent_{task}" for task in _LATENT_LOSS_TASKS},
    **{
        f"loss_{name}_weighted": f"{name}_weighted"
        for name in _OBSERVATION_LOSS_COMPONENTS
    },
    "loss_sird_supervision_weighted": "sird_supervision_weighted",
    **{f"loss_{name}": name for name in _REGULARIZER_COMPONENTS},
    **{f"loss_latent_{task}_raw": f"latent_{task}_raw" for task in _LATENT_LOSS_TASKS},
    **{
        f"loss_latent_{task}_component_weighted": f"latent_{task}_weighted"
        for task in _LATENT_LOSS_TASKS
    },
}


def _add_delay_kernel_metrics(
    *,
    metrics: dict[str, Any],
    model: torch.nn.Module,
) -> None:
    """Add final kernel-state diagnostics for the observation heads.

    Run-level model-state metrics for ``test_summary.json``; the granular row
    schema is unaffected. Clinical heads route through ``DelayKernel`` and the
    wastewater head through ``SheddingConvolution``; both expose the same
    normalized-kernel and Gamma diagnostics.
    """
    model = getattr(model, "_orig_mod", model)
    kernels: dict[str, DelayKernel | SheddingConvolution] = {}
    for head_name in ("hosp", "cases", "deaths"):
        kernel = getattr(
            getattr(model, f"{head_name}_head", None), "delay_kernel", None
        )
        if isinstance(kernel, DelayKernel):
            kernels[head_name] = kernel
    ww_kernel = getattr(getattr(model, "ww_head", None), "shedding_conv", None)
    if isinstance(ww_kernel, SheddingConvolution):
        kernels["ww"] = ww_kernel
    for head_name, kernel in kernels.items():
        if kernel.parameterization not in {"simplex", "gamma"}:
            continue
        trainable_count = sum(
            parameter.numel()
            for parameter in kernel.parameters()
            if parameter.requires_grad
        )
        metrics[f"kernel_{head_name}_trainable_parameter_count"] = trainable_count
        metrics[f"kernel_{head_name}_mean_delay_days"] = kernel.get_mean_delay()
        metrics[f"kernel_{head_name}_delay_std_days"] = kernel.get_delay_std()
        if kernel.parameterization == "gamma":
            shape, scale = kernel.get_gamma_parameters()
            metrics[f"kernel_{head_name}_gamma_shape"] = shape
            metrics[f"kernel_{head_name}_gamma_scale"] = scale
            boundary_hits = kernel.get_gamma_boundary_hits()
            metrics[f"kernel_{head_name}_gamma_shape_lower_bound_hit"] = int(
                boundary_hits["shape_lower"]
            )
            metrics[f"kernel_{head_name}_gamma_shape_upper_bound_hit"] = int(
                boundary_hits["shape_upper"]
            )
            metrics[f"kernel_{head_name}_gamma_scale_lower_bound_hit"] = int(
                boundary_hits["scale_lower"]
            )
            metrics[f"kernel_{head_name}_gamma_scale_upper_bound_hit"] = int(
                boundary_hits["scale_upper"]
            )


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
    granular_output_path: Path | None = None,
    granular_eval_format: str | None = None,
    granular_metadata: dict[str, Any] | None = None,
    relevant_source_mapping: dict[str, str] | None = None,
    node_metrics_target: str = "hospitalizations",
    log_every: int = 10,
    latent_information: bool = False,
    region_sets: dict[str, frozenset[str]] | None = None,
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
        granular_output_path: Optional path to save granular per-example error rows
        granular_eval_format: Optional output format override ("parquet" or "csv")
        granular_metadata: Optional metadata to write alongside granular output
        relevant_source_mapping: Exact target-to-mechanistic-source mapping used
            for quantitative observation-latent coherence metrics and persisted
            qualitative plot overlays. This does not alter pathway attribution.
        node_metrics_target: Canonical target name for the legacy node metrics CSV
        log_every: Progress logging cadence in batches
        latent_information: Compute the within-window latent-information
            indicators (``latent_info_*``). They move every batch to CPU, so
            enable them for final evaluations, not per-epoch validation.
        region_sets: Named region-ID sets; each adds ``region_set_<name>_*``
            probabilistic metrics restricted to its regions' windows.

    Returns:
        Tuple of (mean_loss, metrics_dict, node_mae_dict) where:
        - mean_loss: Average loss per batch
        - metrics_dict: Dictionary of computed metrics
        - node_mae_dict: Dictionary mapping target name -> node_id -> average MAE
    """
    logger.info(f"{split_name} evaluation started...")
    if node_metrics_target not in CANONICAL_TARGET_NAMES.values():
        raise ValueError(
            f"Unknown node_metrics_target {node_metrics_target!r}; expected one of "
            f"{tuple(CANONICAL_TARGET_NAMES.values())}."
        )
    resolved_relevant_sources = _resolve_relevant_source_mapping(
        relevant_source_mapping
    )

    # Device-local accumulators - avoid sync until end. A compiled model also
    # gets compiled metric steps: eager updates launch hundreds of tiny kernels
    # per batch, which made validation cost more than training.
    compile_metrics = hasattr(model, "_orig_mod") and device.type == "cuda"
    loss_ledger = _LossLedger(
        ("total", *_LOSS_METRIC_COMPONENTS.values()), device=device
    )
    latent_metrics = TorchLatentCoherenceAccumulator(
        device=device, compile_step=compile_metrics
    )
    latent_plausibility_metrics = TorchLatentPlausibilityAccumulator(
        device=device, compile_step=compile_metrics
    )
    latent_diagnostic_metrics = {
        protocol: TorchLatentProtocolAccumulator(
            protocol=protocol, device=device, compile_step=compile_metrics
        )
        for protocol in ("oracle_x0", "teacher_forced_one_step")
    }
    latent_information_metrics = (
        TorchLatentInformationAccumulator(
            week_total_heads=[
                head
                for head, source in _WEEK_TOTAL_FLOW_SOURCES.items()
                if resolved_relevant_sources[head] == source
            ]
        )
        if latent_information
        else None
    )

    num_batches = len(loader)
    effective_log_every = max(1, log_every)
    dataset = getattr(loader, "dataset", None)
    temporal_coords = getattr(dataset, "_temporal_coords", None)
    region_ids = getattr(dataset, "_region_ids", None)
    region_labels = getattr(dataset, "_region_labels", None)
    members = region_set_members(region_sets, region_ids, device) if region_sets else {}
    resolved_granular_format = (
        infer_granular_eval_format(granular_output_path, granular_eval_format)
        if granular_output_path is not None
        else None
    )
    granular_writer = (
        GranularEvalWriter(
            path=granular_output_path,
            split_name=split_name,
            output_format=resolved_granular_format,
        )
        if granular_output_path is not None
        else None
    )
    target_scorers = {
        head: TargetScorer(
            head,
            criterion=criterion,
            device=device,
            horizon=horizon,
            compile_step=compile_metrics,
            region_set_members=members,
            latent_information=latent_information_metrics,
            granular_writer=granular_writer,
            temporal_coords=temporal_coords,
            region_ids=region_ids,
            region_labels=region_labels,
        )
        for head in CANONICAL_TARGET_NAMES
    }

    model_was_training = model.training
    model.eval()
    # A torch.compile wrapper exposes the module's methods, but forward_batch
    # would then run the eager forward. Prepare inputs on the module and call
    # the wrapper so evaluation uses the compiled forward.
    forward_model = cast(EpiForecaster, getattr(model, "_orig_mod", model))
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
                forward_inputs = forward_model.forward_inputs(
                    batch_data=batch_data,
                    region_embeddings=region_embeddings,
                    mask_cases=criterion.mask_input_cases,
                    mask_ww=criterion.mask_input_ww,
                    mask_hosp=criterion.mask_input_hosp,
                    mask_deaths=criterion.mask_input_deaths,
                )
                model_outputs = model(**forward_inputs)
                targets_dict = forward_model.batch_targets(batch_data)
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )

                current_stage = "loss_components"
                supervision = criterion.compute_observation_supervision(
                    targets_dict,
                    device=device,
                    batch_data=batch_data,
                )
                components = criterion.compute_components(
                    model_outputs,
                    targets_dict,
                    batch_data,
                    observation_supervision=supervision,
                )
                loss_ledger.update(components)
                latent_metrics.update(
                    model_outputs=model_outputs,
                    targets=targets_dict,
                )
                # Plausibility indicators use every window, with or without latent
                # targets, so they are valid on real data.
                ascertainment_inputs = {}
                for head_name, flow_key in _ASCERTAINMENT_LATENT_FLOWS.items():
                    head_target = supervision[head_name]["target"]
                    head_weights = supervision[head_name]["weights"]
                    if head_target is None or head_weights is None:
                        continue
                    ascertainment_inputs[head_name] = (
                        drop_nowcast(
                            model_outputs[flow_key], int(head_target.shape[1])
                        ),
                        head_target,
                        head_weights,
                    )
                latent_plausibility_metrics.update(
                    model_outputs=model_outputs,
                    ascertainment=ascertainment_inputs,
                )
                del ascertainment_inputs
                for protocol, (
                    diagnostic_predictions,
                    diagnostic_targets,
                ) in build_latent_diagnostic_protocols(
                    model=forward_model,
                    model_outputs=model_outputs,
                    targets=targets_dict,
                ).items():
                    latent_diagnostic_metrics[protocol].update(
                        predictions=diagnostic_predictions,
                        targets=diagnostic_targets,
                    )
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

                for head, scorer in target_scorers.items():
                    forecast = build_target_forecast(
                        head,
                        model_outputs=model_outputs,
                        targets=targets_dict,
                        supervision=supervision[head],
                        criterion=criterion,
                        latent_source=resolved_relevant_sources[head],
                        horizon=horizon,
                    )
                    if forecast is not None:
                        scorer.update(forecast, batch_data=batch_data)
                    del forecast
                current_stage = "granular_write_complete"
                if should_log_batch:
                    logger.info(
                        "[eval][progress] split=%s batch=%d/%d stage=%s",
                        split_name,
                        batch_idx,
                        num_batches,
                        current_stage,
                    )
                del components, supervision, model_outputs, targets_dict, batch_data

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

    mean_loss = loss_ledger.mean("total")
    target_scores = {head: scorer.finalize() for head, scorer in target_scorers.items()}

    per_head_node_metrics = {
        CANONICAL_TARGET_NAMES[head]: score.node_stats
        for head, score in target_scores.items()
        if score.node_stats is not None
    }
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
        per_head_metadata["relevant_source_mapping"] = resolved_relevant_sources
        write_metadata_sidecar(
            per_head_node_metrics_csv_path,
            per_head_metadata,
            schema_version=PER_HEAD_NODE_METRICS_SCHEMA_VERSION,
        )

    if granular_output_path is not None:
        metadata = {
            "dataset_path": getattr(dataset, "aligned_data_path", None),
            "granular_csv_path": granular_output_path,
            "granular_eval_path": granular_output_path,
            "granular_eval_format": resolved_granular_format,
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
        metadata["relevant_source_mapping"] = resolved_relevant_sources
        write_granular_metadata_sidecar(
            granular_output_path,
            metadata,
        )

    joint_obs_loss_total = loss_ledger.mean(
        *(f"{name}_weighted" for name in _OBSERVATION_LOSS_COMPONENTS)
    )
    metrics: dict[str, Any] = {
        # Generic primary metric: aliased joint observation loss
        "mae": joint_obs_loss_total,
        "joint_obs_loss_total": joint_obs_loss_total,
        # Joint loss components (averaged per batch, same reduction as mean_loss)
        **{
            key: loss_ledger.mean(component)
            for key, component in _LOSS_METRIC_COMPONENTS.items()
        },
        **{
            f"loss_latent_{task}_total_weighted": loss_ledger.mean(
                f"latent_{task}_weighted", factor=criterion.latent_total_weight
            )
            for task in _LATENT_LOSS_TASKS
        },
    }
    region_set_metrics: dict[str, dict[str, Any]] = {name: {} for name in members}
    for score in target_scores.values():
        metrics.update(score.metrics)
        for name, set_metrics in score.region_set_metrics.items():
            region_set_metrics[name].update(set_metrics)
    _add_region_set_metrics(metrics=metrics, region_set_metrics=region_set_metrics)
    metrics.update(latent_metrics.finalize())
    metrics.update(latent_plausibility_metrics.finalize())
    if latent_information_metrics is not None:
        metrics.update(latent_information_metrics.finalize())
    for accumulator in latent_diagnostic_metrics.values():
        metrics.update(accumulator.finalize())
    _add_delay_kernel_metrics(metrics=metrics, model=model)

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
    granular_output_path: Path | None = None,
    granular_eval_format: str | None = None,
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
        granular_output_path: Optional path to save granular per-example error rows
        granular_eval_format: Optional output format override ("parquet" or "csv")
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
    explicit_granular_output_path = granular_output_path is not None

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
        "[eval] Building %s loader from dataset=%s resolved_device=%s batch_size=%s "
        "num_workers=%s prefetch_factor=%s pin_memory=%s",
        split,
        config.data.dataset_path,
        resolved_device,
        batch_size if batch_size is not None else config.training.eval_batch_size,
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

    resolved_granular_format = granular_eval_format
    if granular_output_path is None and config.output.write_granular_eval:
        granular_output_path = (
            checkpoint_path.parent.parent
            / config.output.resolve_granular_eval_filename(split=split)
        )
    if resolved_granular_format is None and not explicit_granular_output_path:
        resolved_granular_format = getattr(config.output, "granular_eval_format", None)

    criterion = get_loss_from_config(
        config.training.loss,
        latent_scales=latent_scales_from_checkpoint(checkpoint),
        data_config=config.data,
        observation_heads_config=config.model.observation_heads,
        input_window_length=config.model.input_window_length,
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
            granular_output_path=granular_output_path,
            granular_eval_format=resolved_granular_format,
            node_metrics_target=node_metrics_target,
            log_every=log_every,
            latent_information=True,
            region_sets=(
                load_manifest_region_sets(Path(config.training.crossval_manifest_path))
                if split == "test" and config.training.crossval_manifest_path
                else None
            ),
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
            relevant_source_mapping=(
                config.model.observation_heads.relevant_source_mapping()
            ),
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
