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
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
import zarr.errors

from data.collate import collate_epidataset_batch
from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD
from utils.normalization import unscale_forecasts
from models.configs import EpiForecasterConfig, LossConfig
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


class UnscaledMSELoss(ForecastLoss):
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
        diff = pred_unscaled - targets_unscaled
        return (diff**2).mean()


class CompositeLoss(ForecastLoss):
    def __init__(self, components: list[tuple[ForecastLoss, float]]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _weight in components])
        self.loss_fns: list[ForecastLoss] = [loss for loss, _weight in components]
        self.weights = [float(weight) for _loss, weight in components]

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        total = predictions.new_zeros(())
        for loss_fn, weight in zip(self.loss_fns, self.weights, strict=False):
            if weight == 0:
                continue
            total = total + weight * loss_fn.forward(
                predictions, targets, target_mean, target_scale
            )
        return total


class JointInferenceLoss(nn.Module):
    """
    Joint inference loss combining wastewater, hospitalization, and SIR physics losses.

    This loss is designed for the joint inference framework where the model outputs
    latent SIR states and observation predictions rather than direct forecasts.
    """

    def __init__(
        self,
        w_ww: float = 1.0,
        w_hosp: float = 1.0,
        w_cases: float = 1.0,
        w_deaths: float = 1.0,
        w_sir: float = 0.1,
    ):
        super().__init__()
        self.w_ww = w_ww
        self.w_hosp = w_hosp
        self.w_cases = w_cases
        self.w_deaths = w_deaths
        self.w_sir = w_sir

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Always exclude non-finite targets from supervision to avoid NaN loss/gradients.
        finite_mask = torch.isfinite(target).to(prediction.dtype)
        if mask is None:
            mask_f = finite_mask
        else:
            mask_f = mask.to(prediction.dtype) * finite_mask

        denom = mask_f.sum()
        if denom.item() <= 0:
            # Keep graph connectivity for all-masked batches.
            return prediction.sum() * 0.0

        target_clean = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        sq = (prediction - target_clean) ** 2
        return (sq * mask_f).sum() / denom

    def forward(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        components = self.compute_components(model_outputs, targets)
        return components["total"]

    def compute_components(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        """
        Compute joint inference loss components.

        Args:
            model_outputs: Dict from EpiForecaster.forward() containing:
                - pred_ww: [B, H] predicted wastewater
                - pred_hosp: [B, H] predicted hospitalizations
                - pred_cases: [B, H] predicted reported cases
                - pred_deaths: [B, H] predicted deaths
                - physics_residual: [B, H] SIR dynamics residual
            targets: Dict containing target tensors:
                - ww: [B, H] wastewater targets (optional)
                - hosp: [B, H] hospitalization targets (optional)
                - cases: [B, H] reported cases targets (optional)
                - deaths: [B, H] deaths targets (optional)

        Returns:
            Dict with unweighted and weighted component losses plus total:
                - ww, hosp, cases, deaths, sir
                - ww_weighted, hosp_weighted, cases_weighted, deaths_weighted, sir_weighted
                - total
        """
        # Keep total loss attached to model graph, even if all components are masked out.
        total_loss = model_outputs["pred_ww"].sum() * 0.0
        ww_loss = model_outputs["pred_ww"].sum() * 0.0
        hosp_loss = model_outputs["pred_ww"].sum() * 0.0
        cases_loss = model_outputs["pred_ww"].sum() * 0.0
        deaths_loss = model_outputs["pred_ww"].sum() * 0.0
        sir_loss = model_outputs["pred_ww"].sum() * 0.0

        # Wastewater loss
        if self.w_ww > 0 and targets.get("ww") is not None:
            ww_loss = self._masked_mse(
                model_outputs["pred_ww"], targets["ww"], targets.get("ww_mask")
            )
            total_loss = total_loss + self.w_ww * ww_loss

        # Hospitalization loss
        if self.w_hosp > 0 and targets.get("hosp") is not None:
            hosp_loss = self._masked_mse(
                model_outputs["pred_hosp"], targets["hosp"], targets.get("hosp_mask")
            )
            total_loss = total_loss + self.w_hosp * hosp_loss

        # Cases loss (reported cases observation)
        if targets.get("cases") is not None:
            cases_loss = self._masked_mse(
                model_outputs["pred_cases"], targets["cases"], targets.get("cases_mask")
            )
            if self.w_cases > 0:
                total_loss = total_loss + self.w_cases * cases_loss

        # Deaths loss (mortality observation)
        if targets.get("deaths") is not None:
            deaths_loss = self._masked_mse(
                model_outputs["pred_deaths"],
                targets["deaths"],
                targets.get("deaths_mask"),
            )
            if self.w_deaths > 0:
                total_loss = total_loss + self.w_deaths * deaths_loss

        # SIR physics loss (always computed from residual)
        if self.w_sir > 0:
            physics_residual = model_outputs["physics_residual"]
            ww_mask = targets.get("ww_mask")
            hosp_mask = targets.get("hosp_mask")
            cases_mask = targets.get("cases_mask")
            deaths_mask = targets.get("deaths_mask")

            combined_mask: torch.Tensor | None = None
            masks = [
                m
                for m in [ww_mask, hosp_mask, cases_mask, deaths_mask]
                if m is not None
            ]
            if masks:
                combined_mask = masks[0]
                for m in masks[1:]:
                    combined_mask = torch.maximum(combined_mask, m)

            if combined_mask is None:
                sir_loss = physics_residual.mean()
            else:
                sir_loss = self._masked_mse(
                    physics_residual,
                    torch.zeros_like(physics_residual),
                    combined_mask,
                )
            total_loss = total_loss + self.w_sir * sir_loss

        ww_weighted = self.w_ww * ww_loss
        hosp_weighted = self.w_hosp * hosp_loss
        cases_weighted = self.w_cases * cases_loss
        deaths_weighted = self.w_deaths * deaths_loss
        sir_weighted = self.w_sir * sir_loss

        return {
            "ww": ww_loss,
            "hosp": hosp_loss,
            "cases": cases_loss,
            "deaths": deaths_loss,
            "sir": sir_loss,
            "ww_weighted": ww_weighted,
            "hosp_weighted": hosp_weighted,
            "cases_weighted": cases_weighted,
            "deaths_weighted": deaths_weighted,
            "sir_weighted": sir_weighted,
            "total": total_loss,
        }


def get_loss_function(name: str) -> ForecastLoss:
    name_lower = name.lower()
    if name_lower == "mse":
        return WrappedTorchLoss(nn.MSELoss())
    elif name_lower == "mse_unscaled":
        return UnscaledMSELoss()
    elif name_lower in ("mae", "l1"):
        return WrappedTorchLoss(nn.L1Loss())
    elif name_lower == "smape":
        return SMAPELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_loss_from_config(
    loss_config: LossConfig | None,
) -> ForecastLoss | JointInferenceLoss:
    if loss_config is None:
        return get_loss_function("smape")
    name_lower = loss_config.name.lower()
    if name_lower == "joint_inference":
        # Joint inference loss for SIR + observation heads
        joint_cfg = loss_config.joint
        return JointInferenceLoss(
            w_ww=joint_cfg.w_ww,
            w_hosp=joint_cfg.w_hosp,
            w_cases=joint_cfg.w_cases,
            w_deaths=joint_cfg.w_deaths,
            w_sir=joint_cfg.w_sir,
        )
    if name_lower == "composite":
        if not loss_config.components:
            raise ValueError("Composite loss requires components")
        components: list[tuple[ForecastLoss, float]] = []
        for component in loss_config.components:
            components.append((get_loss_function(component.name), component.weight))
        return CompositeLoss(components)
    return get_loss_function(loss_config.name)


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
) -> tuple[EpiForecaster, EpiForecasterConfig, dict[str, Any]]:
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
        sir_physics=config.model.sir_physics,
        observation_heads=config.model.observation_heads,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolve_device(device))
    return model, config, checkpoint


def split_nodes(config: EpiForecasterConfig) -> tuple[list[int], list[int], list[int]]:
    """Match the node holdout split logic used during training."""
    train_split = 1 - config.training.val_split - config.training.test_split
    if not config.data.run_id:
        raise ValueError("run_id must be specified in config")
    aligned_dataset = EpiDataset.load_canonical_dataset(
        Path(config.data.dataset_path),
        run_id=config.data.run_id,
        run_id_chunk_size=config.data.run_id_chunk_size,
    )
    N = aligned_dataset[REGION_COORD].size
    all_nodes = np.arange(N)
    if config.data.use_valid_targets:
        valid_mask = EpiDataset.get_valid_nodes(
            dataset_path=Path(config.data.dataset_path),
            run_id=config.data.run_id,
        )
        all_nodes = all_nodes[valid_mask]
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
    split_key = split.lower()
    if split_key not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    if config.training.split_strategy == "time":
        train_end: str = config.training.train_end_date or ""
        val_end: str = config.training.val_end_date or ""
        test_end: str | None = config.training.test_end_date
        _train_dataset, val_dataset, test_dataset = EpiDataset.create_temporal_splits(
            config=config,
            train_end_date=train_end,
            val_end_date=val_end,
            test_end_date=test_end,
        )
        dataset = val_dataset if split_key == "val" else test_dataset
    else:
        train_nodes, val_nodes, test_nodes = split_nodes(config)
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
            collate_epidataset_batch,
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
                target_nodes = batch["TargetNode"]
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
        criterion = get_loss_from_config(config.training.loss)
        if not isinstance(criterion, JointInferenceLoss):
            raise ValueError(
                "Evaluation now requires JointInferenceLoss. "
                "Set training.loss.name=joint_inference in the config."
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
    hosp_mae_sum = torch.tensor(0.0, device=device)
    hosp_mse_sum = torch.tensor(0.0, device=device)
    hosp_smape_sum = torch.tensor(0.0, device=device)
    hosp_total_count = 0
    hosp_target_mean_acc = torch.tensor(0.0, device=device)
    hosp_target_m2 = torch.tensor(0.0, device=device)

    per_h_mae_sum = torch.zeros(horizon, device=device)
    per_h_mse_sum = torch.zeros(horizon, device=device)
    per_h_count_sum = torch.zeros(horizon, device=device)

    ww_mae_sum = torch.tensor(0.0, device=device)
    ww_mse_sum = torch.tensor(0.0, device=device)
    ww_smape_sum = torch.tensor(0.0, device=device)
    ww_total_count = 0
    ww_target_mean_acc = torch.tensor(0.0, device=device)
    ww_target_m2 = torch.tensor(0.0, device=device)
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

    epsilon = 1e-6
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

                model_outputs, targets_dict = forward_model.forward_batch(
                    batch_data=batch_data,
                    region_embeddings=region_embeddings,
                )

                components = criterion.compute_components(model_outputs, targets_dict)
                loss = components["total"]
                total_loss += loss.detach()
                loss_ww_sum += components["ww"].detach()
                loss_hosp_sum += components["hosp"].detach()
                loss_cases_sum += components["cases"].detach()
                loss_deaths_sum += components["deaths"].detach()
                loss_sir_sum += components["sir"].detach()
                loss_ww_weighted_sum += components["ww_weighted"].detach()
                loss_hosp_weighted_sum += components["hosp_weighted"].detach()
                loss_cases_weighted_sum += components["cases_weighted"].detach()
                loss_deaths_weighted_sum += components["deaths_weighted"].detach()
                loss_sir_weighted_sum += components["sir_weighted"].detach()

                pred_hosp = model_outputs.get("pred_hosp")
                hosp_targets = targets_dict.get("hosp")
                hosp_mask = targets_dict.get("hosp_mask")
                if pred_hosp is not None and hosp_targets is not None:
                    if hosp_mask is None:
                        hosp_mask = torch.ones_like(hosp_targets)
                    mask = hosp_mask.to(pred_hosp.dtype)

                    # Clean targets for metric calculation
                    hosp_targets_clean = torch.nan_to_num(hosp_targets, nan=0.0)

                    diff = pred_hosp - hosp_targets_clean
                    abs_diff = diff.abs()
                    hosp_mae_sum += (abs_diff * mask).sum()
                    hosp_mse_sum += ((diff**2) * mask).sum()
                    hosp_smape_sum += (
                        2
                        * abs_diff
                        / (pred_hosp.abs() + hosp_targets_clean.abs() + epsilon)
                        * mask
                    ).sum()

                    # Per-node MAE - keep tensors on device until end
                    valid_per_sample = mask.sum(dim=1) > 0
                    per_sample_mae = (abs_diff * mask).sum(dim=1) / mask.sum(
                        dim=1
                    ).clamp_min(1.0)
                    target_nodes = batch_data["TargetNode"]
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

                    # Welford's algorithm for variance (device-local)
                    # Use cleaned targets masked by validity
                    flat_targets = (
                        hosp_targets_clean[mask > 0].detach().float().reshape(-1)
                    )
                    batch_count = flat_targets.numel()
                    if batch_count > 0:
                        batch_mean = flat_targets.mean()
                        batch_m2 = ((flat_targets - batch_mean) ** 2).sum()

                        delta = batch_mean - hosp_target_mean_acc
                        new_count = hosp_total_count + batch_count
                        hosp_target_mean_acc += delta * batch_count / new_count
                        hosp_target_m2 += (
                            batch_m2
                            + (delta**2) * (hosp_total_count * batch_count) / new_count
                        )
                        hosp_total_count = new_count

                    per_h_mae_sum += (abs_diff * mask).sum(dim=0)
                    per_h_mse_sum += ((diff**2) * mask).sum(dim=0)
                    per_h_count_sum += mask.sum(dim=0)

                pred_ww = model_outputs.get("pred_ww")
                ww_targets = targets_dict.get("ww")
                ww_mask = targets_dict.get("ww_mask")
                if pred_ww is not None and ww_targets is not None:
                    if ww_mask is None:
                        ww_mask = torch.ones_like(ww_targets)
                    mask_ww = ww_mask.to(pred_ww.dtype)

                    # Clean targets for metric calculation
                    ww_targets_clean = torch.nan_to_num(ww_targets, nan=0.0)

                    diff_ww = pred_ww - ww_targets_clean
                    abs_diff_ww = diff_ww.abs()
                    ww_mae_sum += (abs_diff_ww * mask_ww).sum()
                    ww_mse_sum += ((diff_ww**2) * mask_ww).sum()
                    ww_smape_sum += (
                        2
                        * abs_diff_ww
                        / (pred_ww.abs() + ww_targets_clean.abs() + epsilon)
                        * mask_ww
                    ).sum()

                    flat_targets_ww = (
                        ww_targets_clean[mask_ww > 0].detach().float().reshape(-1)
                    )
                    batch_count_ww = flat_targets_ww.numel()
                    if batch_count_ww > 0:
                        batch_mean_ww = flat_targets_ww.mean()
                        batch_m2_ww = ((flat_targets_ww - batch_mean_ww) ** 2).sum()

                        delta_ww = batch_mean_ww - ww_target_mean_acc
                        new_count_ww = ww_total_count + batch_count_ww
                        ww_target_mean_acc += delta_ww * batch_count_ww / new_count_ww
                        ww_target_m2 += (
                            batch_m2_ww
                            + (delta_ww**2)
                            * (ww_total_count * batch_count_ww)
                            / new_count_ww
                        )
                        ww_total_count = new_count_ww

    finally:
        if model_was_training:
            model.train()

    # Final sync - transfer metrics to CPU once
    mean_loss = (total_loss / max(1, num_batches)).item()
    if hosp_total_count:
        mean_mae = (hosp_mae_sum / max(1, hosp_total_count)).item()
        mean_rmse = math.sqrt((hosp_mse_sum / max(1, hosp_total_count)).item())
        mean_smape = (hosp_smape_sum / max(1, hosp_total_count)).item()
        ss_res = hosp_mse_sum.item()
        ss_tot = hosp_target_m2.item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        mean_mae = float("nan")
        mean_rmse = float("nan")
        mean_smape = float("nan")
        r2 = float("nan")

    if ww_total_count:
        mean_mae_ww = (ww_mae_sum / max(1, ww_total_count)).item()
        mean_rmse_ww = math.sqrt((ww_mse_sum / max(1, ww_total_count)).item())
        mean_smape_ww = (ww_smape_sum / max(1, ww_total_count)).item()
        ss_res_ww = ww_mse_sum.item()
        ss_tot_ww = ww_target_m2.item()
        r2_ww = 1 - ss_res_ww / ss_tot_ww if ss_tot_ww > 0 else float("nan")
    else:
        mean_mae_ww = float("nan")
        mean_rmse_ww = float("nan")
        mean_smape_ww = float("nan")
        r2_ww = float("nan")

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

    if hosp_total_count:
        per_h_denom = per_h_count_sum.clamp_min(1.0)
        per_h_mae = (per_h_mae_sum / per_h_denom).tolist()
        per_h_rmse = (per_h_mse_sum / per_h_denom).sqrt().tolist()
    else:
        per_h_mae = []
        per_h_rmse = []

    metrics = {
        "mae": mean_mae,
        "rmse": mean_rmse,
        "smape": mean_smape,
        "r2": r2,
        "mae_per_h": per_h_mae,
        "rmse_per_h": per_h_rmse,
        # Hospitalization metrics in log1p(per-100k) space
        "mae_hosp_log1p_per_100k": mean_mae,
        "rmse_hosp_log1p_per_100k": mean_rmse,
        "smape_hosp_log1p_per_100k": mean_smape,
        "r2_hosp_log1p_per_100k": r2,
        # Wastewater metrics in log1p(per-100k) space
        "mae_ww_log1p_per_100k": mean_mae_ww,
        "rmse_ww_log1p_per_100k": mean_rmse_ww,
        "smape_ww_log1p_per_100k": mean_smape_ww,
        "r2_ww_log1p_per_100k": r2_ww,
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
        history_length=int(config.model.history_length),
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
            history_length=int(config.model.history_length),
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
        device: Device to use for evaluation
    log_dir: Optional W&B run directory for forecast plots
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
        config, split=split, device=device, batch_size=batch_size
    )
    dataset = cast(EpiDataset, loader.dataset)
    logger.info(f"[eval] {split} samples: {len(dataset)}")

    # Run evaluation - returns node_mae_dict as third value
    eval_loss = float("nan")
    eval_metrics: dict[str, Any] = {}
    node_mae_dict: dict[int, float] = {}
    try:
        criterion = get_loss_from_config(config.training.loss)
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
