"""
Subgraph Forecaster Trainer

K=1 subgraph-only trainer for inductive GraphSAGE forecasting.
Handles windowing via TemporalSubgraphLoader, target extraction for target nodes,
AMP, early stopping, LR scheduling, metrics, and checkpointing.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# -----------------------
# Configuration and Types
# -----------------------


@dataclass
class SubgraphTrainerConfig:
    # Training
    epochs: int = 50
    patience: int = 20
    grad_clip: float | None = 1.0
    amp: bool = False

    # Temporal windowing
    sequence_length: int = 1
    forecast_horizon: int = 7

    # Optimizer/Scheduler
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    plateau_factor: float = 0.7
    plateau_patience: int = 10

    # Misc
    seed: int | None = 42
    save_best: bool = True
    save_last: bool = True


# target_extractor(target_subgraphs, target_nodes) -> [num_target_nodes, horizon]
TargetExtractor = Callable[[Sequence[Data], torch.Tensor], torch.Tensor]


# -----------------------
# Utilities
# -----------------------


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_target_extractor(
    target_subgraphs: Sequence[Data], target_nodes: torch.Tensor
) -> torch.Tensor:
    """
    Extract targets for target nodes from subgraph target sequence.

    ASSUMES target nodes are the first `len(target_nodes)` nodes in each subgraph,
    which is a convention of PyG's NeighborLoader. For a safer but slower version
    that verifies node indices, see `subgraph_aware_target_extractor`.

    Returns shape [num_target_nodes, horizon]
    """
    num_targets = len(target_nodes)
    targets: list[torch.Tensor] = []
    for tg in target_subgraphs:
        # Take the label from last feature if available; fallback to first
        if tg.x.shape[1] > 3:
            targets.append(tg.x[:num_targets, -1])
        else:
            targets.append(tg.x[:num_targets, 0])
    return torch.stack(targets, dim=1)


def cases_target_extractor(
    target_sequences: Sequence[Data],
    original_target_nodes: torch.Tensor,
    cases_tensor: Optional[torch.Tensor] = None,
    time_indices: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    Extract COVID case targets for specific target nodes from cases tensor.

    This extractor uses actual COVID case data instead of mobility features,
    providing the correct epidemiological forecasting targets.

    Args:
        target_sequences: A sequence of original temporal graph `Data` objects.
            Each Data.x has shape [num_nodes, feature_dim]
        original_target_nodes: A tensor containing the node indices of the target
            nodes in the original graph. Shape: [num_target_nodes]
        cases_tensor: Aligned cases tensor of shape [num_nodes, num_timepoints]
        time_indices: List of time indices corresponding to target_sequences

    Returns:
        A tensor of target values of shape `[num_targets, horizon]`.
        Where num_targets = len(original_target_nodes) and horizon = len(target_sequences)
    """
    if cases_tensor is None:
        raise ValueError("cases_tensor must be provided for cases_target_extractor")

    if time_indices is None:
        # Default to consecutive time indices
        time_indices = list(range(len(target_sequences)))

    num_targets = len(original_target_nodes)
    horizon = len(target_sequences)

    # Validate input shapes
    assert cases_tensor.ndim == 2, f"Expected 2D cases tensor, got {cases_tensor.shape}"
    assert cases_tensor.shape[0] > original_target_nodes.max().item(), (
        "Target node indices out of bounds for cases tensor"
    )
    assert len(time_indices) == horizon, (
        f"Time indices length {len(time_indices)} != horizon {horizon}"
    )

    # Extract cases for target nodes and time indices
    # shape: [num_targets, horizon]
    targets = cases_tensor[original_target_nodes][:, time_indices]

    return targets


def direct_target_extractor(
    target_sequences: Sequence[Data], original_target_nodes: torch.Tensor
) -> torch.Tensor:
    """
    Extract targets for specific target nodes directly from original temporal graphs.

    This extractor directly indexes into the original graph features using the
    target node indices, avoiding the complexity and potential errors of subgraph
    extraction for target values.

    Args:
        target_sequences: A sequence of original temporal graph `Data` objects.
            Each Data.x has shape [num_nodes, feature_dim]
        original_target_nodes: A tensor containing the node indices of the target
            nodes in the original graph. Shape: [num_target_nodes]

    Returns:
        A tensor of target values of shape `[num_targets, horizon]`.
        Where num_targets = len(original_target_nodes) and horizon = len(target_sequences)
    """
    num_targets = len(original_target_nodes)
    horizon = len(target_sequences)
    targets: list[torch.Tensor] = []

    for target_graph in target_sequences:
        # Validate input shapes
        assert target_graph.x.ndim == 2, (
            f"Expected 2D node features, got {target_graph.x.shape}"
        )
        assert original_target_nodes.max() < target_graph.x.shape[0], (
            "Target node indices out of bounds"
        )

        # Directly extract features for target nodes from original graph
        # No need for complex n_id mapping since we're using the original graph

        # Take the label from last feature if available; fallback to first
        if target_graph.x.shape[1] > 3:
            # shape: [num_targets, 1]
            graph_targets = target_graph.x[original_target_nodes, -1]
        else:
            # shape: [num_targets, 1]
            graph_targets = target_graph.x[original_target_nodes, 0]

        targets.append(graph_targets)

    # Stack along horizon dimension: [num_targets, horizon]
    return torch.stack(targets, dim=1)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Compute scalar metrics over all target nodes and horizons.

    Args:
        pred: Predicted values. Shape: [num_predictions, forecast_horizon]
        target: Ground truth values. Shape: [num_predictions, forecast_horizon]

    Returns:
        Dictionary of metric names and their values.
    """
    # Validate input shapes
    assert pred.shape == target.shape, (
        f"Shape mismatch: pred {pred.shape}, target {target.shape}"
    )
    assert pred.ndim == 2, f"Expected 2D predictions, got {pred.shape}"

    num_predictions, forecast_horizon = pred.shape

    metrics: dict[str, float] = {}

    # Flatten for overall metrics (treat all predictions and horizons equally)
    pred_flat = rearrange(pred, "n h -> (n h)")  # [num_predictions * forecast_horizon]
    target_flat = rearrange(
        target, "n h -> (n h)"
    )  # [num_predictions * forecast_horizon]

    mse = F.mse_loss(pred_flat, target_flat)
    mae = F.l1_loss(pred_flat, target_flat)
    rmse = torch.sqrt(mse)
    epsilon = 1e-8
    mape = (
        torch.mean(torch.abs((target_flat - pred_flat) / (target_flat + epsilon)))
        * 100.0
    )

    # R² calculation
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    metrics["mse"] = float(mse.item())
    metrics["mae"] = float(mae.item())
    metrics["rmse"] = float(rmse.item())
    metrics["mape"] = float(mape.item())
    metrics["r2"] = float(r2.item())

    # Add per-horizon metrics for detailed analysis
    for h in range(forecast_horizon):
        h_pred, h_target = pred[:, h], target[:, h]
        metrics[f"mse_h{h + 1}"] = float(F.mse_loss(h_pred, h_target).item())
        metrics[f"mae_h{h + 1}"] = float(F.l1_loss(h_pred, h_target).item())

    return metrics


# -----------------------
# Subgraph Trainer
# -----------------------


class ForecasterSubgraphTrainer:
    """
    Trainer for inductive GraphSAGE-style forecasters.

    Model interface expectation:
      outputs = model.forward_subgraph(
          mobility_sequence=[subgraph_t0, ..., subgraph_t{seq_len-1}],
          target_node_indices=target_nodes,
          edar_sequence=None,
          edar_muni_mask=optional_mask
      )
      outputs must include "case_count_forecast": [num_target_nodes, horizon]
    """

    def __init__(
        self,
        model: nn.Module,
        config: SubgraphTrainerConfig,
        device: str | torch.device = "auto",
        output_dir: str | None = None,
        optimizer: Optimizer | None = None,
        scheduler: ReduceLROnPlateau | None = None,
        target_extractor: TargetExtractor = default_target_extractor,
        edar_muni_mask: torch.Tensor | None = None,
        edar_biomarker_loader=None,
        logger: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.cfg = config
        self.device = self._resolve_device(device)
        self.output_dir = Path(output_dir) if output_dir else None
        self.target_extractor = target_extractor
        self.edar_muni_mask = edar_muni_mask
        self.edar_biomarker_loader = edar_biomarker_loader
        self.log = logger if logger is not None else print

        set_seed(self.cfg.seed)
        self.model.to(self.device)

        self.optimizer = optimizer or Adam(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = scheduler or ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.plateau_factor,
            patience=self.cfg.plateau_patience,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / "trainer_config.json", "w") as f:
                json.dump(asdict(self.cfg), f, indent=2)

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ---- Public API

    def fit(
        self,
        train_graphs: Sequence[Data],
        val_graphs: Sequence[Data],
        subgraph_loader,
    ) -> dict:
        """
        Train using k=1 subgraph batches produced by subgraph_loader.

        Args:
            train_graphs: Sequence of temporal graph Data objects for training.
                Each Data has x [num_nodes, feature_dim], edge_index [2, num_edges]
            val_graphs: Sequence of temporal graph Data objects for validation.
            subgraph_loader: TemporalSubgraphLoader for creating k=1 subgraph batches

        Returns:
            Training history dictionary with epochs, losses, learning rates, etc.
        """
        best_val = float("inf")
        patience = 0
        history = {
            "epochs": [],
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
            "best_epoch": None,
            "early_stopping_epoch": None,
        }

        best_path = self.output_dir / "best_model.pt" if self.output_dir else None
        last_path = self.output_dir / "last_model.pt" if self.output_dir else None

        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_loss = self._epoch_train_subgraph(train_graphs, subgraph_loader)

            self.model.eval()
            with torch.no_grad():
                val_loss = self._epoch_val_subgraph(val_graphs, subgraph_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            history["epochs"].append(epoch)
            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["learning_rates"].append(current_lr)

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                patience = 0
                history["best_epoch"] = epoch
                if self.cfg.save_best and best_path:
                    torch.save(self.model.state_dict(), best_path)
            else:
                patience += 1

            self.log(
                f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}"
            )

            if patience >= self.cfg.patience:
                self.log(f"Early stopping at epoch {epoch}")
                history["early_stopping_epoch"] = epoch
                break

        if self.cfg.save_last and last_path:
            torch.save(self.model.state_dict(), last_path)

        if self.output_dir:
            with open(self.output_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)

        return history

    def evaluate(
        self,
        graphs: Sequence[Data],
        subgraph_loader,
    ) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
        """
        Evaluate on subgraph batches across provided graphs.

        Args:
            graphs: Sequence of temporal graph Data objects for evaluation
            subgraph_loader: TemporalSubgraphLoader for creating k=1 subgraph batches

        Returns:
            Tuple of (metrics_dict, predictions_array, targets_array)
            - metrics: Dictionary of evaluation metrics
            - predictions: numpy array of shape [total_predictions, forecast_horizon]
            - targets: numpy array of shape [total_predictions, forecast_horizon]
        """
        self.model.eval()
        preds, tgts = [], []
        seq_len, horizon = self.cfg.sequence_length, self.cfg.forecast_horizon

        with torch.no_grad():
            batch_iter = subgraph_loader.create_temporal_batches(
                graphs,
                sequence_length=seq_len,
                forecast_horizon=horizon,
            )
            time_idx = 0
            for input_subgraphs, target_sequences, target_nodes in batch_iter:
                if len(input_subgraphs) == 0:
                    continue

                # Move to device
                input_subgraphs = [g.to(self.device) for g in input_subgraphs]
                target_sequences = [g.to(self.device) for g in target_sequences]
                target_nodes = target_nodes.to(self.device)

                # Extract targets: [num_target_nodes, horizon]
                target = self.target_extractor(target_sequences, target_nodes).to(
                    self.device
                )

                # Forward pass
                outputs = self._forward_subgraph(
                    input_subgraphs, target_nodes, time_idx
                )
                pred = outputs["case_count_forecast"]  # [num_target_nodes, horizon]

                # Validate output shapes
                assert pred.shape == target.shape, (
                    f"Prediction shape mismatch: {pred.shape} vs {target.shape}"
                )

                preds.append(pred.cpu())
                tgts.append(target.cpu())
                time_idx += 1

        if len(preds) == 0:
            return {}, None, None

        # Concatenate all batches
        pred_all = torch.cat(preds, dim=0)  # [total_predictions, horizon]
        tgt_all = torch.cat(tgts, dim=0)  # [total_predictions, horizon]

        metrics = compute_metrics(pred_all, tgt_all)
        return metrics, pred_all.numpy(), tgt_all.numpy()

    # ---- Internals

    def _epoch_train_subgraph(self, graphs: Sequence[Data], subgraph_loader) -> float:
        """Train for one epoch using subgraph batches.

        Args:
            graphs: Sequence of temporal graph Data objects
            subgraph_loader: TemporalSubgraphLoader for creating batches

        Returns:
            Average loss across all batches
        """
        total, count = 0.0, 0
        seq_len, horizon = self.cfg.sequence_length, self.cfg.forecast_horizon

        batch_iter = subgraph_loader.create_temporal_batches(
            graphs,
            sequence_length=seq_len,
            forecast_horizon=horizon,
        )
        time_idx = 0
        for input_subgraphs, target_sequences, target_nodes in batch_iter:
            if len(input_subgraphs) == 0:
                continue

            # Debug: Check target_sequences type before moving to device
            logger.debug(f"Target sequences length: {len(target_sequences)}")
            if len(target_sequences) > 0:
                logger.debug(f"Target sequence type: {type(target_sequences[0])}")
                logger.debug(f"Target sequence[0] type: {type(target_sequences[0])}")
                if isinstance(target_sequences[0], list):
                    logger.debug(
                        f"Target sequence[0] is a list with length: {len(target_sequences[0])}"
                    )
                    if len(target_sequences[0]) > 0:
                        logger.debug(
                            f"Target sequence[0][0] type: {type(target_sequences[0][0])}"
                        )
                else:
                    logger.debug(f"Target sequence[0]: {target_sequences[0]}")

            # Handle the case where target_sequences is list[list[Data]]
            # We need to flatten it to list[Data] for the target extractor
            if len(target_sequences) > 0 and isinstance(target_sequences[0], list):
                # This is the case where each target_sequence is a list of horizon steps
                # We need to take the first element as the representative target sequence
                logger.debug("Flattening target_sequences structure")
                target_sequences = target_sequences[
                    0
                ]  # Take first window's target sequence

            # Move to device
            input_subgraphs = [g.to(self.device) for g in input_subgraphs]
            target_sequences = [g.to(self.device) for g in target_sequences]
            target_nodes = target_nodes.to(self.device)

            # Extract targets: [num_target_nodes, horizon]
            target = self.target_extractor(target_sequences, target_nodes).to(
                self.device
            )

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                outputs = self._forward_subgraph(
                    input_subgraphs, target_nodes, time_idx
                )
                pred = outputs["case_count_forecast"]  # [num_target_nodes, horizon]

                # Validate prediction shape
                assert pred.shape == target.shape, (
                    f"Prediction shape mismatch: {pred.shape} vs {target.shape}"
                )

                loss = F.mse_loss(pred, target)

            self.scaler.scale(loss).backward()

            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += float(loss.item())
            count += 1
            time_idx += 1

        return total / max(1, count)

    def _epoch_val_subgraph(self, graphs: Sequence[Data], subgraph_loader) -> float:
        total, count = 0.0, 0
        seq_len, horizon = self.cfg.sequence_length, self.cfg.forecast_horizon

        batch_iter = subgraph_loader.create_temporal_batches(
            graphs,
            sequence_length=seq_len,
            forecast_horizon=horizon,
        )
        time_idx = 0
        for input_subgraphs, target_sequences, target_nodes in batch_iter:
            if len(input_subgraphs) == 0:
                continue

            input_subgraphs = [g.to(self.device) for g in input_subgraphs]
            target_sequences = [g.to(self.device) for g in target_sequences]
            target_nodes = target_nodes.to(self.device)

            target = self.target_extractor(target_sequences, target_nodes).to(
                self.device
            )

            outputs = self._forward_subgraph(input_subgraphs, target_nodes, time_idx)
            pred = outputs["case_count_forecast"]
            loss = F.mse_loss(pred, target)

            total += float(loss.item())
            count += 1
            time_idx += 1

        return total / max(1, count)

    def _forward_subgraph(
        self,
        mobility_sequence: list[Data],
        target_node_indices: torch.Tensor,
        time_index: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model with proper tensor shape handling.

        Args:
            mobility_sequence: List of subgraph Data objects for input sequence.
                Each Data has x [num_subgraph_nodes, feature_dim],
                edge_index [2, num_subgraph_edges]
            target_node_indices: Indices of target nodes within subgraphs.
                Shape: [num_target_nodes]
            time_index: Time index for temporal alignment with EDAR data

        Returns:
            Dictionary containing model outputs, must include 'case_count_forecast'
            with shape [num_target_nodes, forecast_horizon]
        """
        seq_len = len(mobility_sequence)
        num_target_nodes = len(target_node_indices)

        # DualGraphForecaster with EDAR support
        if hasattr(self.model, "dual_graph_encoder"):
            # Create EDAR subgraph sequence if biomarker loader is available
            edar_subgraph_sequence = None
            if self.edar_biomarker_loader is not None:
                edar_subgraph_sequence = []
                for _ in range(seq_len):
                    # Use time index for temporal alignment
                    actual_time_idx = min(
                        time_index, len(self.edar_biomarker_loader.time_index) - 1
                    )
                    edar_graph = self.edar_biomarker_loader.create_edar_graph(
                        actual_time_idx
                    )
                    edar_subgraph_sequence.append(edar_graph.to(self.device))

            # Prepare EDAR mask if available
            edar_muni_mask_device = (
                self.edar_muni_mask.to(self.device)
                if self.edar_muni_mask is not None
                else None
            )

            outputs = self.model.forward_subgraph(
                mobility_sequence=mobility_sequence,
                target_node_indices=target_node_indices,
                edar_sequence=edar_subgraph_sequence,
                edar_muni_mask=edar_muni_mask_device,
            )

        # SimpleDualGraphForecaster (mobility only)
        elif hasattr(self.model, "mobility_encoder"):
            logger.debug(
                f"Calling forward_subgraph with mobility_sequence length: {len(mobility_sequence)}"
            )
            logger.debug(f"Expected sequence_length: {self.cfg.sequence_length}")
            outputs = self.model.forward_subgraph(
                mobility_sequence=mobility_sequence,
                target_node_indices=target_node_indices,
            )
        else:
            raise ValueError("Model does not support subgraph training interface.")

        # Validate output shape
        if "case_count_forecast" not in outputs:
            raise ValueError("Model output must include 'case_count_forecast' key")

        case_forecast = outputs["case_count_forecast"]
        expected_shape = (num_target_nodes, self.cfg.forecast_horizon)

        if case_forecast.shape != expected_shape:
            raise ValueError(
                f"Case count forecast shape mismatch: expected {expected_shape}, "
                f"got {case_forecast.shape}"
            )

        return outputs
