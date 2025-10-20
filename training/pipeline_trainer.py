"""
Pipeline Trainer for Graph Neural Network Epidemiological Forecasting.

This module provides a high-level orchestration class that combines data management,
model creation, training, and evaluation into a single pipeline.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.subgraph_loader import TemporalSubgraphLoader
from models.dual_graph_forecaster import create_dual_graph_forecaster
from utils.plotting import generate_all_plots, print_results_table

from .data_manager import DataManager
from .forecaster_trainer import (
    ForecasterSubgraphTrainer,
    SubgraphTrainerConfig,
    cases_target_extractor,
)

logger = logging.getLogger(__name__)


def create_cases_target_extractor_wrapper(cases_tensor: torch.Tensor):
    """
    Create a wrapper function for cases target extraction with fixed cases tensor.

    Args:
        cases_tensor: Aligned cases tensor of shape [num_nodes, num_timepoints]

    Returns:
        Wrapper function compatible with ForecasterSubgraphTrainer
    """

    def cases_target_extractor_wrapper(target_sequences, original_target_nodes):
        """
        Wrapper function that extracts COVID case targets for specific time windows.

        Args:
            target_sequences: List of temporal graph Data objects (defines time window)
            original_target_nodes: Tensor of target node indices

        Returns:
            Cases tensor for the target nodes and time window
        """
        # Extract time indices from the target sequences
        # For now, assume consecutive time indices starting from 0
        # In a more sophisticated implementation, you'd extract actual time indices
        time_indices = list(range(len(target_sequences)))

        return cases_target_extractor(
            target_sequences=target_sequences,
            original_target_nodes=original_target_nodes,
            cases_tensor=cases_tensor,
            time_indices=time_indices,
        )

    return cases_target_extractor_wrapper


def log_model_summary(model, logger, model_name="Model"):
    """
    Log detailed model parameter summary with per-module breakdown.

    Particularly useful for ablation studies to understand parameter
    distribution across different model components.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"{model_name} Parameter Summary:")
    logger.info("+" + "-" * 60 + "+" + "-" * 15 + "+" + "-" * 10 + "+")
    logger.info(f"| {'Module':<58} | {'Parameters':<13} | {'%':<8} |")
    logger.info("+" + "-" * 60 + "+" + "-" * 15 + "+" + "-" * 10 + "+")

    # Collect module statistics
    module_stats = []

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                percentage = (module_params / total_params) * 100
                module_stats.append((name, module_params, percentage))

    # Sort by parameter count (descending)
    module_stats.sort(key=lambda x: x[1], reverse=True)

    # Log each module
    for name, params, percentage in module_stats:
        # Truncate long module names
        display_name = name if len(name) <= 58 else name[:55] + "..."
        logger.info(f"| {display_name:<58} | {params:>11,} | {percentage:>6.1f}% |")

    logger.info("+" + "-" * 60 + "+" + "-" * 15 + "+" + "-" * 10 + "+")
    logger.info(f"| {'TOTAL':<58} | {total_params:>11,} | {'100.0%':<8} |")
    logger.info(
        f"| {'Trainable':<58} | {trainable_params:>11,} | {(trainable_params / total_params) * 100:>6.1f}% |"
    )
    logger.info("+" + "-" * 60 + "+" + "-" * 15 + "+" + "-" * 10 + "+")


class PipelineTrainer:
    """
    High-level orchestration class for the complete training pipeline.

    This class combines DataManager and ForecasterSubgraphTrainer to provide
    a clean, single-entry-point interface for running the complete pipeline.
    """

    def __init__(self, args, timestamped_dir: str, latest_dir: str):
        self.args = args
        self.timestamped_dir = Path(timestamped_dir)
        self.latest_dir = Path(latest_dir)
        self.device = self._setup_device()

        # Initialize components
        self.data_manager = DataManager()
        self.dataset: Optional[dict[str, Any]] = None
        self.temporal_graphs = None
        self.model = None
        self.trainer: Optional[ForecasterSubgraphTrainer] = None

    @classmethod
    def from_args(cls, args) -> "PipelineTrainer":
        """
        Factory method to create PipelineTrainer from CLI arguments.

        Args:
            args: CLI arguments from parse_arguments()

        Returns:
            Configured PipelineTrainer instance
        """
        # Setup output directories
        timestamped_dir, latest_dir = cls._setup_output_directories(args.output_dir)
        return cls(args, timestamped_dir, latest_dir)

    @staticmethod
    def _setup_output_directories(base_output_dir: str) -> tuple[str, str]:
        """
        Create timestamped output directory and symlink to latest.

        Args:
            base_output_dir: Base directory for outputs

        Returns:
            Tuple of (timestamped_dir, latest_dir)
        """
        base_path = Path(base_output_dir)
        base_path.mkdir(exist_ok=True)

        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_dir = base_path / f"run_{timestamp}"
        timestamped_dir.mkdir(exist_ok=True)

        # Create/update latest directory
        latest_dir = base_path / "latest"
        if latest_dir.exists():
            if latest_dir.is_symlink():
                latest_dir.unlink()
            else:
                shutil.rmtree(latest_dir)
        latest_dir.mkdir(exist_ok=True)

        logger.debug(f"Created output directory: {timestamped_dir}")
        logger.debug(f"Latest results available at: {latest_dir}")

        return str(timestamped_dir), str(latest_dir)

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.args.device)

        logger.debug(f"Using device: {device}")
        return device

    def _create_model(self):
        """Create and configure the dual graph forecaster.

        Creates model with proper tensor shape validation and documentation.
        Key tensor shapes:
        - Input mobility features: [num_nodes, mobility_feature_dim]
        - EDAR features (if used): [num_edars, edar_input_dim]
        - Output forecasts: [num_nodes, forecast_horizon]
        """
        logger.debug("Creating dual graph forecaster")

        # Get dimensions from dataset
        mobility_feature_dim = self.dataset["node_feature_dim"]
        num_nodes = self.dataset["num_nodes"]
        logger.debug(
            f"Mobility input dimension: {mobility_feature_dim} (nodes: {num_nodes})"
        )

        # Validate input dimensions
        assert mobility_feature_dim > 0, (
            f"Invalid mobility feature dimension: {mobility_feature_dim}"
        )
        assert num_nodes > 0, f"Invalid number of nodes: {num_nodes}"

        # Determine EDAR input dimension if using EDAR data
        edar_input_dim = 32  # Default placeholder
        if (
            self.dataset.get("use_edar_data", False)
            and self.dataset.get("edar_biomarker_loader") is not None
        ):
            # Get actual feature dimension from biomarker loader
            edar_biomarker_loader = self.dataset["edar_biomarker_loader"]
            if len(edar_biomarker_loader.time_index) > 0:
                sample_features = edar_biomarker_loader.get_temporal_features_tensor(0)
                edar_input_dim = sample_features.shape[1]
                logger.debug(
                    f"EDAR input dimension from biomarker data: {edar_input_dim}"
                )
                # Validate EDAR feature dimension
                assert edar_input_dim > 0, (
                    f"Invalid EDAR feature dimension: {edar_input_dim}"
                )

        # Model configuration with shape annotations
        config = {
            "hidden_dim": self.args.hidden_dim,  # [num_nodes, hidden_dim]
            "num_layers": self.args.num_layers,
            "forecast_horizon": self.args.forecast_horizon,  # [num_nodes, horizon]
            "sequence_length": 1,  # [seq_len, num_nodes, features] - k=1 subgraph training
            "dropout": self.args.dropout,
            "aggregator": self.args.aggregator,
            "edar_hidden_dim": self.args.edar_hidden_dim,  # [num_edars, edar_hidden_dim]
            "edar_input_dim": edar_input_dim,  # [num_edars, edar_input_dim]
        }

        # Create dual graph forecaster
        self.model = create_dual_graph_forecaster(
            config=config,
            mobility_feature_dim=mobility_feature_dim,
            use_edar_data=self.dataset.get("use_edar_data", False),
            edar_attention_loader=self.dataset.get("edar_attention_loader"),
        )

        self.model = self.model.to(self.device)

        # Log detailed model summary
        model_type = "Dual Graph Forecaster"
        if self.dataset.get("use_edar_data", False):
            model_type += f" (EDAR-enabled, {self.dataset.get('n_edars', 0)} EDARs)"
        else:
            model_type += " (Mobility-only)"

        log_model_summary(self.model, logger, model_type)

    def _create_trainer(self):
        """Create and configure the ForecasterSubgraphTrainer."""
        logger.debug("Creating ForecasterSubgraphTrainer")

        # Create trainer configuration
        trainer_config = SubgraphTrainerConfig(
            epochs=self.args.epochs,
            patience=20,
            grad_clip=1.0,
            amp=False,
            sequence_length=1,  # k=1 subgraph training
            forecast_horizon=self.args.forecast_horizon,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            plateau_factor=0.7,
            plateau_patience=10,
            seed=getattr(self.args, "seed", 42),
            save_best=self.args.save_model,
            save_last=self.args.save_model,
        )

        # Setup optimizer and scheduler
        optimizer = Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=10)

        # Get EDAR attention mask if available
        edar_muni_mask = None
        if self.dataset and self.dataset.get("use_edar_data", False):
            edar_attention_loader = self.dataset.get("edar_attention_loader")
            if edar_attention_loader:
                # Use extended attention mask if available, otherwise fallback to original
                if edar_attention_loader.is_extended():
                    edar_muni_mask = (
                        edar_attention_loader.get_extended_attention_tensor()
                    )
                    logger.debug(
                        f"Using extended EDAR attention mask: {edar_muni_mask.shape}"
                    )
                else:
                    edar_muni_mask = edar_attention_loader.get_attention_tensor()
                    logger.debug(
                        f"Using original EDAR attention mask: {edar_muni_mask.shape}"
                    )

        # Get cases tensor for target extraction
        cases_tensor = self.dataset.get("cases_tensor")
        if cases_tensor is None:
            raise ValueError(
                "Cases tensor not found in dataset. Cannot proceed with COVID case forecasting."
            )

        # Create cases target extractor
        cases_target_wrapper = create_cases_target_extractor_wrapper(cases_tensor)

        # Create trainer
        self.trainer = ForecasterSubgraphTrainer(
            model=self.model,
            config=trainer_config,
            device=self.device,
            output_dir=str(self.timestamped_dir),
            optimizer=optimizer,
            scheduler=scheduler,
            target_extractor=cases_target_wrapper,  # Use cases-based target extractor
            edar_muni_mask=edar_muni_mask,
            edar_biomarker_loader=self.dataset.get("edar_biomarker_loader"),
            logger=logger.info,
        )

    def _save_results(
        self, metrics: dict[str, float], training_history: dict
    ) -> dict[str, Any]:
        """
        Save training results and generate visualizations.

        Args:
            metrics: Evaluation metrics
            training_history: Training history dictionary

        Returns:
            Complete results dictionary
        """
        # Compile results
        results = {
            "args": vars(self.args),
            "dataset_info": {
                "num_nodes": self.dataset["num_nodes"],
                "node_feature_dim": self.dataset["node_feature_dim"],
                "edge_feature_dim": self.dataset["edge_feature_dim"],
                "num_zones": len(self.dataset["region_ids"]),
                "sample_zones": self.dataset["region_ids"][:10],
            },
            "evaluation_metrics": metrics,
            "training_history": training_history,
            "output_dir": str(self.timestamped_dir),
        }

        # Save to timestamped directory
        with open(self.timestamped_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Copy to latest directory
        shutil.copy(
            self.timestamped_dir / "results.json", self.latest_dir / "results.json"
        )

        return results

    def _generate_plots(
        self, predictions, targets, metrics: dict[str, float], training_history: dict
    ):
        """Generate and save all plots.

        Args:
            predictions: numpy array of predictions with shape [num_samples, forecast_horizon]
            targets: numpy array of targets with shape [num_samples, forecast_horizon]
            metrics: dictionary of evaluation metrics
            training_history: dictionary containing training history data
        """
        if not self.args.no_plots and predictions is not None and targets is not None:
            # Validate tensor shapes before plotting
            assert predictions.shape == targets.shape, (
                f"Prediction/target shape mismatch: {predictions.shape} vs {targets.shape}"
            )
            assert predictions.ndim == 2, (
                f"Expected 2D predictions, got {predictions.shape}"
            )
            assert predictions.shape[1] == self.args.forecast_horizon, (
                f"Prediction horizon mismatch: expected {self.args.forecast_horizon}, "
                f"got {predictions.shape[1]}"
            )

            generate_all_plots(
                predictions=predictions,
                targets=targets,
                metrics=metrics,
                forecast_horizon=self.args.forecast_horizon,
                output_dir=str(self.timestamped_dir),
                region_ids=self.dataset["region_ids"],
                training_history=training_history,
            )
            # Copy plots to latest directory
            plot_files = [
                "forecast_time_series.png",
                "residual_analysis.png",
                "metrics_summary.png",
                "training_history.png",
                "prediction_scatter.png",
                "attention_alignment.png",
                "residual_choropleth.png",
                "seen_unseen_performance.png",
                "results_table.csv",
            ]
            for plot_file in plot_files:
                src_path = self.timestamped_dir / plot_file
                if src_path.exists():
                    shutil.copy(src_path, self.latest_dir / plot_file)
        else:
            # Still print results table even without plots
            print_results_table(metrics, self.dataset["region_ids"])

    def run(self) -> dict[str, Any]:
        """
        Run the complete training pipeline.

        Orchestrates the entire training process with proper tensor shape validation
        and documentation throughout the pipeline.

        Key tensor shapes throughout the pipeline:
        - Input mobility graphs: [num_nodes, node_feature_dim]
        - Temporal sequences: [seq_len, num_nodes, feature_dim]
        - Model predictions: [num_target_nodes, forecast_horizon] (COVID cases)
        - Evaluation targets: [num_target_nodes, forecast_horizon] (COVID cases)

        Returns:
            Results dictionary with metrics, training history, and metadata
        """
        logger.info("Starting complete training pipeline")

        # 1. Data loading and processing
        logger.debug("Step 1: Data loading and processing")
        self.dataset, self.temporal_graphs = self.data_manager.load_and_process_data(
            self.args
        )

        # Validate dataset structure
        assert "num_nodes" in self.dataset, "Dataset missing num_nodes key"
        assert "node_feature_dim" in self.dataset, (
            "Dataset missing node_feature_dim key"
        )
        assert len(self.temporal_graphs) > 0, "No temporal graphs loaded"

        # Validate cases data
        assert "cases_tensor" in self.dataset, (
            "Dataset missing cases_tensor key - COVID cases data required"
        )
        cases_tensor = self.dataset["cases_tensor"]
        assert cases_tensor.shape[0] == self.dataset["num_nodes"], (
            f"Cases tensor nodes {cases_tensor.shape[0]} != dataset nodes {self.dataset['num_nodes']}"
        )
        assert cases_tensor.shape[1] >= len(self.temporal_graphs), (
            f"Cases tensor timepoints {cases_tensor.shape[1]} < temporal graphs {len(self.temporal_graphs)}"
        )

        logger.info(
            f"Cases data validation passed: {cases_tensor.shape[0]} municipalities, {cases_tensor.shape[1]} timepoints"
        )

        num_nodes = self.dataset["num_nodes"]
        feature_dim = self.dataset["node_feature_dim"]
        logger.debug(f"Dataset validated: {num_nodes} nodes, {feature_dim} features")

        # 2. Model creation
        logger.debug("Step 2: Model creation")
        self._create_model()

        # 3. Trainer setup
        logger.debug("Step 3: Trainer setup")
        self._create_trainer()

        # 4. Data splitting
        logger.debug("Step 4: Data splitting")
        train_graphs, val_graphs, test_graphs = (
            self.data_manager.get_train_val_test_splits(self.temporal_graphs, self.args)
        )

        # Validate data splits
        assert len(train_graphs) > 0, "Empty training set"
        assert len(val_graphs) > 0, "Empty validation set"
        assert len(test_graphs) > 0, "Empty test set"
        logger.debug(
            f"Data splits - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}"
        )

        # 5. Create subgraph loader
        logger.debug("Step 5: Subgraph loader setup")
        subgraph_loader = TemporalSubgraphLoader(
            num_neighbors=self.args.subgraph_num_neighbors,
            batch_size=self.args.batch_size,
            shuffle=True,
            degree_balanced_sampling=True,
        )

        # Get loader statistics
        loader_stats = subgraph_loader.get_batch_stats(self.temporal_graphs)
        logger.debug("Subgraph loader statistics:")
        for key, value in loader_stats.items():
            logger.debug(f"  {key}: {value}")

        # 6. Training
        logger.info("Step 6: Model training")
        training_history = self.trainer.fit(train_graphs, val_graphs, subgraph_loader)

        # Validate training history
        assert "train_losses" in training_history, (
            "Training history missing train_losses"
        )
        assert "val_losses" in training_history, "Training history missing val_losses"
        assert len(training_history["train_losses"]) > 0, "No training losses recorded"

        # 7. Evaluation
        logger.info("Step 7: Model evaluation")
        metrics, predictions, targets = self.trainer.evaluate(
            test_graphs, subgraph_loader
        )

        # Validate evaluation results
        if predictions is not None:
            assert predictions.shape[1] == self.args.forecast_horizon, (
                f"Prediction horizon mismatch: expected {self.args.forecast_horizon}, "
                f"got {predictions.shape[1]}"
            )
        if targets is not None:
            assert targets.shape[1] == self.args.forecast_horizon, (
                f"Target horizon mismatch: expected {self.args.forecast_horizon}, "
                f"got {targets.shape[1]}"
            )

        # 8. Save results and generate visualizations
        logger.debug("Step 8: Saving results")
        results = self._save_results(metrics, training_history)
        self._generate_plots(predictions, targets, metrics, training_history)

        # 9. Final logging
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results saved to: {self.timestamped_dir}")
        logger.info(f"Latest results available at: {self.latest_dir}")

        return results
