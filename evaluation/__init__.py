"""Evaluation module for EpiForecaster.

This module provides evaluation utilities, loss functions, metrics, and
plotting capabilities for epidemiological forecasting models.

Key submodules:
- losses: Loss functions for training and evaluation
- metrics: Metric accumulation and computation
- loaders: Model and data loader construction
- selection: Node selection utilities
- eval_loop: Core evaluation loop implementation
- epiforecaster_eval: High-level evaluation pipelines
"""

# Core evaluation components
from evaluation.eval_loop import eval_checkpoint, evaluate_loader
from evaluation.loaders import build_loader_from_config, load_model_from_checkpoint
from evaluation.selection import (
    _GLOBAL_RNG,
    select_nodes_by_loss,
    topk_target_nodes_by_mae,
)

# High-level evaluation pipelines
from evaluation.epiforecaster_eval import (
    evaluate_checkpoint_topk_forecasts,
    plot_forecasts_from_csv,
)

# Re-export from plotting for backwards compatibility
from plotting.forecast_plots import generate_forecast_plots

__all__ = [
    # Loaders
    "load_model_from_checkpoint",
    "build_loader_from_config",
    # Selection
    "select_nodes_by_loss",
    "topk_target_nodes_by_mae",
    "_GLOBAL_RNG",
    # Evaluation
    "evaluate_loader",
    # High-level pipelines
    "eval_checkpoint",
    "evaluate_checkpoint_topk_forecasts",
    "generate_forecast_plots",
    "plot_forecasts_from_csv",
]
