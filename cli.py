"""
Simplified Click-based CLI interface for Graph Neural Network Epidemiological Forecasting.

This module provides a streamlined command-line interface with only two main commands:
- `preprocess`: Run offline preprocessing pipeline to create canonical datasets
- `train`: Train models using preprocessed canonical datasets

This replaces the complex variant-specific CLI with configuration-driven training.
"""

import logging
import math
import traceback
from datetime import datetime
from pathlib import Path

import click

from data.preprocess import OfflinePreprocessingPipeline, PreprocessingConfig
from data.preprocess.region_graph_preprocessor import (
    RegionGraphPreprocessConfig,
    RegionGraphPreprocessor,
)
from training.epiforecaster_trainer import (
    EpiForecasterConfig,
    EpiForecasterTrainer,
)
from training.region2vec_trainer import Region2VecTrainer, RegionTrainerConfig
from utils.logging import setup_logging

VALID_DEVICES = ["auto", "cpu", "cuda", "mps"]


@click.group()
@click.version_option()
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging (includes model forward-pass debug logs).",
)
def cli(debug: bool):
    """Graph Neural Network for Epidemiological Forecasting.

    A streamlined toolkit for preprocessing data and training graph neural networks
    for epidemiological forecasting using canonical datasets.
    """
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=level)


@cli.group("preprocess")
def preprocess_group():
    """Run preprocessing pipelines for datasets or region graphs."""
    pass


@preprocess_group.command("regions")
@click.option(
    "--geojson",
    type=click.Path(path_type=Path),
    default=Path("data/files/geo/fl_municipios_catalonia.geojson"),
    show_default=True,
    help="Path to region boundary GeoJSON file",
)
@click.option(
    "--population",
    type=click.Path(path_type=Path),
    default=Path("data/files/fl_population_por_municipis.csv"),
    show_default=True,
    help="Path to CSV with population per region",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("outputs/region_graph/region_graph.zarr"),
    show_default=True,
    help="Output Zarr directory for the region graph",
)
@click.option(
    "--geojson-id-field",
    default="id",
    show_default=True,
    help="Column in GeoJSON that identifies regions",
)
@click.option(
    "--population-id-field",
    default="id",
    show_default=True,
    help="Column in population CSV that matches the GeoJSON IDs",
)
@click.option(
    "--population-value-field",
    default="d.population",
    show_default=True,
    help="Population column name in the CSV",
)
@click.option(
    "--mobility-zarr",
    type=click.Path(path_type=Path),
    default=Path("data/files/mobility.zarr"),
    show_default=True,
    help="Path to mobility Zarr dataset for OD flows",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Start date (YYYY-MM-DD) for averaging OD flows; defaults to last 30 days",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="End date (YYYY-MM-DD) for averaging OD flows; defaults to dataset max",
)
def preprocess_regions(
    geojson: Path,
    population: Path,
    output: Path,
    geojson_id_field: str,
    population_id_field: str,
    population_value_field: str,
    mobility_zarr: Path,
    start_date: datetime | None,
    end_date: datetime | None,
):
    """Build region features, adjacency, and flows from geospatial sources."""
    try:
        start_date_str = start_date.date().isoformat() if start_date else None
        end_date_str = end_date.date().isoformat() if end_date else None
        config = RegionGraphPreprocessConfig(
            geojson_path=geojson,
            population_csv_path=population,
            geojson_id_field=geojson_id_field,
            population_id_field=population_id_field,
            population_value_field=population_value_field,
            output_path=output,
            mobility_zarr_path=mobility_zarr,
            start_date=start_date_str,
            end_date=end_date_str,
        )
        preprocessor = RegionGraphPreprocessor(config)
        result = preprocessor.run()

        click.echo(f"\n{'=' * 60}")
        click.echo("✅ REGION GRAPH PREPROCESSING COMPLETED")
        click.echo(f"{'=' * 60}")
        click.echo(f"Regions: {result['num_regions']}")
        click.echo(f"Features per region: {result['feature_dim']}")
        click.echo(f"Edges: {result['num_edges']}")
        click.echo(f"Output saved to: {result['output_path']}")
        click.echo(f"{'=' * 60}")
    except Exception as exc:
        click.echo(f"❌ Region preprocessing failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@preprocess_group.command("epiforecaster")
@click.option(
    "--config", required=True, help="Path to preprocessing configuration file"
)
def preprocess_epiforecaster(config: str):
    """Run the standard EpiForecaster preprocessing pipeline."""
    try:
        preprocess_config = PreprocessingConfig.from_file(config)

        logger = logging.getLogger(__name__)
        logger.info(f"Loading configuration from: {config}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Configuration summary:")
            summary = preprocess_config.summary()
            for section, info in summary.items():
                logger.debug(f"  {section}: {info}")

        pipeline = OfflinePreprocessingPipeline(preprocess_config)
        output_path = pipeline.run()

        logger = logging.getLogger(__name__)
        logger.info(f"\n{'=' * 60}")
        logger.info("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Dataset saved to: {output_path}")
        logger.info(f"Dataset name: {preprocess_config.dataset_name}")
        logger.info("You can now train models using:")
        logger.info(
            "  uv run python -m cli train epiforecaster --config <training_config>"
        )
        logger.debug(
            f"  (Make sure your training config contains: data.dataset_path: {output_path})"
        )
        logger.info(f"{'=' * 60}")

    except Exception as exc:
        click.echo(f"❌ Preprocessing failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@cli.group("train")
def train_group():
    """Train forecasting models or specialized submodules."""
    pass


@cli.group("eval")
def eval_group():
    """Evaluate trained checkpoints and generate plots."""
    pass


@eval_group.command("epiforecaster")
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to a saved `.pt` checkpoint (e.g., best_model.pt).",
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"], case_sensitive=False),
    default="val",
    show_default=True,
    help="Which split to evaluate for top-k node selection.",
)
@click.option(
    "--topk",
    type=int,
    default=5,
    show_default=True,
    help="Number of best (lowest MAE) target nodes to visualize.",
)
@click.option(
    "--window",
    type=click.Choice(["last"], case_sensitive=False),
    default="last",
    show_default=True,
    help="Which window to plot for each selected node.",
)
@click.option(
    "--device",
    type=click.Choice(VALID_DEVICES),
    default="auto",
    show_default=True,
    help="Device to use for evaluation.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output image path (e.g., outputs/plots/topk_val.png).",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress evaluation progress logs.",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional TensorBoard log dir for eval metrics.",
)
def eval_epiforecaster(
    checkpoint: Path,
    split: str,
    topk: int,
    window: str,
    device: str,
    output: Path | None,
    quiet: bool,
    log_dir: Path | None,
):
    """Evaluate an EpiForecaster checkpoint and plot top-k forecasts."""
    try:
        from evaluation.epiforecaster_eval import evaluate_checkpoint_topk_forecasts

        resolved_log_dir = _resolve_eval_log_dir(checkpoint, log_dir)
        result = evaluate_checkpoint_topk_forecasts(
            checkpoint_path=checkpoint,
            split=split,
            k=topk,
            device=device,
            window=window,
            output_path=output,
            log_dir=resolved_log_dir,
        )

        topk_nodes = result["topk_nodes"]
        click.echo(
            f"Top-{len(topk_nodes)} target nodes (by MAE on {split}): {topk_nodes}"
        )
        loss = result.get("eval_loss", float("nan"))
        metrics = result.get("eval_metrics", {})
        summary = _format_eval_summary(loss, metrics)
        click.echo(f"\nEval summary ({split}):\n{summary}")
        if resolved_log_dir is not None:
            click.echo(f"Eval metrics logged to: {resolved_log_dir}")
        if output is not None:
            click.echo(f"Saved plot to: {output}")
    except Exception as exc:  # pragma: no cover - Click will handle reporting
        click.echo(f"❌ Evaluation failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


def _resolve_eval_log_dir(checkpoint: Path, log_dir: Path | None) -> Path | None:
    if log_dir is not None:
        return log_dir

    parts = checkpoint.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx > 0:
            return Path(*parts[:idx])
    return None


def _format_eval_summary(loss: float, metrics: dict) -> str:
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
    lines = ["Metric  Value"]
    lines.append("------  ------")
    for name, value in rows:
        lines.append(f"{name:<6}  {value}")
    return "\n".join(lines)


@train_group.command("regions")
@click.option("--config", required=True, help="Path to region training configuration")
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Override the number of region pretraining epochs",
)
@click.option(
    "--device",
    type=click.Choice(VALID_DEVICES),
    default="auto",
    help="Override the compute device for region training",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom directory for embedding artifacts",
)
@click.option(
    "--no-cluster",
    is_flag=True,
    help="Skip post-training agglomerative clustering",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Load configuration and initialize the trainer without running optimization",
)
def train_regions(
    config: str,
    epochs: int | None,
    device: str | None,
    output_dir: Path | None,
    no_cluster: bool,
    dry_run: bool,
):
    """Train the Region2Vec-style region embedder via configuration."""
    try:
        region_config = RegionTrainerConfig.from_file(config)

        trainer = Region2VecTrainer(region_config)

        logger = logging.getLogger(__name__)
        if dry_run:
            summary = trainer.describe()
            logger.info("Region embedder configuration:")
            for key, value in summary.items():
                logger.info(f"  - {key}: {value}")
            click.echo("Dry run complete. No training executed.")
            return

        results = trainer.run()
        logger.info(f"\n{'=' * 60}")
        logger.info("✅ REGION EMBEDDING TRAINING COMPLETED")
        logger.info(f"{'=' * 60}")
        logger.info(f"Best loss: {results['best_loss']:.6f}")
        logger.info(f"Epochs run: {results['epochs']}")
        artifacts = results.get("artifacts", {})
        embedding_path = artifacts.get("embedding_path")
        metrics_path = artifacts.get("metrics_path")
        cluster_path = artifacts.get("cluster_labels_path")
        if embedding_path:
            logger.info(f"Embeddings saved to: {embedding_path}")
        if metrics_path:
            logger.info(f"Metrics written to: {metrics_path}")
        if cluster_path:
            logger.info(f"Cluster labels saved to: {cluster_path}")
        logger.info(f"{'=' * 60}")

    except Exception as exc:  # pragma: no cover - Click will handle reporting
        click.echo(f"❌ Region training failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@train_group.command("epiforecaster")
@click.option("--config", required=True, help="Path to training configuration file")
@click.option("--model-id", default="", help="Model id for logging/checkpoints")
@click.option("--resume", is_flag=True, help="Resume training from a saved checkpoint")
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Maximum number of batches to run (for smoke testing)",
)
def train_epiforecaster(
    config: str, model_id: str, resume: bool, max_batches: int | None
):
    """Train EpiForecaster model."""
    _run_forecaster_training(config, model_id, resume, max_batches)


def _run_forecaster_training(
    config: str, model_id: str, resume: bool, max_batches: int | None
) -> None:
    """Execute the original unified forecaster training workflow."""
    try:
        trainer_config = EpiForecasterConfig.from_file(config)
        # Dataset path should be read from the config file
        if model_id:
            trainer_config.training.model_id = model_id
        trainer_config.training.resume = resume

        # Override max_batches if specified
        if max_batches is not None:
            trainer_config.training.max_batches = max_batches

        logger = logging.getLogger(__name__)
        logger.info(f"Loading training configuration from: {config}")
        logger.debug(f"Training model: {trainer_config.model.type}")
        logger.debug(f"Using dataset: {trainer_config.data.dataset_path}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Configuration:")
            logger.debug(f"  - Epochs: {trainer_config.training.epochs}")
            logger.debug(f"  - Batch size: {trainer_config.training.batch_size}")
            logger.debug(f"  - Learning rate: {trainer_config.training.learning_rate}")
            if trainer_config.training.model_id:
                logger.debug(f"  - Model ID: {trainer_config.training.model_id}")
            if trainer_config.training.resume:
                logger.debug("  - Resume: enabled")

        trainer = EpiForecasterTrainer(trainer_config)
        results = trainer.run()

        logger.info(f"\n{'=' * 60}")
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"Total epochs trained: {results['total_epochs']}")
        logger.info(f"Model parameters: {results['model_info']['parameters']:,}")
        logger.info(
            f"Trainable parameters: {results['model_info']['trainable_parameters']:,}"
        )
        log_dir = (
            Path(trainer.config.output.log_dir)
            / trainer.config.output.experiment_name
            / trainer.model_id
        )
        logger.info(f"Training logs saved to: {log_dir}")
        logger.info(f"{'=' * 60}")

    except Exception as exc:  # pragma: no cover - CLI handles presentation
        click.echo(f"❌ Training failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@cli.command("info")
@click.option("--dataset", help="Path to preprocessed dataset (.zarr file)")
@click.option("--validate", is_flag=True, help="Validate dataset integrity")
def dataset_info(dataset: str, validate: bool):
    """Display information about preprocessed datasets."""
    try:
        if not dataset:
            click.echo("❌ Dataset path is required", err=True)
            return

        dataset_path = Path(dataset)
        if not dataset_path.exists():
            click.echo(f"❌ Dataset not found: {dataset}", err=True)
            return

        # Load dataset index
        from data.dataset_storage import DatasetStorage

        logger = logging.getLogger(__name__)

        if validate:
            logger.info(f"Validating dataset: {dataset}")
            validation_result = DatasetStorage.validate_dataset(dataset_path)

            if validation_result["valid"]:
                logger.info("✅ Dataset validation passed")
            else:
                logger.warning("❌ Dataset validation failed:")
                for issue in validation_result["issues"]:
                    logger.warning(f"  - {issue}")

        # Get dataset metadata
        dataset_info = DatasetStorage.load_dataset(dataset_path)
        metadata = dataset_info["metadata"]

        logger.info(f"\n{'=' * 50}")
        logger.info("DATASET INFORMATION")
        logger.info(f"{'=' * 50}")
        logger.info(f"Name: {metadata['dataset_name']}")
        logger.info(f"Created: {metadata['created_at']}")
        logger.info(f"Schema version: {metadata['schema_version']}")

        logger.info("\n📊 DIMENSIONS:")
        logger.info(f"  Timepoints: {metadata['num_timepoints']}")
        logger.info(f"  Nodes: {metadata['num_nodes']}")
        logger.info(f"  Edges: {metadata['num_edges']}")
        logger.info(f"  Feature dimension: {metadata['feature_dim']}")
        logger.info(f"  Forecast horizon: {metadata['forecast_horizon']}")

        logger.info("\n📁 AVAILABLE FEATURES:")
        logger.info("  Node features: ✅")
        logger.info(f"  Edge attributes: {'✅' if metadata['has_edge_attr'] else '❌'}")
        logger.info(
            f"  Region embeddings: {'✅' if metadata['has_region_embeddings'] else '❌'}"
        )
        logger.info(f"  EDAR data: {'✅' if metadata['has_edar_data'] else '❌'}")

        if "time_range" in metadata:
            logger.info("\n📅 TEMPORAL RANGE:")
            logger.info(f"  Start: {metadata['time_range']['start']}")
            logger.info(f"  End: {metadata['time_range']['end']}")

        logger.info(f"{'=' * 50}")

    except Exception as e:
        click.echo(f"❌ Failed to get dataset info: {str(e)}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(e)) from e


@cli.command("list-datasets")
@click.option(
    "--data-dir",
    default="data/processed",
    help="Directory containing processed datasets",
)
def list_datasets(data_dir: str):
    """List all available preprocessed datasets."""
    try:
        from data.dataset_storage import DatasetStorage

        data_dir_path = Path(data_dir)
        if not data_dir_path.exists():
            click.echo(f"❌ Data directory not found: {data_dir}", err=True)
            return

        logger = logging.getLogger(__name__)
        logger.info(f"Scanning for datasets in: {data_dir}")
        dataset_index = DatasetStorage.create_dataset_index(data_dir_path)

        if not dataset_index:
            logger.info("No datasets found.")
            return

        logger.info(f"\n{'=' * 60}")
        logger.info(f"AVAILABLE DATASETS ({len(dataset_index)} found)")
        logger.info(f"{'=' * 60}")

        for name, info in dataset_index.items():
            logger.info(f"\n📦 {name}")
            logger.info(f"  Path: {info['path']}")
            logger.info(f"  Timepoints: {info['num_timepoints']}")
            logger.info(f"  Nodes: {info['num_nodes']}")
            logger.info(f"  Forecast horizon: {info['forecast_horizon']}")
            logger.info(f"  Created: {info['created_at']}")

            features = []
            if info["has_edge_attr"]:
                features.append("mobility")
            if info["has_region_embeddings"]:
                features.append("embeddings")
            if info["has_edar_data"]:
                features.append("EDAR")

            if features:
                logger.info(f"  Features: {', '.join(features)}")

        logger.info(f"{'=' * 60}")

    except Exception as e:
        click.echo(f"❌ Failed to list datasets: {str(e)}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    cli()
