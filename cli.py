"""
Simplified Click-based CLI interface for Graph Neural Network Epidemiological Forecasting.

This module provides a streamlined command-line interface with only two main commands:
- `preprocess`: Run offline preprocessing pipeline to create canonical datasets
- `train`: Train models using preprocessed canonical datasets

This replaces the complex variant-specific CLI with configuration-driven training.
"""

import logging
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
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def preprocess_epiforecaster(config: str, verbose: bool):
    """Run the standard EpiForecaster preprocessing pipeline."""
    try:
        preprocess_config = PreprocessingConfig.from_file(config)

        if verbose:
            print(f"Loading configuration from: {config}")
            print("Configuration summary:")
            summary = preprocess_config.summary()
            for section, info in summary.items():
                print(f"  {section}: {info}")
            print()

        pipeline = OfflinePreprocessingPipeline(preprocess_config)
        output_path = pipeline.run()

        print(f"\n{'=' * 60}")
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'=' * 60}")
        print(f"Dataset saved to: {output_path}")
        print(f"Dataset name: {preprocess_config.dataset_name}")
        print("You can now train models using:")
        print(
            f"  epiforecaster train --dataset {output_path} --config <training_config>"
        )
        print(f"{'=' * 60}")

    except Exception as exc:
        click.echo(f"❌ Preprocessing failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@cli.group("train")
def train_group():
    """Train forecasting models or specialized submodules."""
    pass


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
@click.option("--verbose", is_flag=True, help="Print detailed setup information")
def train_regions(
    config: str,
    epochs: int | None,
    device: str | None,
    output_dir: Path | None,
    no_cluster: bool,
    dry_run: bool,
    verbose: bool,
):
    """Train the Region2Vec-style region embedder via configuration."""
    try:
        region_config = RegionTrainerConfig.from_file(config)

        trainer = Region2VecTrainer(region_config)

        if verbose or dry_run:
            summary = trainer.describe()
            click.echo("Region embedder configuration:")
            for key, value in summary.items():
                click.echo(f"  - {key}: {value}")

        if dry_run:
            click.echo("Dry run complete. No training executed.")
            return

        results = trainer.run()
        click.echo(f"\n{'=' * 60}")
        click.echo("✅ REGION EMBEDDING TRAINING COMPLETED")
        click.echo(f"{'=' * 60}")
        click.echo(f"Best loss: {results['best_loss']:.6f}")
        click.echo(f"Epochs run: {results['epochs']}")
        artifacts = results.get("artifacts", {})
        embedding_path = artifacts.get("embedding_path")
        metrics_path = artifacts.get("metrics_path")
        cluster_path = artifacts.get("cluster_labels_path")
        if embedding_path:
            click.echo(f"Embeddings saved to: {embedding_path}")
        if metrics_path:
            click.echo(f"Metrics written to: {metrics_path}")
        if cluster_path:
            click.echo(f"Cluster labels saved to: {cluster_path}")
        click.echo(f"{'=' * 60}")

    except Exception as exc:  # pragma: no cover - Click will handle reporting
        click.echo(f"❌ Region training failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@train_group.command("epiforecaster")
@click.option("--config", required=True, help="Path to training configuration file")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show tqdm progress bars during training",
)
def train_epiforecaster(config: str, verbose: bool, progress: bool):
    """Train the EpiForecaster model."""
    _run_forecaster_training(config, verbose, progress)


def _run_forecaster_training(config: str, verbose: bool, progress: bool) -> None:
    """Execute the original unified forecaster training workflow."""
    try:
        trainer_config = EpiForecasterConfig.from_file(config)
        # Dataset path should be read from the config file

        trainer_config.training.use_tqdm = progress

        if verbose:
            print(f"Loading training configuration from: {config}")
            print(f"Training model: {trainer_config.model.type}")
            print(f"Using dataset: {trainer_config.data.dataset_path}")
            print("Configuration:")
            print(f"  - Epochs: {trainer_config.training.epochs}")
            print(f"  - Batch size: {trainer_config.training.batch_size}")
            print(f"  - Learning rate: {trainer_config.training.learning_rate}")
            print()

        trainer = EpiForecasterTrainer(trainer_config)
        results = trainer.run()

        print(f"\n{'=' * 60}")
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'=' * 60}")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Total epochs trained: {results['total_epochs']}")
        print(f"Model parameters: {results['model_info']['parameters']:,}")
        print(
            f"Trainable parameters: {results['model_info']['trainable_parameters']:,}"
        )
        print(f"Training logs saved to: {trainer.config.output.log_dir}")
        print(f"{'=' * 60}")

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

        if validate:
            print(f"Validating dataset: {dataset}")
            validation_result = DatasetStorage.validate_dataset(dataset_path)

            if validation_result["valid"]:
                print("✅ Dataset validation passed")
            else:
                print("❌ Dataset validation failed:")
                for issue in validation_result["issues"]:
                    print(f"  - {issue}")

        # Get dataset metadata
        dataset_info = DatasetStorage.load_dataset(dataset_path)
        metadata = dataset_info["metadata"]

        print(f"\n{'=' * 50}")
        print("DATASET INFORMATION")
        print(f"{'=' * 50}")
        print(f"Name: {metadata['dataset_name']}")
        print(f"Created: {metadata['created_at']}")
        print(f"Schema version: {metadata['schema_version']}")

        print("\n📊 DIMENSIONS:")
        print(f"  Timepoints: {metadata['num_timepoints']}")
        print(f"  Nodes: {metadata['num_nodes']}")
        print(f"  Edges: {metadata['num_edges']}")
        print(f"  Feature dimension: {metadata['feature_dim']}")
        print(f"  Forecast horizon: {metadata['forecast_horizon']}")

        print("\n📁 AVAILABLE FEATURES:")
        print("  Node features: ✅")
        print(f"  Edge attributes: {'✅' if metadata['has_edge_attr'] else '❌'}")
        print(
            f"  Region embeddings: {'✅' if metadata['has_region_embeddings'] else '❌'}"
        )
        print(f"  EDAR data: {'✅' if metadata['has_edar_data'] else '❌'}")

        if "time_range" in metadata:
            print("\n📅 TEMPORAL RANGE:")
            print(f"  Start: {metadata['time_range']['start']}")
            print(f"  End: {metadata['time_range']['end']}")

        print(f"{'=' * 50}")

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

        print(f"Scanning for datasets in: {data_dir}")
        dataset_index = DatasetStorage.create_dataset_index(data_dir_path)

        if not dataset_index:
            print("No datasets found.")
            return

        print(f"\n{'=' * 60}")
        print(f"AVAILABLE DATASETS ({len(dataset_index)} found)")
        print(f"{'=' * 60}")

        for name, info in dataset_index.items():
            print(f"\n📦 {name}")
            print(f"  Path: {info['path']}")
            print(f"  Timepoints: {info['num_timepoints']}")
            print(f"  Nodes: {info['num_nodes']}")
            print(f"  Forecast horizon: {info['forecast_horizon']}")
            print(f"  Created: {info['created_at']}")

            features = []
            if info["has_edge_attr"]:
                features.append("mobility")
            if info["has_region_embeddings"]:
                features.append("embeddings")
            if info["has_edar_data"]:
                features.append("EDAR")

            if features:
                print(f"  Features: {', '.join(features)}")

        print(f"{'=' * 60}")

    except Exception as e:
        click.echo(f"❌ Failed to list datasets: {str(e)}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    cli()
