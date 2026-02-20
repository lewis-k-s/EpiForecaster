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
import wandb

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
from utils.platform import is_slurm_cluster

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


@click.group()
@click.version_option()
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging (includes model forward-pass debug logs).",
)
def preprocess_cli(debug: bool):
    """Run preprocessing pipelines for datasets or region graphs.

    Direct usage: uv run preprocess <command>
    Via main:     uv run main preprocess <command>
    """
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=level)


@preprocess_cli.command("regions")
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


@preprocess_cli.command("epiforecaster")
@click.option(
    "--config", required=True, help="Path to preprocessing configuration file"
)
def preprocess_epiforecaster(config: str):
    """Run the standard EpiForecaster preprocessing pipeline."""
    try:
        preprocess_config = PreprocessingConfig.from_file(config)
        if preprocess_config.env == "mn5" and not is_slurm_cluster():
            raise click.ClickException(
                "Production preprocessing configs (env=mn5) must run on SLURM."
            )

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


@click.group()
@click.version_option()
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging (includes model forward-pass debug logs).",
)
def train_cli(debug: bool):
    """Train forecasting models or specialized submodules.

    Direct usage: uv run train <command>
    Via main:     uv run main train <command>
    """
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=level)


# Backward compatibility: add preprocess and train as subgroups of main
cli.add_command(preprocess_cli, name="preprocess")
cli.add_command(train_cli, name="train")


@cli.group("eval")
def eval_group():
    """Evaluate trained checkpoints and generate plots."""
    pass


@cli.group("plot")
def plot_group():
    """Generate forecast plots from evaluation results."""
    pass


@eval_group.command("epiforecaster")
@click.option(
    "--experiment",
    type=str,
    default=None,
    help="Experiment name (e.g., 'local_full'). Used to auto-resolve paths.",
)
@click.option(
    "--run",
    type=str,
    default=None,
    help="Run ID (e.g., 'run_1767364191170741000'). Used with --experiment.",
)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to checkpoint. Optional if --experiment and --run are provided.",
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"], case_sensitive=False),
    default="val",
    show_default=True,
    help="Which split to evaluate.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output image path.",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional CSV output path for node-level metrics (node_id, mae, num_samples).",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional W&B run directory for eval metrics.",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values (e.g., training.device=cpu, training.val_workers=4)",
)
@click.option(
    "--eval-batch-size",
    type=int,
    default=None,
    help="Override evaluation batch size (useful for memory-heavy mobility graphs).",
)
@click.option(
    "--compare-baselines",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Optional baseline metrics CSV (fold or aggregate) to compare against current "
        "checkpoint metrics."
    ),
)
def eval_epiforecaster(
    experiment: str | None,
    run: str | None,
    checkpoint: Path | None,
    split: str,
    output: Path | None,
    output_csv: Path | None,
    log_dir: Path | None,
    override: tuple[str, ...],
    eval_batch_size: int | None,
    compare_baselines: Path | None,
):
    """Evaluate an EpiForecaster checkpoint and generate quartile-based forecast plots.

    Use --override to customize evaluation settings like workers, device, or sampling:

    \b
    --override training.val_workers=4      # Use 4 workers for eval dataloader
    --override training.device=cpu          # Use CPU instead of GPU
    --override training.num_forecast_samples=6  # Sample 6 nodes per quartile
    """
    try:
        # Resolve paths from experiment/run if provided
        if experiment and run:
            from utils.run_discovery import (
                resolve_checkpoint_path,
                get_eval_output_dir,
                list_available_runs,
            )

            if checkpoint is None:
                try:
                    checkpoint = resolve_checkpoint_path(
                        experiment_name=experiment,
                        run_id=run,
                    )
                except FileNotFoundError as e:
                    click.echo(f"❌ Checkpoint not found: {e}", err=True)
                    click.echo(
                        f"\n{list_available_runs(experiment_name=experiment)}", err=True
                    )
                    raise click.ClickException("Cannot resolve checkpoint path")

            # Auto-resolve output paths to eval directory
            eval_dir = get_eval_output_dir(experiment_name=experiment, run_id=run)
            if output is None:
                output = eval_dir / f"{split}_forecasts.png"
            if output_csv is None:
                output_csv = eval_dir / f"{split}_node_metrics.csv"

            click.echo("Resolved output paths:")
            click.echo(f"  Checkpoint: {checkpoint}")
            click.echo(f"  Plot: {output}")
            click.echo(f"  CSV: {output_csv}")
        elif checkpoint is not None and experiment is None and run is None:
            # User provided --checkpoint but not --experiment/--run
            from utils.run_discovery import (
                extract_run_from_checkpoint_path,
                prompt_to_save_eval,
                get_eval_output_dir,
            )

            extracted = extract_run_from_checkpoint_path(checkpoint)

            if extracted is not None:
                extracted_experiment, extracted_run = extracted
                click.echo(
                    f"Detected: experiment={extracted_experiment}, run={extracted_run}"
                )
                should_save = prompt_to_save_eval(
                    extracted_experiment, extracted_run, default=True
                )

                if should_save:
                    eval_dir = get_eval_output_dir(
                        experiment_name=extracted_experiment, run_id=extracted_run
                    )
                    if output is None:
                        output = eval_dir / f"{split}_forecasts.png"
                    if output_csv is None:
                        output_csv = eval_dir / f"{split}_node_metrics.csv"
                else:
                    click.echo(
                        "Skipping persistence. Use --output to specify save location."
                    )
            else:
                # Could not extract - default to current directory
                click.echo("Could not auto-detect experiment/run from checkpoint path.")
                if output is None:
                    output = Path(f"{split}_forecasts.png")
                if output_csv is None:
                    output_csv = Path(f"{split}_node_metrics.csv")
        elif checkpoint is None:
            raise click.ClickException(
                "Must provide either --checkpoint or both --experiment and --run"
            )

        # Suppress zarr logging spam
        logging.getLogger("zarr").setLevel(logging.WARNING)
        logging.getLogger("numcodecs").setLevel(logging.WARNING)

        from evaluation.epiforecaster_eval import (
            eval_checkpoint,
            select_nodes_by_loss,
            generate_forecast_plots,
        )

        # Step 1: Evaluate (with config overrides if provided)
        eval_result = eval_checkpoint(
            checkpoint_path=checkpoint,
            split=split,
            log_dir=log_dir,
            overrides=list(override) if override else None,
            output_csv_path=output_csv,
            batch_size=eval_batch_size,
        )

        # Get samples_per_group from config (default 3)
        samples_per_group = eval_result["config"].training.num_forecast_samples

        # Step 2: Select nodes (quartile strategy)
        node_groups = select_nodes_by_loss(
            node_mae=eval_result["node_mae"],
            strategy="quartile",
            samples_per_group=samples_per_group,
        )

        # Step 3: Generate plots (use "last" window by default)
        plot_result = generate_forecast_plots(
            model=eval_result["model"],
            loader=eval_result["loader"],
            node_groups=node_groups,
            window="last",
            output_path=output,
            log_dir=log_dir,
        )

        # Step 4: Show results
        total_nodes = len(plot_result["selected_nodes"])
        click.echo(f"Selected {total_nodes} nodes from quartiles:")
        for group_name, nodes in plot_result["node_groups"].items():
            click.echo(f"  {group_name}: {len(nodes)} nodes")

        # Show eval metrics
        loss = eval_result["eval_loss"]
        metrics = eval_result["eval_metrics"]
        summary = _format_eval_summary(loss, metrics)
        click.echo(f"\nEval summary ({split}):\n{summary}")

        if compare_baselines is not None:
            from evaluation.baseline_eval import compare_model_metrics_against_baselines

            if output_csv is not None:
                delta_csv = output_csv.with_name(f"{split}_baseline_deltas.csv")
            elif output is not None:
                delta_csv = output.with_name(f"{split}_baseline_deltas.csv")
            else:
                delta_csv = Path(f"{split}_baseline_deltas.csv")

            compare_model_metrics_against_baselines(
                eval_metrics=metrics,
                baseline_results_csv=compare_baselines,
                output_csv=delta_csv,
            )
            click.echo(f"Saved baseline deltas to: {delta_csv}")

        if log_dir is not None:
            click.echo(f"Eval metrics logged to: {log_dir}")
        if output is not None:
            click.echo(f"Saved plot to: {output}")
        if output_csv is not None:
            click.echo(f"Saved node metrics to: {output_csv}")
        if wandb.run is not None:
            wandb.finish()
    except Exception as exc:
        click.echo(f"❌ Evaluation failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@eval_group.command("baselines")
@click.option("--config", required=True, help="Path to training configuration file.")
@click.option(
    "--models",
    type=click.Choice(
        ["tiered", "exp_smoothing", "var_cross_target", "all"],
        case_sensitive=False,
    ),
    default="tiered",
    show_default=True,
    help="Baseline model family to evaluate.",
)
@click.option(
    "--rolling-folds",
    type=int,
    default=5,
    show_default=True,
    help="Number of expanding rolling-origin folds.",
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"], case_sensitive=False),
    default="test",
    show_default=True,
    help="Which split to evaluate baselines on.",
)
@click.option(
    "--seasonal-period",
    type=int,
    default=7,
    show_default=True,
    help="Seasonal period used by seasonal-naive and SARIMA-family baselines.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory where baseline artifacts will be written.",
)
@click.option(
    "--disable-sparsity-bins",
    is_flag=True,
    help="Disable sparsity-decile stratified reporting.",
)
def eval_baselines(
    config: str,
    models: str,
    rolling_folds: int,
    split: str,
    seasonal_period: int,
    output_dir: Path,
    disable_sparsity_bins: bool,
):
    """Run fair rolling-origin baseline benchmarking for EpiForecaster targets."""
    try:
        if rolling_folds <= 0:
            raise click.ClickException("--rolling-folds must be positive.")
        if seasonal_period <= 0:
            raise click.ClickException("--seasonal-period must be positive.")

        cfg = EpiForecasterConfig.load(config)

        from evaluation.baseline_eval import run_baseline_evaluation

        selected_models = (
            ["tiered", "exp_smoothing", "var_cross_target"]
            if models.lower() == "all"
            else [models.lower()]
        )

        artifacts = run_baseline_evaluation(
            config=cfg,
            models=selected_models,
            config_path=config,
            output_dir=output_dir,
            split=split,
            rolling_folds=rolling_folds,
            seasonal_period=seasonal_period,
            include_sparsity_bins=not disable_sparsity_bins,
        )

        click.echo("Baseline evaluation completed.")
        for name, path in artifacts.items():
            click.echo(f"  {name}: {path}")
    except Exception as exc:
        click.echo(f"❌ Baseline evaluation failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@plot_group.command("forecasts")
@click.option(
    "--experiment",
    type=str,
    default=None,
    help="Experiment name. Used with --run to auto-resolve checkpoint path.",
)
@click.option(
    "--run",
    type=str,
    default=None,
    help="Run ID. Used with --experiment to auto-resolve checkpoint path.",
)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to model checkpoint. Required if not using --experiment/--run.",
)
@click.option(
    "--nodes",
    type=str,
    default="random:5",
    show_default=True,
    help=(
        "Node selection strategy: 'random:N', 'quartile:N', 'best:N', 'worst:N'. "
        "Quartile/best/worst require a mini-eval to compute per-node MAE."
    ),
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"], case_sensitive=False),
    default="test",
    show_default=True,
    help="Which split to use for plotting.",
)
@click.option(
    "--window",
    type=click.Choice(["last", "random"], case_sensitive=False),
    default="random",
    show_default=True,
    help="Which time window to plot: 'last' (final window) or 'random' (sample).",
)
@click.option(
    "--device",
    type=click.Choice(VALID_DEVICES),
    default="auto",
    show_default=True,
    help="Device to use for inference.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output image path.",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values (e.g., training.device=cuda, training.val_workers=0)",
)
def plot_forecasts(
    experiment: str | None,
    run: str | None,
    checkpoint: Path | None,
    nodes: str,
    split: str,
    window: str,
    device: str,
    output: Path | None,
    override: tuple[str, ...],
):
    """Generate forecast plots from checkpoint with flexible node selection.

    \b
    Node selection strategies (--nodes):
      random:N    - Sample N random nodes (fast, no eval needed)
      quartile:N  - N nodes per MAE quartile (requires mini-eval)
      best:N      - N best-performing nodes (requires mini-eval)
      worst:N     - N worst-performing nodes (requires mini-eval)

    \b
    Examples:
      # Quick random sample (no eval)
      uv run cli plot forecasts --checkpoint model.pt --nodes random:8

      # Quartile-based from test split
      uv run cli plot forecasts -e myexp -r 12345 --nodes quartile:2 --split test

      # Best/worst performers on validation
      uv run cli plot forecasts --checkpoint model.pt --nodes best:3 --nodes worst:3
    """
    try:
        if experiment and run:
            from utils.run_discovery import (
                get_eval_output_dir,
                resolve_checkpoint_path,
                list_available_runs,
            )

            if checkpoint is None:
                try:
                    checkpoint = resolve_checkpoint_path(
                        experiment_name=experiment,
                        run_id=run,
                    )
                except FileNotFoundError as e:
                    click.echo(f"❌ Checkpoint not found: {e}", err=True)
                    click.echo(
                        f"\n{list_available_runs(experiment_name=experiment)}", err=True
                    )
                    raise click.ClickException("Cannot resolve checkpoint path")

            if output is None:
                eval_dir = get_eval_output_dir(experiment_name=experiment, run_id=run)
                output = eval_dir / f"{split}_forecasts.png"

        if checkpoint is None:
            raise click.ClickException(
                "Must provide --checkpoint or both --experiment and --run"
            )

        logging.getLogger("zarr").setLevel(logging.WARNING)
        logging.getLogger("numcodecs").setLevel(logging.WARNING)

        from evaluation.epiforecaster_eval import (
            build_loader_from_config,
            evaluate_loader,
            generate_forecast_plots,
            get_loss_from_config,
            load_model_from_checkpoint,
            select_nodes_by_loss,
        )

        click.echo(f"Loading checkpoint: {checkpoint}")
        overrides = list(override) if override else None
        model, config, _ckpt = load_model_from_checkpoint(
            checkpoint, device=device, overrides=overrides
        )
        click.echo(
            f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)"
        )

        click.echo(f"Building {split} loader...")
        loader, region_embeddings = build_loader_from_config(
            config, split=split, device=device
        )
        click.echo(f"Dataset: {len(loader.dataset)} samples")

        strategy, k = _parse_nodes_option(nodes)

        node_mae: dict[int, float] | None = None
        if strategy in ("quartile", "best", "worst"):
            click.echo(f"Running mini-eval for {strategy} selection...")
            criterion = get_loss_from_config(
                config.training.loss,
                data_config=config.data,
                forecast_horizon=config.model.forecast_horizon,
            )
            _, _, node_mae = evaluate_loader(
                model=model,
                loader=loader,
                criterion=criterion,
                horizon=config.model.forecast_horizon,
                device=next(model.parameters()).device,
                region_embeddings=region_embeddings,
                split_name=split.capitalize(),
            )

        if strategy == "random":
            dataset = loader.dataset
            all_nodes = list(dataset._valid_window_starts_by_node.keys())
            import random

            random.seed(42)
            selected = random.sample(all_nodes, min(k, len(all_nodes)))
            node_groups = {"Random": selected}
        else:
            node_groups = select_nodes_by_loss(
                node_mae=node_mae or {},
                strategy=strategy,
                k=k,
                samples_per_group=k,
            )

        total_nodes = sum(len(v) for v in node_groups.values())
        click.echo(f"Selected {total_nodes} nodes via {strategy}:")
        for group_name, group_nodes in node_groups.items():
            click.echo(f"  {group_name}: {len(group_nodes)} nodes")

        click.echo(f"Generating plots (window={window})...")
        generate_forecast_plots(
            model=model,
            loader=loader,
            node_groups=node_groups,
            window=window,
            output_path=output,
        )

        if output is not None:
            click.echo(f"Saved figure to: {output}")

        if wandb.run is not None:
            wandb.finish()

    except Exception as exc:
        click.echo(f"❌ Plot generation failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


def _parse_nodes_option(nodes: str) -> tuple[str, int]:
    """Parse --nodes option like 'random:5' into (strategy, k)."""
    parts = nodes.split(":")
    if len(parts) != 2:
        raise click.ClickException(
            f"Invalid --nodes format: '{nodes}'. Expected 'strategy:N' (e.g., 'random:5')"
        )
    strategy, k_str = parts[0].lower(), parts[1]
    valid_strategies = ("random", "quartile", "best", "worst")
    if strategy not in valid_strategies:
        raise click.ClickException(
            f"Invalid strategy: '{strategy}'. Valid: {', '.join(valid_strategies)}"
        )
    try:
        k = int(k_str)
        if k < 1:
            raise ValueError("k must be positive")
    except ValueError as e:
        raise click.ClickException(f"Invalid count in --nodes: {k_str}. {e}") from e
    return strategy, k


def _format_eval_summary(loss: float, metrics: dict) -> str:
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
    lines = ["Metric  Value"]
    lines.append("------  ------")
    for name, value in rows:
        lines.append(f"{name:<6}  {value}")
    return "\n".join(lines)


@train_cli.command("regions")
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
    output_dir: Path | None,  # noqa: ARG001
    no_cluster: bool,  # noqa: ARG001
    dry_run: bool,
):
    """Train the Region2Vec-style region embedder via configuration."""
    try:
        region_config = RegionTrainerConfig.from_file(config)
        if region_config.env == "mn5" and not is_slurm_cluster():
            raise click.ClickException(
                "Production training configs (env=mn5) must run on SLURM."
            )

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


@train_cli.command("epiforecaster")
@click.option("--config", required=True, help="Path to training configuration file")
@click.option("--model-id", default="", help="Model id for logging/checkpoints")
@click.option("--resume", is_flag=True, help="Resume training from a saved checkpoint")
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Maximum number of batches to run (for smoke testing)",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values using dotted keys (e.g., training.learning_rate=0.001)",
)
def train_epiforecaster(
    config: str,
    model_id: str,
    resume: bool,
    max_batches: int | None,
    override: tuple[str, ...],
):
    """Train EpiForecaster model."""
    _run_forecaster_training(config, model_id, resume, max_batches, override)


def _run_forecaster_training(
    config: str,
    model_id: str,
    resume: bool,
    max_batches: int | None,
    overrides: tuple[str, ...] = (),
) -> None:
    """Execute the original unified forecaster training workflow."""
    try:
        override_list = list(overrides)

        if model_id:
            override_list.append(f"training.model_id={model_id}")
        if resume:
            override_list.append("training.resume=true")
        if max_batches is not None:
            override_list.append(f"training.max_batches={max_batches}")

        trainer_config = EpiForecasterConfig.load(
            config, overrides=override_list if override_list else None
        )
        if trainer_config.env == "mn5" and not is_slurm_cluster():
            raise click.ClickException(
                "Production training configs (env=mn5) must run on SLURM."
            )

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
def dataset_info(dataset: str):
    """Display information about preprocessed datasets."""
    try:
        if not dataset:
            click.echo("❌ Dataset path is required", err=True)
            return

        dataset_path = Path(dataset)
        if not dataset_path.exists():
            click.echo(f"❌ Dataset not found: {dataset}", err=True)
            return

        logger = logging.getLogger(__name__)

        # Open the zarr dataset and read basic metadata
        import zarr

        z = zarr.open(dataset_path, mode="r")

        logger.info(f"\n{'=' * 50}")
        logger.info("DATASET INFORMATION")
        logger.info(f"{'=' * 50}")
        logger.info(f"Path: {dataset_path}")

        # Print arrays info
        if hasattr(z, "arrays"):
            logger.info("\n📊 ARRAYS:")
            for name, arr in z.arrays():  # type: ignore[attr-defined]
                logger.info(f"  {name}: {arr.shape}")

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
        data_dir_path = Path(data_dir)
        if not data_dir_path.exists():
            click.echo(f"❌ Data directory not found: {data_dir}", err=True)
            return

        logger = logging.getLogger(__name__)
        logger.info(f"Scanning for datasets in: {data_dir}")

        # Find all .zarr files/directories
        zarr_files = list(data_dir_path.glob("*.zarr")) + [
            p
            for p in data_dir_path.iterdir()
            if p.is_dir() and (p / ".zarray").exists()
        ]

        if not zarr_files:
            logger.info("No datasets found.")
            return

        logger.info(f"\n{'=' * 60}")
        logger.info(f"AVAILABLE DATASETS ({len(zarr_files)} found)")
        logger.info(f"{'=' * 60}")

        for zarr_path in zarr_files:
            logger.info(f"\n📦 {zarr_path.name}")
            logger.info(f"  Path: {zarr_path}")

        logger.info(f"{'=' * 60}")

    except Exception as e:
        click.echo(f"❌ Failed to list datasets: {str(e)}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    cli()
