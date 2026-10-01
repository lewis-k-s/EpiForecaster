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
import pandas as pd
import wandb

from data.preprocess import OfflinePreprocessingPipeline, PreprocessingConfig
from data.preprocess.region_graph_preprocessor import (
    RegionGraphPreprocessConfig,
    RegionGraphPreprocessor,
)
from models.configs import EpiForecasterConfig
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
    help="Start date (YYYY-MM-DD) for averaging OD flows; defaults to first available baseline week",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="End date (YYYY-MM-DD) for averaging OD flows; defaults to start date plus baseline week",
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


@click.group()
@click.version_option()
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging (includes model forward-pass debug logs).",
)
def eval_cli(debug: bool):
    """Evaluate trained checkpoints and generate plots.

    Direct usage: uv run eval <command>
    Via main:     uv run main eval <command>
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
def plot_cli(debug: bool):
    """Generate forecast plots from evaluation results.

    Direct usage: uv run plot <command>
    Via main:     uv run main plot <command>
    """
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=level)


# Backward compatibility: add preprocess, train, eval, plot as subgroups of main
cli.add_command(preprocess_cli, name="preprocess")
cli.add_command(train_cli, name="train")
cli.add_command(eval_cli, name="eval")
cli.add_command(plot_cli, name="plot")


@eval_cli.command("epiforecaster")
@click.option(
    "--checkpoint", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"], case_sensitive=False),
    default="val",
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for canonical evaluation artifacts.",
)
@click.option("--summary-json", type=click.Path(path_type=Path), default=None)
@click.option(
    "--granular/--no-granular",
    default=False,
    help="Write Parquet rows and example-support artifacts.",
)
@click.option("--eval-batch-size", type=click.IntRange(min=1), default=None)
@click.option(
    "--override",
    "overrides",
    multiple=True,
    help="Typed configuration override, such as training.max_batches=8.",
)
def eval_epiforecaster(
    checkpoint: Path,
    split: str,
    output: Path | None,
    summary_json: Path | None,
    granular: bool,
    eval_batch_size: int | None,
    overrides: tuple[str, ...],
) -> None:
    """Evaluate a native checkpoint without starting or resuming training."""
    from evaluation.native_checkpoint import evaluate_native_checkpoint

    output_dir = output or (
        summary_json.parent if summary_json is not None else Path("outputs/evaluation")
    )
    effective_overrides = list(overrides)
    if eval_batch_size is not None:
        effective_overrides.append(f"training.batch_size={eval_batch_size}")
    try:
        summary = evaluate_native_checkpoint(
            checkpoint_path=checkpoint,
            split=split.lower(),
            output_dir=output_dir,
            overrides=effective_overrides,
            granular=granular,
            summary_path=summary_json or output_dir / f"{split.lower()}_summary.json",
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Native {split} joint WIS: {summary['joint_wis']:.6f}")


@eval_cli.command("baselines")
@click.option("--config", required=True, help="Path to training configuration file.")
@click.option(
    "--models",
    type=click.Choice(
        ["sarima", "sarimax", "exp_smoothing", "last_observed", "varmax", "all"],
        case_sensitive=False,
    ),
    default="sarima",
    show_default=True,
    help="Baseline model family to evaluate.",
)
@click.option(
    "--split",
    type=click.Choice(["val", "test", "both"], case_sensitive=False),
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
    split: str,
    seasonal_period: int,
    output_dir: Path,
    disable_sparsity_bins: bool,
):
    """Run same-slice baseline benchmarking for EpiForecaster targets."""
    try:
        if seasonal_period <= 0:
            raise click.ClickException("--seasonal-period must be positive.")

        cfg = EpiForecasterConfig.load(config)

        from evaluation.baseline_eval import run_baseline_evaluation

        selected_models = (
            ["exp_smoothing", "last_observed", "sarima", "sarimax", "varmax"]
            if models.lower() == "all"
            else [models.lower()]
        )

        split_key = split.lower()
        split_list = ["val", "test"] if split_key == "both" else [split_key]

        for split_name in split_list:
            run_output_dir = (
                output_dir / split_name if len(split_list) > 1 else output_dir
            )
            artifacts = run_baseline_evaluation(
                config=cfg,
                models=selected_models,
                config_path=config,
                output_dir=run_output_dir,
                split=split_name,
                seasonal_period=seasonal_period,
            )

            click.echo(f"Baseline evaluation completed ({split_name}).")
            for name, path in artifacts.items():
                click.echo(f"  {name}: {path}")

            usage_path = artifacts.get("baseline_model_usage")
            if usage_path is not None and usage_path.exists():
                usage_df = pd.read_csv(usage_path)
                if not usage_df.empty:
                    click.echo("Selected model usage:")
                    for row in usage_df.itertuples(index=False):
                        click.echo(
                            f"  {row.model} -> {row.selected_model}: {int(row.count)}"
                        )
    except Exception as exc:
        click.echo(f"❌ Baseline evaluation failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


@plot_cli.command("forecasts")
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
        "Selection strategy: 'random:N', 'quartile:N', 'best:N', 'worst:N', "
        "'window_quartile:N', 'window_worst:N'. Quartile/best/worst require a "
        "mini-eval to compute per-node MAE. Window strategies use granular eval rows."
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
@click.option(
    "--granular-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Granular eval CSV to use for window-based selection.",
)
@click.option(
    "--include-sird-latents",
    is_flag=True,
    help="Also save a joint SIRHD latent trajectory figure for selected forecasts.",
)
@click.option(
    "--include-relevant-latent",
    is_flag=True,
    help=(
        "Overlay each target's relevant mechanistic source from canonical granular "
        "evaluation rows."
    ),
)
@click.option(
    "--node-metric-target",
    type=click.Choice(
        ["hospitalizations", "wastewater", "cases", "deaths"],
        case_sensitive=False,
    ),
    default="hospitalizations",
    show_default=True,
    help="Explicit target to use for node-based MAE selection.",
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
    granular_csv: Path | None,
    include_sird_latents: bool,
    include_relevant_latent: bool,
    node_metric_target: str,
):
    """Generate forecast plots from checkpoint with flexible node selection.

    \b
    Node selection strategies (--nodes):
      random:N    - Sample N random nodes (fast, no eval needed)
      quartile:N  - N nodes per MAE quartile (requires mini-eval)
      best:N      - N best-performing nodes (requires mini-eval)
      worst:N     - N worst-performing nodes (requires mini-eval)
      window_quartile:N - N exact (node, window) examples per MAE quartile from granular eval
      window_worst:N - N highest-MAE exact (node, window) examples from granular eval

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
            load_model_from_checkpoint,
            select_nodes_by_loss,
        )
        from evaluation.selection import (
            load_window_selection_specs_from_granular,
            select_windows_by_loss,
            select_worst_windows_by_loss,
        )
        from evaluation.losses import get_loss_from_config
        from data.latent_supervision import latent_scales_from_checkpoint

        click.echo(f"Loading checkpoint: {checkpoint}")
        overrides = list(override) if override else None
        model, config, loaded_checkpoint = load_model_from_checkpoint(
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

        node_mae_by_target: dict[str, dict[int, float]] | None = None
        window_groups = None
        if strategy in ("quartile", "best", "worst"):
            click.echo(f"Running mini-eval for {strategy} selection...")
            criterion = get_loss_from_config(
                config.training.loss,
                latent_scales=latent_scales_from_checkpoint(loaded_checkpoint),
                data_config=config.data,
                observation_heads_config=config.model.observation_heads,
                input_window_length=config.model.input_window_length,
                forecast_horizon=config.model.forecast_horizon,
            )
            _, _, node_mae_by_target = evaluate_loader(
                model=model,
                loader=loader,
                criterion=criterion,
                horizon=config.model.forecast_horizon,
                device=next(model.parameters()).device,
                region_embeddings=region_embeddings,
                split_name=split.capitalize(),
                node_metrics_target=node_metric_target,
                relevant_source_mapping=(
                    config.model.observation_heads.relevant_source_mapping()
                ),
            )

        if strategy == "random":
            dataset = loader.dataset
            all_nodes = list(dataset._valid_window_starts_by_node.keys())
            import random

            random.seed(42)
            selected = random.sample(all_nodes, min(k, len(all_nodes)))
            node_groups = {"Random": selected}
        elif strategy in ("window_quartile", "window_worst"):
            if granular_csv is None:
                granular_csv = (
                    checkpoint.parent.parent
                    / config.output.resolve_granular_eval_filename(split=split)
                )
            if not granular_csv.exists():
                raise click.ClickException(
                    f"{strategy} selection requires granular eval rows, not found: {granular_csv}"
                )
            click.echo(f"Loading granular selection candidates from: {granular_csv}")
            window_specs = load_window_selection_specs_from_granular(
                granular_csv=granular_csv,
                split=split,
            )
            if not window_specs:
                raise click.ClickException(
                    f"No granular window candidates found in {granular_csv}"
                )
            if strategy == "window_worst":
                window_groups = select_worst_windows_by_loss(
                    window_specs=window_specs,
                    k=k,
                )
            else:
                window_groups = select_windows_by_loss(
                    window_specs=window_specs,
                    samples_per_group=k,
                )
            node_groups = None
        else:
            node_groups = select_nodes_by_loss(
                node_mae=(node_mae_by_target or {}).get(node_metric_target, {}),
                target_name=node_metric_target,
                strategy=strategy,
                k=k,
                samples_per_group=k,
            )

        if window_groups is not None:
            total_nodes = sum(len(v) for v in window_groups.values())
            click.echo(f"Selected {total_nodes} node-window examples via {strategy}:")
            for group_name, group_specs in window_groups.items():
                click.echo(f"  {group_name}: {len(group_specs)} windows")
        else:
            assert node_groups is not None
            total_nodes = sum(len(v) for v in node_groups.values())
            click.echo(f"Selected {total_nodes} nodes via {strategy}:")
            for group_name, group_nodes in node_groups.items():
                click.echo(f"  {group_name}: {len(group_nodes)} nodes")

        click.echo(f"Generating plots (window={window})...")
        if include_relevant_latent:
            if granular_csv is None:
                granular_csv = (
                    checkpoint.parent.parent
                    / config.output.resolve_granular_eval_filename(split=split)
                )
            if not granular_csv.exists():
                raise click.ClickException(
                    "Relevant-latent overlays require canonical granular eval rows, "
                    f"not found: {granular_csv}"
                )
        generate_forecast_plots(
            model=model,
            loader=loader,
            node_groups=node_groups,
            window_groups=window_groups,
            window=window,
            output_path=output,
            include_sird_latents=include_sird_latents,
            include_relevant_latent=include_relevant_latent,
            granular_eval_path=granular_csv,
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
    valid_strategies = (
        "random",
        "quartile",
        "best",
        "worst",
        "window_quartile",
        "window_worst",
    )
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


@plot_cli.command("granular-compare")
@click.option(
    "--baseline",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Baseline granular eval artifact (Parquet or CSV).",
)
@click.option(
    "--candidate",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Candidate granular eval artifact (Parquet or CSV).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory where paired delta tables and plots will be written.",
)
@click.option(
    "--min-join-coverage",
    type=float,
    default=1.0,
    show_default=True,
    help="Minimum required matched-row coverage for each input artifact.",
)
@click.option(
    "--rolling-window",
    type=int,
    default=7,
    show_default=True,
    help="Rolling window size for time-series uplift plots.",
)
def plot_granular_compare(
    baseline: Path,
    candidate: Path,
    output_dir: Path,
    min_join_coverage: float,
    rolling_window: int,
) -> None:
    """Compare two granular eval artifacts and render paired uplift artifacts."""
    try:
        from dataviz.granular_comparison import compare_granular_csvs

        artifacts = compare_granular_csvs(
            baseline_csv=baseline,
            candidate_csv=candidate,
            output_dir=output_dir,
            min_join_coverage=min_join_coverage,
            rolling_window=rolling_window,
        )

        click.echo(
            "Granular comparison complete: "
            f"{artifacts['matched_rows']} matched rows "
            f"({artifacts['baseline_rows']} baseline, {artifacts['candidate_rows']} candidate)"
        )
        for name, path in artifacts["tables"].items():
            click.echo(f"  {name}: {path}")
        for name, path in artifacts["plots"].items():
            click.echo(f"  {name}: {path}")
    except Exception as exc:
        click.echo(f"❌ Granular comparison failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


def _format_eval_summary(loss: float, metrics: dict) -> str:
    def _fmt(value: float | None) -> str:
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.6f}"

    rows = [
        ("Loss", _fmt(loss)),
        ("Joint Obs Loss", _fmt(metrics.get("mae"))),
        ("Hosp MAE", _fmt(metrics.get("mae_hosp_log1p_per_100k"))),
        ("WW MAE", _fmt(metrics.get("mae_ww_log1p_per_100k"))),
        ("Cases MAE", _fmt(metrics.get("mae_cases_log1p_per_100k"))),
        ("Deaths MAE", _fmt(metrics.get("mae_deaths_log1p_per_100k"))),
    ]
    lines = ["Metric  Value"]
    lines.append("------  ------")
    for name, value in rows:
        lines.append(f"{name:<6}  {value}")
    return "\n".join(lines)


def _json_metric_value(value: object) -> object:
    """Convert scalar or structured evaluation metrics to JSON-native values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metric_value(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_metric_value(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_metric_value(item())
    raise TypeError(f"Unsupported evaluation metric value: {type(value).__name__}")


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
        if epochs is not None:
            region_config.training.epochs = epochs
            region_config.training.train_until_plateau = False
        if device and device != "auto":
            region_config.training.device = device
        if output_dir is not None:
            region_config.output.output_dir = output_dir
        if no_cluster:
            region_config.clustering.enabled = False

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
@click.option(
    "--resume",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to checkpoint to resume training from",
)
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
    resume: Path | None,
    max_batches: int | None,
    override: tuple[str, ...],
):
    """Train EpiForecaster model."""
    _run_forecaster_training(config, resume, max_batches, override)


def _run_forecaster_training(
    config: str,
    resume: Path | None,
    max_batches: int | None,
    overrides: tuple[str, ...] = (),
) -> None:
    """Execute the native heterogeneous forecaster."""
    try:
        override_list = list(overrides)

        if resume is not None:
            override_list.append(f"training.resume_checkpoint_path={resume}")
        if max_batches is not None:
            override_list.append(f"training.max_batches={max_batches}")

        trainer_config = EpiForecasterConfig.load(
            config, overrides=override_list if override_list else None
        )
        if trainer_config.env == "mn5" and not is_slurm_cluster():
            raise click.ClickException(
                "Production training configs (env=mn5) must run on SLURM."
            )

        _run_native_forecaster_training(trainer_config, config)

    except Exception as exc:  # pragma: no cover - CLI handles presentation
        click.echo(f"❌ Training failed: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        raise click.ClickException(str(exc)) from exc


def _run_native_forecaster_training(
    trainer_config: EpiForecasterConfig,
    config_path: str,
) -> None:
    """Execute source-native clinical and EDAR training.

    This worktree uses the heterogeneous model only.
    """
    from training.native_trainer import NativeTrainer

    logger = logging.getLogger(__name__)
    logger.info(f"Native-support mode enabled; loading config from: {config_path}")
    logger.info(f"Native root: {trainer_config.native.root}")

    trainer = NativeTrainer(trainer_config)
    results = trainer.run()

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ NATIVE TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"{'=' * 60}")
    for epoch, losses in enumerate(results["train_loss_trace"]):
        logger.info(
            f"Epoch {epoch + 1} train losses: {['%.6f' % loss for loss in losses]}"
        )
    selection = results.get("best_val_selection_score")
    if selection is not None:
        logger.info(
            "Best validation selection: "
            f"{results['best_val_selection_metric']}={selection:.6f}"
        )
    joint = results.get("best_val_joint_loss")
    if joint is not None:
        logger.info(f"Validation joint loss at selected checkpoint: {joint:.6f}")
    else:
        logger.info("Validation loss: n/a")
    logger.info(f"Model root: {results['model_root']}")
    logger.info(f"Model parameters: {results['model_info']['parameters']:,}")
    logger.info(f"Wall time: {results['wall_time_s']:.1f}s")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    cli()
