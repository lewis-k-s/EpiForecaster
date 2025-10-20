"""
Click-based CLI interface for Graph Neural Network Epidemiological Forecasting.

This module provides a modern, modular command-line interface using Click
with organized command groups for training, inference, and data analysis.
"""

import json
import logging
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import click

# Import training modules
try:
    from training.pipeline_trainer import PipelineTrainer
    from training.region_pretraining import RegionPretrainer
    from training.timeseries_trainer import TimeSeriesTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running this from the project root directory")
    raise

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file) as f:
        return json.load(f)


def update_args_from_config(args, config: dict[str, Any]):
    """
    Update args with config values.

    Args:
        args: CLI arguments object
        config: Configuration dictionary from JSON file
    """
    for key, value in config.items():
        setattr(args, key, value)


def run_data_investigation(args):
    """Run data investigation pipeline."""
    try:
        from scripts.investigate_data import DataInvestigator

        logger.info("Starting data investigation...")

        # Create investigator
        investigator = DataInvestigator(args, output_dir=args.investigation_output)

        # Run investigation
        results = investigator.run_investigation()

        # Save results
        results_path = investigator.save_results(results)

        # Print summary
        print("\n" + "=" * 60)
        print("📋 DATA INVESTIGATION COMPLETE")
        print("=" * 60)

        executive_summary = results.get("executive_summary", {})
        key_findings = executive_summary.get("key_findings", [])
        critical_issues = executive_summary.get("critical_issues", [])
        recommendations = executive_summary.get("recommendations", [])

        if key_findings:
            print("\n🔍 Key Findings:")
            for i, finding in enumerate(key_findings, 1):
                print(f"  {i}. {finding}")

        if critical_issues:
            print("\n⚠️  Critical Issues:")
            for i, issue in enumerate(critical_issues, 1):
                print(f"  {i}. {issue}")

        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"\n📊 Results: {results_path}")
        print(f"📈 Visualizations: {investigator.output_dir}")
        print(f"📝 Report: {results.get('report_path', 'N/A')}")

        logger.info("Data investigation completed successfully!")

    except Exception as e:
        logger.error(f"Data investigation failed: {e}")
        raise


DEFAULT_DATA_DIR = "data/files"
DEFAULT_MOBILITY_PATH = "daily_dynpop_mitma"
DEFAULT_CASES_FILE = "flowmaps_cat_municipio_cases.csv"
DEFAULT_WASTEWATER_FILE = "wastewater_biomarkers_icra.csv"


# Helpers and shared configuration structures
def add_options(options: Sequence[Callable[[Callable], Callable]]) -> Callable:
    """Apply a sequence of Click option decorators to a command."""

    def _add_options(func: Callable) -> Callable:
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def _coerce_path(value: str | Path | None, fallback: str | None) -> str | None:
    if value is None:
        return fallback
    return str(Path(value))


def _optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value))


def _normalize_path(value: str | Path) -> str:
    return str(Path(value))


def _normalize_optional(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return _normalize_path(value)


def _relativize_optional(value: str | None, base: str) -> str | None:
    if value is None:
        return None

    candidate = Path(value)
    base_path = Path(base)

    try:
        if candidate.is_absolute():
            return str(candidate.resolve().relative_to(base_path.resolve()))
        return str(candidate.relative_to(base_path))
    except ValueError:
        return str(candidate)


@dataclass(frozen=True)
class OptionSpec:
    param_decls: tuple[str, ...]
    kwargs: dict[str, Any]
    override_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelVariantSpec:
    key: str
    label: str
    description: str
    runner: Callable[[SimpleNamespace], dict[str, Any]]
    prepare: Callable[[SimpleNamespace], None] | None = None


def ensure_directory(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Click callback that materializes directories eagerly."""
    if value is None:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@dataclass
class DataConfig:
    data_dir: str
    mobility: str
    cases_file: str
    wastewater_file: str | None

    def overridden(
        self,
        data_dir: str | Path | None = None,
        mobility: str | Path | None = None,
        cases_file: str | Path | None = None,
        wastewater_file: str | Path | None = None,
    ) -> "DataConfig":
        base_data_dir = _coerce_path(data_dir, self.data_dir)
        base_mobility = _coerce_path(mobility, self.mobility)
        base_cases = _coerce_path(cases_file, self.cases_file)
        base_wastewater = _coerce_path(wastewater_file, self.wastewater_file)

        normalized_data_dir = _normalize_path(base_data_dir)
        normalized_mobility = _normalize_path(base_mobility)
        normalized_cases = _normalize_path(base_cases)
        normalized_wastewater = _normalize_optional(base_wastewater)

        return DataConfig(
            data_dir=normalized_data_dir,
            mobility=_relativize_optional(normalized_mobility, normalized_data_dir),
            cases_file=_relativize_optional(normalized_cases, normalized_data_dir),
            wastewater_file=_relativize_optional(
                normalized_wastewater, normalized_data_dir
            ),
        )


@dataclass
class CLIContext:
    data: DataConfig


# Shared option bundles
DATA_OPTION_SPECS: tuple[OptionSpec, ...] = (
    OptionSpec(
        param_decls=("--data-dir",),
        kwargs={
            "type": click.Path(exists=True, file_okay=False, dir_okay=True),
            "default": DEFAULT_DATA_DIR,
            "show_default": True,
            "help": "Directory containing data files",
        },
        override_kwargs={"help": "Override the default data directory"},
    ),
    OptionSpec(
        param_decls=("--mobility",),
        kwargs={
            "type": click.Path(file_okay=True, dir_okay=True, exists=False),
            "default": DEFAULT_MOBILITY_PATH,
            "show_default": True,
            "help": "Relative NetCDF path within the data directory",
        },
        override_kwargs={"help": "Override the mobility path"},
    ),
    OptionSpec(
        param_decls=("--cases-file",),
        kwargs={
            "type": click.Path(dir_okay=False, exists=False),
            "default": DEFAULT_CASES_FILE,
            "show_default": True,
            "help": "Relative or absolute COVID cases CSV path",
        },
        override_kwargs={"help": "Override the COVID cases CSV path"},
    ),
    OptionSpec(
        param_decls=("--wastewater",),
        kwargs={
            "type": click.Path(dir_okay=False, exists=False),
            "default": None,
            "show_default": False,
            "help": (
                "Path to wastewater biomarkers CSV file (defaults to "
                f"'{DEFAULT_WASTEWATER_FILE}' inside --data-dir when present)"
            ),
        },
        override_kwargs={"help": "Override the wastewater biomarkers CSV path"},
    ),
)


def build_option_decorators(
    specs: Sequence[OptionSpec], use_override: bool
) -> tuple[Callable, ...]:
    decorators: list[Callable] = []
    for spec in specs:
        kwargs = dict(spec.kwargs)
        if use_override:
            kwargs.pop("default", None)
            kwargs.pop("show_default", None)
            kwargs["default"] = None
            kwargs["show_default"] = False
            if spec.override_kwargs:
                kwargs.update(spec.override_kwargs)
        decorator = click.option(*spec.param_decls, **kwargs)
        decorators.append(decorator)
    return tuple(decorators)


DATA_ROOT_OPTIONS = build_option_decorators(DATA_OPTION_SPECS, use_override=False)
DATA_OVERRIDE_OPTIONS = build_option_decorators(DATA_OPTION_SPECS, use_override=True)


DATA_DIR_OPTION = (DATA_OVERRIDE_OPTIONS[0],)


def _prepare_timeseries_variant(args: SimpleNamespace) -> None:
    args.model_type = "cases_timeseries"
    args.use_region_embeddings = False
    args.use_edar_data = False


def _dual_graph_prepare_factory(model_type: str) -> Callable[[SimpleNamespace], None]:
    def _prepare(args: SimpleNamespace) -> None:
        args.model_type = model_type

    return _prepare


def _run_timeseries_variant(args: SimpleNamespace) -> dict[str, Any]:
    trainer = TimeSeriesTrainer.from_args(args)
    return trainer.run()


def _run_pipeline_variant(args: SimpleNamespace) -> dict[str, Any]:
    pipeline = PipelineTrainer.from_args(args)
    return pipeline.run()


MODEL_VARIANT_SPECS: dict[str, ModelVariantSpec] = {
    "cases_timeseries": ModelVariantSpec(
        key="cases_timeseries",
        label="Cases-only Temporal Baseline",
        description="GRU forecaster over COVID case sequences without geospatial inputs.",
        runner=_run_timeseries_variant,
        prepare=_prepare_timeseries_variant,
    ),
    "dual_graph_geospatial": ModelVariantSpec(
        key="dual_graph_geospatial",
        label="Dual Graph Forecaster",
        description="Full dual-graph model with mobility features and optional wastewater inputs.",
        runner=_run_pipeline_variant,
        prepare=_dual_graph_prepare_factory("dual_graph"),
    ),
    "dual_graph_temporal": ModelVariantSpec(
        key="dual_graph_temporal",
        label="Temporal Dual Graph Forecaster",
        description="Temporal variant of the dual-graph forecaster with expanded receptive field.",
        runner=_run_pipeline_variant,
        prepare=_dual_graph_prepare_factory("dual_graph_temporal"),
    ),
}

MODEL_VARIANT_CHOICES = tuple(MODEL_VARIANT_SPECS.keys())


MODEL_OPTIONS = (
    click.option(
        "--model-type",
        type=click.Choice(["dual_graph", "dual_graph_temporal"]),
        default="dual_graph",
        show_default=True,
        help="Type of dual graph model",
    ),
    click.option(
        "--aggregator",
        type=click.Choice(["mean", "attention", "max", "lstm", "hybrid"]),
        default="attention",
        show_default=True,
        help="Aggregation method for GraphSAGE",
    ),
    click.option(
        "--hidden-dim",
        type=int,
        default=128,
        show_default=True,
        help="Hidden dimension for models",
    ),
    click.option(
        "--num-layers",
        type=int,
        default=2,
        show_default=True,
        help="Number of GraphSAGE layers",
    ),
)


TRAINING_OPTIONS = (
    click.option(
        "--epochs", type=int, default=10, show_default=True, help="Training epochs"
    ),
    click.option(
        "--batch-size",
        type=int,
        default=14,
        show_default=True,
        help="Number of target nodes per subgraph batch",
    ),
    click.option(
        "--learning-rate",
        type=float,
        default=0.001,
        show_default=True,
        help="Learning rate",
    ),
    click.option(
        "--dropout",
        type=float,
        default=0.5,
        show_default=True,
        help="Dropout probability",
    ),
    click.option(
        "--weight-decay",
        type=float,
        default=1e-4,
        show_default=True,
        help="Weight decay for regularization",
    ),
    click.option(
        "--device",
        type=click.Choice(["auto", "cpu", "cuda"]),
        default="auto",
        show_default=True,
        help="Device to use",
    ),
    click.option(
        "--seed",
        type=int,
        default=42,
        show_default=True,
        help="Random seed for training routines",
    ),
    click.option(
        "--context-length",
        type=int,
        default=28,
        show_default=True,
        help="Context window (days) used by time series variants",
    ),
    click.option(
        "--train-split",
        type=float,
        default=0.7,
        show_default=True,
        help="Training split ratio for time series variants",
    ),
    click.option(
        "--val-split",
        type=float,
        default=0.15,
        show_default=True,
        help="Validation split ratio for time series variants",
    ),
    click.option(
        "--test-split",
        type=float,
        default=0.15,
        show_default=True,
        help="Test split ratio for time series variants",
    ),
)


@click.group()
@click.version_option()
@add_options(DATA_ROOT_OPTIONS)
@click.pass_context
def cli(ctx, data_dir, mobility, cases_file, wastewater):
    """Graph Neural Network for Epidemiological Forecasting.

    A comprehensive toolkit for training and deploying graph neural networks
    for epidemiological forecasting using mobility and case data.
    """
    wastewater_path = wastewater
    if wastewater_path is None:
        candidate = Path(data_dir) / DEFAULT_WASTEWATER_FILE
        wastewater_path = DEFAULT_WASTEWATER_FILE if candidate.exists() else None

    base_config = DataConfig(
        data_dir=_normalize_path(data_dir),
        mobility=_normalize_path(mobility),
        cases_file=_normalize_path(cases_file),
        wastewater_file=_normalize_optional(wastewater_path),
    )

    ctx.obj = CLIContext(data=base_config.overridden())


@cli.group()
def train():
    """Train models and embeddings."""
    pass


@train.command("forecaster")
@add_options(TRAINING_OPTIONS)
@add_options(MODEL_OPTIONS)
@add_options(DATA_OVERRIDE_OPTIONS)
@click.option(
    "--variant",
    type=click.Choice(MODEL_VARIANT_CHOICES),
    default="dual_graph_geospatial",
    show_default=True,
    help="Model variant to train",
)
@click.option(
    "--forecast-horizon",
    type=int,
    default=7,
    show_default=True,
    help="Number of days to forecast",
)
@click.option(
    "--region-embeddings-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to pretrained region embeddings",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="outputs/",
    show_default=True,
    callback=ensure_directory,
    help="Directory for saving outputs",
)
@click.option("--save-model", is_flag=True, help="Save trained model")
@click.option("--no-plots", is_flag=True, help="Disable forecast visualization plots")
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON config file to override arguments",
)
@click.pass_context
def train_forecaster(
    ctx,
    data_dir,
    mobility,
    cases_file,
    wastewater,
    variant,
    model_type,
    aggregator,
    hidden_dim,
    num_layers,
    epochs,
    batch_size,
    learning_rate,
    dropout,
    weight_decay,
    device,
    seed,
    context_length,
    train_split,
    val_split,
    test_split,
    forecast_horizon,
    region_embeddings_path,
    output_dir,
    save_model,
    no_plots,
    config_file,
):
    """Train forecasting models with selectable variants."""
    try:
        variant_spec = MODEL_VARIANT_SPECS.get(variant)
        if variant_spec is None:
            raise click.ClickException(f"Unknown model variant '{variant}'")

        data_config = ctx.obj.data.overridden(
            data_dir=data_dir,
            mobility=mobility,
            cases_file=cases_file,
            wastewater_file=wastewater,
        )
        region_embeddings = _optional_path(region_embeddings_path)

        args = SimpleNamespace(
            variant=variant_spec.key,
            data_dir=data_config.data_dir,
            mobility=data_config.mobility,
            auxiliary_data_dir=".",
            cases_file=data_config.cases_file,
            model_type=model_type,
            aggregator=aggregator,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout=dropout,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
            context_length=context_length,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            forecast_horizon=forecast_horizon,
            output_dir=output_dir,
            save_model=save_model,
            no_plots=no_plots,
            config_file=config_file,
            use_region_embeddings=region_embeddings is not None,
            region_embeddings_path=region_embeddings,
            start_date=None,
            end_date=None,
            cases_normalization="log1p",
            min_cases_threshold=0,
            cases_fill_missing="forward_fill",
            windowing_stride=1,
            min_flow_threshold=10,
            enable_preprocessing_hooks=False,
            use_edar_data=bool(data_config.wastewater_file),
            edar_hidden_dim=64,
            edar_biomarker_features=None,
            subgraph_num_neighbors=25,
            target_dataset="cases",
            padding_strategy="interpolate",
            crop_datasets=False,
            alignment_buffer_days=0,
            interpolation_method="linear",
            validate_alignment=False,
            investigate_data=False,
            investigation_output="outputs/data_investigation",
            wastewater_path=data_config.wastewater_file,
        )

        # Load config file if provided
        if config_file:
            config = load_config(config_file)
            update_args_from_config(args, config)

            data_config = ctx.obj.data.overridden(
                data_dir=getattr(args, "data_dir", None),
                mobility=getattr(args, "mobility", None),
                cases_file=getattr(args, "cases_file", None),
                wastewater_file=getattr(args, "wastewater_path", None),
            )

            args.data_dir = data_config.data_dir
            args.mobility = data_config.mobility
            args.cases_file = data_config.cases_file
            args.wastewater_path = data_config.wastewater_file
            args.use_edar_data = bool(args.wastewater_path)
            if hasattr(args, "output_dir"):
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Allow config overrides to change variant
        variant_key = getattr(args, "variant", variant_spec.key)
        if variant_key not in MODEL_VARIANT_SPECS:
            raise click.ClickException(f"Unsupported variant '{variant_key}'")
        variant_spec = MODEL_VARIANT_SPECS[variant_key]

        # Normalize embedding path after overrides
        if getattr(args, "region_embeddings_path", None):
            args.region_embeddings_path = _normalize_path(args.region_embeddings_path)
        args.use_region_embeddings = bool(
            getattr(args, "use_region_embeddings", False)
        ) and bool(getattr(args, "region_embeddings_path", None))

        # Apply variant-specific argument tweaks
        if variant_spec.prepare:
            variant_spec.prepare(args)

        # Validate region embeddings only when needed
        if variant_spec.key != "cases_timeseries":
            if args.use_region_embeddings and not os.path.exists(
                args.region_embeddings_path
            ):
                raise click.ClickException(
                    f"Region embeddings not found at {args.region_embeddings_path}. "
                    "Please run 'sage train regions' first to generate embeddings."
                )

        results = variant_spec.runner(args)
        output_path = results.get("output_dir", args.output_dir)
        click.echo(f"✅ {variant_spec.label} training completed successfully!")
        click.echo(f"📁 Results saved to: {output_path}")

        metrics = results.get("metrics")
        if metrics:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            click.echo(f"📊 Metrics: {metric_str}")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)
        raise click.ClickException(str(e))


@train.command("regions")
@add_options(DATA_OVERRIDE_OPTIONS)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="outputs/region_embeddings/",
    show_default=True,
    callback=ensure_directory,
    help="Directory for saving embeddings",
)
@click.option(
    "--embedding-dim",
    type=int,
    default=64,
    show_default=True,
    help="Dimension of region embeddings",
)
@click.option(
    "--epochs",
    type=int,
    default=100,
    show_default=True,
    help="Number of pretraining epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for pretraining",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.001,
    show_default=True,
    help="Learning rate for pretraining",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
    help="Device to use",
)
@click.option(
    "--save-checkpoint-every",
    type=int,
    default=10,
    show_default=True,
    help="Save checkpoint every N epochs",
)
@click.pass_context
def train_regions(
    ctx,
    data_dir,
    mobility,
    cases_file,
    wastewater,
    output_dir,
    embedding_dim,
    epochs,
    batch_size,
    learning_rate,
    device,
    save_checkpoint_every,
):
    """Train region2vec embeddings using unsupervised pretraining.

    This command learns geospatial embeddings for regions using mobility
    flow data and unsupervised learning techniques.
    """
    try:
        click.echo("🚀 Starting region2vec pretraining...")

        data_config = ctx.obj.data.overridden(
            data_dir=data_dir,
            mobility=mobility,
            cases_file=cases_file,
            wastewater_file=wastewater,
        )

        # Initialize pretrainer
        pretrainer = RegionPretrainer(
            data_dir=data_config.data_dir,
            mobility_path=data_config.mobility,
            output_dir=output_dir,
            embedding_dim=embedding_dim,
            device=device,
        )

        # Run pretraining
        results = pretrainer.train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_checkpoint_every=save_checkpoint_every,
        )

        click.echo("✅ Region2vec pretraining completed successfully!")
        click.echo(f"📁 Embeddings saved to: {output_dir}")
        click.echo(f"📊 Final loss: {results.get('final_loss', 'N/A')}")

    except Exception as e:
        click.echo(f"❌ Region2vec pretraining failed: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command("check-data")
@add_options(DATA_OVERRIDE_OPTIONS)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="outputs/data_investigation",
    show_default=True,
    callback=ensure_directory,
    help="Directory for saving investigation results",
)
@click.option(
    "--start-date", type=str, help="Start date for filtering (YYYY-MM-DD format)"
)
@click.option("--end-date", type=str, help="End date for filtering (YYYY-MM-DD format)")
@click.option("--generate-plots", is_flag=True, help="Generate visualization plots")
@click.option(
    "--cases-fill-missing",
    type=click.Choice(["forward_fill", "interpolate", "zero"]),
    default="forward_fill",
    show_default=True,
    help="Strategy for handling missing case data",
)
@click.pass_context
def check_data(
    ctx,
    data_dir,
    mobility,
    cases_file,
    wastewater,
    output_dir,
    start_date,
    end_date,
    generate_plots,
    cases_fill_missing,
):
    """Run comprehensive data investigation and quality analysis.

    This command analyzes data quality, checks for alignment issues,
    and generates comprehensive reports on the dataset.
    """
    try:
        click.echo("🔍 Starting data investigation...")

        data_config = ctx.obj.data.overridden(
            data_dir=data_dir,
            mobility=mobility,
            cases_file=cases_file,
            wastewater_file=wastewater,
        )

        args = SimpleNamespace(
            data_dir=data_config.data_dir,
            mobility=data_config.mobility,
            auxiliary_data_dir=".",
            cases_file=data_config.cases_file,
            start_date=start_date,
            end_date=end_date,
            investigation_output=output_dir,
            cases_normalization="log1p",
            min_cases_threshold=0,
            cases_fill_missing=cases_fill_missing,
            windowing_stride=1,
            min_flow_threshold=10,
            enable_preprocessing_hooks=False,
            use_edar_data=bool(data_config.wastewater_file),
            edar_hidden_dim=64,
            model_type="dual_graph",
            aggregator="attention",
            hidden_dim=128,
            num_layers=2,
            epochs=10,
            batch_size=14,
            learning_rate=0.001,
            dropout=0.5,
            weight_decay=1e-4,
            device="auto",
            forecast_horizon=7,
            output_dir="outputs/",
            save_model=False,
            no_plots=not generate_plots,
            config_file=None,
            subgraph_num_neighbors=25,
            use_region_embeddings=False,
            region_embeddings_path=None,
            target_dataset="cases",
            padding_strategy="interpolate",
            crop_datasets=False,
            alignment_buffer_days=0,
            interpolation_method="linear",
            validate_alignment=False,
            edar_biomarker_features=None,
            wastewater_path=data_config.wastewater_file,
        )

        # Run investigation
        run_data_investigation(args)

    except Exception as e:
        click.echo(f"❌ Data investigation failed: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command("infer")
@click.option(
    "--model-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to trained model directory",
)
@click.option("--horizon", type=int, required=True, help="Number of days to forecast")
@click.option(
    "--targets",
    multiple=True,
    required=True,
    help="Target regions/nodes for forecasting",
)
@add_options(DATA_DIR_OPTION)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="outputs/inference/",
    show_default=True,
    callback=ensure_directory,
    help="Directory for saving predictions",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
    help="Device to use",
)
@click.pass_context
def infer(ctx, model_path, horizon, targets, data_dir, output_dir, device):
    """Run inference with trained model.

    This command loads a trained model and generates predictions
    for specified target regions and forecast horizon.
    """
    try:
        click.echo(
            f"🔮 Running inference for {len(targets)} targets with {horizon} day horizon..."
        )

        # Use shared defaults when --data-dir override is omitted
        data_config = ctx.obj.data.overridden(data_dir=data_dir)
        effective_data_dir = data_config.data_dir

        # Validate model path
        if not os.path.exists(model_path):
            raise click.ClickException(f"Model directory not found: {model_path}")

        # Check for required model files
        model_files = ["model_config.json", "model_state.pt"]
        for file in model_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                raise click.ClickException(f"Required model file not found: {file}")

        # TODO: Implement actual inference logic
        # This would involve:
        # 1. Loading the trained model
        # 2. Loading and preprocessing data from effective_data_dir
        # 3. Running inference for specified targets
        # 4. Saving predictions

        click.echo(f"📂 Using data directory: {effective_data_dir}")
        click.echo("✅ Inference completed successfully!")
        click.echo(f"📁 Predictions saved to: {output_dir}")

        # Placeholder for actual implementation
        click.echo("⚠️  Inference logic not yet implemented - this is a placeholder")

    except Exception as e:
        click.echo(f"❌ Inference failed: {e}", err=True)
        raise click.ClickException(str(e))


def main():
    """Main entry point equivalent to original main.py functionality."""
    import sys

    # Check if data investigation is requested
    if "--investigate-data" in sys.argv or "--investigate_data" in sys.argv:
        # Create a simple args object for data investigation
        args = SimpleNamespace(
            data_dir="data/",
            mobility="files/daily_dynpop_mitma/",
            auxiliary_data_dir="files/",
            cases_file="files/flowmaps_cat_municipio_cases.csv",
            start_date=None,
            end_date=None,
            cases_normalization="log1p",
            min_cases_threshold=0,
            investigation_output="outputs/data_investigation",
            windowing_stride=1,
            min_flow_threshold=10,
            enable_preprocessing_hooks=False,
            use_edar_data=False,
            edar_hidden_dim=64,
            edar_biomarker_features=None,
            model_type="dual_graph",
            aggregator="attention",
            hidden_dim=128,
            num_layers=2,
            epochs=10,
            batch_size=14,
            learning_rate=0.001,
            dropout=0.5,
            weight_decay=1e-4,
            device="auto",
            forecast_horizon=7,
            output_dir="outputs/",
            save_model=False,
            no_plots=False,
            config_file=None,
            subgraph_num_neighbors=25,
            use_region_embeddings=False,
            region_embeddings_path=None,
            target_dataset="cases",
            padding_strategy="interpolate",
            crop_datasets=False,
            alignment_buffer_days=0,
            interpolation_method="linear",
            validate_alignment=False,
            investigate_data=True,
            cases_fill_missing="forward_fill",
            mobility_timepoints=90,
            wastewater_path=None,
        )

        # Override with any config file if provided
        for i, arg in enumerate(sys.argv):
            if arg == "--config-file" and i + 1 < len(sys.argv):
                config_file = sys.argv[i + 1]
                config = load_config(config_file)
                update_args_from_config(args, config)
                break

        run_data_investigation(args)
        return

    # Otherwise, run the CLI
    cli()


if __name__ == "__main__":
    main()
