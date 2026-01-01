from dataclasses import dataclass, field
from pathlib import Path

import yaml

GNN_TYPES = ["gcn", "gat"]
FORECASTER_HEAD_TYPES = ["transformer"]


@dataclass
class SmoothingConfig:
    """Temporal smoothing configuration for case data."""

    enabled: bool = False
    window: int = 5
    smoothing_type: str = "none"

    def __post_init__(self) -> None:
        valid_types = {"none", "rolling_mean", "rolling_sum"}
        if self.smoothing_type not in valid_types:
            raise ValueError(
                f"Invalid smoothing_type: {self.smoothing_type}. "
                f"Valid options: {sorted(valid_types)}"
            )
        if self.window <= 0:
            raise ValueError("smoothing.window must be positive")


@dataclass
class ProfilerConfig:
    """Lightweight toggle for torch.profiler sampling during training."""

    enabled: bool = False
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1
    # Optional cap on the number of *training* batches to profile at the start of
    # each epoch. When reached, the profiler is shut off and training continues.
    profile_batches: int | None = None
    # Where to write profiler traces. Use "auto" to place traces inside the
    # TensorBoard run directory so they appear alongside scalars for that run.
    log_dir: str = "auto"
    record_memory: bool = True
    with_stack: bool = False


@dataclass
class ModelVariant:
    cases: bool = field(default=True)
    regions: bool = field(default=False)
    biomarkers: bool = field(default=False)
    mobility: bool = field(default=False)


@dataclass
class DataConfig:
    """Dataset configuration loaded from ``data`` YAML block."""

    dataset_path: str = ""
    regions_data_path: str = ""
    # .pt file containing region2vec encoder model weights
    region2vec_path: str = ""
    # Minimum incoming mobility flow to include a node in the neighborhood mask.
    mobility_threshold: float = 0.0
    # Use valid_targets mask from dataset to filter target nodes
    use_valid_targets: bool = False
    # Sliding window stride for training samples
    window_stride: int = 1
    # Maximum allowed missing values in a history window
    missing_permit: int = 0
    # Temporal smoothing configuration for case data
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    # Log transformation for cases and biomarkers
    log_scale: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.smoothing, dict):
            self.smoothing = SmoothingConfig(**self.smoothing)

        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")

        if self.missing_permit < 0:
            raise ValueError("missing_permit must be non-negative")


@dataclass
class ModelConfig:
    """Model selection plus parameter payload from the ``model`` YAML block."""

    type: ModelVariant

    biomarkers_dim: int
    cases_dim: int
    mobility_embedding_dim: int
    region_embedding_dim: int

    # -- seq sizes --#
    history_length: int
    forecast_horizon: int

    # -- graph params --#
    max_neighbors: int
    gnn_depth: int = 2

    # -- static/temporal covariates --#
    use_population: bool = True
    population_dim: int = 1

    # -- module choices --#
    gnn_module: str = ""
    forecaster_head: str = "transformer"

    # pretrained region2vec encoder model weights
    region2vec_path: str = ""

    def __post_init__(self) -> None:
        assert isinstance(self.type, dict), "type must be a dictionary"
        self.type = ModelVariant(**self.type)

        if self.type.mobility:
            if not self.gnn_module:
                raise ValueError("Mobility is enabled but GNN module is not specified")
            if self.gnn_module not in GNN_TYPES:
                raise ValueError(f"Invalid GNN module: {self.gnn_module}")

        if self.use_population and self.population_dim <= 0:
            raise ValueError(
                "population_dim must be positive when use_population is True"
            )

        assert self.forecaster_head in FORECASTER_HEAD_TYPES, (
            f"Invalid forecaster head: {self.forecaster_head}"
        )


@dataclass
class TrainingParams:
    """Trainer hyper-parameters from ``training`` YAML block."""

    epochs: int = 100
    batch_size: int = 32
    max_batches: int | None = None
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    model_id: str = ""
    resume: bool = False
    scheduler_type: str = "cosine"
    gradient_clip_value: float = 1.0
    early_stopping_patience: int = 10
    nan_loss_patience: int | None = None
    val_split: float = 0.2
    test_split: float = 0.1
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    eval_frequency: int = 5
    eval_metrics: list[str] = field(default_factory=lambda: ["mse", "mae", "rmse"])
    # forecast plotting during validation/test evaluation
    plot_forecasts: bool = True
    num_forecast_samples: int = (
        3  # Number of samples per category (best, worst, random)
    )
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self) -> None:
        if self.resume and not self.model_id:
            raise ValueError(
                "model_id must be provided when resume is True. If you are not resuming, leave model_id empty."
            )


@dataclass
class OutputConfig:
    """Logging and checkpoint settings from the ``output`` YAML block."""

    log_dir: str = "outputs/training"
    experiment_name: str = "epiforecaster_experiment"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    save_best_only: bool = True


@dataclass
class EpiForecasterConfig:
    """Structured configuration mirroring the training YAML schema.

    The YAML loader (`EpiForecasterTrainerConfig.from_file`) hydrates each block into the
    matching dataclass:

    - ``data``      -> :class:`DataConfig`
    - ``model``     -> :class:`ModelConfig`
    - ``training``  -> :class:`TrainingParams`
    - ``output``    -> :class:`OutputConfig`

    For backward compatibility, properties expose legacy flat attributes such as
    ``model_type`` or ``learning_rate`` so the remainder of the trainer can stay
    simple while we retain a typed, well-documented configuration surface.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_file(cls, config_path: str) -> "EpiForecasterConfig":
        """Load configuration from YAML file located at ``config_path``."""

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        data_cfg = DataConfig(**config_dict.get("data", {}))

        # Handle nested model config structure (params may be nested)
        model_dict = config_dict.get("model", {}).copy()
        if "params" in model_dict:
            params = model_dict.pop("params")
            model_dict.update(params)

        model_cfg = ModelConfig(**model_dict)
        training_dict = config_dict.get("training", {}).copy()
        profiler_dict = training_dict.pop("profiler", {})
        training_cfg = TrainingParams(
            **training_dict, profiler=ProfilerConfig(**profiler_dict)
        )
        output_cfg = OutputConfig(**config_dict.get("output", {}))

        return cls(
            model=model_cfg,
            data=data_cfg,
            training=training_cfg,
            output=output_cfg,
        )
