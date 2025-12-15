from dataclasses import dataclass, field
from pathlib import Path

import yaml

GNN_TYPES = ["gcn", "gat"]
FORECASTER_HEAD_TYPES = ["transformer"]


@dataclass
class ProfilerConfig:
    """Lightweight toggle for torch.profiler sampling during training."""

    enabled: bool = False
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1
    profile_batches: int | None = None
    log_dir: str = "outputs/profiler"
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
    """Dataset configuration loaded from the ``data`` YAML block."""

    dataset_path: str = ""
    regions_data_path: str = ""
    # .pt file containing the region2vec encoder model weights
    region2vec_path: str = ""


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

        assert self.forecaster_head in FORECASTER_HEAD_TYPES, (
            f"Invalid forecaster head: {self.forecaster_head}"
        )


@dataclass
class TrainingParams:
    """Trainer hyper-parameters from the ``training`` YAML block."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    scheduler_type: str = "cosine"
    gradient_clip_value: float = 1.0
    early_stopping_patience: int = 10
    val_split: float = 0.2
    test_split: float = 0.1
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    eval_frequency: int = 5
    eval_metrics: list[str] = field(default_factory=lambda: ["mse", "mae", "rmse"])
    use_tqdm: bool = True
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


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
