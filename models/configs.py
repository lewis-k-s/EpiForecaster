from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EgoGraphParams:
    """Parameters for building ego-graph datasets.

    Mirrors the ``model.params.ego_graph_params`` block in YAML configs so the
    trainer knows how to materialize the neighborhood-centric dataset variant
    used by mobility-aware models.

    Attributes:
        history_length: Number of historical steps (``L``) to include per sample.
        min_flow_threshold: Minimum mobility flow required for a neighbor edge.
        max_neighbors: Maximum number of incoming neighbors kept per ego graph.
        use_mobility: Whether to read mobility tensors from the dataset.
        device: Torch device string used while constructing the dataset.
    """

    history_length: int = 14
    min_flow_threshold: float = 10.0
    max_neighbors: int = 20
    use_mobility: bool = True
    device: str = "cpu"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EgoGraphParams":
        if not data:
            return cls()
        return cls(
            history_length=int(data.get("history_length", data.get("L", 14))),
            min_flow_threshold=float(data.get("min_flow_threshold", 10.0)),
            max_neighbors=int(data.get("max_neighbors", 20)),
            use_mobility=bool(data.get("use_mobility", True)),
            device=str(data.get("device", "cpu"))
            if data.get("device") is not None
            else "cpu",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "history_length": self.history_length,
            "min_flow_threshold": self.min_flow_threshold,
            "max_neighbors": self.max_neighbors,
            "use_mobility": self.use_mobility,
            "device": self.device,
        }


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
    region_embeddings_path: str = ""


@dataclass
class ModelConfig:
    """Model selection plus parameter payload from the ``model`` YAML block."""

    type: ModelVariant = field(
        default_factory=lambda: ModelVariant(
            cases=True, regions=False, biomarkers=False, mobility=False
        )
    )
    params: dict[str, Any] = field(default_factory=dict)
    ego_graph_params: EgoGraphParams = field(default_factory=EgoGraphParams)

    def __post_init__(self) -> None:
        # Normalize model variant payloads (string presets or dict flags) into
        # the strongly typed ``ModelVariant`` structure expected downstream.
        self.type = ModelVariant(**self.type)

        if not isinstance(self.params, dict):
            raise TypeError(
                "model.params should be a mapping. "
                f"Received {type(self.params).__name__}"
            )

        params_ego = self.params.pop("ego_graph_params", None)
        self.ego_graph_params = self._coerce_ego_graph_params(self.ego_graph_params)

        if params_ego is not None:
            self.ego_graph_params = EgoGraphParams.from_dict(params_ego)

    @staticmethod
    def _coerce_ego_graph_params(
        value: EgoGraphParams | dict[str, Any] | None,
    ) -> EgoGraphParams:
        if isinstance(value, EgoGraphParams) or value is None:
            return value or EgoGraphParams()
        if isinstance(value, dict):
            return EgoGraphParams.from_dict(value)
        raise TypeError(
            "ego_graph_params must be a mapping or EgoGraphParams instance. "
            f"Received {type(value).__name__}"
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
    validation_split: float = 0.2
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    eval_frequency: int = 5
    eval_metrics: list[str] = field(default_factory=lambda: ["mse", "mae", "rmse"])


@dataclass
class OutputConfig:
    """Logging and checkpoint settings from the ``output`` YAML block."""

    log_dir: str = "outputs/training"
    experiment_name: str = "epiforecaster_experiment"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    save_best_only: bool = True


@dataclass
class EpiForecasterTrainerConfig:
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
    def from_file(cls, config_path: str) -> "EpiForecasterTrainerConfig":
        """Load configuration from YAML file located at ``config_path``."""

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        data_cfg = DataConfig(**config_dict.get("data", {}))
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        training_cfg = TrainingParams(**config_dict.get("training", {}))
        output_cfg = OutputConfig(**config_dict.get("output", {}))

        return cls(
            model=model_cfg,
            data=data_cfg,
            training=training_cfg,
            output=output_cfg,
        )
