from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from omegaconf import MISSING, DictConfig, OmegaConf

GNN_TYPES = ["gcn", "gat"]
FORECASTER_HEAD_TYPES = ["transformer"]


# Kept for backwards compatibility with old checkpoints - DO NOT USE
@dataclass
class SmoothingConfig:
    """Deprecated: Unused smoothing config for backwards compatibility only."""

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
class CurriculumPhaseConfig:
    """Configuration for a single curriculum phase."""

    start_epoch: int
    end_epoch: int
    synth_ratio: float  # Ratio of synthetic samples (0.0 to 1.0)
    mode: str  # "time_major" or "node_major"

    def __post_init__(self) -> None:
        if self.start_epoch >= self.end_epoch:
            raise ValueError(
                f"start_epoch ({self.start_epoch}) must be less than end_epoch ({self.end_epoch})"
            )
        if not 0.0 <= self.synth_ratio <= 1.0:
            raise ValueError(f"synth_ratio must be in [0, 1], got {self.synth_ratio}")
        if self.mode not in {"time_major", "node_major"}:
            raise ValueError(
                f"mode must be 'time_major' or 'node_major', got {self.mode}"
            )


@dataclass
class CurriculumConfig:
    """Curriculum training configuration from ``training.curriculum`` YAML block."""

    enabled: bool = False
    # Number of synthetic runs to sample from per epoch (1-2 recommended for locality)
    active_runs: int = 1
    # Contiguous windows per run before rotating to next run
    chunk_size: int = 512
    # How to select runs: "round_robin" or "random"
    run_sampling: str = "round_robin"
    # List of phase configs defining the curriculum schedule
    schedule: list[CurriculumPhaseConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_run_sampling = {"round_robin", "random"}
        if self.run_sampling not in valid_run_sampling:
            raise ValueError(
                f"run_sampling must be one of {valid_run_sampling}, got {self.run_sampling}"
            )
        if self.active_runs < 1:
            raise ValueError(f"active_runs must be >= 1, got {self.active_runs}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")

        # Validate schedule phases don't overlap
        for i, phase in enumerate(self.schedule):
            for j, other in enumerate(self.schedule):
                if i != j and not (
                    phase.end_epoch <= other.start_epoch
                    or phase.start_epoch >= other.end_epoch
                ):
                    raise ValueError(
                        f"Curriculum phases {i} and {j} have overlapping epoch ranges"
                    )


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
    # Dataset sample ordering: "node" (node-major) or "time" (time-major)
    sample_ordering: str = "node"
    # Log transformation for cases and biomarkers
    log_scale: bool = False
    # Mobility preprocessing configuration
    mobility_log_scale: bool = True
    mobility_clip_range: tuple[float, float] = (-8.0, 8.0)
    mobility_scale_epsilon: float = 1e-6
    # Lags for mobility-weighted case features (e.g. [1, 7, 14])
    mobility_lags: list[int] = field(default_factory=list)
    # Whether to use mobility-weighted lagged case features (imported risk)
    # Lag features are value-only (no mask/age channels) for efficiency
    use_imported_risk: bool = False
    # Chunk size for run_id dimension when loading multi-run synthetic datasets
    # Set to -1 to load all runs at once (full loading, previous behavior)
    # Set to a small value (e.g., 5) for memory-efficient processing on low-resource machines
    # Set to a larger value (e.g., 50) for better performance on high-resource machines
    run_id_chunk_size: int = -1  # Default: -1 means load all runs (no chunking)
    # Single run_id for filtering dataset (e.g., "real", "synth_run_001")
    # Required when curriculum training is disabled
    run_id: str = "real"

    def __post_init__(self) -> None:
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")

        if self.use_imported_risk and not self.mobility_lags:
            raise ValueError(
                "use_imported_risk=True requires mobility_lags to be non-empty. "
                "Specify at least one lag value (e.g., [1, 7, 14])."
            )

        if self.missing_permit < 0:
            raise ValueError("missing_permit must be non-negative")

        if len(self.mobility_clip_range) != 2:
            raise ValueError("mobility_clip_range must be a tuple of (min, max)")

        if self.mobility_scale_epsilon <= 0:
            raise ValueError("mobility_scale_epsilon must be positive")

        valid_orderings = {"node", "time"}
        if self.sample_ordering not in valid_orderings:
            raise ValueError(
                f"sample_ordering must be one of: {sorted(valid_orderings)}"
            )

        # Validate run_id is present and non-empty
        if not self.run_id or not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string. "
                "This is required for valid_targets filtering with multi-run datasets."
            )


@dataclass
class ModelConfig:
    """Model selection plus parameter payload from the ``model`` YAML block."""

    type: ModelVariant

    mobility_embedding_dim: int
    region_embedding_dim: int

    # -- seq sizes --#
    history_length: int
    forecast_horizon: int

    # -- graph params --#
    # DEPRECATED: Now uses full graphs with k-hop feature masking
    # max_neighbors is kept for config compatibility but not used
    # Neighborhood size is determined by gnn_depth
    max_neighbors: int

    # -- dimensionality --#
    # 3 variants × 4 channels (value/mask/censor/age) + 1 has_data = 13
    biomarkers_dim: int = 13
    cases_dim: int = 3  # (value, mask, age) - age = days since last measurement
    # GNN depth determines message passing depth AND feature masking radius
    # - gnn_depth=0: No masking (all nodes contribute)
    # - gnn_depth=1: 1-hop neighbors (direct neighbors only)
    # - gnn_depth=2: 2-hop neighbors (neighbors of neighbors)
    #
    # IMPLEMENTATION NOTE:
    # Full graphs are always used with complete edge connectivity. K-hop limiting
    # is achieved via FEATURE MASKING: nodes outside k-hops have their initial
    # features zeroed out. The target node is always included (never masked).
    #
    # IMPORTANT: Due to message passing over the full graph topology, the GNN's
    # computational receptive field may extend beyond k-hops. For example, with
    # gnn_depth=2, information from nodes beyond 2-hops can still influence the
    # target via intermediate propagation through unmasked nodes. This is a known
    # design choice - true k-hop receptive field would require building k-hop
    # subgraphs instead of full graphs.
    gnn_depth: int = 2
    gnn_hidden_dim: int = 32

    # -- static/temporal covariates --#
    use_population: bool = True
    population_dim: int = 1

    # -- module choices --#
    gnn_module: str = ""
    forecaster_head: str = "transformer"

    # -- forecaster head params --#
    head_d_model: int = 128
    head_n_heads: int = 4
    head_num_layers: int = 3
    head_dropout: float = 0.1

    # pretrained region2vec encoder model weights
    region2vec_path: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.type, (dict, DictConfig)):
            self.type = ModelVariant(**self.type)  # type: ignore[arg-type]

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
    gradient_accumulation_steps: int = 1
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
    # Split strategy: "node" (default, region holdouts) or "time" (temporal splits)
    split_strategy: str = "node"
    # Temporal split boundaries (YYYY-MM-DD format) when split_strategy="time"
    train_end_date: str | None = None
    val_end_date: str | None = None
    test_end_date: str | None = None
    # Curriculum training configuration
    # When enabled, uses mixed synthetic/real data sampling
    # When disabled, uses single run_id from data config
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    device: str = "auto"
    num_workers: int = 4
    val_workers: int = 0
    prefetch_factor: int | None = 4
    pin_memory: bool = True
    eval_frequency: int = 5
    eval_metrics: list[str] = field(default_factory=lambda: ["mse", "mae", "rmse"])
    # forecast plotting during validation/test evaluation
    plot_forecasts: bool = True
    num_forecast_samples: int = (
        3  # Number of samples per category (best, worst, random)
    )
    grad_norm_log_frequency: int = 5
    progress_log_frequency: int = (
        10  # Log progress every N steps to reduce CPU-GPU sync
    )
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    # Precision settings for Tensor Core optimization
    enable_tf32: bool = True
    enable_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"  # "bfloat16" or "float16"

    def __post_init__(self) -> None:
        if self.resume and not self.model_id:
            raise ValueError(
                "model_id must be provided when resume is True. If you are not resuming, leave model_id empty."
            )

        # Validate split_strategy
        valid_strategies = {"node", "time"}
        if self.split_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid split_strategy: {self.split_strategy}. "
                f"Valid options: {sorted(valid_strategies)}"
            )

        # Validate temporal split requirements
        if self.split_strategy == "time":
            import logging

            logger = logging.getLogger(__name__)
            missing_dates = []
            if self.train_end_date is None:
                missing_dates.append("train_end_date")
            if self.val_end_date is None:
                missing_dates.append("val_end_date")

            if missing_dates:
                raise ValueError(
                    f"When split_strategy='time', the following dates are required: {', '.join(missing_dates)}"
                )

            # Validate date format (YYYY-MM-DD)
            import re

            date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            for date_name, date_val in [
                ("train_end_date", self.train_end_date),
                ("val_end_date", self.val_end_date),
                ("test_end_date", self.test_end_date),
            ]:
                if date_val is not None and not date_pattern.match(date_val):
                    raise ValueError(
                        f"{date_name} must be in YYYY-MM-DD format, got: {date_val}"
                    )

            # Warn that val_split/test_split are ignored
            if self.val_split != 0.2 or self.test_split != 0.1:
                logger.warning(
                    "split_strategy='time': val_split and test_split are ignored; "
                    "using train_end_date, val_end_date, test_end_date instead"
                )

        # Validate curriculum configuration
        if self.curriculum.enabled:
            if not self.curriculum.schedule:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "curriculum.enabled=True but schedule is empty. "
                    "Curriculum training will have no effect."
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

    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    training: TrainingParams = MISSING
    output: OutputConfig = MISSING

    @classmethod
    def from_file(cls, config_path: str) -> "EpiForecasterConfig":
        """Load configuration from YAML file located at ``config_path``."""
        return cls.load(config_path)

    @classmethod
    def apply_overrides(
        cls, config: "EpiForecasterConfig", overrides: list[str]
    ) -> "EpiForecasterConfig":
        """Apply dotted-key overrides to an existing configuration object.

        Args:
            config: Existing EpiForecasterConfig instance (e.g., loaded from checkpoint).
            overrides: List of dotted-key overrides like ``["training.learning_rate=0.001"]``.

        Returns:
            EpiForecasterConfig instance with overrides applied.
        """
        from omegaconf import OmegaConf

        # Convert to dict, apply overrides with OmegaConf, then reconstruct
        config_dict = config.to_dict()
        config_cfg = OmegaConf.create(config_dict)
        override_cfg = OmegaConf.from_dotlist(overrides)
        merged_cfg = OmegaConf.merge(config_cfg, override_cfg)

        # Use OmegaConf.to_container to get a proper dict with nested objects converted
        merged_dict = OmegaConf.to_container(merged_cfg)
        assert merged_dict is not None and isinstance(merged_dict, dict)

        # Recreate nested config objects and replace top-level config
        return cls(
            model=ModelConfig(**merged_dict["model"]),
            data=DataConfig(**merged_dict["data"]),
            training=TrainingParams(**merged_dict["training"]),
            output=OutputConfig(**merged_dict["output"]),
        )

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        *,
        overrides: list[str] | None = None,
        strict: bool = True,
    ) -> "EpiForecasterConfig":
        """Load configuration from YAML file with optional dotted-key overrides.

        Args:
            config_path: Path to YAML configuration file.
            overrides: List of dotted-key overrides like ``["training.learning_rate=0.001"]``.
            strict: If True, reject unknown keys in the configuration (raises StructuredConfigError).

        Returns:
            EpiForecasterConfig instance with all validation from ``__post_init__`` applied.

        Example:
            >>> cfg = EpiForecasterConfig.load(
            ...     "configs/train_epifor_full.yaml",
            ...     overrides=["training.learning_rate=0.001", "data.log_scale=true"],
            ...     strict=True,
            ... )
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text())
        if raw is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        model_dict = (raw.get("model", {}) or {}).copy()
        if "params" in model_dict:
            params = model_dict.pop("params")
            if params:
                model_dict.update(params)

        cfg_dict = {
            "model": model_dict,
            "data": raw.get("data", {}) or {},
            "training": raw.get("training", {}) or {},
            "output": raw.get("output", {}) or {},
        }

        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.create(cfg_dict)
        merged = OmegaConf.merge(schema, file_cfg)

        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, override_cfg)

        if strict:
            OmegaConf.set_struct(merged, True)

        cfg: EpiForecasterConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

        return cfg

    def to_dict(self) -> dict:
        """Serialize configuration to a plain dictionary for YAML export."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "EpiForecasterConfig":
        """Reconstruct config from plain dict (e.g., loaded from checkpoint).

        This is the inverse of :meth:`to_dict` and is used when loading configs
        from checkpoints where they are stored as dictionaries instead of pickled
        objects. This makes checkpoints robust to config class changes.

        Args:
            config_dict: Plain dictionary with keys 'model', 'data', 'training', 'output'.

        Returns:
            Reconstructed EpiForecasterConfig instance.
        """
        # Reconstruct nested dataclass for curriculum
        training_dict = config_dict["training"].copy()
        if "curriculum" in training_dict:
            training_dict["curriculum"] = CurriculumConfig(**training_dict["curriculum"])

        return cls(
            model=ModelConfig(**config_dict["model"]),
            data=DataConfig(**config_dict["data"]),
            training=TrainingParams(**training_dict),
            output=OutputConfig(**config_dict["output"]),
        )
