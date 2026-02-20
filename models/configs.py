from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from typing import Any, cast

import yaml
from omegaconf import MISSING, DictConfig, OmegaConf

GNN_TYPES = ["gcn", "gat"]
FORECASTER_HEAD_TYPES = ["transformer"]
POSITIONAL_ENCODING_TYPES = ["sinusoidal", "learned"]


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
    # Which epochs to profile. If None, profiles all epochs. Use [1, 2] to limit.
    profile_epochs: list[int] | None = None
    # Where to write profiler traces. Use "auto" to place traces inside the
    # experiment log directory so they appear alongside other artifacts.
    log_dir: str = "auto"
    record_memory: bool = True
    with_stack: bool = False


@dataclass
class LossComponentConfig:
    """Single loss component for composite objectives."""

    name: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("loss component weight must be non-negative")


@dataclass
class JointLossConfig:
    """Loss weights for joint inference training."""

    # Default balance with higher weight on low-noise signals
    w_hosp: float = 0.4
    w_deaths: float = 0.25
    w_cases: float = 0.15
    w_ww: float = 0.15
    w_sir: float = 0.05
    # Relative per-timestep weights for imputed (mask=0) supervision.
    # 0.0 disables imputed supervision; observed (mask=1) always has weight 1.0.
    ww_imputed_weight: float = 0.25
    hosp_imputed_weight: float = 0.25
    cases_imputed_weight: float = 0.25
    deaths_imputed_weight: float = 0.25
    # Cap on absolute physics residual before squaring in SIR loss.
    # Prevents occasional residual spikes from dominating gradients.
    sir_residual_clip: float = 1.0e3

    def __post_init__(self) -> None:
        for name, value in [
            ("w_ww", self.w_ww),
            ("w_hosp", self.w_hosp),
            ("w_cases", self.w_cases),
            ("w_deaths", self.w_deaths),
            ("w_sir", self.w_sir),
            ("ww_imputed_weight", self.ww_imputed_weight),
            ("hosp_imputed_weight", self.hosp_imputed_weight),
            ("cases_imputed_weight", self.cases_imputed_weight),
            ("deaths_imputed_weight", self.deaths_imputed_weight),
            ("sir_residual_clip", self.sir_residual_clip),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        lsum = sum([self.w_deaths, self.w_hosp, self.w_cases, self.w_ww, self.w_sir])
        if lsum > 1:
            print(f"WARNING: composite loss sum {lsum} > 1")


@dataclass
class LossConfig:
    """Loss configuration from ``training.loss`` YAML block."""

    name: str = "joint_inference"
    components: list[LossComponentConfig] = field(default_factory=list)
    # Joint inference loss weights (used when training with SIR + observation heads)
    joint: JointLossConfig = field(default_factory=JointLossConfig)

    def __post_init__(self) -> None:
        if isinstance(self.components, list):
            normalized: list[LossComponentConfig] = []
            for component in self.components:
                if isinstance(component, LossComponentConfig):
                    normalized.append(component)
                elif isinstance(component, (dict, DictConfig)):
                    normalized.append(LossComponentConfig(**component))
                else:
                    raise ValueError(
                        "loss.components must be LossComponentConfig or dict entries"
                    )
            self.components = normalized

        if isinstance(self.joint, (dict, DictConfig)):
            self.joint = JointLossConfig(**self.joint)


@dataclass
class CurriculumPhaseConfig:
    """Configuration for a single curriculum phase."""

    start_epoch: int
    end_epoch: int
    synth_ratio: float  # Ratio of synthetic samples (0.0 to 1.0)
    mode: str = "time_major"  # "time_major" or "node_major"
    # Sparsity filtering for progressive curriculum (optional)
    min_sparsity: float | None = None  # Minimum sparsity (0.0-1.0)
    max_sparsity: float | None = None  # Maximum sparsity (0.0-1.0)

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

        # Validate sparsity bounds
        if self.min_sparsity is not None and self.max_sparsity is not None:
            if self.min_sparsity > self.max_sparsity:
                raise ValueError(
                    f"min_sparsity ({self.min_sparsity}) > max_sparsity ({self.max_sparsity})"
                )
        for val in [self.min_sparsity, self.max_sparsity]:
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"Sparsity must be in [0.0, 1.0], got {val}")


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
class SIRPhysicsConfig:
    """SIR dynamics configuration for the physics roll-forward core."""

    dt: float = 1.0
    enforce_nonnegativity: bool = True
    enforce_mass_conservation: bool = True
    # Parameter bounds for numerical stability (prevent gradient explosion)
    beta_min: float = 0.01
    beta_max: float = 2.0
    gamma_min: float = 0.05
    gamma_max: float = 0.5
    mortality_min: float = 1e-4
    mortality_max: float = 0.05
    # Physics residual clipping at source (prevents extreme gradients)
    residual_clip: float = 1e4

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.beta_min >= self.beta_max:
            raise ValueError(
                f"beta_min ({self.beta_min}) must be < beta_max ({self.beta_max})"
            )
        if self.gamma_min >= self.gamma_max:
            raise ValueError(
                f"gamma_min ({self.gamma_min}) must be < gamma_max ({self.gamma_max})"
            )
        if self.mortality_min >= self.mortality_max:
            raise ValueError(
                f"mortality_min ({self.mortality_min}) must be < mortality_max ({self.mortality_max})"
            )
        if self.residual_clip <= 0:
            raise ValueError(
                f"residual_clip must be positive, got {self.residual_clip}"
            )


@dataclass
class ObservationHeadConfig:
    """Configuration for observation heads (wastewater, hospitalization, cases, deaths)."""

    # Kernel lengths
    kernel_length_ww: int = 14
    kernel_length_hosp: int = 21
    kernel_length_cases: int = 14
    kernel_length_deaths: int = 21

    # Learnable parameters
    learnable_kernel_ww: bool = True
    learnable_kernel_hosp: bool = True
    learnable_kernel_cases: bool = True
    learnable_kernel_deaths: bool = True
    learnable_scale_ww: bool = True
    learnable_scale_hosp: bool = True
    learnable_scale_cases: bool = True
    learnable_scale_deaths: bool = True

    # Residual connection settings
    residual_mode: str = "additive"  # "additive" | "modulation"
    residual_scale: float = 0.2  # alpha in pred = base + alpha * residual
    residual_hidden_dim: int = 32
    residual_layers: int = 2
    residual_dropout: float = 0.1

    # Observation context dimension
    obs_context_dim: int = 96

    def __post_init__(self) -> None:
        for name, value in [
            ("kernel_length_ww", self.kernel_length_ww),
            ("kernel_length_hosp", self.kernel_length_hosp),
            ("kernel_length_cases", self.kernel_length_cases),
            ("kernel_length_deaths", self.kernel_length_deaths),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.residual_scale < 0:
            raise ValueError(
                f"residual_scale must be non-negative, got {self.residual_scale}"
            )
        if self.residual_hidden_dim <= 0:
            raise ValueError(
                f"residual_hidden_dim must be positive, got {self.residual_hidden_dim}"
            )
        if self.residual_layers < 0:
            raise ValueError(
                f"residual_layers must be non-negative, got {self.residual_layers}"
            )
        if not 0 <= self.residual_dropout <= 1:
            raise ValueError(
                f"residual_dropout must be in [0, 1], got {self.residual_dropout}"
            )
        if self.obs_context_dim <= 0:
            raise ValueError(
                f"obs_context_dim must be positive, got {self.obs_context_dim}"
            )

        valid_modes = {"additive", "modulation"}
        if self.residual_mode not in valid_modes:
            raise ValueError(
                f"residual_mode must be one of {valid_modes}, got {self.residual_mode}"
            )


@dataclass
class DataConfig:
    """Dataset configuration loaded from ``data`` YAML block."""

    dataset_path: str = ""
    # Optional separate path for real data (used in curriculum training if real/synth are split)
    real_dataset_path: str = ""
    regions_data_path: str = ""
    # .pt file containing region2vec encoder model weights
    region2vec_path: str = ""
    # Minimum incoming mobility flow to include a node in the neighborhood mask.
    mobility_threshold: float = 0.0
    # Use valid_targets mask from dataset to filter target nodes
    use_valid_targets: bool = False
    # Sliding window stride for training samples
    window_stride: int = 1
    # Maximum allowed missing values by target for both history and horizon windows.
    # Preferred schema:
    #   missing_permit:
    #     biomarkers_joint: 24
    #     cases: 24
    #     hospitalizations: 24
    #     deaths: 24
    #
    missing_permit: dict[str, int] = field(default_factory=dict)
    # Dataset sample ordering: "node" (node-major) or "time" (time-major)
    sample_ordering: str = "node"
    # Log transformation for cases (used by legacy CasesPreprocessor only)
    log_scale: bool = False
    # Mobility preprocessing configuration (data already log1p-transformed from pipeline)
    mobility_clip_range: tuple[float, float] = (-8.0, 8.0)
    mobility_scale_epsilon: float = 1e-6
    # Lags for mobility-weighted case features (e.g. [1, 7, 14])
    mobility_lags: list[int] = field(default_factory=list)
    # Whether to use mobility-weighted lagged case features (imported risk)
    # Lag features are value-only (no mask/age channels) for efficiency
    use_imported_risk: bool = False
    # Chunk size for run_id dimension when loading multi-run synthetic datasets
    # Set to -1 to load all runs at once (full loading, previous behavior)
    # Chunk size for run_id dimension. Default is 1 to ensure memory-efficient loading.
    # Each chunk loads one run at a time, preventing OOM when dataset has many runs.
    # This should NOT be set to -1 (load all runs) as it causes memory issues.
    run_id_chunk_size: int = 1  # Default: 1 means load one run at a time
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

        valid_targets = {
            "biomarkers_joint",
            "cases",
            "hospitalizations",
            "deaths",
        }
        for name, permit in self.missing_permit.items():
            if name not in valid_targets:
                raise ValueError(
                    f"missing_permit has unsupported key '{name}'. "
                    f"Expected one of: {sorted(valid_targets)}"
                )
            if int(permit) < 0:
                raise ValueError(f"missing_permit['{name}'] must be non-negative")

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
        if (
            not self.run_id
            or not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string. "
                "This is required for valid_targets filtering with multi-run datasets."
            )

    def resolve_missing_permit_map(self) -> dict[str, int]:
        """Resolve per-target missing permits used for window inclusion and loss gating.

        Returns keys in dataset/loss naming:
        - cases
        - hospitalizations
        - deaths
        - wastewater
        """
        return {
            "cases": int(self.missing_permit.get("cases", 0)),
            "hospitalizations": int(self.missing_permit.get("hospitalizations", 0)),
            "deaths": int(self.missing_permit.get("deaths", 0)),
            "wastewater": int(self.missing_permit.get("biomarkers_joint", 0)),
        }


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
    include_day_of_week: bool = False
    include_holidays: bool = False
    temporal_covariates_dim: int = field(init=False)

    # -- module choices --#
    gnn_module: str = ""
    forecaster_head: str = "transformer"

    # -- forecaster head params --#
    head_d_model: int = 128
    head_n_heads: int = 4
    head_num_layers: int = 3
    head_dropout: float = 0.1
    head_positional_encoding: str = "sinusoidal"

    # -- SIR physics and observation heads --#
    sir_physics: SIRPhysicsConfig = field(default_factory=SIRPhysicsConfig)
    observation_heads: ObservationHeadConfig = field(
        default_factory=ObservationHeadConfig
    )

    # pretrained region2vec encoder model weights
    region2vec_path: str = ""

    def __post_init__(self) -> None:
        self.temporal_covariates_dim = (2 if self.include_day_of_week else 0) + (
            1 if self.include_holidays else 0
        )

        if isinstance(self.type, (dict, DictConfig)):
            self.type = ModelVariant(**self.type)  # type: ignore[arg-type]

        if isinstance(self.sir_physics, (dict, DictConfig)):
            self.sir_physics = SIRPhysicsConfig(**self.sir_physics)

        if isinstance(self.observation_heads, (dict, DictConfig)):
            self.observation_heads = ObservationHeadConfig(**self.observation_heads)

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
        if self.head_positional_encoding not in POSITIONAL_ENCODING_TYPES:
            raise ValueError(
                "head_positional_encoding must be one of "
                f"{POSITIONAL_ENCODING_TYPES}, got {self.head_positional_encoding!r}"
            )


@dataclass
class TrainingParams:
    """Trainer hyper-parameters from ``training`` YAML block."""

    epochs: int = 100
    seed: int | None = 42  # Random seed for reproducibility (None = non-deterministic)
    batch_size: int = 32
    # Preserve sample_ordering within each batch and shuffle batch order each epoch.
    # When False, training iterates through batches in deterministic dataset order.
    shuffle_train_batches: bool = True
    gradient_accumulation_steps: int = 1
    max_batches: int | None = None
    learning_rate: float = 1.0e-3
    optimizer: str = "adamw"
    weight_decay: float = 1.0e-5
    model_id: str = ""
    resume: bool = False
    scheduler_type: str = "cosine"
    warmup_steps: int = 0
    gradient_clip_value: float = 5.0
    early_stopping_patience: int | None = 10  # None = disabled
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
    val_workers: int = 4
    test_workers: int = (
        0  # Test loader workers (typically 0 since test runs once at end)
    )
    # Keep DataLoader workers alive across epochs when using fork+CUDA.
    # Disabling this can re-fork workers after CUDA init and cause hangs.
    persistent_workers: bool = True
    prefetch_factor: int | None = 4
    pin_memory: bool = True
    eval_frequency: int = 5
    eval_metrics: list[str] = field(default_factory=lambda: ["mse", "mae", "rmse"])
    # Loss configuration (single or composite)
    loss: LossConfig = field(default_factory=LossConfig)
    # forecast plotting during validation/test evaluation
    plot_forecasts: bool = True
    num_forecast_samples: int = (
        3  # Number of samples per category (best, worst, random)
    )
    grad_norm_log_frequency: int = 10
    progress_log_frequency: int = (
        10  # Log progress every N steps to reduce CPU-GPU sync
    )
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    # Precision settings for Tensor Core optimization
    enable_tf32: bool = True
    enable_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"  # Only "bfloat16" is supported
    # Parameter dtype (currently only float32 is supported)
    parameter_dtype: str = "float32"
    # Optimizer epsilon (None = use default 1e-8)
    optimizer_eps: float | None = None
    # Gradient debugging (toggleable, disabled by default for zero overhead)
    enable_gradient_debug: bool = False
    gradient_debug_log_dir: str | None = None  # Auto-set if None and enabled

    def __post_init__(self) -> None:
        if self.resume and not self.model_id:
            raise ValueError(
                "model_id must be provided when resume is True. If you are not resuming, leave model_id empty."
            )

        if isinstance(self.loss, (dict, DictConfig)):
            loss_dict = cast(dict[str, Any], self.loss)
            self.loss = LossConfig(**loss_dict)
        if isinstance(self.curriculum, (dict, DictConfig)):
            curriculum_dict = cast(dict[str, Any], self.curriculum)
            self.curriculum = CurriculumConfig(**curriculum_dict)
        if isinstance(self.profiler, (dict, DictConfig)):
            profiler_dict = cast(dict[str, Any], self.profiler)
            self.profiler = ProfilerConfig(**profiler_dict)

        valid_loss_names = {
            "smape",
            "mse",
            "mae",
            "l1",
            "mse_unscaled",
            "composite",
            "joint_inference",
        }
        loss_name = (self.loss.name or "").lower()
        if loss_name not in valid_loss_names:
            raise ValueError(
                f"Invalid loss.name: {self.loss.name}. Valid options: {sorted(valid_loss_names)}"
            )
        if loss_name == "composite":
            if not self.loss.components:
                raise ValueError("loss.components must be provided for composite loss")
            for component in self.loss.components:
                comp_name = (component.name or "").lower()
                if comp_name not in valid_loss_names - {"composite"}:
                    raise ValueError(
                        f"Invalid loss component: {component.name}. "
                        f"Valid options: {sorted(valid_loss_names - {'composite'})}"
                    )
        elif self.loss.components:
            raise ValueError(
                "loss.components is only valid when loss.name is 'composite'"
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

        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        valid_optimizers = {"adam", "adamw"}
        optimizer_name = self.optimizer.lower()
        if optimizer_name not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}. "
                f"Valid options: {sorted(valid_optimizers)}"
            )
        self.optimizer = optimizer_name

        # Validate curriculum configuration
        if self.curriculum.enabled:
            if not self.curriculum.schedule:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "curriculum.enabled=True but schedule is empty. "
                    "Curriculum training will have no effect."
                )

        # Validate precision settings
        # Only float32 parameters are supported
        if self.parameter_dtype != "float32":
            raise ValueError(
                f"Unsupported parameter_dtype: '{self.parameter_dtype}'. "
                "Only 'float32' is supported."
            )

        # Only bfloat16 autocast is supported (FP16 removed)
        if self.enable_mixed_precision:
            mp_dtype = self.mixed_precision_dtype.lower()
            if mp_dtype == "float16":
                raise ValueError(
                    "FP16 is no longer supported. "
                    "Use 'bfloat16' for autocast or disable mixed precision "
                    "by setting enable_mixed_precision=false."
                )
            if mp_dtype != "bfloat16":
                raise ValueError(
                    f"Unsupported mixed_precision_dtype: '{mp_dtype}'. "
                    "Only 'bfloat16' is supported."
                )

        # Validate optimizer epsilon if provided
        if self.optimizer_eps is not None and self.optimizer_eps <= 0:
            raise ValueError(
                f"optimizer_eps must be positive, got {self.optimizer_eps}"
            )


@dataclass
class OutputConfig:
    """Logging and checkpoint settings from the ``output`` YAML block."""

    log_dir: str = "outputs/training"
    experiment_name: str = "epiforecaster_experiment"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    save_best_only: bool = True
    wandb_project: str = "epiforecaster"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_mode: str = "online"


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
    env: str | None = None

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
            "env": raw.get("env"),
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

        def _filter_init_fields(config_class, data_dict: dict) -> dict:
            """Filter dict to only include fields that are in the class __init__."""
            # Get the set of field names that have init=True
            init_fields = {f.name for f in fields(config_class) if f.init}
            return {k: v for k, v in data_dict.items() if k in init_fields}

        # Reconstruct nested dataclass for curriculum
        training_dict = config_dict["training"].copy()
        if "curriculum" in training_dict:
            training_dict["curriculum"] = CurriculumConfig(
                **training_dict["curriculum"]
            )

        return cls(
            model=ModelConfig(**_filter_init_fields(ModelConfig, config_dict["model"])),
            data=DataConfig(**_filter_init_fields(DataConfig, config_dict["data"])),
            training=TrainingParams(
                **_filter_init_fields(TrainingParams, training_dict)
            ),
            output=OutputConfig(
                **_filter_init_fields(OutputConfig, config_dict["output"])
            ),
            env=config_dict.get("env"),
        )
