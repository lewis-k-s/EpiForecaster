"""Checkpoint restore: read a trainer checkpoint, build the model, load weights.

Every path that puts saved weights into an ``EpiForecaster`` uses this module:
trainer resume, trainer init checkpoint, trainer best checkpoint for final
evaluation, and evaluation loading. The legacy-checkpoint migrations therefore
run in one fixed order, and a model built for evaluation is the model the
trainer builds from the same config.

Weights always load into the uncompiled model. Checkpoints written by a
compiled model carry a ``_orig_mod.`` key prefix, which restore removes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from models.configs import (
    EpiForecasterConfig,
    ModelConfig,
    ResidualScheduleConfig,
    resolve_residual_schedule_phase,
)
from models.epiforecaster import EpiForecaster
from utils.device import resolve_device
from utils.precision_policy import validate_old_checkpoint_compatible

logger = logging.getLogger(__name__)

_REQUIRED_CHECKPOINT_KEYS = ("model_state_dict", "config")
_COMPILE_PREFIX = "_orig_mod."
_MOBILITY_INPUT_WEIGHT_KEYS = (
    "mobility_gnn.layers.0.lin.weight",
    "mobility_gnn.skips.0.weight",
)
_DISTRIBUTION_PARAMETER_MARKERS = (
    ".log_distribution_scale",
    ".distribution_scale_proj.",
)
_DISABLED_RESIDUAL_SCHEDULE_STATE = {
    "enabled": False,
    "phase_index": None,
    "phase_start_epoch": None,
    "phase_end_epoch": None,
    "effective_max_log_location_fraction": None,
}


@dataclass(frozen=True)
class Checkpoint:
    """A trainer checkpoint read from disk.

    ``payload`` is the saved dictionary. It always has ``model_state_dict``
    and ``config``; the other keys depend on the run that saved it.
    """

    path: Path
    payload: dict[str, Any]

    def config(self) -> EpiForecasterConfig:
        """Parse the saved config through the current config schema."""
        raw = self.payload["config"]
        if isinstance(raw, dict):
            return EpiForecasterConfig.from_dict(raw)
        if isinstance(raw, EpiForecasterConfig):
            # Older checkpoints pickled the config object. Rehydrate it so
            # config blocks added since then get their defaults.
            return EpiForecasterConfig.from_dict(raw.to_dict())
        raise ValueError(
            f"Checkpoint config has invalid type: {type(raw).__name__}. "
            "Expected EpiForecasterConfig or dict."
        )


@dataclass(frozen=True)
class RestoreReport:
    """What ``restore_weights`` changed to fit a checkpoint to the model.

    ``optimizer_state_compatible`` is false when a model parameter was padded
    or initialized from the model's defaults: optimizer state saved for the
    old parameters no longer matches.
    """

    migrated_keys: tuple[tuple[str, str], ...]
    initialized_keys: tuple[str, ...]
    padded_keys: tuple[str, ...]
    optimizer_state_compatible: bool


def read_checkpoint(path: Path | str) -> Checkpoint:
    """Read a trainer checkpoint and check that it has weights and a config."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {checkpoint_path}")
    missing_keys = [key for key in _REQUIRED_CHECKPOINT_KEYS if key not in payload]
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}. "
            "This checkpoint may be from an incompatible version or corrupted."
        )
    return Checkpoint(path=checkpoint_path, payload=payload)


def build_model(model_config: ModelConfig, *, device: torch.device) -> EpiForecaster:
    """Build the EpiForecaster that ``model_config`` describes."""
    return EpiForecaster(
        variant_type=model_config.type,
        sir_physics=model_config.sir_physics,
        init_weights=model_config.init_weights,
        observation_heads=model_config.observation_heads,
        temporal_input_dim=model_config.cases_dim,
        biomarkers_dim=model_config.biomarkers_dim,
        region_embedding_dim=model_config.region_embedding_dim,
        mobility_embedding_dim=model_config.mobility_embedding_dim,
        gnn_depth=model_config.gnn_depth,
        sequence_length=model_config.input_window_length,
        forecast_horizon=model_config.forecast_horizon,
        use_population=model_config.use_population,
        population_dim=model_config.population_dim,
        device=device,
        gnn_module=model_config.gnn_module,
        gnn_hidden_dim=model_config.gnn_hidden_dim,
        head_d_model=model_config.head_d_model,
        head_n_heads=model_config.head_n_heads,
        head_num_layers=model_config.head_num_layers,
        head_dropout=model_config.head_dropout,
        head_positional_encoding=model_config.head_positional_encoding,
        temporal_covariates_dim=model_config.temporal_covariates_dim,
        vaccination_dim=model_config.vaccination_dim,
        strict=model_config.strict,
    )


def restore_weights(model: torch.nn.Module, checkpoint: Checkpoint) -> RestoreReport:
    """Load the checkpoint weights into the uncompiled ``model``.

    Legacy checkpoints are fitted to the current model in this order: remove
    the compile prefix, pad mobility input weights, rename frozen clinical
    Gamma kernels, seed missing Gamma metadata, and initialize missing
    distribution parameters. The load is strict. Checkpoints with FP16
    precision state and models with non-float32 parameters are rejected.
    """
    validate_old_checkpoint_compatible(checkpoint.payload)
    model_state = model.state_dict()
    if any(key.startswith(_COMPILE_PREFIX) for key in model_state):
        raise ValueError("restore_weights needs the uncompiled model")
    saved_config = checkpoint.payload["config"]

    state = _strip_compile_prefix(checkpoint.payload["model_state_dict"])
    state, padded_keys = _pad_legacy_mobility_input_weights(state, model_state)
    state, migrated_keys = _migrate_legacy_frozen_clinical_gamma_keys(
        state, model_state, saved_config
    )
    state, metadata_keys = _initialize_missing_frozen_gamma_metadata(
        state, model_state, saved_config
    )
    state, distribution_keys = _initialize_missing_distribution_parameters(
        state, model_state
    )
    model.load_state_dict(state)

    non_fp32 = sorted(
        {str(p.dtype) for p in model.parameters() if p.dtype != torch.float32}
    )
    if non_fp32:
        raise ValueError(
            f"Checkpoint dtype mismatch: model has {non_fp32} parameters, but "
            "only float32 parameters are supported."
        )

    initialized_keys = metadata_keys + distribution_keys
    parameter_names = {name for name, _ in model.named_parameters()}
    report = RestoreReport(
        migrated_keys=migrated_keys,
        initialized_keys=initialized_keys,
        padded_keys=padded_keys,
        optimizer_state_compatible=parameter_names.isdisjoint(
            initialized_keys + padded_keys
        ),
    )
    if migrated_keys:
        logger.info(
            "Migrated %d legacy frozen clinical Gamma kernel keys", len(migrated_keys)
        )
    if distribution_keys:
        logger.info(
            "Initialized %d distribution parameters missing from legacy checkpoint",
            len(distribution_keys),
        )
    return report


def restore_residual_budget(model: EpiForecaster, checkpoint: Checkpoint) -> None:
    """Restore the checkpointed residual log-location budget fail-closed.

    The budget buffer is not persistent, so it comes from the
    ``residual_schedule_state`` provenance, which must agree with the
    checkpoint's own schedule at its saved epoch. A checkpoint without a
    schedule block predates residual schedules; it keeps the unrestricted
    residual path only when it also has no schedule provenance.
    """
    payload = checkpoint.payload
    schedule_dict = _saved_residual_schedule(payload["config"])
    saved_state = payload.get("residual_schedule_state")
    if schedule_dict is None:
        if saved_state is not None:
            raise ValueError(
                "Legacy checkpoint carries residual schedule provenance "
                f"without a residual schedule: {saved_state!r}"
            )
        return

    schedule = ResidualScheduleConfig(**schedule_dict)
    if not schedule.enabled:
        if saved_state is not None and saved_state != _DISABLED_RESIDUAL_SCHEDULE_STATE:
            raise ValueError(
                "Checkpoint carries a disabled residual schedule with "
                f"contradictory provenance: {saved_state!r}"
            )
        return

    if saved_state is None:
        raise ValueError(
            "Checkpoint was trained with an enabled residual schedule but "
            "carries no residual_schedule_state provenance; refusing to "
            "use an unknown residual budget."
        )
    saved_epoch = int(payload.get("epoch", -1))
    expected_index, expected_phase = resolve_residual_schedule_phase(
        schedule, saved_epoch
    )
    expected_state = {
        "enabled": expected_phase is not None,
        "phase_index": expected_index,
        "phase_start_epoch": (
            None if expected_phase is None else expected_phase.start_epoch
        ),
        "phase_end_epoch": None if expected_phase is None else expected_phase.end_epoch,
        "effective_max_log_location_fraction": (
            None
            if expected_phase is None
            else float(expected_phase.max_log_location_fraction)
        ),
    }
    if saved_state != expected_state:
        raise ValueError(
            "Checkpoint residual schedule provenance is inconsistent with "
            f"the saved epoch {saved_epoch}: checkpoint={saved_state!r}, "
            f"expected={expected_state!r}"
        )
    model.set_observation_residual_max_log_location_fraction(
        saved_state["effective_max_log_location_fraction"]
    )
    logger.info(
        "Restored residual log-location budget "
        "max_log_location_fraction=%s from checkpoint provenance",
        saved_state["effective_max_log_location_fraction"],
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "auto",
    overrides: list[str] | None = None,
) -> tuple[EpiForecaster, EpiForecasterConfig, dict[str, Any]]:
    """Load an EpiForecaster and its config from a trainer checkpoint.

    ``overrides`` are dotted-key config overrides applied before the model is
    built, so they can change the architecture. Returns the model on
    ``device``, the config, and the checkpoint payload.
    """
    checkpoint = read_checkpoint(checkpoint_path)
    config = checkpoint.config()
    if overrides:
        config = EpiForecasterConfig.apply_overrides(config, overrides)
        logger.info("Applied %d config overrides before model creation", len(overrides))
    resolved_device = resolve_device(device)
    model = build_model(config.model, device=resolved_device)
    restore_weights(model, checkpoint)
    restore_residual_budget(model, checkpoint)
    model.to(resolved_device)
    return model, config, checkpoint.payload


def _saved_residual_schedule(saved_config: object) -> dict[str, Any] | None:
    if isinstance(saved_config, dict):
        return saved_config.get("training", {}).get("residual_schedule")
    if isinstance(saved_config, EpiForecasterConfig) and hasattr(
        saved_config.training, "residual_schedule"
    ):
        return asdict(saved_config.training.residual_schedule)
    return None


def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix(_COMPILE_PREFIX): value for key, value in state.items()}


def _pad_legacy_mobility_input_weights(
    state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Zero-pad old mobility input weights when node feature channels were added."""
    patched = dict(state)
    padded_keys: list[str] = []
    for key in _MOBILITY_INPUT_WEIGHT_KEYS:
        saved = patched.get(key)
        expected = model_state.get(key)
        if saved is None or expected is None or saved.shape == expected.shape:
            continue
        if saved.ndim != 2 or expected.ndim != 2:
            continue
        if saved.shape[0] != expected.shape[0] or saved.shape[1] > expected.shape[1]:
            continue
        padded = expected.new_zeros(expected.shape)
        padded[:, : saved.shape[1]] = saved.to(dtype=expected.dtype)
        patched[key] = padded
        padded_keys.append(key)
        logger.info(
            "Padded legacy checkpoint weight %s from %s to %s",
            key,
            tuple(saved.shape),
            tuple(expected.shape),
        )
    return patched, tuple(padded_keys)


def _migrate_legacy_frozen_clinical_gamma_keys(
    state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    saved_config: object,
) -> tuple[dict[str, torch.Tensor], tuple[tuple[str, str], ...]]:
    """Rename pre-cutover frozen clinical Gamma buffers.

    Frozen clinical Gamma kernels previously used ``*.delay_kernel.kernel``.
    The current distinct ``gamma_kernel`` key prevents simplex/Gamma
    collisions. Migration therefore requires checkpoint configuration proving
    that the saved clinical head was frozen Gamma; simplex checkpoints remain
    strict failures, and wastewater shedding keys never migrate.
    """
    saved_observation_heads = _checkpoint_observation_heads_config(saved_config)
    migrated: list[tuple[str, str]] = []
    patched = dict(state)
    for new_key in model_state:
        if not new_key.endswith(".delay_kernel.gamma_kernel"):
            continue
        head_name = new_key.split(".", 1)[0].removesuffix("_head")
        parameterization = _config_value(
            saved_observation_heads, f"kernel_parameterization_{head_name}"
        )
        learnable = _config_value(
            saved_observation_heads, f"learnable_kernel_{head_name}"
        )
        if parameterization != "gamma" or learnable is not False:
            continue
        if new_key in patched:
            continue
        legacy_key = new_key.removesuffix(".gamma_kernel") + ".kernel"
        if legacy_key not in patched:
            continue
        patched[new_key] = patched.pop(legacy_key)
        migrated.append((legacy_key, new_key))
    return patched, tuple(migrated)


def _initialize_missing_frozen_gamma_metadata(
    state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    saved_config: object,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Seed Gamma shape and scale buffers absent from pre-metadata checkpoints."""
    saved_observation_heads = _checkpoint_observation_heads_config(saved_config)
    missing_keys = tuple(
        key
        for key in model_state
        if key.endswith((".gamma_shape_value", ".gamma_scale_value"))
        and key not in state
    )
    if not missing_keys:
        return state, ()
    patched = dict(state)
    for key in missing_keys:
        head_name = key.split(".", 1)[0].removesuffix("_head")
        parameter_name = (
            "kernel_gamma_shape_"
            if key.endswith(".gamma_shape_value")
            else "kernel_gamma_scale_"
        ) + head_name
        saved_value = _config_value(saved_observation_heads, parameter_name)
        if saved_value is None:
            patched[key] = model_state[key].detach().clone()
        else:
            patched[key] = model_state[key].new_tensor(float(saved_value))
    return patched, missing_keys


def _initialize_missing_distribution_parameters(
    state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Initialize distribution parameters absent from point-forecast checkpoints."""
    missing_keys = tuple(
        key
        for key in model_state
        if key not in state
        and any(marker in key for marker in _DISTRIBUTION_PARAMETER_MARKERS)
    )
    if not missing_keys:
        return state, ()
    patched = dict(state)
    for key in missing_keys:
        patched[key] = model_state[key].detach().clone()
    return patched, missing_keys


def _checkpoint_observation_heads_config(config: object) -> object | None:
    if isinstance(config, dict):
        model = config.get("model")
        return model.get("observation_heads") if isinstance(model, dict) else None
    model = getattr(config, "model", None)
    return getattr(model, "observation_heads", None)


def _config_value(config: object | None, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)
