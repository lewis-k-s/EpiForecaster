from __future__ import annotations

import torch

_DISTRIBUTION_PARAMETER_MARKERS = (
    ".log_distribution_scale",
    ".distribution_scale_proj.",
)


def initialize_missing_distribution_parameters(
    *,
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Initialize distribution parameters absent from legacy point checkpoints."""
    missing_keys = tuple(
        key
        for key in model_state_dict
        if key not in state_dict
        and any(marker in key for marker in _DISTRIBUTION_PARAMETER_MARKERS)
    )
    if not missing_keys:
        return state_dict, ()

    patched = dict(state_dict)
    for key in missing_keys:
        patched[key] = model_state_dict[key].detach().clone()
    return patched, missing_keys


def migrate_legacy_frozen_clinical_gamma_keys(
    *,
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    saved_config: object | None = None,
) -> tuple[dict[str, torch.Tensor], tuple[tuple[str, str], ...]]:
    """Migrate pre-cutover frozen clinical Gamma buffer names.

    Frozen clinical Gamma kernels previously used ``*.delay_kernel.kernel``.
    The current distinct ``gamma_kernel`` key prevents simplex/Gamma
    collisions. Migration therefore requires checkpoint configuration proving
    that the saved clinical head was frozen Gamma; simplex checkpoints remain
    strict failures, and wastewater shedding keys never migrate.
    """
    saved_observation_heads = _checkpoint_observation_heads_config(saved_config)
    migrated: list[tuple[str, str]] = []
    patched = dict(state_dict)
    for new_key in model_state_dict:
        if not new_key.endswith(".delay_kernel.gamma_kernel"):
            continue
        key_without_compile_prefix = new_key.removeprefix("_orig_mod.")
        head_name = key_without_compile_prefix.split(".", 1)[0].removesuffix("_head")
        parameterization = _config_value(
            saved_observation_heads,
            f"kernel_parameterization_{head_name}",
        )
        learnable = _config_value(
            saved_observation_heads,
            f"learnable_kernel_{head_name}",
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


def initialize_missing_frozen_gamma_metadata(
    *,
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    saved_config: object | None = None,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Seed metadata absent from pre-metadata frozen Gamma checkpoints."""
    saved_observation_heads = _checkpoint_observation_heads_config(saved_config)
    missing_keys = tuple(
        key
        for key in model_state_dict
        if key.endswith((".gamma_shape_value", ".gamma_scale_value"))
        and key not in state_dict
    )
    if not missing_keys:
        return state_dict, ()
    patched = dict(state_dict)
    for key in missing_keys:
        key_without_compile_prefix = key.removeprefix("_orig_mod.")
        head_name = key_without_compile_prefix.split(".", 1)[0].removesuffix("_head")
        parameter_name = (
            "kernel_gamma_shape_"
            if key.endswith(".gamma_shape_value")
            else "kernel_gamma_scale_"
        ) + head_name
        saved_value = _config_value(saved_observation_heads, parameter_name)
        if saved_value is None:
            patched[key] = model_state_dict[key].detach().clone()
        else:
            patched[key] = model_state_dict[key].new_tensor(float(saved_value))
    return patched, missing_keys


def _checkpoint_observation_heads_config(config: object | None) -> object | None:
    if isinstance(config, dict):
        model = config.get("model")
        return model.get("observation_heads") if isinstance(model, dict) else None
    model = getattr(config, "model", None)
    return getattr(model, "observation_heads", None)


def _config_value(config: object | None, name: str) -> object | None:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)
