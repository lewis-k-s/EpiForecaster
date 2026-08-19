from __future__ import annotations

import pytest
import torch

from utils.checkpoint_compat import (
    initialize_missing_distribution_parameters,
    initialize_missing_frozen_gamma_metadata,
    migrate_legacy_frozen_clinical_gamma_keys,
)


def test_initialize_missing_distribution_parameters_is_selective() -> None:
    state_dict = {"core.weight": torch.tensor([1.0])}
    model_state_dict = {
        "core.weight": torch.tensor([2.0]),
        "core.bias": torch.tensor([3.0]),
        "ww_head.log_distribution_scale": torch.tensor([-1.0]),
        "ww_head.distribution_scale_proj.weight": torch.tensor([[0.0]]),
        "ww_head.distribution_scale_proj.bias": torch.tensor([0.5]),
    }

    patched, initialized_keys = initialize_missing_distribution_parameters(
        state_dict=state_dict,
        model_state_dict=model_state_dict,
    )

    assert patched["core.weight"].item() == 1.0
    assert "core.bias" not in patched
    assert set(initialized_keys) == {
        "ww_head.log_distribution_scale",
        "ww_head.distribution_scale_proj.weight",
        "ww_head.distribution_scale_proj.bias",
    }
    for key in initialized_keys:
        assert torch.equal(patched[key], model_state_dict[key])
        assert patched[key].data_ptr() != model_state_dict[key].data_ptr()


def test_migrate_legacy_frozen_clinical_gamma_keys() -> None:
    legacy = torch.tensor([1.0, 2.0])
    state_dict = {
        "hosp_head.delay_kernel.kernel": legacy,
        "ww_head.shedding_conv.kernel": torch.tensor([3.0, 4.0]),
    }
    model_state_dict = {
        "hosp_head.delay_kernel.gamma_kernel": torch.zeros(2),
        "ww_head.shedding_conv.gamma_kernel": torch.zeros(2),
    }

    patched, migrated = migrate_legacy_frozen_clinical_gamma_keys(
        state_dict=state_dict,
        model_state_dict=model_state_dict,
        saved_config={
            "model": {
                "observation_heads": {
                    "kernel_parameterization_hosp": "gamma",
                    "learnable_kernel_hosp": False,
                }
            }
        },
    )

    assert patched["hosp_head.delay_kernel.gamma_kernel"] is legacy
    assert "hosp_head.delay_kernel.kernel" not in patched
    assert migrated == (
        (
            "hosp_head.delay_kernel.kernel",
            "hosp_head.delay_kernel.gamma_kernel",
        ),
    )
    assert patched["ww_head.shedding_conv.kernel"].tolist() == [3.0, 4.0]
    assert "ww_head.shedding_conv.gamma_kernel" not in patched


def test_migrate_legacy_gamma_does_not_overwrite_new_key() -> None:
    state_dict = {
        "cases_head.delay_kernel.kernel": torch.tensor([1.0]),
        "cases_head.delay_kernel.gamma_kernel": torch.tensor([2.0]),
    }
    model_state_dict = {
        "cases_head.delay_kernel.gamma_kernel": torch.zeros(1),
    }

    patched, migrated = migrate_legacy_frozen_clinical_gamma_keys(
        state_dict=state_dict,
        model_state_dict=model_state_dict,
        saved_config={
            "model": {
                "observation_heads": {
                    "kernel_parameterization_cases": "gamma",
                    "learnable_kernel_cases": False,
                }
            }
        },
    )

    assert migrated == ()
    assert patched == state_dict


def test_migrate_legacy_gamma_rejects_unknown_simplex_checkpoint() -> None:
    state_dict = {"hosp_head.delay_kernel.kernel": torch.tensor([1.0])}
    model_state_dict = {"hosp_head.delay_kernel.gamma_kernel": torch.zeros(1)}

    patched, migrated = migrate_legacy_frozen_clinical_gamma_keys(
        state_dict=state_dict,
        model_state_dict=model_state_dict,
        saved_config={
            "model": {
                "observation_heads": {
                    "kernel_parameterization_hosp": "simplex",
                    "learnable_kernel_hosp": False,
                }
            }
        },
    )

    assert patched == state_dict
    assert migrated == ()


def test_migrate_legacy_gamma_handles_compiled_model_prefix() -> None:
    state_dict = {"_orig_mod.hosp_head.delay_kernel.kernel": torch.tensor([1.0])}
    model_state_dict = {"_orig_mod.hosp_head.delay_kernel.gamma_kernel": torch.zeros(1)}

    patched, migrated = migrate_legacy_frozen_clinical_gamma_keys(
        state_dict=state_dict,
        model_state_dict=model_state_dict,
        saved_config={
            "model": {
                "observation_heads": {
                    "kernel_parameterization_hosp": "gamma",
                    "learnable_kernel_hosp": False,
                }
            }
        },
    )

    assert "_orig_mod.hosp_head.delay_kernel.kernel" not in patched
    assert "_orig_mod.hosp_head.delay_kernel.gamma_kernel" in patched
    assert migrated == (
        (
            "_orig_mod.hosp_head.delay_kernel.kernel",
            "_orig_mod.hosp_head.delay_kernel.gamma_kernel",
        ),
    )


def test_missing_gamma_metadata_uses_saved_config_prior() -> None:
    state_dict: dict[str, torch.Tensor] = {}
    model_state_dict = {
        "deaths_head.delay_kernel.gamma_shape_value": torch.tensor(1.0),
        "deaths_head.delay_kernel.gamma_scale_value": torch.tensor(1.0),
    }

    patched, missing = initialize_missing_frozen_gamma_metadata(
        state_dict=state_dict,
        model_state_dict=model_state_dict,
        saved_config={
            "model": {
                "observation_heads": {
                    "kernel_gamma_shape_deaths": 12.25,
                    "kernel_gamma_scale_deaths": 4.0 / 7.0,
                }
            }
        },
    )

    assert missing == (
        "deaths_head.delay_kernel.gamma_shape_value",
        "deaths_head.delay_kernel.gamma_scale_value",
    )
    assert patched["deaths_head.delay_kernel.gamma_shape_value"].item() == 12.25
    assert patched[
        "deaths_head.delay_kernel.gamma_scale_value"
    ].item() == pytest.approx(4.0 / 7.0)
