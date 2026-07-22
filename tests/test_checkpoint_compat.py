from __future__ import annotations

import torch

from utils.checkpoint_compat import initialize_missing_distribution_parameters


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
