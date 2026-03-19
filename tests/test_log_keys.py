from __future__ import annotations

from utils.log_keys import (
    TENSORBOARD_SCALARS,
    build_curriculum_metric_key,
    build_eval_metric_key,
    build_grad_snapshot_head_key,
    build_gradnorm_obs_key,
    build_horizon_metric_key,
    build_loss_key,
    infer_observation_head_from_name,
    parse_grad_snapshot_head_key,
)


def test_metric_key_builders_return_canonical_names() -> None:
    assert build_loss_key(split="val") == "loss_val"
    assert build_loss_key(component="ww") == "loss_ww"
    assert build_loss_key(split="test", component="cases", weighted=True) == (
        "loss_test_cases_weighted"
    )
    assert build_eval_metric_key("mae", "val") == "mae_val"
    assert build_horizon_metric_key("rmse", "test", "w2") == "rmse_test_w2"
    assert build_curriculum_metric_key("sparsity", "epoch") == "train_sparsity_epoch"
    assert build_gradnorm_obs_key("deaths") == "gradnorm_obs_deaths"


def test_grad_snapshot_head_key_round_trip() -> None:
    key = build_grad_snapshot_head_key("cases", "zero_when_active_rate")
    assert key == "grad_snapshot_head_cases_zero_when_active_rate"
    assert parse_grad_snapshot_head_key(key) == ("cases", "zero_when_active_rate")
    assert parse_grad_snapshot_head_key("grad_snapshot_global_norm") is None


def test_infer_observation_head_from_name_handles_wrapped_and_unwrapped_names() -> None:
    assert infer_observation_head_from_name("ww_head.scale") == "ww"
    assert infer_observation_head_from_name("_orig_mod.hosp_head.alpha") == "hosp"
    assert infer_observation_head_from_name("backbone.obs_context_projection.0.weight") is None


def test_tensorboard_scalars_are_single_canonical_tags() -> None:
    assert TENSORBOARD_SCALARS["loss_train"] == "Loss/Train_step"
    assert TENSORBOARD_SCALARS["gradnorm_backbone"] == "GradNorm/Backbone"
