from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from scripts.optuna_epiforecaster_worker import (
    _categorical_choices,
    _compute_worker_seed,
    _decode_categorical_value,
    _overrides_to_dotlist,
    suggest_epiforecaster_params,
)


class _StubTrial:
    """Configurable mock that tracks all suggest calls."""

    def __init__(
        self,
        categorical_values: dict[str, object] | None = None,
        float_values: dict[str, float] | None = None,
        int_values: dict[str, int] | None = None,
    ):
        self._categorical_values = categorical_values or {}
        self._float_values = float_values or {}
        self._int_values = int_values or {}
        self.params: dict[str, object] = {}
        self.suggest_calls: list[tuple[str, str, Any]] = []

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
        self.suggest_calls.append(("float", name, (low, high, log)))
        value = self._float_values.get(name, low)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.suggest_calls.append(("int", name, (low, high)))
        value = self._int_values.get(name, low)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: tuple[object, ...]) -> object:
        self.suggest_calls.append(("categorical", name, choices))
        value = self._categorical_values.get(name, choices[0])
        self.params[name] = value
        return value


def _base_cfg_stub(**overrides: Any) -> SimpleNamespace:
    """Create config stub with sensible defaults, overrideable."""
    model_defaults = dict(
        type=SimpleNamespace(mobility=False, regions=False),
        temporal_covariates_dim=0,
        include_day_of_week=True,
        include_holidays=True,
        input_window_length=60,
        forecast_horizon=28,
    )
    model_overrides = overrides.pop("model", {})
    model_defaults.update(model_overrides)

    training_defaults = dict(
        loss=SimpleNamespace(name="mse"),
    )
    training_overrides = overrides.pop("training", {})
    training_defaults.update(training_overrides)

    return SimpleNamespace(
        model=SimpleNamespace(**model_defaults),
        training=SimpleNamespace(**training_defaults),
    )


def _mobility_cfg_stub(**overrides: Any) -> SimpleNamespace:
    """Config stub with mobility enabled."""
    model_overrides = overrides.setdefault("model", {})
    model_overrides["type"] = SimpleNamespace(mobility=True, regions=False)
    return _base_cfg_stub(**overrides)


def _temporal_covariates_cfg_stub(**overrides: Any) -> SimpleNamespace:
    """Config stub with temporal covariates enabled."""
    model_overrides = overrides.setdefault("model", {})
    model_overrides["temporal_covariates_dim"] = 8
    return _base_cfg_stub(**overrides)


def _joint_loss_cfg_stub(**overrides: Any) -> SimpleNamespace:
    """Config stub with joint_inference loss."""
    joint_overrides = overrides.pop("joint", {})
    joint_defaults = {
        "adaptive_scheme": "gradnorm",
        "w_sir": 1.0,
        "gradnorm_obs_weight_sum": 0.95,
    }
    joint_defaults.update(joint_overrides)
    training_overrides = overrides.setdefault("training", {})
    training_overrides["loss"] = SimpleNamespace(
        name="joint_inference",
        joint=SimpleNamespace(**joint_defaults),
    )
    return _base_cfg_stub(**overrides)


class TestComputeWorkerSeed:
    def test_returns_base_seed_without_slurm_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _compute_worker_seed(42) == 42

    def test_offsets_by_slurm_array_task_id(self) -> None:
        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "3"}):
            assert _compute_worker_seed(42) == 45

    def test_offsets_by_zero_task_id(self) -> None:
        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "0"}):
            assert _compute_worker_seed(42) == 42

    def test_different_task_ids_produce_different_seeds(self) -> None:
        base_seed = 100
        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "0"}):
            seed_0 = _compute_worker_seed(base_seed)
        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1"}):
            seed_1 = _compute_worker_seed(base_seed)
        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "2"}):
            seed_2 = _compute_worker_seed(base_seed)

        assert seed_0 == 100
        assert seed_1 == 101
        assert seed_2 == 102
        assert len({seed_0, seed_1, seed_2}) == 3


class TestCategoricalChoices:
    def test_encodes_list_to_json(self) -> None:
        assert _categorical_choices(([1, 2],)) == ("[1, 2]",)

    def test_encodes_tuple_to_json(self) -> None:
        assert _categorical_choices(((1, 2),)) == ("[1, 2]",)

    def test_encodes_dict_to_json(self) -> None:
        result = _categorical_choices(({"a": 1},))
        assert result == ('{"a": 1}',)

    def test_passes_through_string(self) -> None:
        assert _categorical_choices(("foo",)) == ("foo",)

    def test_passes_through_int(self) -> None:
        assert _categorical_choices((1, 2)) == (1, 2)

    def test_passes_through_float(self) -> None:
        assert _categorical_choices((1.5,)) == (1.5,)

    def test_passes_through_bool(self) -> None:
        assert _categorical_choices((True, False)) == (True, False)

    def test_passes_through_none(self) -> None:
        assert _categorical_choices((None,)) == (None,)

    def test_mixed_choices(self) -> None:
        result = _categorical_choices(([1, 2], "foo", 3, {"a": 1}))
        assert result == ("[1, 2]", "foo", 3, '{"a": 1}')


class TestDecodeCategoricalValue:
    def test_decodes_json_list(self) -> None:
        assert _decode_categorical_value("[1, 2]") == [1, 2]

    def test_decodes_json_dict(self) -> None:
        assert _decode_categorical_value('{"a": 1}') == {"a": 1}

    def test_passes_through_non_json_string(self) -> None:
        assert _decode_categorical_value("foo") == "foo"

    def test_passes_through_int(self) -> None:
        assert _decode_categorical_value(42) == 42

    def test_passes_through_list(self) -> None:
        assert _decode_categorical_value([1, 2]) == [1, 2]

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON categorical value"):
            _decode_categorical_value("[invalid")


class TestOverridesToDotlist:
    def test_converts_key_value_pairs(self) -> None:
        result = _overrides_to_dotlist({"a": 1, "b": "foo"})
        assert set(result) == {"a=1", "b=foo"}

    def test_formats_bool_as_lowercase(self) -> None:
        assert _overrides_to_dotlist({"a": True, "b": False}) == ["a=true", "b=false"]

    def test_formats_list_as_str(self) -> None:
        result = _overrides_to_dotlist({"a": [1, 2]})
        assert result == ["a=[1, 2]"]

    def test_formats_tuple_as_str(self) -> None:
        result = _overrides_to_dotlist({"a": (1, 2)})
        assert result == ["a=(1, 2)"]

    def test_skips_none_values(self) -> None:
        result = _overrides_to_dotlist({"a": 1, "b": None, "c": 2})
        assert set(result) == {"a=1", "c=2"}

    def test_formats_float(self) -> None:
        result = _overrides_to_dotlist({"lr": 0.001})
        assert result == ["lr=0.001"]


class TestSuggestEpiforecasterParams:
    # NOTE: batch_size/gradient_accumulation_steps are no longer swept.
    # They were fixed at 32/1 based on production config. Test removed.

    def test_training_knobs_learning_rate(self) -> None:
        trial = _StubTrial()
        suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "training.learning_rate" in trial.params
        call = next(c for c in trial.suggest_calls if c[1] == "training.learning_rate")
        assert call[0] == "float"
        assert call[2][0] == pytest.approx(1e-5, rel=0.1)
        assert call[2][1] == pytest.approx(3e-3, rel=0.1)

    def test_training_knobs_weight_decay(self) -> None:
        trial = _StubTrial()
        suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "training.weight_decay" in trial.params
        call = next(c for c in trial.suggest_calls if c[1] == "training.weight_decay")
        assert call[0] == "float"

    def test_data_knobs_sample_ordering(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "data.sample_ordering" in overrides
        assert overrides["data.sample_ordering"] in ("node", "time")

    def test_data_knobs_mobility_threshold(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "data.mobility_threshold" in overrides

    def test_full_worker_does_not_suggest_missing_permit(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        missing_calls = [c for c in trial.suggest_calls if "missing_permit" in c[1]]
        assert missing_calls == []
        missing_override_keys = [
            k for k in overrides if k.startswith("data.missing_permit.")
        ]
        assert missing_override_keys == []

    def test_mobility_model_knobs_gnn_depth(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(
            trial=trial, base_cfg=_mobility_cfg_stub()
        )
        assert "model.gnn_depth" in overrides
        call = next(c for c in trial.suggest_calls if c[1] == "model.gnn_depth")
        assert call[0] == "int"
        assert call[2] == (1, 4)

    def test_mobility_model_knobs_gnn_module(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(
            trial=trial, base_cfg=_mobility_cfg_stub()
        )
        assert "model.gnn_module" in overrides
        assert overrides["model.gnn_module"] in ("gcn", "gat")

    def test_mobility_model_knobs_embedding_dim(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(
            trial=trial, base_cfg=_mobility_cfg_stub()
        )
        assert "model.mobility_embedding_dim" in overrides
        assert overrides["model.mobility_embedding_dim"] in (16, 32, 64, 128)

    def test_no_mobility_skips_gnn_knobs(self) -> None:
        trial = _StubTrial()
        suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        gnn_calls = [c for c in trial.suggest_calls if "gnn" in c[1]]
        assert gnn_calls == []

    def test_temporal_covariates_enabled_sets_flags(self) -> None:
        trial = _StubTrial(categorical_values={"model.use_temporal_covariates": True})
        cfg = _temporal_covariates_cfg_stub()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        assert overrides["model.include_day_of_week"] is True
        assert overrides["model.include_holidays"] is True

    def test_temporal_covariates_disabled_clears_flags(self) -> None:
        trial = _StubTrial(categorical_values={"model.use_temporal_covariates": False})
        cfg = _temporal_covariates_cfg_stub()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        assert overrides["model.include_day_of_week"] is False
        assert overrides["model.include_holidays"] is False

    def test_no_temporal_covariates_skips_flag_overrides(self) -> None:
        trial = _StubTrial()
        suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        cov_calls = [c for c in trial.suggest_calls if "temporal_covariates" in c[1]]
        assert cov_calls == []

    def test_head_positional_encoding(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "model.head_positional_encoding" in overrides
        assert overrides["model.head_positional_encoding"] in ("sinusoidal", "learned")

    def test_init_weights_group_params(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())

        assert "model.init_weights.rezero_init" in overrides
        assert "model.init_weights.rate_head_final_gain" in overrides
        assert "model.init_weights.initial_state_final_gain" in overrides
        assert "model.init_weights.obs_context_final_gain" in overrides

        assert overrides["model.init_weights.rezero_init"] in (1.0e-3, 3.0e-3, 1.0e-2)
        assert overrides["model.init_weights.rate_head_final_gain"] in (
            5.0e-3,
            1.0e-2,
            2.0e-2,
        )
        assert overrides["model.init_weights.initial_state_final_gain"] in (
            5.0e-3,
            1.0e-2,
            2.0e-2,
        )
        assert overrides["model.init_weights.obs_context_final_gain"] in (
            0.25,
            0.5,
            1.0,
        )

    def test_joint_inference_no_static_loss_weight_overrides(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        loss_weight_overrides = [
            k for k in overrides if k.startswith("training.loss.joint.w_")
        ]
        assert loss_weight_overrides == []

    def test_joint_inference_observation_heads_params(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)

        assert "model.observation_heads.residual_scale" in overrides
        assert "model.observation_heads.residual_hidden_dim" in overrides
        assert "model.observation_heads.residual_layers" in overrides
        assert "model.observation_heads.residual_dropout" in overrides
        assert "model.observation_heads.obs_context_dim" in overrides
        assert "model.observation_heads.residual_mode" in overrides
        assert "model.observation_heads.learnable_kernel_ww" in overrides
        assert "model.observation_heads.learnable_kernel_hosp" in overrides
        assert overrides["model.observation_heads.learnable_kernel_ww"] in (
            False,
            True,
        )
        assert overrides["model.observation_heads.learnable_kernel_hosp"] in (
            False,
            True,
        )

    def test_joint_inference_gradnorm_params_when_enabled(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub(joint={"adaptive_scheme": "gradnorm"})
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)

        assert "training.loss.joint.gradnorm_alpha" in overrides
        assert "training.loss.joint.gradnorm_weight_lr" in overrides
        assert "training.loss.joint.gradnorm_warmup_steps" in overrides
        assert "training.loss.joint.gradnorm_update_every" in overrides
        assert "training.loss.joint.gradnorm_ema_decay" in overrides
        assert "training.loss.joint.gradnorm_min_weight" in overrides

    def test_joint_inference_skips_gradnorm_params_when_disabled(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub(joint={"adaptive_scheme": "none"})
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        gradnorm_overrides = [
            k for k in overrides if k.startswith("training.loss.joint.gradnorm_")
        ]
        assert gradnorm_overrides == []

    def test_joint_inference_residual_scale_bounds(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub()
        suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        call = next(
            c
            for c in trial.suggest_calls
            if c[1] == "init_weights.observation_residual_scale"
        )
        assert call[0] == "float"
        assert call[2][0] == pytest.approx(0.03)
        assert call[2][1] == pytest.approx(0.2)

    def test_joint_inference_residual_layers_bounds(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub()
        suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        call = next(
            c
            for c in trial.suggest_calls
            if c[1] == "model.observation_heads.residual_layers"
        )
        assert call[0] == "int"
        assert call[2] == (1, 4)

    def test_joint_inference_residual_mode_options(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        assert overrides["model.observation_heads.residual_mode"] in (
            "additive",
            "modulation",
        )

    def test_non_joint_inference_skips_observation_heads(self) -> None:
        trial = _StubTrial()
        suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        obs_calls = [c for c in trial.suggest_calls if "observation_heads" in c[1]]
        assert obs_calls == []
