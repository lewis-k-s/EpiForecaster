from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.hpo.hpsearch_worker import (
    _categorical_choices,
    _decode_categorical_value,
    _overrides_to_dotlist,
    _remaining_worker_trials,
    _worker_outcome_error,
    objective,
    suggest_epiforecaster_params,
)
from training.training_dynamics import TrainingDynamicsWriter
from scripts.hpo.trial_artifacts import sha256_file


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
        self.number = 0
        self.user_attrs: dict[str, object] = {}
        self.study = SimpleNamespace(best_value=0.4)

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
        self.suggest_calls.append(("float", name, (low, high, log)))
        value = self._float_values.get(name, low)
        self.params[name] = value
        return value

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value

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


def test_worker_outcome_error_rejects_missing_and_failed_trials() -> None:
    assert _worker_outcome_error([]) is not None
    assert _worker_outcome_error([(2, "FAIL")]) == (
        "HPO worker recorded failed trial(s): 2"
    )
    assert _worker_outcome_error([(0, "COMPLETE"), (1, "PRUNED")]) is None


def test_worker_trial_budget_resumes_only_its_completed_trials() -> None:
    trials = [
        SimpleNamespace(
            number=0,
            state=SimpleNamespace(name="COMPLETE"),
            user_attrs={"epiforecaster_worker_id": 2},
        ),
        SimpleNamespace(
            number=1,
            state=SimpleNamespace(name="PRUNED"),
            user_attrs={"epiforecaster_worker_id": 2},
        ),
        SimpleNamespace(
            number=2,
            state=SimpleNamespace(name="COMPLETE"),
            user_attrs={"epiforecaster_worker_id": 1},
        ),
    ]
    assert _remaining_worker_trials(SimpleNamespace(trials=trials), 2, 4) == 2


@pytest.mark.parametrize("state", ["FAIL", "RUNNING", "WAITING"])
def test_worker_trial_budget_rejects_unresolved_owned_trials(state: str) -> None:
    study = SimpleNamespace(
        trials=[
            SimpleNamespace(
                number=7,
                state=SimpleNamespace(name=state),
                user_attrs={"epiforecaster_worker_id": 3},
            )
        ]
    )
    with pytest.raises(RuntimeError, match=f"7:{state}"):
        _remaining_worker_trials(study, 3, 1)


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
        loss=SimpleNamespace(
            name="joint_inference",
            joint=SimpleNamespace(adaptive_scheme="gradnorm"),
        ),
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
        "gradnorm_obs_weight_sum": 0.95,
    }
    joint_defaults.update(joint_overrides)
    training_overrides = overrides.setdefault("training", {})
    training_overrides["loss"] = SimpleNamespace(
        name="joint_inference",
        joint=SimpleNamespace(**joint_defaults),
    )
    return _base_cfg_stub(**overrides)


def test_categorical_trial_values_survive_cli_override_encoding() -> None:
    encoded = _categorical_choices(([1, 2], {"heads": ["ww"]}, True))
    decoded = [_decode_categorical_value(value) for value in encoded]

    assert decoded == [[1, 2], {"heads": ["ww"]}, True]
    assert _overrides_to_dotlist(
        {
            "model.layers": decoded[0],
            "training.enabled": decoded[2],
            "unused": None,
        }
    ) == ["model.layers=[1, 2]", "training.enabled=true"]
    with pytest.raises(ValueError, match="Invalid JSON categorical value"):
        _decode_categorical_value("[invalid")


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

    def test_lr_only_search_space_samples_only_learning_rate(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(
            trial=trial,
            base_cfg=_mobility_cfg_stub(),
            search_space="lr_only",
        )
        assert overrides == {"training.learning_rate": pytest.approx(1e-5)}
        assert trial.suggest_calls == [
            ("float", "training.learning_rate", (1e-5, 3e-3, True))
        ]

    def test_training_loss_objective_not_sampled_by_default(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "training.loss.joint.observation_loss" not in overrides

    def test_training_loss_objective_sampled_when_requested(self) -> None:
        trial = _StubTrial(
            categorical_values={"training.loss.joint.observation_loss": "nll"}
        )
        overrides = suggest_epiforecaster_params(
            trial=trial,
            base_cfg=_base_cfg_stub(),
            sample_training_loss_objective=True,
        )
        assert overrides["training.loss.joint.observation_loss"] == "nll"
        call = next(
            c
            for c in trial.suggest_calls
            if c[1] == "training.loss.joint.observation_loss"
        )
        assert call[0] == "categorical"
        assert call[2] == ("nll", "crps")

    def test_data_knobs_sample_ordering(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "data.sample_ordering" in overrides
        assert overrides["data.sample_ordering"] in ("node", "time")

    def test_data_knobs_mobility_min_flow_raw(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "data.mobility_min_flow_raw" in overrides

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
        assert call[2] == (1, 2)

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

    def test_joint_inference_skips_gradnorm_params_when_disabled(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub(joint={"adaptive_scheme": "none"})
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        gradnorm_overrides = [
            k for k in overrides if k.startswith("training.loss.joint.gradnorm_")
        ]
        assert gradnorm_overrides == []

    def test_joint_inference_residual_mode_options(self) -> None:
        trial = _StubTrial()
        cfg = _joint_loss_cfg_stub()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=cfg)
        assert overrides["model.observation_heads.residual_mode"] in (
            "additive",
            "modulation",
        )

    def test_base_stub_also_suggests_observation_heads(self) -> None:
        trial = _StubTrial()
        overrides = suggest_epiforecaster_params(trial=trial, base_cfg=_base_cfg_stub())
        assert "model.observation_heads.residual_hidden_dim" in overrides


def test_objective_returns_selection_score_and_records_joint_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = _joint_loss_cfg_stub(joint={"observation_loss": "mae"})
    cfg.native = SimpleNamespace(fold_manifest="")
    monkeypatch.setattr("scripts.hpo.hpsearch_worker.build_for_config", lambda *args: None)
    cfg.output = SimpleNamespace(log_dir=str(tmp_path), experiment_name="study")
    cfg.training.epochs = 1
    captured: dict[str, object] = {}

    class _FakeConfig:
        @staticmethod
        def from_file(_path: str):
            return cfg

        @staticmethod
        def load(_path: str, overrides: list[str]):
            captured["overrides"] = overrides
            return cfg

    class _FakeTrainer:
        def __init__(
            self,
            config,
            trial=None,
            pruning_start_epoch=10,
            model_id=None,
        ):
            captured["trainer_config"] = config
            captured["trainer_trial"] = trial
            captured["pruning_start_epoch"] = pruning_start_epoch
            captured["constructor_model_id"] = model_id
            self.config = config
            self.model_id = model_id
            self.experiment_dir = (
                Path(self.config.output.log_dir)
                / self.config.output.experiment_name
                / self.model_id
            )
            self.experiment_dir.mkdir(parents=True)

        def run(self):
            captured["trainer_model_id"] = self.model_id
            checkpoint = (
                Path(self.config.output.log_dir)
                / self.config.output.experiment_name
                / self.model_id
                / "checkpoints/best_model.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"deterministic checkpoint")
            writer = TrainingDynamicsWriter(
                self.experiment_dir,
                progress_log_frequency=10,
                grad_norm_log_frequency=10,
                mobility_enabled=True,
            )
            writer.append(
                step=0,
                epoch=0,
                values={"loss_train_step": 1.0},
            )
            writer.finalize("complete")
            return {
                "best_val_selection_score": 0.25,
                "best_val_selection_metric": "mean_target_mae",
                "best_val_joint_loss": 1.75,
                "best_val_series_metrics": {
                    "mae_ww_log1p_per_100k": 0.2,
                    "mae_hosp_log1p_per_100k": 0.3,
                },
            }

    monkeypatch.setattr("scripts.hpo.hpsearch_worker.EpiForecasterConfig", _FakeConfig)
    monkeypatch.setattr(
        "scripts.hpo.hpsearch_worker.NativeTrainer", _FakeTrainer
    )

    def _fake_suggest_params(
        *,
        trial,
        base_cfg,
        sample_training_loss_objective=False,
        search_space="default",
    ):
        return {"training.learning_rate": 1.0e-3}

    monkeypatch.setattr(
        "scripts.hpo.hpsearch_worker.suggest_epiforecaster_params",
        _fake_suggest_params,
    )

    trial = _StubTrial()
    value = objective(
        trial,
        base_config_path=tmp_path / "config.yaml",
        study_name="study",
        run_root=tmp_path,
        fixed_epochs=1,
        fixed_max_batches=None,
        seed=None,
        cli_overrides=["training.loss.joint.observation_loss=mae"],
        pruning_start_epoch=3,
        sample_training_loss_objective=False,
        search_space="default",
        worker_id=4,
    )

    assert value == pytest.approx(0.25)
    assert trial.user_attrs["best_val_selection_score"] == pytest.approx(0.25)
    assert trial.user_attrs["best_val_selection_metric"] == "mean_target_mae"
    assert trial.user_attrs["best_val_joint_loss"] == pytest.approx(1.75)
    assert trial.user_attrs["training_loss_objective"] == "mae"
    assert trial.user_attrs["best_val_mae_ww_log1p_per_100k"] == pytest.approx(0.2)
    assert captured["pruning_start_epoch"] == 3
    assert captured["constructor_model_id"] == "trial-000000"
    assert captured["trainer_model_id"] == "trial-000000"

    trial_dir = tmp_path / "study" / "trial-000000"
    checkpoint = trial_dir / "checkpoints/best_model.pt"
    checkpoint_sha256 = sha256_file(checkpoint)
    assert trial.user_attrs["trial_dir"] == str(trial_dir)
    assert trial.user_attrs["best_model_path"] == str(checkpoint)
    assert trial.user_attrs["best_model_sha256"] == checkpoint_sha256
    assert trial.user_attrs["epiforecaster_worker_id"] == 4

    summary_path = trial_dir / "optuna_trial.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["value"] == pytest.approx(0.25)
    assert summary["best_val_selection_score"] == pytest.approx(0.25)
    assert summary["best_val_joint_loss"] == pytest.approx(1.75)
    assert summary["training_loss_objective"] == "mae"
    assert summary["best_model_path"] == str(checkpoint)
    assert summary["best_model_sha256"] == checkpoint_sha256
