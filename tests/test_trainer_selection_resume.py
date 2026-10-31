"""Tests for CRPS auxiliary checkpoint, resume policy, and probe persistence."""

import inspect
import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.configs import TrainingParams
from training.epiforecaster_trainer import EpiForecasterTrainer

HEADS = ["ww", "hosp", "cases", "deaths"]


def _crps_trainer(observation_loss: str = "nll") -> EpiForecasterTrainer:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(selection_required_heads=HEADS),
        output=SimpleNamespace(save_checkpoints=True),
    )
    trainer.criterion = SimpleNamespace(observation_loss=observation_loss)
    trainer.best_val_crps = float("inf")
    trainer.best_val_selection_score = 7.0
    trainer.best_val_selection_metric = "mean_target_wis"
    trainer.saved = []
    trainer._save_checkpoint = lambda *a, **k: trainer.saved.append((a, k))
    trainer._status = lambda *a, **k: trainer.__dict__.setdefault("logs", []).append(
        (a, k)
    )
    return trainer


def test_retain_best_crps_requires_probabilistic_loss() -> None:
    params = TrainingParams(retain_best_crps_checkpoint=True)
    assert params.loss.joint.observation_loss == "nll"
    for loss in ["mse", "rmse", "mae"]:
        with pytest.raises(ValueError, match="probabilistic"):
            TrainingParams(
                retain_best_crps_checkpoint=True,
                loss={"joint": {"observation_loss": loss}},
            )
    TrainingParams(
        retain_best_crps_checkpoint=True,
        loss={"joint": {"observation_loss": "crps"}},
    )


def test_invalid_crps_epoch_skips_auxiliary_save_with_warning() -> None:
    trainer = _crps_trainer()
    good = {f"crps_{h}_log1p_per_100k": 2.0 for h in HEADS}
    missing = {f"crps_{h}_log1p_per_100k": 1.0 for h in HEADS[:-1]}
    invalid = dict(good, invalid_prediction_count_ww=3)
    nonfinite = dict(good, crps_ww_log1p_per_100k=float("nan"))
    for metrics in (missing, invalid, nonfinite):
        trainer._maybe_save_crps_checkpoint(0, 1.0, metrics)
    assert trainer.saved == []
    assert trainer.best_val_crps == float("inf")
    assert len(trainer.logs) == 3
    assert all(a[1] == logging.WARNING for a, _ in trainer.logs)
    trainer._maybe_save_crps_checkpoint(1, 1.0, good)
    assert len(trainer.saved) == 1


def test_crps_save_follows_primary_bookkeeping_in_epoch_loop() -> None:
    source = inspect.getsource(EpiForecasterTrainer._run_training)
    crps = source.index("self._maybe_save_crps_checkpoint(")
    assert crps > source.index("self.best_val_selection_score = val_selection_score")
    assert crps > source.index('self.training_history["epoch_times"].append')


def _resume_trainer(config_metric: str, saved_metric: str, score: float = 1.0):
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(selection_metric=config_metric)
    )
    trainer.best_val_selection_score = score
    trainer.best_val_selection_metric = saved_metric
    trainer._resume_selection_metric = None
    return trainer


def test_resume_explicit_metric_mismatch_raises() -> None:
    trainer = _resume_trainer("mean_target_wis", "mean_target_crps")
    with pytest.raises(ValueError, match="selection metric mismatch"):
        trainer._validate_resumed_selection_metric("mean_target_wis")
    trainer = _resume_trainer("mean_target_wis", "mean_target_wis")
    trainer._validate_resumed_selection_metric("mean_target_wis")


def test_resume_auto_compares_on_first_validation() -> None:
    trainer = _resume_trainer("auto", "mean_target_wis")
    trainer._resume_selection_metric = trainer.best_val_selection_metric
    with pytest.raises(ValueError, match="selection metric mismatch"):
        trainer._validate_resumed_selection_metric("mean_target_crps")
    trainer._resume_selection_metric = "mean_target_crps"
    trainer._validate_resumed_selection_metric("mean_target_crps")
    assert trainer._resume_selection_metric is None


def test_patience_and_rng_state_round_trip() -> None:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = EpiForecasterTrainer._capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    EpiForecasterTrainer._restore_rng_state(state)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected
    EpiForecasterTrainer._restore_rng_state(None)


def test_probe_names_strip_compile_prefix_and_persist_only_when_active() -> None:
    source = inspect.getsource(EpiForecasterTrainer._init_gradient_norm_groups)
    assert '.replace("_orig_mod.", "")' in source
    run_source = inspect.getsource(EpiForecasterTrainer.setup_logging)
    assert "pathway_gradient_frequency > 0" in run_source


def test_checkpoint_saves_patience_and_rng_state(tmp_path) -> None:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    model = torch.nn.Linear(1, 1)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.scheduler = None
    trainer.global_step = 3
    trainer.best_val_loss = 1.0
    trainer.best_val_crps = 1.0
    trainer.best_val_selection_score = 1.0
    trainer.best_val_selection_metric = "mean_target_mae"
    trainer.best_val_joint_loss = 1.0
    trainer.best_val_series_metrics = {}
    trainer.training_history = {}
    trainer.patience_counter = 4
    trainer.latent_scale_state = None
    trainer.precision_policy = SimpleNamespace(
        param_dtype=torch.float32,
        autocast_enabled=False,
        autocast_dtype=None,
        optimizer_eps=1e-8,
    )
    trainer._active_residual_schedule_state = None
    trainer._get_config_for_save = lambda: {}
    trainer._status = lambda *a, **k: None
    trainer.checkpoint_dir = tmp_path
    trainer._save_checkpoint(0, 1.0)
    checkpoint = torch.load(
        next(tmp_path.glob("checkpoint_epoch_*.pt")), weights_only=False
    )
    assert checkpoint["patience_counter"] == 4
    assert set(checkpoint["rng_state"]) == {"python", "numpy", "torch"}
