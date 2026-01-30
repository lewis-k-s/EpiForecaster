from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from training.epiforecaster_trainer import EpiForecasterTrainer


def _make_trainer_stub(config) -> EpiForecasterTrainer:
    trainer = EpiForecasterTrainer.__new__(EpiForecasterTrainer)
    trainer.config = config
    trainer._status = lambda *_args, **_kwargs: None
    return trainer


@pytest.mark.epiforecaster
def test_resolve_model_id_prefers_config(monkeypatch) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(model_id="explicit"),
        output=SimpleNamespace(log_dir="unused", experiment_name="exp"),
    )
    trainer = _make_trainer_stub(config)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert trainer._resolve_model_id() == "explicit"


@pytest.mark.epiforecaster
def test_resolve_model_id_falls_back_to_slurm(monkeypatch) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(model_id=""),
        output=SimpleNamespace(log_dir="unused", experiment_name="exp"),
    )
    trainer = _make_trainer_stub(config)
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    assert trainer._resolve_model_id() == "456"


@pytest.mark.epiforecaster
def test_resolve_model_id_interactive_slurm_uses_datetime(monkeypatch) -> None:
    """Interactive SLURM sessions should use datetime ID, not SLURM_JOB_ID."""
    config = SimpleNamespace(
        training=SimpleNamespace(model_id=""),
        output=SimpleNamespace(log_dir="unused", experiment_name="exp"),
    )
    trainer = _make_trainer_stub(config)
    # Simulate interactive SLURM session
    monkeypatch.setenv("SLURM_JOB_ID", "35320487")
    monkeypatch.setenv("SLURM_JOB_NAME", "interactive")
    result = trainer._resolve_model_id()
    assert result.startswith("run_")
    assert result != "35320487"


@pytest.mark.epiforecaster
def test_resolve_model_id_interactive_qos_detection(monkeypatch) -> None:
    """Detect interactive sessions via _interactive in QOS."""
    config = SimpleNamespace(
        training=SimpleNamespace(model_id=""),
        output=SimpleNamespace(log_dir="unused", experiment_name="exp"),
    )
    trainer = _make_trainer_stub(config)
    # Simulate interactive SLURM via QOS
    monkeypatch.setenv("SLURM_JOB_ID", "35320487")
    monkeypatch.setenv("SLURM_JOB_QOS", "acc_interactive")
    result = trainer._resolve_model_id()
    assert result.startswith("run_")
    assert result != "35320487"


@pytest.mark.epiforecaster
def test_resolve_model_id_batch_job_uses_slurm_id(monkeypatch) -> None:
    """Batch jobs should still use SLURM_JOB_ID."""
    config = SimpleNamespace(
        training=SimpleNamespace(model_id=""),
        output=SimpleNamespace(log_dir="unused", experiment_name="exp"),
    )
    trainer = _make_trainer_stub(config)
    # Simulate batch job (no interactive markers)
    monkeypatch.setenv("SLURM_JOB_ID", "35320487")
    monkeypatch.setenv("SLURM_JOB_NAME", "train_epoch")
    monkeypatch.setenv("SLURM_JOB_QOS", "normal")
    assert trainer._resolve_model_id() == "35320487"


@pytest.mark.epiforecaster
def test_find_checkpoint_for_model_id_prefers_best(tmp_path) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(model_id=""),
        output=SimpleNamespace(
            log_dir=str(tmp_path),
            experiment_name="exp",
            save_checkpoints=True,
        ),
    )
    trainer = _make_trainer_stub(config)
    trainer.model_id = "run_001"
    checkpoint_dir = tmp_path / "exp" / trainer.model_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best_path = checkpoint_dir / "best_model.pt"
    best_path.touch()
    (checkpoint_dir / "final_model.pt").touch()
    (checkpoint_dir / "checkpoint_epoch_0005.pt").touch()
    assert trainer._find_checkpoint_for_model_id() == best_path


@pytest.mark.epiforecaster
def test_find_checkpoint_for_model_id_falls_back_to_latest(tmp_path) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(model_id=""),
        output=SimpleNamespace(
            log_dir=str(tmp_path),
            experiment_name="exp",
            save_checkpoints=True,
        ),
    )
    trainer = _make_trainer_stub(config)
    trainer.model_id = "run_002"
    checkpoint_dir = tmp_path / "exp" / trainer.model_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    first = checkpoint_dir / "checkpoint_epoch_0001.pt"
    last = checkpoint_dir / "checkpoint_epoch_0003.pt"
    first.touch()
    last.touch()
    assert trainer._find_checkpoint_for_model_id() == last


@pytest.mark.epiforecaster
def test_resume_from_checkpoint_loads_state(tmp_path) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(model_id=""),
        output=SimpleNamespace(
            log_dir=str(tmp_path),
            experiment_name="exp",
            save_checkpoints=True,
        ),
    )
    trainer = _make_trainer_stub(config)
    trainer.model_id = "run_003"
    trainer.model = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = None
    trainer.best_val_loss = float("inf")
    trainer.training_history = {"train_loss": []}
    trainer.current_epoch = 0

    checkpoint_dir = tmp_path / "exp" / trainer.model_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    for param in trainer.model.parameters():
        torch.nn.init.constant_(param, 0.0)
    original_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    updated_state = {k: v + 1.0 for k, v in original_state.items()}

    torch.save(
        {
            "epoch": 4,
            "model_state_dict": updated_state,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "best_val_loss": 0.123,
            "training_history": {"train_loss": [1.0]},
        },
        checkpoint_path,
    )

    trainer._resume_from_checkpoint()
    assert trainer.current_epoch == 5
    assert trainer.best_val_loss == 0.123
    assert trainer.training_history["train_loss"] == [1.0]
    for key, value in trainer.model.state_dict().items():
        assert torch.allclose(value, updated_state[key])
