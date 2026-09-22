"""Native trainer canonical artifact lifecycle tests (WP-B; plan 12.2d).

Covers the production artifact contract: the per-model-id canonical root,
the three-head validation-selection score, checkpoint reload fidelity, the
training-dynamics terminal statuses, the shared loader worker policy, and
the explicit rejection of unimplemented configuration (W&B, compile,
periodic checkpointing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation.eval_loop import evaluate_native_loader
from models.configs import TrainingParams
from scripts.analysis.analyze_training_dynamics import analyze_training_dynamics
from tests.test_native_trainer_wiring import _trainer_config, _write_native_root
from training.native_trainer import (
    NativeNonFiniteHalt,
    NativeTrainer,
    compute_native_selection_score,
    load_native_checkpoint,
    native_loader_kwargs,
)
from training.training_dynamics import load_training_dynamics_identity

pytestmark = pytest.mark.epiforecaster


def _artifact_config(root: Path) -> object:
    """Trainer config with the canonical artifact outputs enabled."""
    config = _trainer_config(root)
    config.output.save_checkpoints = True
    config.output.write_granular_eval = True
    config.output.granular_eval_format = "csv"
    config.output.granular_eval_filename = "{split}_granular.csv"
    return config


def _run_trainer(root: Path, model_id: str) -> tuple[NativeTrainer, dict]:
    trainer = NativeTrainer(_artifact_config(root), model_id=model_id)
    return trainer, trainer.run()


# ----------------------------------------------------------------------
# (1) Canonical model-root layout and identities
# ----------------------------------------------------------------------


def test_canonical_model_root_layout_and_identity(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    trainer, results = _run_trainer(tmp_path, "canonical_m1")
    model_root = Path(results["model_root"])
    assert model_root == (
        Path(trainer.config.output.log_dir)
        / trainer.config.output.experiment_name
        / "canonical_m1"
    )
    for relative in (
        "training_dynamics.jsonl",
        "training_dynamics_manifest.json",
        "checkpoints/best_model.pt",
        "checkpoints/final_model.pt",
        "val_granular.csv",
        "val_granular.csv.meta.json",
        "val_native_aggregate_metrics.csv",
        "val_native_wis_summary.json",
    ):
        assert (model_root / relative).is_file(), f"missing {relative}"

    identity = load_training_dynamics_identity(model_root)
    assert identity["status"] == "complete"
    assert results["training_dynamics"]["status"] == "complete"

    analysis = analyze_training_dynamics(model_root)
    assert analysis["identity"]["manifest_status"] == "complete"

    assert results["best_val_selection_score"] is not None
    assert results["best_val_selection_metric"] == "mean_target_crps"
    assert results["best_val_joint_loss"] is not None
    assert set(results["best_val_series_metrics"]) == {"cases", "hosp", "deaths"}

    _, record = load_native_checkpoint(model_root / "checkpoints/best_model.pt")
    assert record["model_id"] == "canonical_m1"
    assert record["config"]["native"]["enabled"] is True
    assert record["precision_signature"]["param_dtype"] == "float32"
    assert record["scheduler_state_dict"] is not None


# ----------------------------------------------------------------------
# (2) Checkpoint reload reproduces the best-state validation rows
# ----------------------------------------------------------------------


def test_reload_reproduces_final_validation_metrics(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    trainer, results = _run_trainer(tmp_path, "reload_m1")
    model, _record = load_native_checkpoint(
        Path(results["model_root"]) / "checkpoints/best_model.pt"
    )
    assert model.training is False  # eval mode after reload

    loss, metrics, _aggregate = evaluate_native_loader(
        model=model,
        loader=trainer.val_loader,
        criterion=trainer.criterion,
        device=torch.device("cpu"),
        split_name="Val",
        max_batches=trainer.config.training.max_batches,
        confirmation_guard=trainer.config.native.confirmation_guard,
    )
    assert loss == pytest.approx(float(results["best_val_joint_loss"]), abs=1e-9)
    for key in ("crps_cases", "crps_hosp", "crps_deaths", "wis_cases"):
        assert metrics[key] == pytest.approx(
            float(trainer.last_val_metrics[key]), abs=1e-9, rel=1e-9
        ), f"reload diverged on {key}"


# ----------------------------------------------------------------------
# (3) Fold/seed identities cannot overwrite one another
# ----------------------------------------------------------------------


def test_model_ids_isolate_roots(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    _trainer_a, results_a = _run_trainer(tmp_path, "fold0_seed52")
    _trainer_b, results_b = _run_trainer(tmp_path, "fold1_seed52")

    experiment_dir = Path(results_a["model_root"]).parent
    assert Path(results_a["model_root"]).parent == Path(results_b["model_root"]).parent
    assert sorted(path.name for path in experiment_dir.iterdir()) == [
        "fold0_seed52",
        "fold1_seed52",
    ]
    for results in (results_a, results_b):
        identity = load_training_dynamics_identity(Path(results["model_root"]))
        assert identity["status"] == "complete"
        assert (Path(results["model_root"]) / "checkpoints/best_model.pt").is_file()
    # Nothing run-owned leaks into the shared experiment directory.
    assert not (experiment_dir / "checkpoints").exists()
    assert not (experiment_dir / "training_dynamics.jsonl").exists()


# ----------------------------------------------------------------------
# (4) Validation-selection score chain
# ----------------------------------------------------------------------


def test_selection_score_prefers_crps() -> None:
    metrics = {
        "crps_cases": 1.0,
        "crps_hosp": 2.0,
        "crps_deaths": 3.0,
        "nll_cases": 10.0,
        "mae_cases_log1p_per_100k": 0.5,
    }
    score, metric = compute_native_selection_score(metrics, observation_loss="nll")
    assert metric == "mean_target_crps"
    assert score == pytest.approx(2.0)


def test_selection_score_falls_back_to_nll_then_mae() -> None:
    nll_only = {"nll_cases": 1.0, "nll_hosp": 3.0, "mae_deaths_log1p_per_100k": 0.7}
    score, metric = compute_native_selection_score(nll_only, observation_loss="nll")
    assert metric == "mean_target_nll"
    assert score == pytest.approx(2.0)

    mae_only = {"mae_cases_log1p_per_100k": 0.4, "mae_hosp_log1p_per_100k": 0.6}
    score, metric = compute_native_selection_score(mae_only, observation_loss="nll")
    assert metric == "mean_target_mae"
    assert score == pytest.approx(0.5)

    # CRPS objective never falls through to NLL (municipal chain parity).
    score, metric = compute_native_selection_score(nll_only, observation_loss="crps")
    assert metric == "mean_target_mae"
    assert score == pytest.approx(0.7)


def test_selection_score_skips_nonfinite_and_bool_values() -> None:
    metrics = {
        "crps_cases": float("inf"),
        "crps_hosp": 2.0,
        "crps_deaths": 4.0,
        "crps_extra": True,
    }
    score, metric = compute_native_selection_score(metrics, observation_loss="crps")
    assert metric == "mean_target_crps"
    assert score == pytest.approx(3.0)


# ----------------------------------------------------------------------
# (5) Training-dynamics terminal statuses
# ----------------------------------------------------------------------


class _NaNCriterion:
    def __call__(self, outputs, batch):  # noqa: ANN001
        return {"total": torch.tensor(float("nan"))}


def test_nonfinite_halt_finalizes_manifest(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    trainer = NativeTrainer(_artifact_config(tmp_path), model_id="nan_halt")
    trainer.criterion = _NaNCriterion()
    with pytest.raises(NativeNonFiniteHalt):
        trainer.run()
    manifest = json.loads(
        (trainer.experiment_dir / "training_dynamics_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "nonfinite_halt"
    assert manifest["error"]["type"] == "NativeNonFiniteHalt"


def test_unrelated_failure_finalizes_failed_manifest(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    trainer = NativeTrainer(_artifact_config(tmp_path), model_id="failed_run")

    def _boom(epoch: int) -> list[float]:
        raise RuntimeError("boom")

    trainer._train_epoch = _boom
    with pytest.raises(RuntimeError, match="boom"):
        trainer.run()
    manifest = json.loads(
        (trainer.experiment_dir / "training_dynamics_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["error"]["message"] == "boom"


# ----------------------------------------------------------------------
# (6) Shared loader worker policy
# ----------------------------------------------------------------------


def test_loader_kwargs_worker_modes() -> None:
    zero = native_loader_kwargs(
        TrainingParams(num_workers=0, val_workers=0, enable_mixed_precision=False),
        workers=0,
        device_type="cpu",
    )
    assert zero == {"num_workers": 0, "pin_memory": False}

    positive = native_loader_kwargs(
        TrainingParams(
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
            enable_mixed_precision=False,
        ),
        workers=2,
        device_type="cpu",
    )
    assert positive["num_workers"] == 2
    assert positive["pin_memory"] is False  # CPU never pins
    assert positive["persistent_workers"] is True
    assert positive["prefetch_factor"] == 2
    # The context is present for positive workers ("fork" on Linux; None
    # selects the platform default off Linux — both are valid DataLoader
    # values, mirroring the municipal dataloader factory).
    assert "multiprocessing_context" in positive

    cuda = native_loader_kwargs(
        TrainingParams(pin_memory=True, enable_mixed_precision=False),
        workers=1,
        device_type="cuda",
    )
    assert cuda["pin_memory"] is True

    no_persist = native_loader_kwargs(
        TrainingParams(
            persistent_workers=False,
            prefetch_factor=None,
            enable_mixed_precision=False,
        ),
        workers=2,
        device_type="cpu",
    )
    assert "persistent_workers" not in no_persist
    assert "prefetch_factor" not in no_persist
    assert "multiprocessing_context" in no_persist

    auto = native_loader_kwargs(
        TrainingParams(enable_mixed_precision=False),
        workers=-1,
        device_type="cpu",
    )
    assert auto["num_workers"] >= 1


# ----------------------------------------------------------------------
# (7) Unimplemented configuration is rejected, not ignored
# ----------------------------------------------------------------------


def test_unsupported_configuration_rejected(tmp_path: Path) -> None:
    _write_native_root(tmp_path)

    wandb = _artifact_config(tmp_path)
    wandb.output.wandb_group = "stage4"
    with pytest.raises(ValueError, match="wandb_group"):
        NativeTrainer(wandb, model_id="reject_wandb")

    compiled = _artifact_config(tmp_path)
    compiled.training.compile = True
    with pytest.raises(ValueError, match="torch.compile"):
        NativeTrainer(compiled, model_id="reject_compile")

    periodic = _artifact_config(tmp_path)
    periodic.output.save_best_only = False
    with pytest.raises(ValueError, match="save_best_only"):
        NativeTrainer(periodic, model_id="reject_periodic")
