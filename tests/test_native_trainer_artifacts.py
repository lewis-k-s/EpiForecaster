"""Native trainer canonical artifact lifecycle tests (WP-B; plan 12.2d).

Covers the production artifact contract: the per-model-id canonical root,
the active-head validation-selection score, checkpoint reload fidelity, the
training-dynamics terminal statuses, the shared loader worker policy, and
the explicit rejection of unimplemented configuration (W&B, compile,
periodic checkpointing).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from evaluation.eval_loop import evaluate_native_loader
from models.configs import TrainingParams
from scripts.analysis.analyze_training_dynamics import analyze_training_dynamics
from tests.native_fixtures import _trainer_config, _write_native_root
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


def test_native_info_progress_follows_configured_cadence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_native_root(tmp_path)
    config = _artifact_config(tmp_path)
    config.training.max_batches = 3
    config.training.progress_log_frequency = 2
    trainer = NativeTrainer(config, model_id="progress_cadence")

    with caplog.at_level(logging.INFO, logger="training.native_trainer"):
        trainer.run()

    progress = [
        record.getMessage()
        for record in caplog.records
        if record.name == "training.native_trainer"
        and "native epoch 1 step" in record.getMessage()
    ]
    assert [message.split(" total_loss=")[0] for message in progress] == [
        "native epoch 1 step 2/3",
        "native epoch 1 step 3/3",
    ]


def test_native_training_device_metrics_preserve_weighted_epoch_scores(
    tmp_path: Path,
) -> None:
    _write_native_root(tmp_path)
    config = _artifact_config(tmp_path)
    config.training.max_batches = 3
    trainer = NativeTrainer(config, model_id="device_metrics")
    criterion = trainer.criterion
    expected_losses = []
    rows = [
        {"ww": (1.25, 0), "cases": (2.1, 2), "hosp": (0.0, 0), "deaths": (4.2, 1)},
        {"ww": (2.5, 2), "cases": (3.2, 7), "hosp": (0.0, 0), "deaths": (0.0, 0)},
        {"ww": (4.75, 3), "cases": (5.3, 1), "hosp": (0.0, 0), "deaths": (6.4, 9)},
    ]
    mixed_rows = [(7.1, 1), (8.2, 4), (9.3, 3)]
    expected_sums = dict.fromkeys(rows[0], 0.0)
    expected_counts = dict.fromkeys(rows[0], 0.0)
    expected_mixed_sum = 0.0

    def measured_components(outputs, batch):
        nonlocal expected_mixed_sum
        components = criterion(outputs, batch)
        index = len(expected_losses)
        expected_losses.append(float(components["total"].detach()))
        for head, (score, count) in rows[index].items():
            score_tensor = components["total"].new_tensor(score)
            count_tensor = components["total"].new_tensor(count)
            score_key = "ww_uncensored_objective" if head == "ww" else head
            count_key = "ww_uncensored_count" if head == "ww" else f"{head}_n_eff"
            components[score_key] = score_tensor
            components[count_key] = count_tensor
            expected_sums[head] += float(score_tensor) * float(count_tensor)
            expected_counts[head] += float(count_tensor)
        score, count = mixed_rows[index]
        components["ww"] = components["total"].new_tensor(score)
        components["ww_n_eff"] = components["total"].new_tensor(count)
        expected_mixed_sum += float(components["ww"]) * count
        return components

    trainer.criterion = measured_components
    losses = trainer._train_epoch(0)
    trace = trainer.train_epoch_trace[-1]
    assert losses == expected_losses
    assert trace["joint_loss"] == pytest.approx(sum(expected_losses) / 3)
    for head in rows[0]:
        expected = (
            expected_sums[head] / expected_counts[head]
            if expected_counts[head]
            else None
        )
        key = f"{config.training.loss.joint.observation_loss}_{head}"
        if expected is None:
            assert trace[key] is None
        else:
            assert trace[key] == pytest.approx(expected, rel=1e-12)
        assert trace[f"observed_count_{head}"] == (
            8 if head == "ww" else expected_counts[head]
        )
    assert trace["ww_uncensored_count"] == 5
    assert trace["ww_censored_count"] == 3
    assert trace["ww_mixed_objective"] == pytest.approx(
        expected_mixed_sum / 8, rel=1e-12
    )


def test_native_training_empty_epoch_has_no_scores(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    trainer = NativeTrainer(_artifact_config(tmp_path), model_id="empty_metrics")
    trainer.train_loader = []
    assert trainer._train_epoch(0) == []
    trace = trainer.train_epoch_trace[-1]
    assert trace["joint_loss"] is None
    assert trace["ww_mixed_objective"] is None
    for head in ("ww", "cases", "hosp", "deaths"):
        assert trace[f"observed_count_{head}"] == 0
        assert (
            trace[f"{trainer.config.training.loss.joint.observation_loss}_{head}"]
            is None
        )


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
        "config.json",
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
    dynamics_manifest = json.loads(
        (model_root / "training_dynamics_manifest.json").read_text()
    )
    epoch_trace = dynamics_manifest["summary"]["validation_epoch_trace"]
    assert epoch_trace == results["validation_epoch_trace"]
    assert (
        dynamics_manifest["summary"]["train_epoch_trace"]
        == results["train_epoch_trace"]
    )
    assert len(epoch_trace) == trainer.config.training.epochs
    assert (
        results["train_epoch_trace"][0]["batch_size"]
        == trainer.config.training.batch_size
    )
    assert (
        json.loads((model_root / "config.json").read_text()) == trainer.config.to_dict()
    )
    assert epoch_trace[0]["epoch"] == 1
    assert epoch_trace[0]["step"] == trainer.global_step
    assert epoch_trace[0]["joint_loss"] == pytest.approx(
        results["val_joint_loss_trace"][0]
    )
    assert epoch_trace[0]["selection_metric"] == "mean_target_wis"
    assert results["val_loss_metric"] == "mean_target_wis"
    assert results["val_loss_trace"][0] == epoch_trace[0]["selection_score"]
    assert results["best_val_loss"] == results["best_val_selection_score"]
    assert epoch_trace[0]["observed_count_cases"] > 0
    assert epoch_trace[0]["crps_cases"] is not None
    assert epoch_trace[0]["wis_cases"] == pytest.approx(
        sum(
            epoch_trace[0][f"wis_{component}_cases"]
            for component in (
                "dispersion",
                "underprediction",
                "overprediction",
                "point",
            )
        )
    )
    assert epoch_trace[0]["censored_count_ww"] == 0

    analysis = analyze_training_dynamics(model_root)
    assert analysis["identity"]["manifest_status"] == "complete"

    assert results["best_val_selection_score"] is not None
    assert results["best_val_selection_metric"] == "mean_target_wis"
    assert results["best_val_joint_loss"] is not None
    assert set(results["best_val_series_metrics"]) == {
        "ww",
        "cases",
        "hosp",
        "deaths",
    }
    assert results["best_val_series_metrics"]["ww"] is None
    assert results["best_val_series_metrics"]["cases"] == epoch_trace[0]["wis_cases"]

    _, record = load_native_checkpoint(model_root / "checkpoints/best_model.pt")
    assert record["model_id"] == "canonical_m1"
    assert record["config"]["native"]["enabled"] is True
    assert record["precision_signature"]["param_dtype"] == "float32"
    assert record["scheduler_state_dict"] is not None
    assert record["val_loss_metric"] == "mean_target_wis"
    assert record["val_loss_trace"] == results["val_loss_trace"]
    assert record["val_joint_loss_trace"] == results["val_joint_loss_trace"]


@pytest.mark.parametrize("objective", ["nll", "crps", "mse"])
def test_checkpoint_follows_wis_when_other_scores_disagree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    objective: str,
) -> None:
    _write_native_root(tmp_path)
    config = _artifact_config(tmp_path)
    config.training.epochs = 3
    config.training.loss.joint.observation_loss = objective
    config.output.write_granular_eval = False
    trainer = NativeTrainer(config, model_id="wis_ranking")
    trainer.test_loader = None
    parameter = next(trainer.model.parameters())

    def train_epoch(epoch):
        with torch.no_grad():
            parameter.fill_(epoch + 1)
        trainer.global_step += trainer.steps_per_epoch
        return [1.0] * trainer.steps_per_epoch

    evaluations = iter(
        [
            (3.0, {"wis_cases": 0.5, "crps_cases": 0.1, "observed_count_cases": 2}),
            (2.0, {"wis_cases": 0.2, "crps_cases": 0.2, "observed_count_cases": 2}),
            (1.0, {"wis_cases": 0.4, "crps_cases": 0.3, "observed_count_cases": 2}),
        ]
    )
    monkeypatch.setattr(trainer, "_train_epoch", train_epoch)
    monkeypatch.setattr(trainer, "_evaluate", lambda: next(evaluations))
    result = trainer.run()
    assert result["val_loss_trace"] == [0.5, 0.2, 0.4]
    assert result["val_joint_loss_trace"] == [3.0, 2.0, 1.0]
    assert result["best_val_loss"] == 0.2
    assert result["best_val_joint_loss"] == 2.0
    assert torch.all(parameter == 2.0)
    _, checkpoint = load_native_checkpoint(trainer.checkpoint_dir / "best_model.pt")
    assert checkpoint["epoch"] == 1
    assert checkpoint["best_val_series_metrics"]["cases"] == 0.2
    assert checkpoint["val_loss_trace"] == [0.5, 0.2]
    _, final = load_native_checkpoint(trainer.checkpoint_dir / "final_model.pt")
    assert final["val_loss_trace"] == [0.5, 0.2, 0.4]


def test_diagnostic_test_epoch_trace_does_not_select_checkpoint(
    tmp_path: Path,
) -> None:
    _write_native_root(tmp_path)
    config = _artifact_config(tmp_path)
    trainer = NativeTrainer(config, model_id="diagnostic_trace")
    # Reuse the bounded validation loader to test the diagnostic evaluation path.
    trainer.test_loader = trainer.val_loader
    trainer.config.training.diagnostic_test_trace_epochs = [1]

    results = trainer.run()

    assert len(results["validation_epoch_trace"]) == 1
    assert len(results["test_epoch_trace"]) == 1
    assert results["test_epoch_trace"][0]["epoch"] == 1
    assert results["test_epoch_trace"][0]["crps_cases"] == pytest.approx(
        results["validation_epoch_trace"][0]["crps_cases"]
    )
    assert results["best_val_selection_score"] == pytest.approx(
        results["validation_epoch_trace"][0]["selection_score"]
    )
    manifest = json.loads(
        (Path(results["model_root"]) / "training_dynamics_manifest.json").read_text()
    )
    assert manifest["summary"]["test_epoch_trace"] == results["test_epoch_trace"]


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


def test_selection_score_prefers_wis() -> None:
    metrics = {
        "wis_cases": 1.0,
        "wis_hosp": 2.0,
        "wis_deaths": 3.0,
        "crps_cases": 10.0,
        "nll_cases": 20.0,
        "mae_cases_log1p_per_100k": 0.5,
    }
    # Selection does not receive or depend on the training objective.
    score, metric = compute_native_selection_score(metrics)
    assert metric == "mean_target_wis"
    assert score == pytest.approx(2.0)


def test_selection_score_includes_native_wastewater_when_available() -> None:
    metrics = {"wis_ww": 4.0, "wis_cases": 1.0, "wis_hosp": 2.0, "wis_deaths": 3.0}
    score, metric = compute_native_selection_score(metrics)
    assert metric == "mean_target_wis"
    assert score == pytest.approx(2.5)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_selection_score_rejects_missing_wis_for_observed_head(value) -> None:
    with pytest.raises(ValueError, match="Observed head 'cases'.*WIS"):
        compute_native_selection_score(
            {
                "observed_count_cases": 2,
                "wis_cases": value,
                "wis_hosp": 1.0,
            }
        )


def test_selection_score_excludes_unobserved_heads_and_never_falls_back() -> None:
    score, metric = compute_native_selection_score(
        {
            "wis_cases": 2.0,
            "observed_count_cases": 10,
            "wis_hosp": 4.0,
            "observed_count_hosp": 1,
            "wis_ww": 100.0,
            "observed_count_ww": 0,
        }
    )
    assert metric == "mean_target_wis"
    assert score == pytest.approx(3.0)  # Equal heads, not observed-count weights.
    with pytest.raises(ValueError, match="at least one finite observed WIS"):
        compute_native_selection_score(
            {
                "crps_cases": 1.0,
                "nll_cases": 2.0,
                "mae_cases_log1p_per_100k": 0.5,
            }
        )


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


def test_nonfinite_gradient_halts_before_optimizer_step(tmp_path: Path) -> None:
    """A finite loss with NaN gradients must halt without stepping."""
    _write_native_root(tmp_path)
    trainer = NativeTrainer(_artifact_config(tmp_path), model_id="nan_grad")
    params_before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    hook = next(trainer.model.parameters()).register_hook(
        lambda grad: torch.full_like(grad, float("nan"))
    )
    try:
        with pytest.raises(NativeNonFiniteHalt, match="gradient norm"):
            trainer.run()
    finally:
        hook.remove()
    # The optimizer never consumed the corrupted gradients.
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter.detach(), params_before[name]), name

    manifest = json.loads(
        (trainer.experiment_dir / "training_dynamics_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "nonfinite_halt"
    rows = [
        json.loads(line)
        for line in (trainer.experiment_dir / "training_dynamics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert rows, "halt row must be recorded"
    assert rows[-1]["nonfinite_fields"] == ["grad_norm"]
    assert rows[-1]["grad_norm"] is None
    assert rows[-1]["train_loss"] is not None  # the loss itself was finite


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
    # CPU workers retain the shared platform policy.
    assert "multiprocessing_context" in positive

    cuda = native_loader_kwargs(
        TrainingParams(pin_memory=True, enable_mixed_precision=False),
        workers=1,
        device_type="cuda",
    )
    assert cuda["pin_memory"] is True
    assert cuda["multiprocessing_context"] == "spawn"

    zero_cuda = native_loader_kwargs(
        TrainingParams(pin_memory=True, enable_mixed_precision=False),
        workers=0,
        device_type="cuda",
    )
    assert "multiprocessing_context" not in zero_cuda

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

    online = _artifact_config(tmp_path)
    online.output.wandb_mode = "online"
    with pytest.raises(ValueError, match="wandb_mode"):
        NativeTrainer(online, model_id="reject_wandb_mode")

    compiled = _artifact_config(tmp_path)
    compiled.training.compile = True
    with pytest.raises(ValueError, match="torch.compile"):
        NativeTrainer(compiled, model_id="reject_compile")

    periodic = _artifact_config(tmp_path)
    periodic.output.save_best_only = False
    with pytest.raises(ValueError, match="save_best_only"):
        NativeTrainer(periodic, model_id="reject_periodic")


@pytest.mark.parametrize("batch_multiplier", [1, 2])
def test_native_resume_restores_schedule_and_continues_epoch_history(
    tmp_path: Path, monkeypatch, batch_multiplier: int
) -> None:
    _write_native_root(tmp_path)
    config = _artifact_config(tmp_path)
    config.training.epochs = 3
    config.training.max_batches = None
    config.training.batch_size = 64
    parent = NativeTrainer(config, model_id="resume_parent")
    train_epoch = parent._train_epoch

    def interrupt_after_first(epoch):
        if epoch == 1:
            raise RuntimeError("test interruption")
        return train_epoch(epoch)

    monkeypatch.setattr(parent, "_train_epoch", interrupt_after_first)
    with pytest.raises(RuntimeError, match="test interruption"):
        parent.run()
    checkpoint = parent.checkpoint_dir / "best_model.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Older native checkpoints do not contain per-head training history.
    saved.pop("train_epoch_trace")
    torch.save(saved, checkpoint)
    resumed_config = _artifact_config(tmp_path)
    resumed_config.training.epochs = 3
    resumed_config.training.max_batches = None
    resumed_config.training.batch_size = 64 * batch_multiplier
    resumed_config.training.resume_checkpoint_path = str(checkpoint)
    resumed = NativeTrainer(resumed_config, model_id="resume_child")
    assert resumed.start_epoch == 1
    assert resumed.global_step == parent.global_step
    assert (
        resumed.optimizer.param_groups[0]["lr"]
        == parent.optimizer.param_groups[0]["lr"]
    )
    assert resumed.scheduler.state_dict() == parent.scheduler.state_dict()
    for parameter, original in zip(
        resumed.model.parameters(), parent.model.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, original)
    assert resumed.best_val_selection_score == parent.best_val_selection_score
    assert resumed.resumed_val_trace == saved["val_joint_loss_trace"]
    assert (resumed.checkpoint_dir / "best_model.pt").is_file()
    epochs = []
    original_train_epoch = resumed._train_epoch

    def collect_epoch(epoch):
        epochs.append(epoch)
        return original_train_epoch(epoch)

    monkeypatch.setattr(resumed, "_train_epoch", collect_epoch)
    resumed.run()
    assert epochs == [1, 2]
    assert [row["epoch"] for row in resumed.validation_epoch_trace] == [1, 2, 3]
    assert resumed.validation_epoch_trace[0] == saved["validation_epoch_trace"][0]
    assert resumed.global_step == parent.global_step + 2 * resumed.steps_per_epoch
    if batch_multiplier == 2:
        assert resumed.steps_per_epoch < parent.steps_per_epoch
    assert resumed.scheduler.last_epoch == resumed.global_step
    assert resumed.scheduler.T_max == parent.scheduler.T_max
    broken = dict(saved)
    broken["global_step"] += 1
    broken_path = tmp_path / "broken_resume.pt"
    torch.save(broken, broken_path)
    resumed_config.training.resume_checkpoint_path = str(broken_path)
    with pytest.raises(ValueError, match="completed-step history differs"):
        NativeTrainer(resumed_config, model_id="resume_bad_steps")
    incompatible = _artifact_config(tmp_path)
    incompatible.training.epochs = 4
    incompatible.training.resume_checkpoint_path = str(checkpoint)
    with pytest.raises(ValueError, match="training.epochs differs"):
        NativeTrainer(incompatible, model_id="resume_bad_schedule")
