"""Native diagnostics must preserve optimization and canonical score identities."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from evaluation.aggregate_export import NativeAggregateAccumulator
from models.configs import EpiForecasterConfig, EvaluationConfig, TrainingParams
from tests.native_fixtures import _write_native_root, _write_native_wastewater
from tests.test_native_trainer_artifacts import _artifact_config
from training.native_diagnostics import (
    fixed_training_panel,
    native_component_gradient_norms,
)
from training.native_trainer import NativeTrainer

pytestmark = pytest.mark.epiforecaster


@pytest.mark.parametrize("value", [-1, True, 1.5])
@pytest.mark.parametrize(
    "name", ["training_panel_queries_per_source", "minimum_validation_origins"]
)
def test_panel_limits_reject_invalid_values(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        EvaluationConfig(**{name: value})


def test_diagnostics_config_requires_explicit_panel_enablement() -> None:
    with pytest.raises(ValueError, match="epoch_diagnostics"):
        EvaluationConfig(training_panel_queries_per_source=3)
    with pytest.raises(ValueError, match="component_gradient_diagnostics"):
        TrainingParams(component_gradient_diagnostics="true")
    config = EpiForecasterConfig.apply_overrides(
        _artifact_config(Path("/tmp/unused-native-test")),
        [
            "evaluation.epoch_diagnostics=true",
            "evaluation.training_panel_queries_per_source=3",
            "evaluation.minimum_validation_origins=56",
            "training.component_gradient_diagnostics=true",
        ],
    )
    assert config.evaluation.training_panel_queries_per_source == 3
    assert config.evaluation.minimum_validation_origins == 56
    assert config.training.component_gradient_diagnostics is True


def test_fixed_panel_is_source_stratified_unique_and_rng_free() -> None:
    items = {
        "cases": {("municipality", "A"): list(range(10))},
        "ww": {("edar", "B"): [10, 11], ("edar", "C"): [12]},
    }
    state = torch.get_rng_state().clone()
    assert fixed_training_panel(items, 0) == []
    assert fixed_training_panel(items, 3) == [0, 4, 9, 10, 11, 12]
    assert fixed_training_panel(items, 1) == [5, 11, 12]
    torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)


def test_component_norms_are_disjoint_preclip_and_read_only() -> None:
    model = torch.nn.Module()
    model.spatial_encoder = torch.nn.Linear(2, 1)
    model.input_adapter = torch.nn.Linear(2, 1)
    model.wastewater_head = torch.nn.Linear(2, 1)
    model.base = torch.nn.Module()
    model.base.cases_head = torch.nn.Linear(2, 1)
    model.unused = torch.nn.Linear(2, 1)
    gradients = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if not name.startswith("unused."):
            parameter.grad = torch.full_like(parameter, index + 1.0)
            gradients[name] = parameter.grad.clone()
    norms = native_component_gradient_norms(model)
    assert set(norms) == {
        "gradnorm_native_spatial_encoder",
        "gradnorm_backbone_encoder",
        "gradnorm_obs_ww",
        "gradnorm_obs_cases",
    }
    total_squared = sum(
        float(gradient.square().sum()) for gradient in gradients.values()
    )
    assert sum(norm**2 for norm in norms.values()) == pytest.approx(total_squared)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        else:
            assert parameter.grad is None


def test_validation_origin_guard_applies_after_purging(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    config = _artifact_config(tmp_path)
    config.evaluation.minimum_validation_origins = 56
    with pytest.raises(ValueError, match="origins after purging"):
        NativeTrainer(config, model_id="too_narrow")


def test_origin_blocks_use_canonical_scores_and_keep_masked_targets_out() -> None:
    accumulator = NativeAggregateAccumulator(origin_block_days=28)
    values = torch.tensor([[1.0], [3.0], [100.0]])
    common = dict(
        head="ww",
        source_types=["edar"] * 3,
        source_ids=["A"] * 3,
        predictions=values,
        targets=torch.zeros_like(values),
        raw_predictions=values,
        raw_targets=torch.zeros_like(values),
        weights=torch.tensor([[1.0], [1.0], [0.0]]),
        nll=values,
        crps=values,
        intervals={"80": (values - 4, values + 4)},
        wis_components={
            name: values
            for name in [
                "wis",
                "dispersion",
                "underprediction",
                "overprediction",
                "point",
            ]
        },
    )
    with pytest.raises(ValueError, match="one origin per row"):
        accumulator.update(**common)
    accumulator.update(**common, origin_indices=[27, 28, 56])
    table = accumulator.finalize()
    blocks = table[table.aggregation == "origin_block"].set_index("source_id")
    assert set(blocks.index) == {"0", "28"}
    assert blocks.loc["0", "wis"] == 1.0
    assert blocks.loc["28", "wis"] == 3.0
    assert blocks.observed_count.sum() == 2
    assert blocks.coverage_80.tolist() == [1.0, 1.0]
    assert blocks.interval_width_80.tolist() == [8.0, 8.0]
    assert table[table.aggregation == "head_all"].iloc[0].wis == 2.0


@pytest.mark.parametrize("optional_context", [False, True])
def test_epoch_diagnostics_preserve_training_and_publish_hashed_metrics(
    tmp_path: Path,
    optional_context: bool,
) -> None:
    _write_native_root(tmp_path)
    if optional_context:
        _write_native_wastewater(tmp_path)
    control_config = _artifact_config(tmp_path)
    control_config.native.wastewater_enabled = optional_context
    control_config.native.mobility_enabled = optional_context
    control_config.training.epochs = 2
    control_config.training.run_final_evaluation = False
    control = NativeTrainer(control_config, model_id="control")
    control_result = control.run()
    control_state = torch.load(
        control.checkpoint_dir / "final_model.pt", weights_only=False
    )["model_state_dict"]

    config = _artifact_config(tmp_path)
    config.native.wastewater_enabled = optional_context
    config.native.mobility_enabled = optional_context
    config.training.epochs = 2
    config.training.run_final_evaluation = False
    config.training.component_gradient_diagnostics = True
    config.training.grad_norm_log_frequency = 1
    config.evaluation.epoch_diagnostics = True
    config.evaluation.training_panel_queries_per_source = 3
    config.evaluation.training_objective_diagnostic = True
    config.evaluation.origin_block_days = 28
    diagnostic = NativeTrainer(config, model_id="diagnostic")
    result = diagnostic.run()
    final = torch.load(diagnostic.checkpoint_dir / "final_model.pt", weights_only=False)
    for name, expected in control_state.items():
        torch.testing.assert_close(
            final["model_state_dict"][name], expected, rtol=0, atol=0
        )
    assert result["val_wis_trace"] == control_result["val_wis_trace"]
    assert result["selected_epoch"] == control_result["selected_epoch"]
    assert len(result["training_panel_epoch_trace"]) == 2
    manifest = json.loads(diagnostic.training_dynamics_writer.manifest_path.read_text())
    assert (
        manifest["summary"]["training_panel_epoch_trace"]
        == result["training_panel_epoch_trace"]
    )
    assert final["training_panel_epoch_trace"] == result["training_panel_epoch_trace"]
    assert final["epoch_diagnostic_artifacts"] == result["epoch_diagnostic_artifacts"]
    panel_path = diagnostic.experiment_dir / "diagnostic_panels.json"
    assert (
        hashlib.sha256(panel_path.read_bytes()).hexdigest()
        == result["diagnostic_panels"]["sha256"]
    )
    panels = json.loads(panel_path.read_text())["panels"]
    training = {row["position"] for row in panels["training_panel"]}
    validation = {row["position"] for row in panels["val"]}
    assert training and training.isdisjoint(validation)
    assert len(result["epoch_diagnostic_artifacts"]) == 2
    for artifact in result["epoch_diagnostic_artifacts"]:
        for record in artifact["files"]:
            path = diagnostic.experiment_dir / record["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        aggregate = pd.read_csv(
            diagnostic.experiment_dir
            / "epoch_diagnostics"
            / f"epoch_{artifact['epoch']:03d}"
            / "val_native_aggregate_metrics.csv"
        )
        head = aggregate[
            (aggregate.aggregation == "head_all") & (aggregate.target == "cases")
        ].iloc[0]
        epoch = result["validation_epoch_trace"][artifact["epoch"] - 1]
        assert epoch["coverage_80_cases"] == pytest.approx(head.coverage_80, abs=1e-6)
        assert epoch["interval_width_80_cases"] == pytest.approx(
            head.interval_width_80, rel=1e-6
        )
    samples = [
        json.loads(line)
        for line in diagnostic.training_dynamics_writer.artifact_path.read_text().splitlines()
    ]
    assert any("gradnorm_native_spatial_encoder" in row for row in samples)
    assert all("lr" in row for row in samples)
    assert samples[0]["lr"] > samples[-1]["lr"]
    assert control_result["training_panel_epoch_trace"] == []
    assert control_result["epoch_diagnostic_artifacts"] == []
