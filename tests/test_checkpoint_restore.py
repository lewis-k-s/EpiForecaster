from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.checkpoint_restore import (
    build_model,
    load_model_from_checkpoint,
    read_checkpoint,
    restore_weights,
)
from models.configs import (
    DataConfig,
    EpiForecasterConfig,
    ModelConfig,
    ModelVariant,
    ObservationHeadConfig,
    OutputConfig,
    SIRPhysicsConfig,
    TrainingParams,
)
from models.epiforecaster import EpiForecaster

pytestmark = pytest.mark.epiforecaster

_CPU = torch.device("cpu")


def _config(
    *,
    variant: ModelVariant | None = None,
    observation_heads: ObservationHeadConfig | None = None,
    **model_fields,
) -> EpiForecasterConfig:
    return EpiForecasterConfig(
        model=ModelConfig(
            type=variant or ModelVariant(cases=True),
            mobility_embedding_dim=8,
            region_embedding_dim=8,
            input_window_length=14,
            forecast_horizon=7,
            max_neighbors=5,
            head_d_model=16,
            sir_physics=SIRPhysicsConfig(),
            observation_heads=observation_heads or ObservationHeadConfig(),
            **model_fields,
        ),
        data=DataConfig(dataset_path="unused.zarr", run_id="real"),
        training=TrainingParams(),
        output=OutputConfig(log_dir="unused", experiment_name="restore"),
    )


def _model(config: EpiForecasterConfig, *, seed: int) -> EpiForecaster:
    torch.manual_seed(seed)
    return build_model(config.model, device=_CPU)


def _save(
    tmp_path: Path,
    config: EpiForecasterConfig,
    state: dict[str, torch.Tensor],
    **extra: object,
) -> Path:
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": state, "config": config.to_dict(), **extra}, path)
    return path


def _assert_state_equal(model: torch.nn.Module, expected: dict[str, torch.Tensor]):
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def test_eval_load_builds_the_trained_architecture(tmp_path) -> None:
    config = _config(include_vaccination=True, head_positional_encoding="learned")
    reference = _model(config, seed=0)
    path = _save(tmp_path, config, reference.state_dict())

    model, loaded_config, payload = load_model_from_checkpoint(path, device="cpu")

    assert model.vaccination_dim == 3
    assert loaded_config.model.head_positional_encoding == "learned"
    assert payload["config"] == config.to_dict()
    _assert_state_equal(model, reference.state_dict())


def test_compiled_checkpoint_restores_into_uncompiled_model(tmp_path) -> None:
    config = _config()
    reference = _model(config, seed=0)
    compiled_state = {
        f"_orig_mod.{key}": value for key, value in reference.state_dict().items()
    }
    checkpoint = read_checkpoint(_save(tmp_path, config, compiled_state))
    model = _model(config, seed=1)

    report = restore_weights(model, checkpoint)

    _assert_state_equal(model, reference.state_dict())
    assert report.migrated_keys == ()
    assert report.initialized_keys == ()
    assert report.padded_keys == ()
    assert report.optimizer_state_compatible


def test_point_checkpoint_initializes_distribution_parameters(tmp_path) -> None:
    config = _config()
    reference = _model(config, seed=0)
    state = reference.state_dict()
    distribution_keys = tuple(
        key
        for key in state
        if ".log_distribution_scale" in key or ".distribution_scale_proj." in key
    )
    assert distribution_keys
    point_state = {k: v for k, v in state.items() if k not in distribution_keys}
    checkpoint = read_checkpoint(_save(tmp_path, config, point_state))
    model = _model(config, seed=1)
    defaults = {key: model.state_dict()[key].clone() for key in distribution_keys}

    report = restore_weights(model, checkpoint)

    assert set(report.initialized_keys) == set(distribution_keys)
    assert not report.optimizer_state_compatible
    restored = model.state_dict()
    for key in distribution_keys:
        assert torch.equal(restored[key], defaults[key])
    for key, value in point_state.items():
        assert torch.equal(restored[key], value)


def test_frozen_gamma_checkpoint_migrates_kernel_and_seeds_metadata(tmp_path) -> None:
    heads = ObservationHeadConfig(
        kernel_parameterization_hosp="gamma",
        learnable_kernel_hosp=False,
        kernel_gamma_shape_hosp=7.5,
    )
    config = _config(observation_heads=heads)
    reference = _model(config, seed=0)
    state = dict(reference.state_dict())
    new_key = "hosp_head.delay_kernel.gamma_kernel"
    legacy_key = "hosp_head.delay_kernel.kernel"
    metadata_keys = (
        "hosp_head.delay_kernel.gamma_shape_value",
        "hosp_head.delay_kernel.gamma_scale_value",
    )
    state[legacy_key] = state.pop(new_key)
    for key in metadata_keys:
        del state[key]
    checkpoint = read_checkpoint(_save(tmp_path, config, state))
    model = _model(config, seed=1)

    report = restore_weights(model, checkpoint)

    assert report.migrated_keys == ((legacy_key, new_key),)
    assert set(metadata_keys) <= set(report.initialized_keys)
    assert report.optimizer_state_compatible
    restored = model.state_dict()
    assert torch.equal(restored[new_key], reference.state_dict()[new_key])
    assert restored[metadata_keys[0]].item() == pytest.approx(7.5)


def test_narrow_mobility_input_weight_is_zero_padded(tmp_path) -> None:
    config = _config(variant=ModelVariant(cases=True, mobility=True), gnn_module="gcn")
    reference = _model(config, seed=0)
    state = dict(reference.state_dict())
    key = "mobility_gnn.layers.0.lin.weight"
    narrow = state[key][:, :-1].clone()
    state[key] = narrow
    checkpoint = read_checkpoint(_save(tmp_path, config, state))
    model = _model(config, seed=1)

    report = restore_weights(model, checkpoint)

    assert key in report.padded_keys
    assert not report.optimizer_state_compatible
    restored = model.state_dict()[key]
    assert torch.equal(restored[:, :-1], narrow)
    assert torch.all(restored[:, -1] == 0)


def test_fp16_checkpoint_is_rejected(tmp_path) -> None:
    config = _config()
    reference = _model(config, seed=0)
    path = _save(tmp_path, config, reference.state_dict(), scaler_state_dict={})

    with pytest.raises(ValueError, match="GradScaler"):
        restore_weights(_model(config, seed=1), read_checkpoint(path))


def test_non_float32_model_is_rejected(tmp_path) -> None:
    config = _config()
    reference = _model(config, seed=0)
    checkpoint = read_checkpoint(_save(tmp_path, config, reference.state_dict()))

    with pytest.raises(ValueError, match="dtype mismatch"):
        restore_weights(_model(config, seed=1).double(), checkpoint)


def test_checkpoint_without_config_is_rejected(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {}}, path)

    with pytest.raises(ValueError, match="missing required keys"):
        read_checkpoint(path)
