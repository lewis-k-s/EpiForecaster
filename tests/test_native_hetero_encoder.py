"""ENC-1 tests for typed overlap and directed-OD native context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from data.native_batch import collate_native_batch
from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
from evaluation.losses import NativeJointInferenceLoss
from models.hetero_encoder import HeterogeneousContextEncoder
from models.native_forecaster import NativeEpiForecaster, NativeForecasterConfig
from tests.test_native_dataset import H, L, MUNIS, T, _write_native_root

pytestmark = pytest.mark.epiforecaster


def _mixed_batch(tmp_path: Path):
    _write_native_root(tmp_path, with_od=True)
    dataset = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        NativeDatasetConfig(
            input_window_length=L,
            forecast_horizon=H,
            spatial_encoder="hetero",
            od_source="mitma",
            mobility_enabled=True,
        ),
    )
    positions = {
        (query.source_type, query.source_id, query.forecast_origin): index
        for index, query in enumerate(dataset.queries)
    }
    items = [
        dataset[positions[("municipality", "01001", 14)]],
        dataset[positions[("abs", "101", 14)]],
        dataset[positions[("comarca", "02", 14)]],
    ]
    return items, collate_native_batch(items)


def _model(batch, spatial_encoder: str) -> NativeEpiForecaster:
    return NativeEpiForecaster(
        NativeForecasterConfig(
            f_native=int(batch.hist.shape[2]),
            d_ctx=int(batch.ctx_feat.shape[2]),
            forecast_horizon=H,
            sequence_length=L,
            head_d_model=32,
            head_n_heads=2,
            head_num_layers=1,
            head_dropout=0.0,
            spatial_encoder=spatial_encoder,
            mobility_enabled=True,
            hetero_hidden_dim=8,
            hetero_parameter_budget=10_000,
        )
    )


def _open_gates(model: NativeEpiForecaster) -> None:
    assert model.spatial_encoder is not None
    for message in model.spatial_encoder.overlap_messages.values():
        message.gate.data.fill_(0.5)
    for message in model.spatial_encoder.mobility_messages.values():
        message.gate.data.fill_(0.5)


def test_graph_context_is_grouped_once_per_identical_origin(tmp_path: Path) -> None:
    _items, batch = _mixed_batch(tmp_path)

    assert batch.od_node_features.shape == (1, 3, 3)
    assert batch.od_adjacency.shape == (1, 3, 3)
    assert batch.od_target_weights.shape == (3, 3)
    assert batch.od_group_index.tolist() == [0, 0, 0]
    assert set(batch.relation_ctx) == {"muni_abs", "muni_comarca"}
    assert batch.relation_ctx["muni_abs"][0].abs().sum() > 0
    assert batch.relation_ctx["muni_comarca"][0].abs().sum() > 0
    assert batch.relation_ctx["muni_abs"][1].abs().sum() > 0
    assert batch.relation_ctx["muni_comarca"][2].abs().sum() > 0


def test_zero_init_hetero_is_full_model_equivalent_to_pooled(tmp_path: Path) -> None:
    _items, batch = _mixed_batch(tmp_path)
    torch.manual_seed(23)
    pooled = _model(batch, "pooled").eval()
    torch.manual_seed(23)
    hetero = _model(batch, "hetero").eval()

    with torch.no_grad():
        pooled_outputs = pooled(batch)
        hetero_outputs = hetero(batch)

    assert torch.equal(hetero_outputs["spatial_context"], batch.ctx_feat)
    for key in (
        "beta_t",
        "initial_states",
        "dist_cases_loc",
        "dist_deaths_loc",
        "dist_hosp_weekly_loc",
    ):
        assert torch.equal(hetero_outputs[key], pooled_outputs[key])


def test_every_relation_path_has_gradient_and_history_is_residual(tmp_path: Path) -> None:
    _items, batch = _mixed_batch(tmp_path)
    model = _model(batch, "hetero")
    _open_gates(model)
    hist_before = batch.hist.clone()

    context = model.spatial_encoder(batch)
    context.square().mean().backward()

    assert torch.equal(batch.hist, hist_before)
    assert not torch.equal(context.detach(), batch.ctx_feat)
    assert model.spatial_encoder is not None
    for name, parameter in model.spatial_encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_single_query_matches_mixed_batch_and_both_modes_train(tmp_path: Path) -> None:
    items, mixed = _mixed_batch(tmp_path)
    torch.manual_seed(7)
    pooled = _model(mixed, "pooled")
    torch.manual_seed(7)
    hetero = _model(mixed, "hetero")
    _open_gates(hetero)
    hetero.eval()

    with torch.no_grad():
        mixed_outputs = hetero(mixed)
        for row, item in enumerate(items):
            single_outputs = hetero(collate_native_batch([item]))
            assert single_outputs["spatial_context"][0] == pytest.approx(
                mixed_outputs["spatial_context"][row], abs=1e-6
            )
            assert single_outputs["beta_t"][0] == pytest.approx(
                mixed_outputs["beta_t"][row], abs=1e-6
            )

    criterion = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")
    for model in (pooled, hetero.train()):
        model.zero_grad(set_to_none=True)
        total = criterion(model(mixed), mixed)["total"]
        assert torch.isfinite(total)
        total.backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.parameters()
        )


def test_parameter_budget_is_enforced() -> None:
    encoder = HeterogeneousContextEncoder(
        context_dim=3,
        hidden_dim=8,
        mobility_enabled=True,
        parameter_budget=10_000,
    )
    assert encoder.parameter_count < 10_000
    with pytest.raises(ValueError, match="exceeds budget"):
        HeterogeneousContextEncoder(
            context_dim=3,
            hidden_dim=8,
            mobility_enabled=True,
            parameter_budget=encoder.parameter_count - 1,
        )


def test_mobility_disabled_needs_no_od_artifact_and_trains(tmp_path: Path) -> None:
    _write_native_root(tmp_path, with_od=False)
    dataset = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        NativeDatasetConfig(
            input_window_length=L,
            forecast_horizon=H,
            spatial_encoder="hetero",
            mobility_enabled=False,
        ),
    )
    batch = collate_native_batch([dataset[0]])
    model = NativeEpiForecaster(
        NativeForecasterConfig(
            f_native=int(batch.hist.shape[2]),
            d_ctx=int(batch.ctx_feat.shape[2]),
            forecast_horizon=H,
            sequence_length=L,
            head_d_model=32,
            head_n_heads=2,
            head_num_layers=1,
            head_dropout=0.0,
            spatial_encoder="hetero",
            mobility_enabled=False,
        )
    )
    assert model.spatial_encoder is not None
    assert model.spatial_encoder.mobility_encoder is None
    total = NativeJointInferenceLoss(
        horizon_days=H, observation_loss="nll"
    )(model(batch), batch)["total"]
    assert torch.isfinite(total)
    total.backward()


def test_projected_municipal_od_source_decodes_and_routes(tmp_path: Path) -> None:
    _write_native_root(tmp_path, with_od=False)
    mobility = np.zeros((T, len(MUNIS), len(MUNIS)), dtype=np.float32)
    mobility[14, 1, 0] = 9.0
    xr.Dataset(
        {
            "mobility": (
                ("date", "origin", "target"),
                np.log1p(mobility).astype(np.float16),
            ),
            "zone_id": ("zone", MUNIS),
        },
        coords={
            "date": np.datetime64("2020-03-01", "D")
            + np.arange(T, dtype="timedelta64[D]"),
            "origin": MUNIS,
            "target": MUNIS,
            "zone": np.arange(len(MUNIS)),
        },
    ).to_zarr(tmp_path / "od_municipal.zarr")
    dataset = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        NativeDatasetConfig(
            input_window_length=L,
            forecast_horizon=H,
            spatial_encoder="hetero",
            od_source="municipal",
            mobility_enabled=True,
            od_min_flow_raw=0.0,
        ),
    )
    positions = {
        (query.source_type, query.source_id, query.forecast_origin): index
        for index, query in enumerate(dataset.queries)
    }
    batch = collate_native_batch(
        [
            dataset[positions[("municipality", "01001", 14)]],
            dataset[positions[("abs", "101", 14)]],
            dataset[positions[("comarca", "02", 14)]],
        ]
    )

    assert batch.od_node_features.shape == (1, 4, 3)
    assert batch.od_adjacency[0, 0, 1].item() == pytest.approx(1.0)
    assert batch.od_target_weights[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert batch.od_target_weights[1].sum().item() == pytest.approx(1.0)
    assert batch.od_target_weights[2].sum().item() == pytest.approx(0.6)
    outputs = _model(batch, "hetero")(batch)
    assert torch.isfinite(outputs["spatial_context"]).all()
