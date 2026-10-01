"""Native trainer wiring tests (slice TRN-1).

Covers: the head-balanced deterministic round-robin sampler, the
``NativeSupportConfig`` surface (defaults, ``EpiForecasterConfig`` round
trip, smoke-YAML load), a 2-step forward+backward smoke on a synthetic
native-support fixture (replicating the DS-1 test fixture builder), and
the temporal tail split. Real-artifact acceptance runs through
``uv run train epiforecaster --config configs/train_epifor_smoke.yaml``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
import yaml

from data.native_batch import collate_native_batch
from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
from data.native_source_sets import FoldRoles, SourceSetManifest
from data.query_contract import ForecastQuery
from models.configs import (
    EpiForecasterConfig,
    NativeSupportConfig,
)
from tests.config_fixtures import config_fixture
from models.native_forecaster import NativeEpiForecaster, NativeForecasterConfig
from training.native_trainer import (
    HierarchicalBalancedSampler,
    NativeTrainer,
    source_set_manifest_split,
    temporal_tail_split,
)

from tests.native_fixtures import (
    COVARIATES,
    L,
    H,
    _trainer_config,
    _write_native_root,
    _write_native_wastewater,
)

pytestmark = pytest.mark.epiforecaster

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

# ----------------------------------------------------------------------
# Synthetic native-support fixture (replicated from tests/test_native_dataset.py)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# (1) Head-balanced sampler composition and determinism
# ----------------------------------------------------------------------


def _heads_of(order: Sequence[int]) -> list[str]:
    membership = {
        **{i: "cases" for i in range(10)},
        **{i: "hosp" for i in range(10, 14)},
        **{i: "deaths" for i in range(14, 16)},
    }
    return [membership[i] for i in order]


def test_hierarchical_sampler_balances_heads_then_sources() -> None:
    grouped = {
        "cases": {
            ("municipality", "m1"): list(range(8)),
            ("municipality", "m2"): [8, 9],
        },
        "hosp": {
            ("abs", "a1"): [10, 11, 12],
            ("abs", "a2"): [13],
        },
        "deaths": {
            ("comarca", "c1"): [14],
            ("comarca", "c2"): [15],
        },
    }
    membership = {
        **{index: ("cases", "m1") for index in range(8)},
        8: ("cases", "m2"),
        9: ("cases", "m2"),
        10: ("hosp", "a1"),
        11: ("hosp", "a1"),
        12: ("hosp", "a1"),
        13: ("hosp", "a2"),
        14: ("deaths", "c1"),
        15: ("deaths", "c2"),
    }

    order = list(HierarchicalBalancedSampler(grouped, seed=42))
    routed = [membership[index] for index in order]

    assert len(order) == 30  # 10 samples for each of three heads
    assert Counter(head for head, _source in routed) == {
        "cases": 10,
        "hosp": 10,
        "deaths": 10,
    }
    for head in ("cases", "hosp", "deaths"):
        counts = Counter(
            source for routed_head, source in routed if routed_head == head
        )
        assert max(counts.values()) - min(counts.values()) <= 1
    assert list(HierarchicalBalancedSampler(grouped, seed=42)) == order

    plain = list(HierarchicalBalancedSampler(grouped, seed=42, balanced=False))
    assert sorted(plain) == list(range(16))
    plain_sampler = HierarchicalBalancedSampler(grouped, seed=42, balanced=False)
    first_epoch = list(plain_sampler)
    second_epoch = list(plain_sampler)
    assert sorted(first_epoch) == sorted(second_epoch) == list(range(16))
    assert first_epoch != second_epoch


# ----------------------------------------------------------------------
# (2) NativeSupportConfig defaults, round trip, smoke YAML
# ----------------------------------------------------------------------


def test_native_support_config_defaults() -> None:
    config = NativeSupportConfig()
    assert config.enabled is True
    assert config.root == "data/processed/native_support"
    assert config.spatial_encoder == "hetero"
    assert config.od_source == "mitma"
    assert config.overlap_weighting == "geostat_grid"
    assert config.od_self_flows == "retain"
    assert config.od_min_flow_raw == 10.0
    assert config.mobility_enabled is True
    assert config.wastewater_enabled is True
    assert config.include_vaccination is True
    assert config.hetero_hidden_dim == 16
    assert config.hetero_parameter_budget == 100_000
    assert config.fold_manifest.endswith("native_source_sets/manifest.json")
    assert config.fold_index is None
    assert config.head_balanced_sampling is False
    assert config.confirmation_guard is True
    with pytest.raises(ValueError):
        NativeSupportConfig(spatial_encoder="dense")
    with pytest.raises(ValueError):
        NativeSupportConfig(od_min_flow_raw=-1.0)
    with pytest.raises(ValueError):
        NativeSupportConfig(hetero_hidden_dim=0)
    with pytest.raises(ValueError):
        NativeSupportConfig(hetero_parameter_budget=0)
    with pytest.raises(ValueError):
        NativeSupportConfig(fold_index=-1)


def test_config_roundtrip_and_overrides(tmp_path: Path) -> None:
    smoke_path = tmp_path / "config.yaml"
    source = config_fixture()
    source.training.device = "cpu"
    source.model.input_window_length = 28
    source.model.forecast_horizon = 14
    smoke_path.write_text(yaml.safe_dump(source.to_dict()))
    config = EpiForecasterConfig.load(smoke_path)
    assert config.native.enabled is True
    assert config.native.root == "data/processed/native_support"
    assert config.native.head_balanced_sampling is False
    assert config.native.confirmation_guard is True
    assert config.native.wastewater_enabled is True
    assert config.model.forecast_horizon == 14
    assert config.model.input_window_length == 28
    assert config.training.epochs >= 2
    assert config.training.device == "cpu"

    restored = EpiForecasterConfig.from_dict(config.to_dict())
    assert restored.native == config.native

    # A dict without a native block restores the source-native default.
    legacy = config.to_dict()
    legacy.pop("native")
    assert EpiForecasterConfig.from_dict(legacy).native == NativeSupportConfig()

    # Overrides apply through the native field without breaking struct mode.
    overridden = EpiForecasterConfig.load(
        smoke_path, overrides=["native.head_balanced_sampling=false"]
    )
    assert overridden.native.head_balanced_sampling is False


# ----------------------------------------------------------------------
# (3) Two training steps on the synthetic fixture
# ----------------------------------------------------------------------


def test_native_trainer_construction_guards(tmp_path: Path) -> None:
    _write_native_root(tmp_path)

    municipal = _trainer_config(tmp_path)
    municipal.native.enabled = False
    with pytest.raises(ValueError, match="native.enabled"):
        NativeTrainer(municipal)

    curric = _trainer_config(tmp_path)
    curric.training.curriculum.enabled = True
    with pytest.raises(ValueError, match="Curriculum"):
        NativeTrainer(curric)

    gradnorm = _trainer_config(tmp_path)
    gradnorm.training.loss.joint.adaptive_scheme = "gradnorm"
    with pytest.raises(ValueError, match="GradNorm"):
        NativeTrainer(gradnorm)

    legacy = _trainer_config(tmp_path)
    legacy.data.dataset_path = "data/files/hospitalizations_municipality.csv"
    with pytest.raises(ValueError, match="legacy projected"):
        NativeTrainer(legacy)


# ----------------------------------------------------------------------
# (4) Temporal tail split
# ----------------------------------------------------------------------


def test_temporal_tail_split() -> None:
    origins = [13, 13, 14, 15, 20, 33, 34, 35]
    train, val = temporal_tail_split(origins, val_fraction=0.25)
    # 7 distinct origins -> ceil(0.25 * 7) = 2 val origins {34, 35}.
    assert sorted(origins[i] for i in val) == [34, 35]
    assert train and val
    assert set(train).isdisjoint(val)
    assert set(train + val) <= set(range(len(origins)))
    assert max(origins[i] + 1 for i in train) < min(origins[i] for i in val)

    # Zero fraction keeps everything for training.
    train_all, val_empty = temporal_tail_split(origins, val_fraction=0.0)
    assert len(train_all) == len(origins)
    assert val_empty == []

    # At least one origin is always reserved when val_fraction > 0.
    _, val_single = temporal_tail_split([5, 5, 5], val_fraction=0.1)
    assert len(val_single) == 3  # the only origin is the tail


def test_temporal_tail_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        temporal_tail_split([1, 2, 3], val_fraction=1.0)
    with pytest.raises(ValueError):
        temporal_tail_split([], val_fraction=0.1)


def test_source_set_manifest_split_uses_rotating_roles_and_three_intervals(
    tmp_path: Path,
) -> None:
    queries = [
        ForecastQuery(
            source_type="municipality",
            source_id=source_id,
            forecast_origin=origin,
            horizon_days=2,
            requested_heads=frozenset({"cases"}),
        )
        for origin in (5, 15, 25)
        for source_id in ("test", "validation", "train")
    ]
    manifest = SourceSetManifest(
        path=tmp_path / "source_sets.json",
        num_folds=3,
        assignments={
            ("municipality", "test"): 0,
            ("municipality", "validation"): 1,
            ("municipality", "train"): 2,
        },
        validation_start_origin=10,
        test_start_origin=20,
        forecast_horizon=1,
        provenance={},
        run_roles=(
            FoldRoles(0, (2,), 1, 0),
            FoldRoles(1, (0,), 2, 1),
            FoldRoles(2, (1,), 0, 2),
        ),
    )

    train, validation, test = source_set_manifest_split(
        queries, ["cases"] * len(queries), manifest, run_index=0
    )

    assert train == [2]
    assert validation == [4]
    assert test == [6]


# Sanity: the fixture writer is exercised through the trainer tests above;
# keep one direct check that the dataset indexes all three heads on it.
def test_dataset_config_matches_fixture(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    config = NativeDatasetConfig(
        input_window_length=L, forecast_horizon=H, window_stride=1
    )
    dataset = NativeSupportDataset(tmp_path, tmp_path / "real_with_id.zarr", config)
    assert set(dataset.head_counts) == {"cases", "deaths", "hosp"}


def test_dataset_indexes_one_wastewater_query_per_edar_origin(
    tmp_path: Path,
) -> None:
    _write_native_root(tmp_path)
    _write_native_wastewater(tmp_path)
    config = NativeDatasetConfig(
        input_window_length=L,
        forecast_horizon=H,
        window_stride=1,
        wastewater_enabled=True,
    )
    dataset = NativeSupportDataset(tmp_path, tmp_path / "real_with_id.zarr", config)

    wastewater_indices = [
        index for index in range(len(dataset)) if dataset.head_at(index) == "ww"
    ]
    assert wastewater_indices
    identities = {
        (
            dataset.queries[index].source_id,
            dataset.queries[index].forecast_origin,
        )
        for index in wastewater_indices
    }
    assert len(identities) == len(wastewater_indices)

    item = dataset[wastewater_indices[0]]
    assert item["query"].source_type == "edar"
    assert item["target"].shape == (H, 3)
    assert item["target_mask"].shape == (H, 3)
    assert item["hist"].shape == (L, 6 + 3 + len(COVARIATES))
    assert item["target_extra"]["detection_limit"].shape == (H, 3)
    assert item["target_extra"]["censor_code"].shape == (H, 3)
    assert item["target_extra"]["flow_m3_24h"].shape == (H,)
    assert item["target_extra"]["flow_mask"].shape == (H,)
    assert item["source_pop"] in {1000.0, 2000.0}


def test_hetero_wastewater_accepts_vaccination_relation(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    _write_native_wastewater(tmp_path)
    dataset = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        NativeDatasetConfig(
            input_window_length=L,
            forecast_horizon=H,
            wastewater_enabled=True,
            spatial_encoder="hetero",
            mobility_enabled=False,
        ),
    )
    index = next(i for i in range(len(dataset)) if dataset.head_at(i) == "ww")
    batch = collate_native_batch([dataset[index]])
    assert batch.relation_ctx["muni_edar_catchment"].shape[-1] == 6
    assert batch.relation_ctx["muni_edar_catchment"][:, :, 3:].sum() > 0
    model = NativeEpiForecaster(
        NativeForecasterConfig(
            f_native=batch.hist.shape[-1],
            d_ctx=batch.ctx_feat.shape[-1],
            sequence_length=L,
            forecast_horizon=H,
            spatial_encoder="hetero",
            wastewater_enabled=True,
            mobility_enabled=False,
        )
    )
    assert torch.isfinite(model(batch)["spatial_context"]).all()
