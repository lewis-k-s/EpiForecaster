"""Native trainer wiring tests (slice TRN-1).

Covers: the head-balanced deterministic round-robin sampler, the
``NativeSupportConfig`` surface (defaults, ``EpiForecasterConfig`` round
trip, smoke-YAML load), a 2-step forward+backward smoke on a synthetic
native-support fixture (replicating the DS-1 test fixture builder), and
the temporal tail split. Real-artifact acceptance runs through
``uv run train epiforecaster --config configs/train_epifor_native_smoke.yaml``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import zarr

from data.native_batch import ROUTED_HEADS
from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
from models.configs import (
    DataConfig,
    EpiForecasterConfig,
    ModelConfig,
    ModelVariant,
    NativeSupportConfig,
    OutputConfig,
    TrainingParams,
)
from training.native_trainer import (
    HeadBalancedSampler,
    NativeTrainer,
    temporal_tail_split,
)

pytestmark = pytest.mark.epiforecaster

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

# ----------------------------------------------------------------------
# Synthetic native-support fixture (replicated from tests/test_native_dataset.py)
# ----------------------------------------------------------------------

GRID_START = np.datetime64("2020-03-01", "D")
T = 43
MONDAY_OFFSETS = np.array([1, 8, 15, 22, 29, 36])
L, H = 14, 7

MUNIS = ["01001", "01002", "01003", "01004"]
MUNI_POP = {"01001": 40000.0, "01002": 30000.0, "01003": 20000.0, "01004": 10000.0}
CASE_COUNT = {"01001": 3.0, "01002": 5.0, "01003": 7.0, "01004": 2.0}

ABS_IDS = ["101", "102", "103"]
ABS_POP = {"101": 33000.0, "102": 5000.0, "103": 3000.0}
ABS_VALUE = {"101": [10, 12, 14, 16, 18, 20], "102": [0, 5, 7, 0, 0, 0], "103": [1] * 6}
ABS_MASK = {"101": [1] * 6, "102": [0, 1, 1, 0, 0, 0], "103": [1] * 6}
ABS_STRATUM = {"101": 900.0, "102": 400.0, "103": 100.0}

COMARCAS = ["01", "02"]
COMARCA_POP = {"01": 50000.0, "02": 70000.0}
DEATH_VALUE = {"01": 2.0, "02": 4.0}

COVARIATES = ["dow_sin", "dow_cos", "is_holiday", "lockdown_severity"]


def _cases_value(muni: str) -> np.ndarray:
    value = np.log1p(CASE_COUNT[muni] / MUNI_POP[muni] * 1e5)
    return np.full(T, value, dtype=np.float16)


def _write_native_root(root: Path) -> None:
    """Write native_graph/obs_admissions/obs_deaths + a real_with_id-shaped
    cases zarr (no OD context: ``include_od_context`` is off in TRN-1)."""
    dates = GRID_START + np.arange(T, dtype="timedelta64[D]")

    rows: list[tuple[str, str, str, float]] = []
    for muni in MUNIS:
        rows.append(("municipality", muni, "unobserved", MUNI_POP[muni]))
    for abs_id in ABS_IDS:
        status = "unmapped" if abs_id == "103" else "observed"
        rows.append(("abs", abs_id, status, ABS_POP[abs_id]))
    for comarca in COMARCAS:
        rows.append(("comarca", comarca, "observed", COMARCA_POP[comarca]))
    zones = ["z1", "z2", "z3"]
    for zone in zones:
        rows.append(("mitma", zone, "observed", 1000.0))
    uids = [f"{t}:{i}" for t, i, _, _ in rows]
    node = xr.Dataset(
        {
            "source_id": ("node", [i for _, i, _, _ in rows]),
            "node_type": ("node", [t for t, _, _, _ in rows]),
            "status": ("node", [s for _, _, s, _ in rows]),
            "population": ("node", np.array([p for *_, p in rows], dtype=np.float32)),
            "od_zone_ref": (("od_zone",), zones),
        },
        coords={"node": uids, "od_zone": np.arange(len(zones))},
    )
    node.to_zarr(root / "native_graph.zarr")

    node_index = {(t, i): k for k, (t, i, _, _) in enumerate(rows)}

    def edge(rel: str, pairs: list[tuple[str, str, str, str, float]]) -> None:
        group = zarr.open_group(str(root / "native_graph.zarr"), mode="a").require_group(
            f"edges/{rel}"
        )
        group.create_array(
            "src_idx",
            data=np.array(
                [node_index[(st, si)] for st, si, _, _, _ in pairs], dtype="int32"
            ),
        )
        group.create_array(
            "dst_idx",
            data=np.array(
                [node_index[(dt, di)] for _, _, dt, di, _ in pairs], dtype="int32"
            ),
        )
        group.create_array("weight", data=np.array([w for *_, w in pairs], dtype="float32"))

    edge(
        "muni_abs",
        [
            ("municipality", "01001", "abs", "101", 0.5),
            ("municipality", "01002", "abs", "101", 0.3),
            ("municipality", "01003", "abs", "101", 0.2),
            ("municipality", "01004", "abs", "102", 0.6),
        ],
    )
    edge(
        "muni_comarca",
        [
            ("municipality", "01001", "comarca", "01", 0.7),
            ("municipality", "01002", "comarca", "01", 0.3),
            ("municipality", "01003", "comarca", "02", 0.6),
        ],
    )

    monday_dates = dates[MONDAY_OFFSETS]
    adm = xr.Dataset(
        {
            "value": (
                ("date", "abs_id"),
                np.array(
                    [[ABS_VALUE[a][w] for a in ABS_IDS] for w in range(6)],
                    dtype=np.float32,
                ),
            ),
            "mask": (
                ("date", "abs_id"),
                np.array([[bool(ABS_MASK[a][w]) for a in ABS_IDS] for w in range(6)]),
            ),
            "age": (
                ("date", "abs_id"),
                np.array(
                    [[1 if ABS_MASK[a][w] else 2 for a in ABS_IDS] for w in range(6)],
                    dtype=np.uint8,
                ),
            ),
            "stratum_pop_sum": (
                ("date", "abs_id"),
                np.full((6, len(ABS_IDS)), 0.0, dtype=np.float32)
                + np.array([ABS_STRATUM[a] for a in ABS_IDS], dtype=np.float32),
            ),
        },
        coords={
            "date": monday_dates.astype("datetime64[ns]"),
            "abs_id": ABS_IDS,
            "week_end": ("date", (monday_dates + 6).astype("datetime64[ns]")),
        },
    )
    adm.to_zarr(root / "obs_admissions.zarr")

    deaths = xr.Dataset(
        {
            "value": (
                ("date", "comarca_id"),
                np.stack(
                    [np.full(T, DEATH_VALUE[c], dtype=np.float32) for c in COMARCAS],
                    axis=1,
                ),
            ),
            "mask": (
                ("date", "comarca_id"),
                np.ones((T, len(COMARCAS)), dtype=bool),
            ),
            "age": (("date", "comarca_id"), np.ones((T, len(COMARCAS)), dtype=np.uint8)),
            "population": (
                ("comarca_id",),
                np.array([COMARCA_POP[c] for c in COMARCAS], dtype=np.float32),
            ),
        },
        coords={
            "date": dates.astype("datetime64[ns]"),
            "comarca_id": COMARCAS,
            "comarca_name": (("comarca_id",), [f"name-{c}" for c in COMARCAS]),
        },
    )
    deaths.to_zarr(root / "obs_deaths.zarr")

    cases = xr.Dataset(
        {
            "cases": (
                ("run_id", "date", "region_id"),
                np.stack([_cases_value(m) for m in MUNIS], axis=-1)[None],
            ),
            "cases_mask": (
                ("run_id", "date", "region_id"),
                np.ones((1, T, len(MUNIS)), dtype=bool),
            ),
            "cases_age": (
                ("run_id", "date", "region_id"),
                np.ones((1, T, len(MUNIS)), dtype=np.uint8),
            ),
            "temporal_covariates": (
                ("date", "covariate"),
                (np.arange(T, dtype=np.float32)[:, None] * np.arange(4)[None, :]) / 10.0,
            ),
        },
        coords={
            "run_id": ["real"],
            "date": dates.astype("datetime64[ns]"),
            "region_id": MUNIS,
            "covariate": COVARIATES,
        },
    )
    cases.to_zarr(root / "real_with_id.zarr")


def _trainer_config(root: Path, **native_kwargs: bool) -> EpiForecasterConfig:
    return EpiForecasterConfig(
        model=ModelConfig(
            type=ModelVariant(cases=False, regions=False, biomarkers=False, mobility=False),
            mobility_embedding_dim=16,
            region_embedding_dim=8,
            input_window_length=L,
            forecast_horizon=H,
            max_neighbors=20,
            head_d_model=32,
            head_n_heads=2,
            head_num_layers=1,
        ),
        data=DataConfig(dataset_path=str(root / "real_with_id.zarr")),
        training=TrainingParams(
            epochs=1,
            max_batches=2,
            batch_size=8,
            seed=42,
            split_seed=42,
            device="cpu",
            val_split=0.1,
            scheduler_type="cosine",
            warmup_steps=0,
        ),
        output=OutputConfig(
            experiment_name="native_trainer_test", log_dir=str(root / "outputs")
        ),
        native=NativeSupportConfig(enabled=True, root=str(root), **native_kwargs),
    )


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


def test_head_balanced_sampler_interleaves_and_is_deterministic() -> None:
    head_items = {
        "cases": list(range(10)),
        "hosp": list(range(10, 14)),
        "deaths": list(range(14, 16)),
    }
    order = list(HeadBalancedSampler(head_items, seed=42))
    heads = _heads_of(order)

    # Counts 10/4/2 -> every item appears exactly once.
    assert len(order) == 16
    assert sorted(order) == list(range(16))
    assert heads.count("cases") == 10
    assert heads.count("hosp") == 4
    assert heads.count("deaths") == 2

    # Round-robin interleave: while all heads have stock, each rotation
    # window contains one item of every head.
    assert set(heads[0:3]) == {"cases", "hosp", "deaths"}
    assert set(heads[3:6]) == {"cases", "hosp", "deaths"}
    assert set(heads[6:9]) == {"cases", "hosp"}  # deaths exhausted after 2

    # Deterministic under seed.
    assert list(HeadBalancedSampler(head_items, seed=42)) == order

    # Plain mode: same multiset, seeded shuffle.
    plain = list(HeadBalancedSampler(head_items, seed=42, balanced=False))
    assert sorted(plain) == list(range(16))
    assert list(HeadBalancedSampler(head_items, seed=42, balanced=False)) == plain


# ----------------------------------------------------------------------
# (2) NativeSupportConfig defaults, round trip, smoke YAML
# ----------------------------------------------------------------------


def test_native_support_config_defaults() -> None:
    config = NativeSupportConfig()
    assert config.enabled is False
    assert config.root == "data/processed/native_support"
    assert config.spatial_encoder == "pooled"
    assert config.od_source == "mitma"
    assert config.overlap_weighting == "geostat_grid"
    assert config.od_self_flows == "retain"
    assert config.od_min_flow_raw == 10.0
    assert config.mobility_enabled is True
    assert config.hetero_hidden_dim == 16
    assert config.hetero_parameter_budget == 100_000
    assert config.head_balanced_sampling is True
    assert config.confirmation_guard is True
    with pytest.raises(ValueError):
        NativeSupportConfig(spatial_encoder="dense")
    with pytest.raises(ValueError):
        NativeSupportConfig(od_min_flow_raw=-1.0)
    with pytest.raises(ValueError):
        NativeSupportConfig(hetero_hidden_dim=0)
    with pytest.raises(ValueError):
        NativeSupportConfig(hetero_parameter_budget=0)


def test_config_roundtrip_and_smoke_yaml() -> None:
    smoke_path = CONFIGS_DIR / "train_epifor_native_smoke.yaml"
    config = EpiForecasterConfig.load(smoke_path)
    assert config.native.enabled is True
    assert config.native.root == "data/processed/native_support"
    assert config.native.head_balanced_sampling is True
    assert config.native.confirmation_guard is True
    assert config.model.forecast_horizon == 14
    assert config.model.input_window_length == 28
    assert config.training.epochs >= 2
    assert config.training.device == "cpu"

    restored = EpiForecasterConfig.from_dict(config.to_dict())
    assert restored.native == config.native

    # A dict without a native block restores the municipal defaults.
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


def test_native_trainer_two_steps_finite_and_backward(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    trainer = NativeTrainer(_trainer_config(tmp_path))
    parameter_before = next(trainer.model.parameters()).detach().clone()

    results = trainer.run()

    losses = results["train_loss_trace"][0]
    assert len(losses) == 2  # epochs=1 x max_batches=2 -> exactly 2 steps
    assert all(np.isfinite(losses))
    val_loss = results["val_loss_trace"][0]
    assert val_loss is not None and np.isfinite(val_loss)
    assert set(trainer.last_val_metrics) >= {
        "wis_cases",
        "wis_hosp",
        "wis_deaths",
    }

    # Backward + optimizer step actually updated the model.
    parameter_after = next(trainer.model.parameters()).detach()
    assert not torch.equal(parameter_before, parameter_after)

    # All three routed heads were represented in the index.
    assert set(results["head_counts"]) == set(ROUTED_HEADS)


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
    assert sorted(train + val) == list(range(len(origins)))
    assert max(origins[i] for i in train) < min(origins[i] for i in val)

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


# Sanity: the fixture writer is exercised through the trainer tests above;
# keep one direct check that the dataset indexes all three heads on it.
def test_dataset_config_matches_fixture(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    config = NativeDatasetConfig(
        input_window_length=L, forecast_horizon=H, window_stride=1
    )
    dataset = NativeSupportDataset(tmp_path, tmp_path / "real_with_id.zarr", config)
    assert set(dataset.head_counts) == {"cases", "deaths", "hosp"}
