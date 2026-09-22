"""Native query dataset tests (slice DS-1).

Synthetic fixture mirrors the real artifact layout on a tiny grid: a
Sunday-start shared grid (43 days), Monday week-starts, a
``native_graph.zarr`` with node table + edge subgroup, weekly admissions
and daily deaths stores, a ``real_with_id``-shaped cases zarr, and an
optional ``od_mitma.zarr``. Expected eligibility sets and pooled context
values are hand-derived. The confirmation-guard test (ruling R9) writes a
second grid spanning the reserved window; ``_write_native_root``
parameterizes ``grid_start``/``days`` for it. Real-artifact integration is
skipif-guarded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import zarr

from data.native_dataset import (
    CONFIRMATION_WINDOW,
    SOURCE_TYPE_IDS,
    NativeDatasetConfig,
    NativeSupportDataset,
)

pytestmark = pytest.mark.epiforecaster

# Shared daily grid: 2020-03-01 (Sunday) + 43 days; Mondays at 1, 8, ..., 36.
GRID_START = np.datetime64("2020-03-01", "D")
T = 43
MONDAY_OFFSETS = np.array([1, 8, 15, 22, 29, 36])
L, H = 14, 7
ORIGINS = list(range(L - 1, T - H))  # 13..35, stride 1

# Confirmation-guard fixture grid: 2020-11-22 (Sunday) + 74 days through
# 2021-02-03; the reserved window sits at offsets 39..66, leaving origins
# after it (up to 66) reachable with L + H = 21 days of grid.
GUARD_GRID_START = np.datetime64("2020-11-22", "D")
GUARD_DAYS = 74

MUNIS = ["01001", "01002", "01003", "01004"]
MUNI_POP = {"01001": 40000.0, "01002": 30000.0, "01003": 20000.0, "01004": 10000.0}
CASE_COUNT = {"01001": 3.0, "01002": 5.0, "01003": 7.0, "01004": 2.0}

ABS_IDS = ["101", "102", "103"]
ABS_POP = {"101": 33000.0, "102": 5000.0, "103": 3000.0}
ABS_VALUE = {"101": [10, 12, 14, 16, 18, 20], "102": [0, 5, 7, 0, 0, 0], "103": [1] * 6}
ABS_MASK = {
    "101": [1] * 6,
    "102": [0, 1, 1, 0, 0, 0],
    "103": [1] * 6,
}
ABS_STRATUM = {"101": 900.0, "102": 400.0, "103": 100.0}

COMARCAS = ["01", "02"]
COMARCA_POP = {"01": 50000.0, "02": 70000.0}
DEATH_VALUE = {"01": 2.0, "02": 4.0}
DEATH_GAP = set(range(20, 27))  # comarca 02 unreported days

COVARIATES = ["dow_sin", "dow_cos", "is_holiday", "lockdown_severity"]


def _grid_dates() -> np.ndarray:
    return GRID_START + np.arange(T, dtype="timedelta64[D]")


def _cases_mask(muni: str, days: int = T) -> np.ndarray:
    mask = np.ones(days, dtype=bool)
    if muni == "01003":
        mask[:10] = False
    return mask


def _deaths_mask(comarca: str, days: int = T) -> np.ndarray:
    mask = np.ones(days, dtype=bool)
    if comarca == "02":
        mask[sorted(DEATH_GAP)] = False
    return mask


def _cases_value(muni: str, days: int = T) -> np.ndarray:
    value = np.log1p(CASE_COUNT[muni] / MUNI_POP[muni] * 1e5)
    return np.full(days, value, dtype=np.float16)


def _write_native_root(
    root: Path,
    *,
    with_od: bool = False,
    grid_start: np.datetime64 = GRID_START,
    days: int = T,
) -> None:
    """Write native_graph/obs_admissions/obs_deaths (+ od_mitma) fixtures.

    ``grid_start`` must be a Sunday so ``MONDAY_OFFSETS`` stay Mondays; all
    arrays size to ``days``.
    """
    dates = grid_start + np.arange(days, dtype="timedelta64[D]")

    # -- node table -----------------------------------------------------
    rows: list[tuple[str, str, str, float]] = []
    for muni in MUNIS:  # geometry nodes only; the registry reads cases data
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

    # -- edge groups: (src_idx, dst_idx) with frac_of_dst_pop weights ----
    node_index = {(t, i): k for k, (t, i, _, _) in enumerate(rows)}

    def edge(rel: str, pairs: list[tuple[str, str, str, str, float]]) -> None:
        group = zarr.open_group(
            str(root / "native_graph.zarr"), mode="a"
        ).require_group(f"edges/{rel}")
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
        group.create_array(
            "weight", data=np.array([w for *_, w in pairs], dtype="float32")
        )

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
    if with_od:
        edge(
            "mitma_abs",
            [("mitma", "z1", "abs", "101", 1.0), ("mitma", "z2", "abs", "101", 1.0)],
        )
        edge("mitma_comarca", [("mitma", "z3", "comarca", "02", 1.0)])

    # -- weekly admissions -----------------------------------------------
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

    # -- daily deaths ------------------------------------------------------
    deaths = xr.Dataset(
        {
            "value": (
                ("date", "comarca_id"),
                np.stack(
                    [
                        np.full(days, DEATH_VALUE[c], dtype=np.float32)
                        for c in COMARCAS
                    ],
                    axis=1,
                ),
            ),
            "mask": (
                ("date", "comarca_id"),
                np.stack([_deaths_mask(c, days) for c in COMARCAS], axis=1),
            ),
            "age": (
                ("date", "comarca_id"),
                np.ones((days, len(COMARCAS)), dtype=np.uint8),
            ),
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

    # -- real_with_id-shaped cases store -----------------------------------
    cases = xr.Dataset(
        {
            "cases": (
                ("run_id", "date", "region_id"),
                np.stack([_cases_value(m, days) for m in MUNIS], axis=-1)[None],
            ),
            "cases_mask": (
                ("run_id", "date", "region_id"),
                np.stack([_cases_mask(m, days) for m in MUNIS], axis=-1)[None],
            ),
            "cases_age": (
                ("run_id", "date", "region_id"),
                np.ones((1, days, len(MUNIS)), dtype=np.uint8),
            ),
            "temporal_covariates": (
                ("date", "covariate"),
                (np.arange(days, dtype=np.float32)[:, None] * np.arange(4)[None, :])
                / 10.0,
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

    if with_od:
        mobility = np.zeros((days, 3, 3), dtype=np.float32)
        mobility[14, 1, 0] = 100.0  # z2 -> z1
        mobility[14, 2, 1] = 300.0  # z3 -> z2
        mobility[14, 0, 2] = 600.0  # z1 -> z3
        od = xr.Dataset(
            {"mobility": (("date", "origin", "target"), mobility)},
            coords={
                "date": dates.astype("datetime64[ns]"),
                "origin": zones,
                "target": zones,
                "zone": np.arange(3),
                "zone_id": (("zone",), zones),
            },
        )
        od.to_zarr(root / "od_mitma.zarr")


@pytest.fixture()
def native_root(tmp_path: Path) -> Path:
    _write_native_root(tmp_path)
    return tmp_path


@pytest.fixture()
def config() -> NativeDatasetConfig:
    return NativeDatasetConfig(
        input_window_length=L,
        forecast_horizon=H,
        window_stride=1,
    )


@pytest.fixture()
def dataset(native_root: Path, config: NativeDatasetConfig) -> NativeSupportDataset:
    return NativeSupportDataset(native_root, native_root / "real_with_id.zarr", config)


# Hand-derived eligibility: abs 101 (all weeks reported) only when a full
# Mon-Sun week fits the (o, o+H] target: o+1 must be a Monday.
EXPECTED_ABS_101 = [o for o in ORIGINS if (o + 1) % 7 == 1]  # {14, 21, 28, 35}
EXPECTED_ABS_102 = [14]  # reported weeks 8, 15 only align at origin 14
# comarca 02 (gap days 20..26) tolerates <= 3 missing horizon days.
EXPECTED_COMARCA_02 = [13, 14, 15] + list(range(23, 36))
EXPECTED_MUNI_01003 = list(range(16, 36))  # history needs 7 observed days


def test_head_counts_and_hand_derived_eligibility(
    dataset: NativeSupportDataset,
) -> None:
    assert dataset.head_counts == {"cases": 89, "deaths": 39, "hosp": 5}
    index = dataset.head_index
    assert [o for _, s, o in index["hosp"] if s == "101"] == EXPECTED_ABS_101
    assert [o for _, s, o in index["hosp"] if s == "102"] == EXPECTED_ABS_102
    assert all(s != "103" for _, s, _ in index["hosp"])  # graph-unmapped
    assert [o for _, s, o in index["deaths"] if s == "02"] == EXPECTED_COMARCA_02
    assert [o for _, s, o in index["deaths"] if s == "01"] == ORIGINS
    assert [o for _, s, o in index["cases"] if s == "01003"] == EXPECTED_MUNI_01003
    assert [o for _, s, o in index["cases"] if s == "01001"] == ORIGINS


def test_observed_source_without_audited_population_is_not_indexed(
    tmp_path: Path,
) -> None:
    """A raw-count target cannot be queried without its rate denominator."""
    _write_native_root(tmp_path)
    graph = zarr.open_group(str(tmp_path / "native_graph.zarr"), mode="r+")
    abs_101 = len(MUNIS)
    graph["population"][abs_101] = np.nan

    ds = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        NativeDatasetConfig(input_window_length=L, forecast_horizon=H),
    )

    record = ds.registry.get("abs", "101")
    assert record is not None
    assert record.status == "observed"
    assert record.population is None
    assert all(
        not (query.source_type == "abs" and query.source_id == "101")
        for query in ds.queries
    )


def test_dedup_and_deterministic_ordering(dataset: NativeSupportDataset) -> None:
    identities = [
        (q.source_type, q.source_id, q.forecast_origin) for q in dataset.queries
    ]
    assert len(set(identities)) == len(identities)  # same (source, origin) once
    assert identities == sorted(identities)


def test_first_item_is_abs_101_at_origin_14(dataset: NativeSupportDataset) -> None:
    query = dataset.queries[0]
    assert (query.source_type, query.source_id, query.forecast_origin) == (
        "abs",
        "101",
        14,
    )
    item = dataset[0]
    assert item["head"] == "hosp"
    assert item["source_type_id"] == SOURCE_TYPE_IDS["abs"]


def test_municipal_history_channels_and_raw_target(
    dataset: NativeSupportDataset,
) -> None:
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(dataset.queries)
    }
    idx = identities[("municipality", "01001", 13)]
    item = dataset[idx]
    assert item["hist"].shape == (L, 7)
    assert item["hist"].dtype == torch.float32
    expected_value = float(_cases_value("01001")[13])
    assert item["hist"][:, 0] == pytest.approx(expected_value, rel=1e-3)
    assert torch.all(item["hist"][:, 1] == 1.0)
    assert torch.all(item["hist_mask"])
    covariates = (
        np.arange(0, L, dtype=np.float32)[:, None]
        * np.arange(4, dtype=np.float32)[None, :]
    ) / 10.0
    assert torch.allclose(item["hist"][:, 3:], torch.from_numpy(covariates))
    assert torch.allclose(item["temporal_covariates"], torch.from_numpy(covariates))
    # Raw counts inverted from the stored log1p(per-100k): count = 3.0.
    assert item["target"].shape == (H,)
    assert item["target"] == pytest.approx(torch.full((H,), 3.0), rel=1e-2)
    assert torch.all(item["target_mask"])
    assert item["target_stratum_pop_sum"] is None
    assert item["source_pop"] == MUNI_POP["01001"]
    assert item["target_population_graph"] == MUNI_POP["01001"]
    assert item["source_type_id"] == SOURCE_TYPE_IDS["municipality"]
    # 2020-03-14 is a Saturday.
    assert item["origin_dow"] == 5


def test_origin_dow_sunday_for_weekly_origins(dataset: NativeSupportDataset) -> None:
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(dataset.queries)
    }
    for origin in EXPECTED_ABS_101:
        assert dataset[identities[("abs", "101", origin)]]["origin_dow"] == 6


def test_abs_history_interval_placement(
    dataset: NativeSupportDataset, native_root: Path
) -> None:
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(dataset.queries)
    }
    item = dataset[identities[("abs", "101", 14)]]
    hist = item["hist"]
    # History window covers grid days 1..14: Mondays at day 1 and day 8.
    # Rulings R10/R17 consumption transform: log1p(count / pop_graph * 1e5)
    # with the ABS graph population 33000; the stored value stays raw.
    assert hist[0, 0].item() == pytest.approx(
        np.log1p(10.0 / ABS_POP["101"] * 1e5), rel=1e-6
    )
    assert hist[7, 0].item() == pytest.approx(
        np.log1p(12.0 / ABS_POP["101"] * 1e5), rel=1e-6
    )
    weekly_days = {0, 7}
    for day in range(L):
        if day not in weekly_days:
            assert hist[day, 0].item() == 0.0
            assert not item["hist_mask"][day]
            assert hist[day, 2].item() == 0.0
        else:
            assert item["hist_mask"][day]
            assert hist[day, 1].item() == 1.0
            assert hist[day, 2].item() == 1.0
    # Stores stay raw at rest: re-read the week-1 value from the store.
    adm = xr.open_zarr(native_root / "obs_admissions.zarr")
    stored = float(adm["value"].sel(abs_id="101").values[0])
    assert stored == 10.0
    # The TARGET stays the raw weekly count (14.0, pinned below in
    # test_abs_weekly_target_alignment_and_populations).


def test_abs_weekly_target_alignment_and_populations(
    dataset: NativeSupportDataset,
) -> None:
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(dataset.queries)
    }
    item = dataset[identities[("abs", "101", 14)]]
    # First Monday at/after day 15; H=7 -> single complete week slot.
    assert item["target"].shape == (1,)
    assert item["target"][0].item() == 14.0
    assert item["target_mask"].tolist() == [True]
    assert item["target_stratum_pop_sum"] is not None
    assert item["target_stratum_pop_sum"][0].item() == 900.0
    assert item["source_pop"] == ABS_POP["101"]
    assert item["target_population_graph"] == ABS_POP["101"]


def test_abs_incomplete_week_slot_is_masked(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    config = NativeDatasetConfig(input_window_length=L, forecast_horizon=14)
    ds = NativeSupportDataset(tmp_path, tmp_path / "real_with_id.zarr", config)
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(ds.queries)
    }
    item = ds[identities[("abs", "101", 13)]]
    # Target (13, 27]: week day 15 complete, week day 22 incomplete.
    assert item["target"].shape == (2,)
    assert item["target"].tolist() == [14.0, 0.0]
    assert item["target_mask"].tolist() == [True, False]
    assert item["target_stratum_pop_sum"] is not None
    assert item["target_stratum_pop_sum"].tolist() == [900.0, 0.0]


def test_comarca_history_value_transform_and_raw_target(
    dataset: NativeSupportDataset, native_root: Path
) -> None:
    """Rulings R10/R17: hist is log1p(count / pop_graph * 1e5); the target
    stays the raw count; the store keeps raw counts at rest."""
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(dataset.queries)
    }
    item = dataset[identities[("comarca", "01", 13)]]
    assert item["hist"].shape == (L, 7)
    expected_value = np.log1p(DEATH_VALUE["01"] / COMARCA_POP["01"] * 1e5)
    assert item["hist"][:, 0] == pytest.approx(
        torch.full((L,), expected_value, dtype=torch.float32), rel=1e-6
    )
    assert torch.all(item["hist"][:, 1] == 1.0)
    # TARGET STAYS RAW (the loss owns target-side transforms).
    assert item["target"].shape == (H,)
    assert torch.all(item["target"] == DEATH_VALUE["01"])
    assert torch.all(item["target_mask"])
    assert item["source_pop"] == COMARCA_POP["01"]
    assert item["target_population_graph"] == COMARCA_POP["01"]
    assert item["source_type_id"] == SOURCE_TYPE_IDS["comarca"]
    # Store stays raw at rest: re-read a value from obs_deaths.zarr.
    deaths = xr.open_zarr(native_root / "obs_deaths.zarr")
    stored = float(deaths["value"].sel(comarca_id="01").values[0])
    assert stored == DEATH_VALUE["01"]


def test_comarca_gap_keeps_missing_rows_missing(dataset: NativeSupportDataset) -> None:
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(dataset.queries)
    }
    item = dataset[identities[("comarca", "02", 27)]]
    # History window [14, 27] contains the unreported days 20..26.
    assert not item["hist_mask"][6:13].any()
    assert item["hist"][6:13, 0].eq(0).all()
    assert item["hist_mask"][:6].all()


def test_pooling_raw_fractions_zero_deficit_contribution(
    dataset: NativeSupportDataset,
) -> None:
    """Ruling R6: raw destination fractions, never renormalized.

    A partially covered context contributes ``coverage x feature`` plus a
    zero-valued deficit slot — no renormalization to fake completeness —
    and ``ctx_coverage`` keeps reporting the deficit.
    """
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(dataset.queries)
    }
    full = dataset[identities[("abs", "101", 14)]]
    # weights [0.5, 0.3, 0.2] over munis 01001..01003, coverage 1.0.
    window = slice(1, 15)
    expected = (
        0.5 * _cases_value("01001")[window].astype(np.float32)
        + 0.3 * _cases_value("01002")[window].astype(np.float32)
        + 0.2 * _cases_value("01003")[window].astype(np.float32)
    )
    assert full["ctx_feat"].shape == (L, 3)
    assert full["ctx_feat"][:, 0] == pytest.approx(
        torch.from_numpy(expected), rel=1e-3, abs=1e-4
    )
    assert full["ctx_coverage"] == pytest.approx(1.0)  # deficit slot = 0

    partial = dataset[identities[("abs", "102", 14)]]
    # Single neighbor with raw weight 0.6: contributes 0.6 x series — NOT
    # 1.0 x — with the 0.4 deficit contributing zeros.
    assert partial["ctx_feat"][:, 0] == pytest.approx(
        torch.from_numpy(0.6 * _cases_value("01004")[window].astype(np.float32)),
        rel=1e-3,
        abs=1e-4,
    )
    assert partial["ctx_coverage"] == pytest.approx(0.6)

    municipal = dataset[identities[("municipality", "01001", 13)]]
    assert municipal["ctx_feat"].eq(0).all()  # no municipal overlap neighbors
    assert municipal["ctx_coverage"] == 0.0


def test_determinism_across_instances(native_root: Path) -> None:
    config = NativeDatasetConfig(input_window_length=L, forecast_horizon=H)
    first = NativeSupportDataset(native_root, native_root / "real_with_id.zarr", config)
    second = NativeSupportDataset(
        native_root, native_root / "real_with_id.zarr", config
    )
    assert first.queries == second.queries
    assert first.head_counts == second.head_counts
    for idx in (0, 7, len(first) - 1):
        a, b = first[idx], second[idx]
        assert torch.equal(a["hist"], b["hist"])
        assert torch.equal(a["ctx_feat"], b["ctx_feat"])
        assert torch.equal(a["target"], b["target"])
        assert a["origin_dow"] == b["origin_dow"]


def test_window_stride_controls_origin_spacing(native_root: Path) -> None:
    config = NativeDatasetConfig(
        input_window_length=L, forecast_horizon=H, window_stride=2
    )
    ds = NativeSupportDataset(native_root, native_root / "real_with_id.zarr", config)
    origins = [o for _, s, o in ds.head_index["cases"] if s == "01001"]
    assert origins == list(range(L - 1, T - H, 2))
    assert all(o % 2 == (L - 1) % 2 for o in origins)


def test_od_context_row_gated(tmp_path: Path) -> None:
    _write_native_root(tmp_path, with_od=True)
    off = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        NativeDatasetConfig(input_window_length=L, forecast_horizon=H),
    )
    assert off[0]["ctx_feat"].shape == (L, 3)

    on_config = NativeDatasetConfig(
        input_window_length=L, forecast_horizon=H, include_od_context=True
    )
    on = NativeSupportDataset(tmp_path, tmp_path / "real_with_id.zarr", on_config)
    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(on.queries)
    }
    item = on[identities[("abs", "101", 14)]]
    assert item["ctx_feat"].shape == (L, 4)
    # Incoming share at day 14: (100 + 300) / (100 + 300 + 600) = 0.4.
    assert item["ctx_feat"][:, 3] == pytest.approx(torch.full((L,), 0.4))
    municipal = on[identities[("municipality", "01001", 13)]]
    assert municipal["ctx_feat"][:, 3].eq(0).all()  # no overlapping MITMA zones


def test_confirmation_guard_excludes_reserved_window_origins(
    tmp_path: Path,
) -> None:
    """Ruling R9 (plan section 10.3(10)): guard-on drops every origin whose
    target window ``(origin, origin+H]`` intersects the reserved window;
    guard-off keeps everything."""
    _write_native_root(tmp_path, grid_start=GUARD_GRID_START, days=GUARD_DAYS)
    grid = GUARD_GRID_START + np.arange(GUARD_DAYS, dtype="timedelta64[D]")
    win_start, win_end = CONFIRMATION_WINDOW

    def reserved(origin: int) -> bool:
        target = grid[origin + 1 : origin + H + 1]
        return bool(((target >= win_start) & (target <= win_end)).any())

    def build(guard: bool) -> NativeSupportDataset:
        return NativeSupportDataset(
            tmp_path,
            tmp_path / "real_with_id.zarr",
            NativeDatasetConfig(
                input_window_length=L,
                forecast_horizon=H,
                confirmation_guard=guard,
            ),
        )

    off = build(False)
    on = build(True)
    assert NativeDatasetConfig().confirmation_guard is True  # default on
    assert off.head_counts.keys() == on.head_counts.keys()

    # Guard-off keeps everything; guard-on drops exactly the reserved
    # origins, per head (the guard is index-wide, not per source).
    for head in off.head_counts:
        off_pairs = {(s, o) for _, s, o in off.head_index[head]}
        on_pairs = {(s, o) for _, s, o in on.head_index[head]}
        assert on_pairs == {(s, o) for s, o in off_pairs if not reserved(o)}

    cases_off = {o for _, s, o in off.head_index["cases"] if s == "01001"}
    cases_on = {o for _, s, o in on.head_index["cases"] if s == "01001"}
    # Origin INSIDE the reserved window: excluded.
    assert 42 in cases_off and 42 not in cases_on
    # Origin BEFORE the window whose horizon crosses into it: excluded.
    assert 35 in cases_off and 35 not in cases_on
    # Origin at the window's last day, target fully after it: kept.
    assert 66 in cases_on
    assert reserved(42) and reserved(35) and not reserved(66)


def test_od_threshold_and_self_flow_policy(tmp_path: Path) -> None:
    """Ruling R10: consumption-time od_min_flow_raw threshold and
    od_self_flows policy on a synthetic OD row; the store stays raw."""
    _write_native_root(tmp_path, with_od=True)
    mobility = np.zeros((T, 3, 3), dtype=np.float32)
    # Synthetic OD row at day 14: self-flows (diagonal), small cross-flows
    # below any threshold, one large flow.
    mobility[14] = np.array(
        [[500.0, 50.0, 100.0], [40.0, 200.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    od = xr.Dataset(
        {"mobility": (("date", "origin", "target"), mobility)},
        coords={
            "date": _grid_dates().astype("datetime64[ns]"),
            "origin": ["z1", "z2", "z3"],
            "target": ["z1", "z2", "z3"],
            "zone": np.arange(3),
            "zone_id": (("zone",), ["z1", "z2", "z3"]),
        },
    )
    od.to_zarr(tmp_path / "od_mitma.zarr", mode="w")

    def build(**kwargs: object) -> NativeSupportDataset:
        config = NativeDatasetConfig(
            input_window_length=L,
            forecast_horizon=H,
            include_od_context=True,
            **kwargs,
        )
        return NativeSupportDataset(tmp_path, tmp_path / "real_with_id.zarr", config)

    identities = {
        (q.source_type, q.source_id, q.forecast_origin): i
        for i, q in enumerate(build(confirmation_guard=False).queries)
    }

    def share_of(ds: NativeSupportDataset) -> float:
        item = ds[identities[("abs", "101", 14)]]
        return float(item["ctx_feat"][0, 3])

    # retain + min 10: nothing dropped -> (500+50+40+200)/(500+50+100+40+200).
    retain_small = build(od_min_flow_raw=10.0, od_self_flows="retain")
    assert share_of(retain_small) == pytest.approx(790.0 / 890.0)
    # retain + min 100: the 50 and 40 flows drop -> 700/800.
    retain_large = build(od_min_flow_raw=100.0, od_self_flows="retain")
    assert share_of(retain_large) == pytest.approx(700.0 / 800.0)
    # drop + min 100: the diagonal (500, 200) clears too -> 0 incoming.
    drop_large = build(od_min_flow_raw=100.0, od_self_flows="drop")
    assert share_of(drop_large) == 0.0
    # Raw store untouched by consumption-time policy.
    raw = xr.open_zarr(tmp_path / "od_mitma.zarr")
    assert float(raw["mobility"].values[14, 0, 0]) == 500.0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        NativeDatasetConfig(window_stride=0)
    with pytest.raises(ValueError):
        NativeDatasetConfig(input_window_length=0)
    with pytest.raises(ValueError):
        NativeDatasetConfig(min_history_observed_fraction=1.5)
    with pytest.raises(ValueError):
        NativeDatasetConfig(od_self_flows="keep")
    with pytest.raises(ValueError):
        NativeDatasetConfig(od_min_flow_raw=-1.0)


# ----------------------------------------------------------------------
# Real-artifact integration (skipped when artifacts are absent)
# ----------------------------------------------------------------------

REAL_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "native_support"
)
REAL_CASES = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "real_with_id.zarr"
)

requires_real_artifacts = pytest.mark.skipif(
    not (REAL_ROOT / "native_graph.zarr").exists() or not REAL_CASES.exists(),
    reason="real native-support artifacts not present",
)


@requires_real_artifacts
def test_real_artifacts_index_and_weekly_alignment() -> None:
    config = NativeDatasetConfig(input_window_length=60, forecast_horizon=28)
    ds = NativeSupportDataset(REAL_ROOT, REAL_CASES, config)
    print("head_counts:", ds.head_counts)
    assert sum(ds.head_counts.values()) == len(ds)
    for head, expected_type in (
        ("cases", "municipality"),
        ("hosp", "abs"),
        ("deaths", "comarca"),
    ):
        pairs = ds.head_index[head]
        sources = {s for _, s, _ in pairs}
        possible = len(sources) * len(range(59, 435 - 28, config.window_stride))
        print(
            f"{head}: items={len(pairs)} sources={len(sources)} "
            f"eligibility_rate={len(pairs) / possible:.3f}"
        )
        assert pairs, f"head {head} must index real sources"
        assert all(st == expected_type for st, _, _ in pairs)
        for source_id in sources:
            record = ds.registry.get(expected_type, source_id)
            assert record is not None
            assert record.population is not None
            assert np.isfinite(record.population) and record.population > 0

    # Spot-check one ABS query's weekly target alignment against the store.
    first_abs = next(q for q in ds.queries if q.source_type == "abs")
    item = ds[ds.queries.index(first_abs)]
    adm = xr.open_zarr(REAL_ROOT / "obs_admissions.zarr").compute()
    dates = adm["date"].values.astype("datetime64[D]")
    values = adm["value"].sel(abs_id=first_abs.source_id).values
    masks = adm["mask"].sel(abs_id=first_abs.source_id).values
    origin_date = np.datetime64("2020-03-01", "D") + np.timedelta64(
        int(first_abs.forecast_origin), "D"
    )
    dow = int((origin_date.astype(np.int64) + 3) % 7)
    assert item["origin_dow"] == dow
    first_monday = first_abs.forecast_origin + 1 + (-(dow + 1)) % 7
    for week in range(item["target"].shape[0]):
        start = first_monday + 7 * week
        if start + 6 > first_abs.forecast_origin + 28:
            assert not item["target_mask"][week]
            continue
        date = np.datetime64("2020-03-01", "D") + np.timedelta64(int(start), "D")
        row = int(np.flatnonzero(dates == date)[0])
        if masks[row]:
            assert item["target"][week].item() == pytest.approx(float(values[row]))
            assert item["target_mask"][week]
    assert item["target_stratum_pop_sum"] is not None
