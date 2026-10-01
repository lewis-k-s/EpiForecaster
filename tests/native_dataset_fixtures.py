"""Deterministic native source stores for dataset and graph tests."""

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import zarr

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
    with_od: bool = True,
    grid_start: np.datetime64 = GRID_START,
    days: int = T,
) -> None:
    """Write native_graph/obs_admissions/obs_deaths (+ od_mitma) fixtures.

    ``grid_start`` must be a Sunday so ``MONDAY_OFFSETS`` stay Mondays; all
    arrays size to ``days``.
    """
    pd.DataFrame(
        [
            {
                "src_type": "municipality",
                "src_id": "01001",
                "dst_type": "abs",
                "dst_id": "101",
                "overlap_pop_est": 1000.0,
            }
        ]
    ).to_parquet(root / "overlap_edges.parquet", index=False)
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
            ("abs", "101", "municipality", "01001", 0.5),
            ("abs", "101", "municipality", "01002", 0.5),
            ("abs", "101", "municipality", "01003", 0.5),
            ("abs", "102", "municipality", "01004", 0.5),
        ],
    )
    edge(
        "muni_comarca",
        [
            ("municipality", "01001", "comarca", "01", 0.7),
            ("municipality", "01002", "comarca", "01", 0.3),
            ("municipality", "01003", "comarca", "02", 0.6),
            ("comarca", "01", "municipality", "01001", 1.0),
            ("comarca", "01", "municipality", "01002", 1.0),
            ("comarca", "02", "municipality", "01003", 1.0),
        ],
    )
    if with_od:
        edge(
            "mitma_muni",
            [
                ("municipality", "01001", "mitma", "z1", 1.0),
                ("mitma", "z1", "municipality", "01001", 1.0),
                ("municipality", "01002", "mitma", "z2", 1.0),
                ("mitma", "z2", "municipality", "01002", 1.0),
                ("municipality", "01003", "mitma", "z3", 1.0),
                ("mitma", "z3", "municipality", "01003", 1.0),
            ],
        )
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
                    [np.full(days, DEATH_VALUE[c], dtype=np.float32) for c in COMARCAS],
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

    abs_values = np.stack(
        [
            np.full(days, DEATH_VALUE["01"], dtype=np.float32),
            np.full(days, DEATH_VALUE["02"], dtype=np.float32),
            np.zeros(days, dtype=np.float32),
        ],
        axis=1,
    )
    abs_masks = np.stack(
        [
            _deaths_mask("01", days),
            _deaths_mask("02", days),
            np.zeros(days, dtype=bool),
        ],
        axis=1,
    )
    xr.Dataset(
        {
            "value": (("date", "abs_id"), abs_values),
            "expected": (("date", "abs_id"), np.ones_like(abs_values)),
            "excess": (("date", "abs_id"), abs_values - 1),
            "mask": (("date", "abs_id"), abs_masks),
            "age": (("date", "abs_id"), np.ones_like(abs_values, dtype=np.uint8)),
        },
        coords={"date": dates.astype("datetime64[ns]"), "abs_id": ABS_IDS},
        attrs={"schema": "obs_deaths_abs_excess"},
    ).to_zarr(root / "obs_deaths_abs_excess.zarr")

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
            "vaccination_rate": (
                ("run_id", "date", "region_id"),
                np.broadcast_to(
                    np.arange(days, dtype=np.float32)[None, :, None] / days,
                    (1, days, len(MUNIS)),
                ).copy(),
            ),
            "vaccination_rate_mask": (
                ("run_id", "date", "region_id"),
                np.ones((1, days, len(MUNIS)), dtype=bool),
            ),
            "vaccination_rate_age": (
                ("run_id", "date", "region_id"),
                np.zeros((1, days, len(MUNIS)), dtype=np.uint8),
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
