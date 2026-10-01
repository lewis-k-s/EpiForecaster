"""Small deterministic native datasets for pipeline tests."""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from models.configs import (
    DataConfig,
    EpiForecasterConfig,
    ModelConfig,
    ModelVariant,
    NativeSupportConfig,
    OutputConfig,
    TrainingParams,
)

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
    cases zarr and the small OD fixture used by the native trainer."""
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

    edge(
        "mitma_muni",
        [
            ("mitma", "z1", "municipality", "01001", 1.0),
            ("municipality", "01001", "mitma", "z1", 1.0),
        ],
    )
    edge("mitma_abs", [("mitma", "z1", "abs", "101", 1.0)])
    edge("mitma_comarca", [("mitma", "z1", "comarca", "01", 1.0)])
    xr.Dataset(
        {
            "mobility": (
                ("date", "origin", "target"),
                np.zeros((T, 3, 3), dtype=np.float32),
            )
        },
        coords={
            "date": dates.astype("datetime64[ns]"),
            "origin": zones,
            "target": zones,
            "zone": np.arange(3),
            "zone_id": ("zone", zones),
        },
    ).to_zarr(root / "od_mitma.zarr")

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
            "age": (
                ("date", "comarca_id"),
                np.ones((T, len(COMARCAS)), dtype=np.uint8),
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
            np.full(T, DEATH_VALUE["01"], dtype=np.float32),
            np.full(T, DEATH_VALUE["02"], dtype=np.float32),
            np.zeros(T, dtype=np.float32),
        ],
        axis=1,
    )
    abs_masks = np.ones_like(abs_values, dtype=bool)
    abs_masks[:, 2] = False
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
            "vaccination_rate": (
                ("run_id", "date", "region_id"),
                np.broadcast_to(
                    np.arange(T, dtype=np.float32)[None, :, None] / T,
                    (1, T, len(MUNIS)),
                ).copy(),
            ),
            "vaccination_rate_mask": (
                ("run_id", "date", "region_id"),
                np.ones((1, T, len(MUNIS)), dtype=bool),
            ),
            "vaccination_rate_age": (
                ("run_id", "date", "region_id"),
                np.zeros((1, T, len(MUNIS)), dtype=np.uint8),
            ),
            "temporal_covariates": (
                ("date", "covariate"),
                (np.arange(T, dtype=np.float32)[:, None] * np.arange(4)[None, :])
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


def _write_native_wastewater(root: Path) -> None:
    """Write two EDAR series and their municipality catchment rows."""
    dates = GRID_START + np.arange(T, dtype="timedelta64[D]")
    edar_ids = ["E1", "E2"]
    assays = ["N1", "N2", "IP4"]
    shape = (T, len(edar_ids), len(assays))
    observed = np.zeros(shape, dtype=bool)
    observed[2::7] = True
    concentration = np.zeros(shape, dtype=np.float32)
    concentration[observed] = np.tile(
        np.array([100.0, 200.0, 50.0], dtype=np.float32),
        int(observed.sum() / len(assays)),
    )
    detection_limit = np.full(shape, 10.0, dtype=np.float32)
    censor_code = np.where(observed, 0, 2).astype(np.uint8)
    history = np.full(shape, np.nan, dtype=np.float32)
    history_available = np.zeros(shape, dtype=bool)
    age = np.full(shape, np.nan, dtype=np.float32)
    for edar in range(len(edar_ids)):
        last = np.full(len(assays), np.nan, dtype=np.float32)
        last_day = np.full(len(assays), -1, dtype=np.int32)
        for day in range(T):
            assay_mask = observed[day, edar]
            last[assay_mask] = np.log1p(concentration[day, edar, assay_mask])
            last_day[assay_mask] = day
            available = last_day >= 0
            history[day, edar, available] = last[available]
            history_available[day, edar, available] = True
            age[day, edar, available] = day - last_day[available]
    flow = np.full((T, len(edar_ids)), 1000.0, dtype=np.float32)
    wastewater = xr.Dataset(
        {
            "concentration_raw": (("date", "edar_id", "assay"), concentration),
            "detection_limit": (("date", "edar_id", "assay"), detection_limit),
            "observed_mask": (("date", "edar_id", "assay"), observed),
            "censor_code": (("date", "edar_id", "assay"), censor_code),
            "history_log1p": (("date", "edar_id", "assay"), history),
            "history_available_mask": (
                ("date", "edar_id", "assay"),
                history_available,
            ),
            "history_age_days": (("date", "edar_id", "assay"), age),
            "flow_m3_24h": (("date", "edar_id"), flow),
            "flow_mask": (("date", "edar_id"), np.ones_like(flow, dtype=bool)),
        },
        coords={
            "date": dates.astype("datetime64[ns]"),
            "edar_id": edar_ids,
            "assay": assays,
        },
        attrs={"schema": "native_wastewater_observations.v1"},
    )
    wastewater.to_zarr(root / "obs_wastewater.zarr")
    pd.DataFrame(
        [
            {
                "relation": "muni_edar_catchment",
                "edar_id": "E1",
                "target_type": "municipality",
                "target_id": "01001",
                "catchment_population_est": 1000.0,
                "served_population_est": 1000.0,
                "frac_of_edar_catchment": 1.0,
            },
            {
                "relation": "muni_edar_catchment",
                "edar_id": "E2",
                "target_type": "municipality",
                "target_id": "01002",
                "catchment_population_est": 2000.0,
                "served_population_est": 2000.0,
                "frac_of_edar_catchment": 1.0,
            },
        ]
    ).to_parquet(root / "edar_catchment_edges.parquet", index=False)


def _trainer_config(root: Path, **native_kwargs: bool) -> EpiForecasterConfig:
    native_options = {
        "mobility_enabled": False,
        "wastewater_enabled": False,
        **native_kwargs,
    }
    return EpiForecasterConfig(
        model=ModelConfig(
            type=ModelVariant(
                cases=False, regions=False, biomarkers=False, mobility=False
            ),
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
            num_workers=0,
            val_workers=0,
            enable_mixed_precision=False,
            val_split=0.1,
            scheduler_type="cosine",
            warmup_steps=0,
        ),
        output=OutputConfig(
            experiment_name="native_trainer_test",
            log_dir=str(root / "outputs"),
            wandb_mode="disabled",
        ),
        native=NativeSupportConfig(enabled=True, root=str(root), **native_options),
    )


def pipeline_config(
    root: Path,
    *,
    strict: bool,
    mobility: bool = False,
    vaccination: bool = True,
    wastewater: bool = True,
) -> EpiForecasterConfig:
    """Three sources per resolution and enough time for purged held-out splits."""
    from tests.native_dataset_fixtures import _write_native_root as write_graph_root

    write_graph_root(root, with_od=True)
    _write_native_wastewater(root)
    dates = GRID_START + np.arange(140, dtype="timedelta64[D]")
    for name in ("real_with_id.zarr", "od_mitma.zarr"):
        path = root / name
        original = xr.open_zarr(path).load()
        expanded = original.isel(
            date=np.arange(140) % original.sizes["date"]
        ).assign_coords(date=dates.astype("datetime64[ns]"))
        original.close()
        expanded.to_zarr(path, mode="w")
    weeks = dates[1::7]
    values = np.ones((len(weeks), 3), dtype=np.float32) * np.array([10, 20, 30])
    xr.Dataset(
        {
            "value": (("date", "abs_id"), values),
            "mask": (("date", "abs_id"), np.ones_like(values, dtype=bool)),
            "age": (("date", "abs_id"), np.ones_like(values, dtype=np.uint8)),
            "stratum_pop_sum": (("date", "abs_id"), np.ones_like(values) * 500),
        },
        coords={
            "date": weeks.astype("datetime64[ns]"),
            "abs_id": ABS_IDS,
            "week_end": ("date", (weeks + 6).astype("datetime64[ns]")),
        },
    ).to_zarr(root / "obs_admissions.zarr", mode="w")
    # Only writes use the low-level Zarr API; reads use xarray.
    graph = zarr.open_group(str(root / "native_graph.zarr"), mode="r+")
    graph["status"][len(MUNIS) + 2] = "observed"
    ww_rows = []
    for day, date in enumerate(dates):
        for edar_id in ("E1", "E2", "E3"):
            ww_rows.append(
                {
                    "id mostra": f"{edar_id}-{date}",
                    "depuradora": edar_id,
                    "LD(CG/L)": 10.0,
                    "Cabal últimes 24h(m3)": 1000.0,
                    "Pluja(mm)": 0.0,
                    **{
                        f"{assay}(CG/L)": 2.0 if day % 3 == 0 else 100.0
                        for assay in ("N1", "N2", "IP4")
                    },
                }
            )
    pd.DataFrame(ww_rows).to_csv(root / "wastewater_source.csv", index=False)
    death_rows = []
    for date in pd.date_range("2017-01-01", dates[-1]):
        for source, value in zip(ABS_IDS, (2, 4, 3), strict=True):
            death_rows.append(
                {
                    "data": date,
                    "codi_abs": source,
                    "nom_abs": source,
                    "sexe": "all",
                    "grup_edat": "all",
                    "index_socioeconomic": "all",
                    "casos": value if date >= pd.Timestamp(dates[0]) else 1,
                }
            )
    pd.DataFrame(death_rows).to_csv(root / "deaths_source.csv", index=False)
    process_pipeline_observations(root)
    catchment = pd.read_parquet(root / "edar_catchment_edges.parquet")
    third = catchment.iloc[:1].copy()
    third["edar_id"] = "E3"
    pd.concat([catchment, third]).to_parquet(
        root / "edar_catchment_edges.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "src_type": "municipality",
                "src_id": MUNIS[i],
                "dst_type": "abs",
                "dst_id": ABS_IDS[i],
                "overlap_pop_est": 1000.0,
            }
            for i in range(3)
        ]
    ).to_parquet(root / "overlap_edges.parquet", index=False)
    config = _trainer_config(
        root,
        mobility_enabled=mobility,
        wastewater_enabled=wastewater,
        include_vaccination=vaccination,
    )
    config.model.strict = strict
    config.model.forecast_horizon = 14
    config.model.input_window_length = 28
    config.native.num_folds = 3
    config.native.validation_fraction = 0.2
    config.native.test_fraction = 0.2
    config.native.fold_index = 0
    config.native.fold_manifest = str(root / "splits/manifest.json")
    config.native.head_balanced_sampling = True
    config.training.batch_size = 16
    config.training.max_batches = 2
    config.training.run_final_evaluation = False
    config.training.loss.joint.distribution_family = "student_t"
    config.training.loss.joint.distribution_df = 5.0
    config.training.loss.joint.observation_loss = "crps"
    config.output.save_checkpoints = True
    config.output.write_granular_eval = True
    config.output.granular_eval_format = (
        "csv"  # Standalone DAG exports must still be Parquet.
    )
    return config


def process_pipeline_observations(root: Path) -> None:
    """Run canonical observation processors on the small raw source files."""
    from data.preprocess.processors.abs_excess_deaths_processor import (
        build_abs_excess_deaths,
    )
    from data.preprocess.processors.native_wastewater_processor import (
        build_native_wastewater_observations,
    )

    cases = xr.open_zarr(root / "real_with_id.zarr")
    start, end = pd.Timestamp(cases.date.values[0]), pd.Timestamp(cases.date.values[-1])
    frame = pd.read_csv(
        root / "deaths_source.csv", dtype={"codi_abs": "string"}, parse_dates=["data"]
    )
    build_abs_excess_deaths(frame, ABS_IDS, start=start, end=end).to_zarr(
        root / "obs_deaths_abs_excess.zarr", mode="w"
    )
    build_native_wastewater_observations(
        root / "wastewater_source.csv", start_date=start, end_date=end
    ).to_zarr(root / "obs_wastewater.zarr", mode="w")
