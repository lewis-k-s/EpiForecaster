"""Build the eligibility-backed version-4 native source-set asset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.native_dataset import NativeSupportDataset
from data.native_inputs import native_dataset, query_inventory, split_identity
from data.native_source_sets import (
    QueryEligibilityRecord,
    SourceBalanceRecord,
    assemble_source_set_manifest,
    assign_balanced_source_sets,
    freeze_calendar_split,
    write_source_set_manifest,
)
from models.configs import EpiForecasterConfig

DEFAULT_CONFIG = Path("configs/train_epifor_smoke.yaml")
DEFAULT_OUTPUT = Path("data/processed/native_support/native_source_sets")

HEAD_BY_TYPE = {
    "municipality": "cases",
    "abs": "hosp+deaths",
    "edar": "ww",
}


def _dominant_province_from_edges(
    edges: pd.DataFrame,
    *,
    source_column: str,
    municipality_column: str,
    mass_column: str,
) -> dict[str, str]:
    province_mass: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in edges.itertuples(index=False):
        source_id = str(getattr(row, source_column))
        municipality_id = str(getattr(row, municipality_column))
        province_mass[source_id][municipality_id[:2]] += float(
            getattr(row, mass_column)
        )
    return {
        source_id: max(masses, key=lambda province: (masses[province], province))
        for source_id, masses in province_mass.items()
    }


def _clinical_burden(
    root: Path,
    real_path: Path,
    graph_population: dict[tuple[str, str], float],
    cutoff: np.datetime64,
) -> dict[tuple[str, str], float]:
    burden: dict[tuple[str, str], float] = {}
    cases = xr.open_zarr(real_path)[["cases", "cases_mask"]]
    if "run_id" in cases.dims:
        run_ids = [str(value) for value in cases["run_id"].values]
        cases = cases.isel(run_id=run_ids.index("real") if "real" in run_ids else 0)
    cases = cases.sel(date=cases.date < cutoff)
    for column, source_id in enumerate(map(str, cases["region_id"].values.tolist())):
        population = graph_population.get(("municipality", source_id))
        if population is None or not math.isfinite(population):
            continue
        values = np.asarray(cases["cases"].isel(region_id=column).values, dtype=float)
        mask = np.asarray(cases["cases_mask"].isel(region_id=column).values, dtype=bool)
        burden[("municipality", source_id)] = float(
            np.expm1(values[mask]).sum() * population / 100_000.0
        )
    for source_type, filename, dim_name in (
        ("abs", "obs_admissions.zarr", "abs_id"),
        ("abs", "obs_deaths_abs_excess.zarr", "abs_id"),
    ):
        dataset = xr.open_zarr(root / filename)
        # Weekly totals become available only after the complete reporting week.
        available_date = (
            dataset.week_end if filename == "obs_admissions.zarr" else dataset.date
        )
        dataset = dataset.sel(date=available_date < cutoff)
        for column, source_id in enumerate(map(str, dataset[dim_name].values.tolist())):
            values = np.asarray(dataset["value"].isel({dim_name: column}).values)
            mask = np.asarray(
                dataset["mask"].isel({dim_name: column}).values, dtype=bool
            )
            key = (source_type, source_id)
            burden[key] = burden.get(key, 0.0) + float(values[mask].sum())
    return burden


def build_inventory(
    config: EpiForecasterConfig,
    dataset: NativeSupportDataset | None = None,
) -> tuple[list[SourceBalanceRecord], list[QueryEligibilityRecord], np.ndarray]:
    """Build source-level balance records and the full query inventory."""
    root = Path(config.native.root)
    if dataset is None:
        dataset = native_dataset(config)
    grid = dataset.date_grid
    queries = query_inventory(dataset)
    calendar = freeze_calendar_split(
        queries,
        validation_fraction=config.native.validation_fraction,
        test_fraction=config.native.test_fraction,
        forecast_horizon=config.model.forecast_horizon,
    )
    cutoff = grid[calendar.validation_start_origin]
    clinical_queries = [q for q in queries if q.source_type != "edar"]
    clinical_counts = Counter(
        (query.source_type, query.source_id) for query in clinical_queries
    )

    graph = xr.open_zarr(root / "native_graph.zarr")
    graph_population = {
        (str(node_type), str(source_id)): float(population)
        for node_type, source_id, population in zip(
            graph["node_type"].values.tolist(),
            graph["source_id"].values.tolist(),
            graph["population"].values.tolist(),
            strict=True,
        )
        if np.isfinite(population)
    }
    burden = _clinical_burden(
        root, Path(config.data.dataset_path), graph_population, cutoff
    )
    overlap = pd.read_parquet(root / "overlap_edges.parquet")
    clinical_geography: dict[tuple[str, str], str] = {}
    for source_type in ("abs",):
        rows = overlap[
            (overlap["src_type"] == "municipality")
            & (overlap["dst_type"] == source_type)
        ].rename(
            columns={
                "dst_id": "source_id",
                "src_id": "municipality_id",
                "overlap_pop_est": "mass",
            }
        )
        dominant = _dominant_province_from_edges(
            rows,
            source_column="source_id",
            municipality_column="municipality_id",
            mass_column="mass",
        )
        clinical_geography.update(
            ((source_type, source_id), province)
            for source_id, province in dominant.items()
        )

    records: list[SourceBalanceRecord] = []
    for (source_type, source_id), query_count in sorted(clinical_counts.items()):
        population = graph_population[(source_type, source_id)]
        geography = (
            source_id[:2]
            if source_type == "municipality"
            else clinical_geography.get((source_type, source_id), "unknown")
        )
        records.append(
            SourceBalanceRecord(
                source_type=source_type,
                source_id=source_id,
                head=HEAD_BY_TYPE[source_type],
                eligible_queries=query_count,
                support_population=population,
                target_burden=burden.get((source_type, source_id), 0.0),
                geography=geography,
            )
        )

    if not config.native.wastewater_enabled:
        return records, queries, grid

    wastewater = xr.open_zarr(root / config.native.wastewater_filename)
    wastewater = wastewater.sel(date=wastewater.date < cutoff).compute()
    ww_counts = Counter(q.source_id for q in queries if q.source_type == "edar")
    catchment = pd.read_parquet(root / "edar_catchment_edges.parquet")
    catchment_muni = catchment[catchment["target_type"] == "municipality"]
    catchment_population = catchment_muni.groupby("edar_id")[
        "catchment_population_est"
    ].first()
    edar_geography = _dominant_province_from_edges(
        catchment_muni.rename(
            columns={"target_id": "municipality_id", "served_population_est": "mass"}
        ),
        source_column="edar_id",
        municipality_column="municipality_id",
        mass_column="mass",
    )
    observed = np.asarray(wastewater["observed_mask"].values, dtype=bool)
    censor = np.asarray(wastewater["censor_code"].values)
    flow_mask = np.asarray(wastewater["flow_mask"].values, dtype=bool)
    raw = np.asarray(wastewater["concentration_raw"].values, dtype=float)
    edar_ids = [str(value) for value in wastewater["edar_id"].values.tolist()]
    for column, edar_id in enumerate(edar_ids):
        query_count = ww_counts.get(edar_id, 0)
        if query_count <= 0:
            continue
        site_observed = observed[:, column, :]
        n_observed = int(site_observed.sum())
        sample_days = site_observed.any(axis=1)
        records.append(
            SourceBalanceRecord(
                source_type="edar",
                source_id=edar_id,
                head="ww",
                eligible_queries=query_count,
                support_population=float(catchment_population.loc[edar_id]),
                target_burden=float(np.log1p(raw[:, column, :][site_observed]).sum()),
                geography=edar_geography.get(edar_id, "unknown"),
                assay_observations=n_observed,
                censored_fraction=float(
                    (censor[:, column, :][site_observed] == 1).mean()
                )
                if n_observed
                else 0.0,
                flow_coverage=float(flow_mask[:, column][sample_days].mean())
                if sample_days.any()
                else 0.0,
            )
        )
    return records, queries, grid


def build_for_config(
    config: EpiForecasterConfig, output_path: Path
) -> dict[str, object]:
    """Derive one deterministic DVC-ready split asset from processed inputs."""
    dataset = native_dataset(config)
    records, queries, _grid = build_inventory(config, dataset)
    assignments = assign_balanced_source_sets(
        records, num_folds=config.native.num_folds, seed=config.training.split_seed
    )
    calendar = freeze_calendar_split(
        queries,
        validation_fraction=config.native.validation_fraction,
        test_fraction=config.native.test_fraction,
        forecast_horizon=config.model.forecast_horizon,
        min_queries_per_head=config.native.num_folds,
    )
    payload = assemble_source_set_manifest(
        records,
        assignments,
        calendar,
        num_folds=config.native.num_folds,
        seed=config.training.split_seed,
    )
    payload["provenance"] = split_identity(config, dataset)
    # Reject an asset that cannot provide every active head in each run.
    from data.native_source_sets import rotating_fold_roles, split_queries_for_run

    expected = {q.head for q in queries}
    for run in range(config.native.num_folds):
        for positions in split_queries_for_run(
            queries,
            assignments,
            calendar,
            rotating_fold_roles(run, config.native.num_folds),
        ):
            if {queries[i].head for i in positions} != expected:
                raise ValueError(
                    f"Source-set run {run} lacks a supervised head after purging"
                )
    write_source_set_manifest(payload, output_path)
    return payload


def build(
    config_path: Path,
    output_path: Path,
    *,
    num_folds: int = 10,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, object]:
    config = EpiForecasterConfig.load(config_path)
    config.native.num_folds = num_folds
    config.native.validation_fraction = validation_fraction
    config.native.test_fraction = test_fraction
    return build_for_config(config, output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build eligibility-backed native source sets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    payload = build(
        args.config,
        args.output,
        num_folds=args.num_folds,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": payload["schema"],
                "provenance": payload["provenance"],
                "fold_metrics": payload["fold_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
