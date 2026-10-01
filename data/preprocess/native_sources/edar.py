"""Build source-native EDAR catchment and point-location relations.

The municipality contribution matrix defines an EDAR's epidemiological
catchment. The GeoJSON point defines only the plant location. These two
relations are kept separate: a missing or coastal point never changes target
support, catchment population, fold ownership, or query eligibility.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from data.native_registry import (
    SOURCE_ABS,
    SOURCE_COMARCA,
    SOURCE_MUNICIPALITY,
    EdarRegistryReport,
    build_edar_registry,
    build_municipal_registry,
)
from data.preprocess.native_sources.overlap import load_geometry

CATCHMENT_SCHEMA = "native_edar_catchment.v1"
POINT_SCHEMA = "native_edar_point_locations.v1"

RELATION_BY_TARGET = {
    SOURCE_MUNICIPALITY: "muni_edar_catchment",
    SOURCE_ABS: "edar_abs_catchment",
    SOURCE_COMARCA: "edar_comarca_catchment",
}
POINT_RELATION_BY_TARGET = {
    SOURCE_MUNICIPALITY: "edar_located_in_municipality",
    SOURCE_ABS: "edar_located_in_abs",
    SOURCE_COMARCA: "edar_located_in_comarca",
}

CATCHMENT_EDGE_COLUMNS = (
    "relation",
    "edar_id",
    "target_type",
    "target_id",
    "contribution_ratio",
    "served_population_est",
    "catchment_population_est",
    "frac_of_edar_catchment",
    "frac_of_target_population_served",
    "provenance",
)
POINT_EDGE_COLUMNS = (
    "relation",
    "edar_id",
    "target_type",
    "target_id",
    "provenance",
)

_WEIGHT_TOL = 1e-6

DEFAULT_WASTEWATER_CSV = Path("data/files/wastewater_biomarkers_icra.csv")
DEFAULT_CONTRIBUTION = Path("data/files/edar_muni_edges.nc")
DEFAULT_EDAR_GEOJSON = Path("data/files/geo/EDAR.geojson")
DEFAULT_MUNICIPAL_GEOJSON = Path("data/files/geo/fl_municipios_catalonia.geojson")
DEFAULT_ABS_SHP = Path("data/files/geo/cartografia/ABS.shp")
DEFAULT_COMARCA_GEOJSON = Path(
    "data/files/geo/generated/comarques_igcg_v2r1.geojson"
)
DEFAULT_POPULATION_CSV = Path("data/files/fl_population_por_municipis.csv")
DEFAULT_REFERENCE_CSV = Path("data/files/municipality_id_reference.csv")
DEFAULT_OVERLAP = Path("data/processed/native_support/overlap_edges.parquet")
DEFAULT_OUTPUT_ROOT = Path("data/processed/native_support")


@dataclass(frozen=True)
class EdarCatchmentReport:
    """Counts and allocation coverage for EDAR catchment relations."""

    edar_count: int
    municipality_edge_count: int
    abs_edge_count: int
    comarca_edge_count: int
    total_catchment_population_est: float
    abs_allocation_fraction_min: float
    comarca_allocation_fraction_min: float
    missing_abs_municipalities: tuple[str, ...]
    missing_comarca_municipalities: tuple[str, ...]


@dataclass(frozen=True)
class EdarPointReport:
    """Point-location coverage by administrative target type."""

    geometry_ids: tuple[str, ...]
    located_ids: dict[str, tuple[str, ...]]
    missing_location_ids: dict[str, tuple[str, ...]]


def load_contribution_ratios(path: str | Path) -> xr.DataArray:
    """Load and validate the ``edar_id × home`` contribution matrix."""
    with xr.open_dataset(path) as dataset:
        if "contribution_ratio" not in dataset:
            raise ValueError("EDAR contribution store lacks 'contribution_ratio'")
        contribution = dataset["contribution_ratio"].load()
    if contribution.dims != ("edar_id", "home"):
        raise ValueError(
            "EDAR contribution_ratio dims must be ('edar_id', 'home'), got "
            f"{contribution.dims}"
        )
    edar_ids = [str(value) for value in contribution["edar_id"].values.tolist()]
    municipality_ids = [str(value) for value in contribution["home"].values.tolist()]
    if len(edar_ids) != len(set(edar_ids)):
        raise ValueError("Duplicate edar_id labels in contribution matrix")
    if len(municipality_ids) != len(set(municipality_ids)):
        raise ValueError("Duplicate home labels in contribution matrix")
    values = np.asarray(contribution.values, dtype=np.float64)
    bad = np.isfinite(values) & ((values < 0) | (values > 1 + _WEIGHT_TOL))
    if bad.any():
        raise ValueError("EDAR contribution ratios must be in [0, 1]")
    return contribution.assign_coords(
        edar_id=np.asarray(edar_ids, dtype=str),
        home=np.asarray(municipality_ids, dtype=str),
    )


def _municipality_target_weights(
    overlap: pd.DataFrame, target_type: str
) -> pd.DataFrame:
    """Canonical municipality-to-target population fractions from OVL-1."""
    required = {"src_type", "src_id", "dst_type", "dst_id", "frac_of_src_pop"}
    missing = sorted(required - set(overlap.columns))
    if missing:
        raise ValueError(f"overlap table missing columns: {missing}")
    rows = overlap[
        (overlap["src_type"] == SOURCE_MUNICIPALITY)
        & (overlap["dst_type"] == target_type)
    ][["src_id", "dst_id", "frac_of_src_pop"]].copy()
    rows.columns = ["municipality_id", "target_id", "municipality_fraction"]
    rows["municipality_id"] = rows["municipality_id"].astype(str)
    rows["target_id"] = rows["target_id"].astype(str)
    if rows.duplicated(["municipality_id", "target_id"]).any():
        raise ValueError(f"duplicate municipality->{target_type} overlap rows")
    weights = rows["municipality_fraction"].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(weights) | (weights < 0) | (weights > 1 + _WEIGHT_TOL)
    if bad.any():
        raise ValueError(
            f"municipality->{target_type} fractions must be finite and in [0, 1]"
        )
    return rows


def build_catchment_edges(
    contribution: xr.DataArray,
    municipal_populations: dict[str, float],
    overlap: pd.DataFrame,
) -> tuple[pd.DataFrame, EdarCatchmentReport]:
    """Allocate EDAR catchment population to municipalities, ABS, and comarcas.

    ``contribution_ratio[e, m]`` is the fraction of municipality ``m`` served
    by EDAR ``e``. The estimated population mass is therefore
    ``ratio × audited municipal population``. ABS and comarca relations sum
    that mass through municipality-normalized OVL-1 fractions. Point geometry
    is not used in this calculation.
    """
    contribution = load_contribution_ratios_from_array(contribution)
    long = contribution.to_series().rename("contribution_ratio").reset_index()
    long["edar_id"] = long["edar_id"].astype(str)
    long["home"] = long["home"].astype(str)
    long = long[np.isfinite(long["contribution_ratio"])]
    long = long[long["contribution_ratio"] > 0].copy()
    if long.empty:
        raise ValueError("EDAR contribution matrix has no positive support")

    missing_population = sorted(set(long["home"]) - set(municipal_populations))
    if missing_population:
        raise ValueError(
            "EDAR catchment municipalities lack audited population: "
            f"{missing_population}"
        )
    population = pd.Series(municipal_populations, dtype=float)
    if (~np.isfinite(population) | (population <= 0)).any():
        bad = population.index[(~np.isfinite(population) | (population <= 0))]
        raise ValueError(f"Municipal populations must be positive and finite: {list(bad)}")

    long["municipality_population"] = long["home"].map(population)
    long["served_population_est"] = (
        long["contribution_ratio"] * long["municipality_population"]
    )
    catchment_population = long.groupby("edar_id")["served_population_est"].sum()
    if (catchment_population <= 0).any():
        raise ValueError("Every EDAR with support must have positive catchment population")

    rows: list[dict[str, object]] = []
    for record in long.itertuples(index=False):
        catchment = float(catchment_population.loc[record.edar_id])
        rows.append(
            {
                "relation": RELATION_BY_TARGET[SOURCE_MUNICIPALITY],
                "edar_id": record.edar_id,
                "target_type": SOURCE_MUNICIPALITY,
                "target_id": record.home,
                "contribution_ratio": float(record.contribution_ratio),
                "served_population_est": float(record.served_population_est),
                "catchment_population_est": catchment,
                "frac_of_edar_catchment": float(record.served_population_est)
                / catchment,
                "frac_of_target_population_served": float(
                    record.contribution_ratio
                ),
                "provenance": "edar_municipality_contribution_ratio",
            }
        )

    missing_by_target: dict[str, tuple[str, ...]] = {}
    allocation_fractions: dict[str, pd.Series] = {}
    for target_type in (SOURCE_ABS, SOURCE_COMARCA):
        weights = _municipality_target_weights(overlap, target_type)
        supported_municipalities = set(long["home"])
        mapped_municipalities = set(weights["municipality_id"])
        missing_by_target[target_type] = tuple(
            sorted(supported_municipalities - mapped_municipalities)
        )
        allocated = long.merge(
            weights,
            left_on="home",
            right_on="municipality_id",
            how="inner",
            validate="many_to_many",
        )
        allocated["allocated_population"] = (
            allocated["served_population_est"]
            * allocated["municipality_fraction"]
        )
        grouped = (
            allocated.groupby(["edar_id", "target_id"], as_index=False)[
                "allocated_population"
            ]
            .sum()
            .sort_values(["edar_id", "target_id"], kind="stable")
        )
        target_population = (
            weights.assign(
                municipal_population=weights["municipality_id"].map(population)
            )
            .assign(
                target_population=lambda frame: frame["municipal_population"]
                * frame["municipality_fraction"]
            )
            .groupby("target_id")["target_population"]
            .sum()
        )
        allocated_by_edar = grouped.groupby("edar_id")["allocated_population"].sum()
        allocation_fractions[target_type] = (
            allocated_by_edar / catchment_population
        ).reindex(catchment_population.index, fill_value=0.0)
        for record in grouped.itertuples(index=False):
            catchment = float(catchment_population.loc[record.edar_id])
            target_pop = float(target_population.loc[record.target_id])
            rows.append(
                {
                    "relation": RELATION_BY_TARGET[target_type],
                    "edar_id": record.edar_id,
                    "target_type": target_type,
                    "target_id": record.target_id,
                    "contribution_ratio": np.nan,
                    "served_population_est": float(record.allocated_population),
                    "catchment_population_est": catchment,
                    "frac_of_edar_catchment": float(record.allocated_population)
                    / catchment,
                    "frac_of_target_population_served": float(
                        record.allocated_population
                    )
                    / target_pop,
                    "provenance": (
                        "edar_municipality_contribution_ratio_x_"
                        "municipality_overlap_fraction"
                    ),
                }
            )

    edges = pd.DataFrame(rows, columns=CATCHMENT_EDGE_COLUMNS).sort_values(
        ["relation", "edar_id", "target_id"], kind="stable"
    )
    edges = edges.reset_index(drop=True)
    report = EdarCatchmentReport(
        edar_count=len(catchment_population),
        municipality_edge_count=int(
            (edges["target_type"] == SOURCE_MUNICIPALITY).sum()
        ),
        abs_edge_count=int((edges["target_type"] == SOURCE_ABS).sum()),
        comarca_edge_count=int((edges["target_type"] == SOURCE_COMARCA).sum()),
        total_catchment_population_est=float(catchment_population.sum()),
        abs_allocation_fraction_min=float(allocation_fractions[SOURCE_ABS].min()),
        comarca_allocation_fraction_min=float(
            allocation_fractions[SOURCE_COMARCA].min()
        ),
        missing_abs_municipalities=missing_by_target[SOURCE_ABS],
        missing_comarca_municipalities=missing_by_target[SOURCE_COMARCA],
    )
    return edges, report


def load_contribution_ratios_from_array(contribution: xr.DataArray) -> xr.DataArray:
    """Validate an in-memory contribution matrix with the file contract."""
    if contribution.dims != ("edar_id", "home"):
        raise ValueError(
            "EDAR contribution_ratio dims must be ('edar_id', 'home'), got "
            f"{contribution.dims}"
        )
    values = np.asarray(contribution.values, dtype=np.float64)
    bad = np.isfinite(values) & ((values < 0) | (values > 1 + _WEIGHT_TOL))
    if bad.any():
        raise ValueError("EDAR contribution ratios must be in [0, 1]")
    if len(set(map(str, contribution["edar_id"].values.tolist()))) != contribution.sizes[
        "edar_id"
    ]:
        raise ValueError("Duplicate edar_id labels in contribution matrix")
    if len(set(map(str, contribution["home"].values.tolist()))) != contribution.sizes[
        "home"
    ]:
        raise ValueError("Duplicate home labels in contribution matrix")
    return contribution


def build_point_location_edges(
    points: gpd.GeoDataFrame,
    target_layers: dict[str, tuple[gpd.GeoDataFrame, str]],
    *,
    edar_id_col: str = "edar_id",
) -> tuple[pd.DataFrame, EdarPointReport]:
    """Locate EDAR points in each polygon layer without nearest fallbacks."""
    if edar_id_col not in points:
        raise ValueError(f"EDAR point layer lacks {edar_id_col!r}")
    points = points[[edar_id_col, "geometry"]].copy()
    points[edar_id_col] = points[edar_id_col].astype(str)
    if points[edar_id_col].duplicated().any():
        raise ValueError("EDAR point layer contains duplicate identifiers")
    if points.crs is None:
        raise ValueError("EDAR point layer has no CRS")

    rows: list[dict[str, str]] = []
    located_ids: dict[str, tuple[str, ...]] = {}
    missing_ids: dict[str, tuple[str, ...]] = {}
    point_ids = set(points[edar_id_col])
    for target_type, (layer, target_id_col) in sorted(target_layers.items()):
        if target_type not in POINT_RELATION_BY_TARGET:
            raise ValueError(f"Unsupported EDAR point target type {target_type!r}")
        if target_id_col not in layer:
            raise ValueError(
                f"{target_type} point-location layer lacks {target_id_col!r}"
            )
        target = layer[[target_id_col, "geometry"]].copy().to_crs(points.crs)
        target[target_id_col] = target[target_id_col].astype(str)
        joined = gpd.sjoin(points, target, how="left", predicate="intersects")
        matches = joined.dropna(subset=[target_id_col])[
            [edar_id_col, target_id_col]
        ].drop_duplicates()
        ambiguous = matches.groupby(edar_id_col)[target_id_col].nunique()
        ambiguous_ids = sorted(ambiguous[ambiguous > 1].index.astype(str))
        if ambiguous_ids:
            raise ValueError(
                f"EDAR points intersect multiple {target_type} polygons: "
                f"{ambiguous_ids}"
            )
        for edar_id, target_id in matches.itertuples(index=False):
            rows.append(
                {
                    "relation": POINT_RELATION_BY_TARGET[target_type],
                    "edar_id": str(edar_id),
                    "target_type": target_type,
                    "target_id": str(target_id),
                    "provenance": "edar_point_intersects_polygon",
                }
            )
        found = set(matches[edar_id_col].astype(str))
        located_ids[target_type] = tuple(sorted(found))
        missing_ids[target_type] = tuple(sorted(point_ids - found))

    edges = pd.DataFrame(rows, columns=POINT_EDGE_COLUMNS).sort_values(
        ["relation", "edar_id", "target_id"], kind="stable"
    )
    report = EdarPointReport(
        geometry_ids=tuple(sorted(point_ids)),
        located_ids=located_ids,
        missing_location_ids=missing_ids,
    )
    return edges.reset_index(drop=True), report


def write_edar_support(
    catchment_edges: pd.DataFrame,
    point_edges: pd.DataFrame,
    catchment_report: EdarCatchmentReport,
    point_report: EdarPointReport,
    output_root: str | Path,
    registry_report: EdarRegistryReport | None = None,
) -> None:
    """Write deterministic EDAR relation tables and their audit report."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    catchment_edges.to_parquet(root / "edar_catchment_edges.parquet", index=False)
    point_edges.to_parquet(root / "edar_point_locations.parquet", index=False)
    payload = {
        "catchment_schema": CATCHMENT_SCHEMA,
        "point_schema": POINT_SCHEMA,
        "catchment": asdict(catchment_report),
        "point": asdict(point_report),
    }
    if registry_report is not None:
        payload["registry"] = asdict(registry_report)
    (root / "edar_support_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_real_edar_support(
    *,
    wastewater_csv: Path = DEFAULT_WASTEWATER_CSV,
    contribution_path: Path = DEFAULT_CONTRIBUTION,
    edar_geojson: Path = DEFAULT_EDAR_GEOJSON,
    municipal_geojson: Path = DEFAULT_MUNICIPAL_GEOJSON,
    abs_shp: Path = DEFAULT_ABS_SHP,
    comarca_geojson: Path = DEFAULT_COMARCA_GEOJSON,
    population_csv: Path = DEFAULT_POPULATION_CSV,
    reference_csv: Path = DEFAULT_REFERENCE_CSV,
    overlap_path: Path = DEFAULT_OVERLAP,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    """Build the real EDAR registry audit and both relation tables."""
    _edar_registry, registry_report = build_edar_registry(
        wastewater_csv, contribution_path, edar_geojson
    )
    municipal_registry, _municipal_report = build_municipal_registry(
        population_csv, reference_csv
    )
    populations = {
        record.source_id: float(record.population)
        for record in municipal_registry.records_for(SOURCE_MUNICIPALITY)
        if record.population is not None
    }
    catchment_edges, catchment_report = build_catchment_edges(
        load_contribution_ratios(contribution_path),
        populations,
        pd.read_parquet(overlap_path),
    )

    points = gpd.read_file(edar_geojson).rename(columns={"codi_edar": "edar_id"})
    points = points.to_crs("EPSG:3042")
    municipalities = load_geometry(
        municipal_geojson, "id", "municipality_id", 5
    )
    abs_layer = load_geometry(abs_shp, "CODABSa", "abs_id", 3)
    comarcas = gpd.read_file(comarca_geojson).to_crs("EPSG:3042")
    point_edges, point_report = build_point_location_edges(
        points,
        {
            SOURCE_MUNICIPALITY: (municipalities, "municipality_id"),
            SOURCE_ABS: (abs_layer, "abs_id"),
            SOURCE_COMARCA: (comarcas, "comarca_id"),
        },
    )
    write_edar_support(
        catchment_edges,
        point_edges,
        catchment_report,
        point_report,
        output_root,
        registry_report,
    )
    return {
        "registry": asdict(registry_report),
        "catchment": asdict(catchment_report),
        "point": asdict(point_report),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI for the real source-native EDAR relation artifacts."""
    parser = argparse.ArgumentParser(
        description="Build native EDAR catchment and point-location relations."
    )
    parser.add_argument("--wastewater-csv", type=Path, default=DEFAULT_WASTEWATER_CSV)
    parser.add_argument("--contribution", type=Path, default=DEFAULT_CONTRIBUTION)
    parser.add_argument("--edar-geojson", type=Path, default=DEFAULT_EDAR_GEOJSON)
    parser.add_argument("--overlap", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    summary = build_real_edar_support(
        wastewater_csv=args.wastewater_csv,
        contribution_path=args.contribution,
        edar_geojson=args.edar_geojson,
        overlap_path=args.overlap,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
