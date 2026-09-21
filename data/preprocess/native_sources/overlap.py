"""Geographic overlap engine and edge tables (OVL-1, plan §12.2).

Direct pairwise intersections between the native geometry sets
(municipality ↔ ABS, municipality ↔ comarca via exact membership,
MITMA zone ↔ municipality/ABS/comarca) in EPSG:3042 — never composed
through municipal crosswalks (plan §3, §10.3(2)).

Population weighting (plan §11.7), primary ``geostat_grid`` policy:
    overlap_pop_est(A, B) = Σ_cell TOT_P(cell) · frac(cell ∩ A) · frac(cell ∩ B)
    pop_est(X)            = Σ_cell TOT_P(cell) · frac(cell ∩ X)
with frac(cell ∩ X) = area(cell ∩ X) / area(cell). The
``uniform_density`` fallback runs the identical bilinear estimator with
the municipal polygons (audited ``fl_population_por_municipis``
population, uniform within each municipality) as the surface.

Grid-allocation approach ported from the read-only sibling project
``dasymetric_mob/dasymetric_transform.py``
(``build_population_weight_matrix`` / ``_load_population_grid_geometries``:
populated-cell fractional cover × cell population), reimplemented with
plain geopandas overlays because ``tobler`` is not a dependency here.

Output per plan §11.3: ``data/processed/native_support/overlap_edges.parquet``
— one row per directed overlap pair with both endpoint fractions and
surface coverages. Fractions use surface-implied populations only;
audited municipal population never enters ``geostat_grid`` fractions
(plan §3). Comarca pop_est is the sum over mapped member municipalities
(REG-1 convention), so membership destination fractions sum to 1 up to
the unmapped-member deficit.

Run as ``uv run python -m data.preprocess.native_sources.overlap``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import box

from data.native_registry import build_municipal_registry
from data.preprocess.native_sources.comarca_geometry import (
    _sha256_file,
    load_municipality_comarca_mapping,
)

# Area-only intersections: exact boundary touches produce degenerate
# line/point parts that carry zero area — dropping them is the intent.
warnings.filterwarnings("ignore", message=".*keep_geom_type=True.*", category=UserWarning)

TARGET_CRS = "EPSG:3042"
GRID_VINTAGE = "GEOSTAT 2011"
BOUNDARY_PAD_M = 2000.0

DASYMETRIC_ROOT = Path("/Volumes/HUBSSD/code/meprecisa/dasymetric_mob")
MITMA_SHP_REL = Path("data/mitma_v1/zonificacion-distritos/distritos_mitma.shp")
GEOSTAT_CSV_REL = Path("data/GEOSTAT_grid_POP_1K_2011_V2_0_1.csv")
GRID_GEOM_REL = Path("data/GRID_GEOM/Grid_ETRS89_LAEA_1K-ref_GEOSTAT_POP_2011_V2_0_1.shp")

MUNICIPAL_GEOJSON = Path("data/files/geo/fl_municipios_catalonia.geojson")
ABS_SHP = Path("data/files/geo/cartografia/ABS.shp")
COMARCA_GEOJSON = Path("data/files/geo/generated/comarques_igcg_v2r1.geojson")
MAPPING_CSV = Path("data/files/mpiscatalunya.csv")
FL_POPULATION_CSV = Path("data/files/fl_population_por_municipis.csv")
MUNICIPAL_REFERENCE_CSV = Path("data/files/municipality_id_reference.csv")
OD_MITMA_ZARR = Path("data/processed/native_support/od_mitma.zarr")

DEFAULT_OUT = Path("data/processed/native_support/overlap_edges.parquet")
DEFAULT_REPORT = Path("data/processed/native_support/overlap_report.json")

EDGE_COLUMNS = (
    "src_type", "src_id", "dst_type", "dst_id", "overlap_area_m2",
    "overlap_pop_est", "frac_of_src_pop", "frac_of_dst_pop",
    "coverage_src", "coverage_dst", "provenance",
)
POLICIES = ("geostat_grid", "uniform_density")


# Population surface and the bilinear grid estimator


@dataclass(frozen=True)
class PopulationSurface:
    """Population pieces whose overlap allocation weights every estimate.

    ``pieces`` carries columns ``piece_id`` (str), ``population`` (float)
    and ``piece_area`` (m²) in ``TARGET_CRS``; positional index is the
    ``piece_pos`` used throughout the allocation tables. Pieces must not
    overlap each other (GEOSTAT cells and municipal polygons both
    satisfy this).
    """

    pieces: gpd.GeoDataFrame
    label: str


def _fractional_cover(
    surface: PopulationSurface, targets: gpd.GeoDataFrame, id_col: str
) -> pd.DataFrame:
    """Fractional cover table: rows (piece_pos, target_id, frac).

    ``frac`` is area(piece ∩ target) / area(piece) — the same cell-level
    fractional cover the reference dasymetric transform computes before
    multiplying by cell population.
    """
    pieces = surface.pieces
    left = pieces[["piece_area", "geometry"]].copy()
    left["piece_pos"] = pieces.index
    inter = gpd.overlay(
        left,
        targets[[id_col, "geometry"]],
        how="intersection",
    )
    if inter.empty:
        return pd.DataFrame(columns=["piece_pos", "target_id", "frac"])
    inter["frac"] = inter.geometry.area / inter["piece_area"]
    return (
        inter.rename(columns={id_col: "target_id"})[["piece_pos", "target_id", "frac"]]
        .sort_values(["target_id", "piece_pos"], kind="stable")
        .reset_index(drop=True)
    )


def node_stats(
    surface: PopulationSurface,
    targets: gpd.GeoDataFrame,
    id_col: str,
    frac_cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Surface-implied population, area and coverage per target feature.

    Indexed by target id: ``pop_est`` = Σ_piece pop·frac, ``area_m2`` and
    ``coverage`` = Σ_piece frac·piece_area / area(target) ∈ [0, 1].
    """
    if frac_cache is None or id_col not in frac_cache:
        frac = _fractional_cover(surface, targets, id_col)
        if frac_cache is not None:
            frac_cache[id_col] = frac
    else:
        frac = frac_cache[id_col]
    index = pd.Index(targets[id_col].astype(str), name=id_col)
    stats = pd.DataFrame({"pop_est": 0.0, "covered_m2": 0.0}, index=index)
    if not frac.empty:
        weighted = frac.assign(
            _pop=frac["piece_pos"].map(surface.pieces["population"]) * frac["frac"],
            _cov=frac["piece_pos"].map(surface.pieces["piece_area"]) * frac["frac"],
        )
        stats[["pop_est", "covered_m2"]] = weighted.groupby("target_id")[
            ["_pop", "_cov"]
        ].sum()
    stats["area_m2"] = pd.Series(
        targets.geometry.area.to_numpy(dtype=float), index=index
    )
    stats["coverage"] = (stats["covered_m2"] / stats["area_m2"]).fillna(0.0)
    return stats[["pop_est", "area_m2", "coverage"]]


def overlap_pop_table(
    surface: PopulationSurface,
    frac_a: pd.DataFrame,
    frac_b: pd.DataFrame,
) -> pd.DataFrame:
    """Shared population per pair: Σ_piece pop · frac_a · frac_b.

    Symmetric by construction: swapping the two cover tables yields the
    same pair values.
    """
    left = frac_a.assign(_pop=frac_a["piece_pos"].map(surface.pieces["population"]))
    merged = left.merge(
        frac_b.rename(columns={"target_id": "target_id_b", "frac": "frac_b"}),
        on="piece_pos",
    )
    if merged.empty:
        return pd.DataFrame(columns=["a_id", "b_id", "overlap_pop_est"])
    merged["_w"] = merged["frac"] * merged["frac_b"] * merged["_pop"]
    out = (
        merged.groupby(["target_id", "target_id_b"])["_w"].sum().reset_index()
    )
    out.columns = ["a_id", "b_id", "overlap_pop_est"]
    return out


def polygon_overlap_area(
    a: gpd.GeoDataFrame, a_col: str, b: gpd.GeoDataFrame, b_col: str
) -> pd.DataFrame:
    """Direct polygon intersection area (m²) per id pair — both policies."""
    inter = gpd.overlay(
        a[[a_col, "geometry"]].rename(columns={a_col: "a_id"}),
        b[[b_col, "geometry"]].rename(columns={b_col: "b_id"}),
        how="intersection",
    )
    if inter.empty:
        return pd.DataFrame(columns=["a_id", "b_id", "overlap_area_m2"])
    inter["_area"] = inter.geometry.area
    out = inter.groupby(["a_id", "b_id"])["_area"].sum().reset_index()
    return out.rename(columns={"_area": "overlap_area_m2"})


# Concrete surfaces


def load_geostat_surface(
    grid_geom_path: str | Path,
    grid_csv_path: str | Path,
    study_bounds: tuple[float, float, float, float],
) -> PopulationSurface:
    """Load populated GEOSTAT 1-km cells inside the padded study bounds.

    Mirrors the reference loader: merge cell polygons with population,
    keep ``TOT_P > 0``, project to the metric CRS, clip to the study
    window before any overlay work.
    """
    grid = gpd.read_file(grid_geom_path)
    if "GRD_ID" not in grid.columns:
        raise ValueError(f"Grid geometry missing GRD_ID in {grid_geom_path}")
    minx, miny, maxx, maxy = study_bounds
    window_3035 = (
        gpd.GeoSeries(
            [box(minx - BOUNDARY_PAD_M, miny - BOUNDARY_PAD_M, maxx + BOUNDARY_PAD_M, maxy + BOUNDARY_PAD_M)],
            crs=TARGET_CRS,
        )
        .to_crs(grid.crs)
        .iloc[0]
    )
    keep = grid.sindex.query(window_3035, predicate="intersects")
    grid = grid.iloc[sorted(keep)].to_crs(TARGET_CRS)

    population = pd.read_csv(grid_csv_path, usecols=["GRD_ID", "TOT_P"])
    population = population.drop_duplicates("GRD_ID")
    grid = grid.merge(population, on="GRD_ID", how="inner")
    grid = grid[grid["TOT_P"].notna() & (grid["TOT_P"] > 0)].copy()
    if grid.empty:
        raise ValueError("No populated GEOSTAT cells inside the study bounds")
    grid["piece_area"] = grid.geometry.area
    grid = grid[grid["piece_area"] > 0].reset_index(drop=True)
    pieces = gpd.GeoDataFrame(
        {
            "piece_id": grid["GRD_ID"].astype(str),
            "population": grid["TOT_P"].astype(float),
            "piece_area": grid["piece_area"].astype(float),
            "geometry": grid.geometry,
        },
        crs=TARGET_CRS,
    )
    return PopulationSurface(pieces=pieces, label=f"{GRID_VINTAGE}_1km_cells")


def uniform_density_surface(
    municipalities: gpd.GeoDataFrame,
    audited_pop: dict[str, float],
) -> tuple[PopulationSurface, list[str]]:
    """Municipal polygons carrying audited population, uniform within each.

    Returns the surface and the sorted ids of municipalities without an
    audited population (excluded pieces — their share is reported as
    uncovered deficit by the membership rows).
    """
    pop = pd.Series(
        [audited_pop.get(str(mid)) for mid in municipalities["municipality_id"]],
        index=municipalities.index,
        dtype="Float64",
    )
    missing = sorted(municipalities.loc[pop.isna(), "municipality_id"].astype(str))
    kept = municipalities.loc[pop.notna()].copy()
    pieces = gpd.GeoDataFrame(
        {
            "piece_id": kept["municipality_id"].astype(str),
            "population": pop.dropna().astype(float).to_numpy(),
            "piece_area": kept.geometry.area.to_numpy(dtype=float),
            "geometry": kept.geometry,
        },
        crs=TARGET_CRS,
    )
    return PopulationSurface(pieces=pieces, label="uniform_municipal_density"), missing


# Geometry loaders


def load_geometry(
    path: str | Path,
    id_col: str,
    out_col: str,
    id_width: int,
) -> gpd.GeoDataFrame:
    """Load a geometry layer into ``TARGET_CRS`` with a zfilled string id."""
    gdf = gpd.read_file(path)
    if id_col not in gdf.columns:
        raise ValueError(f"Geometry missing {id_col!r} in {path}")
    gdf = gdf.rename(columns={id_col: out_col})
    gdf[out_col] = gdf[out_col].astype(str).str.strip().str.zfill(id_width)
    return (
        gdf.to_crs(TARGET_CRS).sort_values(out_col).reset_index(drop=True)
    )


def load_od_mitma_zone_ids(zarr_path: str | Path) -> list[str]:
    """The audited MOB-2 MITMA zone registry ids (456 for the real store)."""
    ds = xr.open_zarr(zarr_path)
    zones = sorted({str(z) for z in ds["origin"].values.tolist()})
    if not zones:
        raise ValueError(f"No zone ids in od registry {zarr_path}")
    return zones


def load_od_mitma_prefix_zone_ids(zarr_path: str | Path) -> list[str]:
    """The subset of registry zones carrying a Catalan INE province prefix."""
    ds = xr.open_zarr(zarr_path)
    return sorted(
        {
            str(zone)
            for zone, from_prefix in zip(
                ds["origin"].values.tolist(), ds["from_prefix"].values.tolist()
            )
            if bool(from_prefix)
        }
    )


def load_mitma_zones(
    shp_path: str | Path, zone_ids: Sequence[str]
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """MITMA zone polygons filtered to the registry ids.

    Returns the frame (``mitma_id`` column, ``TARGET_CRS``) and the sorted
    registry ids missing from the shapefile — expected empty; the caller
    decides whether that is fatal.
    """
    gdf = gpd.read_file(shp_path)
    if "ID" not in gdf.columns:
        raise ValueError(f"MITMA geometry missing 'ID' in {shp_path}")
    wanted = list(dict.fromkeys(str(z) for z in zone_ids))
    gdf = gdf.rename(columns={"ID": "mitma_id"})
    gdf["mitma_id"] = gdf["mitma_id"].astype(str).str.strip()
    gdf = gdf[gdf["mitma_id"].isin(wanted)]
    missing = sorted(set(wanted) - set(gdf["mitma_id"]))
    return gdf.to_crs(TARGET_CRS).sort_values("mitma_id").reset_index(drop=True), missing


# Edge-table assembly


def _fraction_rows(
    pairs: pd.DataFrame,
    src_type: str,
    dst_type: str,
    pops: dict[tuple[str, str], float],
    coverages: dict[tuple[str, str], float],
    provenance: str,
) -> list[dict]:
    """Directed edge rows with both endpoint fractions (0.0 when pop 0)."""
    src_pop = pairs["src_id"].map(lambda i: pops.get((src_type, i), 0.0))
    dst_pop = pairs["dst_id"].map(lambda i: pops.get((dst_type, i), 0.0))
    out = pd.DataFrame(
        {
            "src_type": src_type,
            "src_id": pairs["src_id"].to_numpy(),
            "dst_type": dst_type,
            "dst_id": pairs["dst_id"].to_numpy(),
            "overlap_area_m2": pairs["overlap_area_m2"].astype(float).to_numpy(),
            "overlap_pop_est": pairs["overlap_pop_est"].astype(float).to_numpy(),
            "frac_of_src_pop": (pairs["overlap_pop_est"] / src_pop).where(src_pop > 0, 0.0),
            "frac_of_dst_pop": (pairs["overlap_pop_est"] / dst_pop).where(dst_pop > 0, 0.0),
            "coverage_src": pairs["src_id"].map(lambda i: coverages.get((src_type, i), 0.0)),
            "coverage_dst": pairs["dst_id"].map(lambda i: coverages.get((dst_type, i), 0.0)),
            "provenance": provenance,
        }
    )
    return out.to_dict("records")


def _directed_edges(
    pairs: pd.DataFrame,
    type_a: str,
    type_b: str,
    pops: dict[tuple[str, str], float],
    coverages: dict[tuple[str, str], float],
    prov: str,
) -> list[dict]:
    """Both directions of one unordered pair table (reciprocity per §3)."""
    forward = _fraction_rows(
        pairs.rename(columns={"a_id": "src_id", "b_id": "dst_id"}),
        type_a, type_b, pops, coverages, prov,
    )
    backward = _fraction_rows(
        pairs.rename(columns={"b_id": "src_id", "a_id": "dst_id"}),
        type_b, type_a, pops, coverages, prov,
    )
    return forward + backward


def assemble_edge_tables(
    surface: PopulationSurface,
    node_frames: dict[str, gpd.GeoDataFrame],
    membership: pd.DataFrame | None,
    provenance: str,
    include_abs_comarca: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build the directed overlap edge table and QC report.

    ``node_frames`` maps source type → frame with its ``<type>_id`` id
    column. ``membership`` (columns ``municipality_id``, ``comarca_id``)
    yields the exact muni ↔ comarca rows: overlap area is the municipal
    polygon area and the municipal endpoint fraction is exactly 1.0.
    """
    id_cols = {t: f"{t}_id" for t in node_frames}
    frac_cache: dict[str, pd.DataFrame] = {}
    stats = {
        t: node_stats(surface, frame, id_cols[t], frac_cache)
        for t, frame in node_frames.items()
    }
    pops = {
        (t, str(idx)): float(row.pop_est)
        for t, frame in stats.items()
        for idx, row in frame.iterrows()
    }
    coverages = {
        (t, str(idx)): float(row.coverage)
        for t, frame in stats.items()
        for idx, row in frame.iterrows()
    }

    # Comarca population = sum over mapped member municipalities (REG-1
    # convention); makes membership destination fractions sum to 1 up to
    # the unmapped/zero-population member deficit.
    unmapped_members: list[str] = []
    if membership is not None and not membership.empty and "comarca" in node_frames:
        mem = membership.copy()
        mem["municipality_id"] = mem["municipality_id"].astype(str)
        mem["comarca_id"] = mem["comarca_id"].astype(str)
        mem = mem[mem["comarca_id"].isin(stats["comarca"].index)]
        known = mem["municipality_id"].isin(stats["municipality"].index)
        unmapped_members = sorted(mem.loc[~known, "municipality_id"].unique())
        mapped = mem[known]
        member_pop = mapped.assign(
            _pop=mapped["municipality_id"].map(
                {k[1]: v for k, v in pops.items() if k[0] == "municipality"}
            )
        )
        comarca_pop = member_pop.groupby("comarca_id")["_pop"].sum()
        for cid, total in comarca_pop.items():
            pops[("comarca", cid)] = float(total)
    else:
        mapped = None

    edges: list[dict] = []
    relation_stats: dict[str, dict] = {}

    def _pair_table(t_a: str, t_b: str) -> pd.DataFrame:
        area = polygon_overlap_area(
            node_frames[t_a], id_cols[t_a], node_frames[t_b], id_cols[t_b]
        )
        pop = overlap_pop_table(
            surface, frac_cache[id_cols[t_a]], frac_cache[id_cols[t_b]]
        )
        pairs = area.merge(pop, on=["a_id", "b_id"], how="outer")
        pairs["overlap_area_m2"] = pairs["overlap_area_m2"].fillna(0.0)
        pairs["overlap_pop_est"] = pairs["overlap_pop_est"].fillna(0.0)
        # Keep pairs whose polygons merely touch (area 0) but whose shared
        # grid cells still split population between both endpoints.
        keep = (pairs["overlap_area_m2"] > 0) | (pairs["overlap_pop_est"] > 0)
        return pairs[keep]

    intersection_relations = [
        ("municipality", "abs"),
        ("mitma", "municipality"),
        ("mitma", "abs"),
        ("mitma", "comarca"),
    ]
    if include_abs_comarca:
        intersection_relations.append(("abs", "comarca"))
    for t_a, t_b in intersection_relations:
        if t_a not in node_frames or t_b not in node_frames:
            continue
        pairs = _pair_table(t_a, t_b)
        relation_stats[f"{t_a}-{t_b}"] = {
            "pairs": int(len(pairs)),
            "total_overlap_area_m2": float(pairs["overlap_area_m2"].sum()),
            "total_overlap_pop_est": float(pairs["overlap_pop_est"].sum()),
        }
        edges.extend(
            _directed_edges(pairs, t_a, t_b, pops, coverages, provenance)
        )

    if mapped is not None and not mapped.empty:
        muni_area = stats["municipality"]["area_m2"]
        muni_pop = mapped["municipality_id"].map(
            {k[1]: v for k, v in pops.items() if k[0] == "municipality"}
        )
        pairs = pd.DataFrame(
            {
                "a_id": mapped["comarca_id"],
                "b_id": mapped["municipality_id"],
                "overlap_area_m2": mapped["municipality_id"].map(muni_area),
                "overlap_pop_est": muni_pop,
            }
        )
        relation_stats["municipality-comarca"] = {
            "pairs": int(len(pairs)),
            "total_overlap_area_m2": float(pairs["overlap_area_m2"].sum()),
            "total_overlap_pop_est": float(pairs["overlap_pop_est"].sum()),
            "unmapped_member_ids": unmapped_members,
        }
        edges.extend(
            _directed_edges(pairs, "comarca", "municipality", pops, coverages, provenance)
        )

    table = pd.DataFrame(edges, columns=list(EDGE_COLUMNS)).sort_values(
        ["src_type", "src_id", "dst_type", "dst_id"], kind="stable"
    ).reset_index(drop=True)

    report = _qc_report(
        table, stats, node_frames, relation_stats, surface, include_abs_comarca
    )
    return table, report


def _qc_report(
    table: pd.DataFrame,
    stats: dict[str, pd.DataFrame],
    node_frames: dict[str, gpd.GeoDataFrame],
    relation_stats: dict[str, dict],
    surface: PopulationSurface,
    include_abs_comarca: bool,
) -> dict:
    """Coverage, deficit and zero-overlap QC numbers."""
    coverage = {
        t: {
            "mean": float(f["coverage"].mean()),
            "min": float(f["coverage"].min()),
            "nodes": int(len(f)),
        }
        for t, f in stats.items()
    }
    zero_pop = {
        t: sorted(f.index[f["pop_est"] <= 0].tolist()) for t, f in stats.items()
    }
    zero_overlap = {}
    for t, frame in node_frames.items():
        seen = set(table.loc[table["src_type"] == t, "src_id"]) | set(
            table.loc[table["dst_type"] == t, "dst_id"]
        )
        zero_overlap[t] = sorted(set(frame[f"{t}_id"].astype(str)) - seen)

    deficit: dict[str, dict] = {}
    for relation in relation_stats:
        t_a, t_b = relation.split("-", 1)
        sel = table[(table["src_type"] == t_a) & (table["dst_type"] == t_b)]
        sums = sel.groupby("dst_id")["frac_of_dst_pop"].sum()
        gaps = (1.0 - sums).clip(lower=0.0)
        deficit[relation] = {
            "dst_type": t_b,
            "nodes": int(len(sums)),
            "max_deficit": float(gaps.max()) if len(gaps) else 0.0,
            "nodes_with_deficit": int((gaps > 1e-9).sum()),
        }

    return {
        "policy": surface.label,
        "rows_total": int(len(table)),
        "rows_per_relation": relation_stats,
        "surface_pieces": int(len(surface.pieces)),
        "surface_population_total": float(surface.pieces["population"].sum()),
        "coverage": coverage,
        "zero_pop_nodes": zero_pop,
        "zero_overlap_nodes": zero_overlap,
        "destination_deficit": deficit,
        "abs_comarca_included": include_abs_comarca,
    }


def _provenance_string(policy: str) -> str:
    return (
        "overlap.v1"
        f"|policy={policy}"
        f"|grid={GRID_VINTAGE}"
        "|muni=fl_municipios_catalonia.geojson"
        "|abs=ABS.shp"
        "|comarca=comarques_igcg_v2r1.geojson(GEO-1)"
        "|mitma=distritos_mitma.shp"
        "|membership=mpiscatalunya.csv"
    )


def build(
    policy: str = "geostat_grid",
    out_path: str | Path | None = DEFAULT_OUT,
    report_path: str | Path | None = DEFAULT_REPORT,
    dasymetric_root: str | Path = DASYMETRIC_ROOT,
    include_abs_comarca: bool = False,
    write: bool = True,
) -> dict:
    """Run the full overlap build; returns the QC report."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    root = Path(dasymetric_root)

    municipalities = load_geometry(MUNICIPAL_GEOJSON, "id", "municipality_id", 5)
    abs_gdf = load_geometry(ABS_SHP, "CODABSa", "abs_id", 3)
    comarcas = load_geometry(COMARCA_GEOJSON, "comarca_id", "comarca_id", 2)
    zone_ids = load_od_mitma_zone_ids(OD_MITMA_ZARR)
    zones, missing_zones = load_mitma_zones(root / MITMA_SHP_REL, zone_ids)
    if missing_zones:
        raise ValueError(
            f"{len(missing_zones)} od_mitma registry ids missing from MITMA "
            f"geometry: {missing_zones[:10]}..."
        )
    mapping = load_municipality_comarca_mapping(MAPPING_CSV)

    if policy == "geostat_grid":
        surface = load_geostat_surface(
            root / GRID_GEOM_REL,
            root / GEOSTAT_CSV_REL,
            study_bounds=tuple(municipalities.total_bounds),
        )
    else:
        registry, _ = build_municipal_registry(FL_POPULATION_CSV, MUNICIPAL_REFERENCE_CSV)
        audited = {
            r.source_id: float(r.population)
            for r in registry.records_for("municipality")
            if r.population is not None
        }
        surface, _ = uniform_density_surface(municipalities, audited)

    provenance = _provenance_string(policy)
    table, report = assemble_edge_tables(
        surface,
        {
            "municipality": municipalities,
            "abs": abs_gdf,
            "comarca": comarcas,
            "mitma": zones,
        },
        membership=mapping,
        provenance=provenance,
        include_abs_comarca=include_abs_comarca,
    )

    hashes = {
        str(p): _sha256_file(p)
        for p in (
            MUNICIPAL_GEOJSON,
            ABS_SHP,
            COMARCA_GEOJSON,
            MAPPING_CSV,
            root / MITMA_SHP_REL,
            root / GEOSTAT_CSV_REL,
            root / GRID_GEOM_REL,
        )
    }

    # MITMA zones without nonzero municipal population overlap are a
    # legitimate data reality (border zones whose Catalan slivers fall in
    # uninhabited cells) — recorded, never forced (main-agent ruling).
    mitma_rows = table[
        (
            (table["src_type"] == "mitma") & (table["dst_type"] == "municipality")
        ) | (
            (table["src_type"] == "municipality") & (table["dst_type"] == "mitma")
        )
    ]
    covered_zones = set(
        mitma_rows.loc[mitma_rows["overlap_pop_est"] > 0, "src_id"]
    ) | set(mitma_rows.loc[mitma_rows["overlap_pop_est"] > 0, "dst_id"])
    prefix_ids = set(load_od_mitma_prefix_zone_ids(OD_MITMA_ZARR))
    zero_overlap_zones = sorted(set(zone_ids) - covered_zones)
    report["mitma_municipal_overlap"] = {
        "zone_count": len(zone_ids),
        "prefix_zone_count": len(prefix_ids),
        "covered_nonzero": len(covered_zones & set(zone_ids)),
        "zero_overlap_zone_ids": zero_overlap_zones,
        "prefix_zero_overlap_zone_ids": sorted(prefix_ids & set(zero_overlap_zones)),
    }

    report = {
        "provenance": {
            "target_crs": TARGET_CRS,
            "grid_vintage": GRID_VINTAGE,
            "policy": policy,
            "od_mitma_zone_count": len(zone_ids),
            "input_sha256": hashes,
            "provenance_string": provenance,
        },
        **report,
    }

    if write:
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(out_path, index=False)
        if report_path is not None:
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the native overlap edge table (slice OVL-1)."
    )
    parser.add_argument(
        "--policy",
        default="geostat_grid",
        choices=list(POLICIES),
        help="population weighting policy (plan §11.7)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output parquet path")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="QC report path")
    parser.add_argument(
        "--source-root",
        default=str(DASYMETRIC_ROOT),
        help="read-only dasymetric_mob checkout (grid + MITMA geometry)",
    )
    parser.add_argument(
        "--include-abs-comarca",
        action="store_true",
        help="also emit the off-by-default ABS↔comarca relation",
    )
    args = parser.parse_args(argv)
    report = build(
        policy=args.policy,
        out_path=args.out,
        report_path=args.report,
        dasymetric_root=args.source_root,
        include_abs_comarca=args.include_abs_comarca,
    )
    print(
        f"Wrote {args.out}: {report['rows_total']} directed rows over "
        f"{len(report['rows_per_relation'])} relations (policy={args.policy})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
