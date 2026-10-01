"""Assemble the heterogeneous native graph store (slice GRF-1, plan §11.3/§12.2).

Builds ``data/processed/native_support/native_graph.zarr`` (zarr v3 via
``xarray.to_zarr``):

- root group: node table on the ``node`` dim with string coord ``node_uid``
  (``f"{node_type}:{source_id}"``), variables ``node_type``, ``source_id``,
  ``population`` (float32, NaN where no sound denominator exists) and
  ``status`` (``observed``/``unobserved``/``unmapped``), ABS-only
  ``overlap_population_est`` (the GEOSTAT weighting quantity), plus ``od_zone_ref``
  (dim ``od_zone``): the ``od_mitma`` zone ids in OD zone coord order,
  mapping ``"mitma:<id>"`` node uids into OD origin/target indices.
- ``edges/<relation>`` group per relation: flat long layout on dim ``edge``
  with ``src_idx``/``dst_idx`` (int32 indices into ``node``), ``weight``
  (``frac_of_dst_pop``) and ``overlap_pop_est`` (float32). Both directions
  of every OVL-1 row are kept, per-row direction preserved, rows ordered by
  ``(src_idx, dst_idx)``.

Node set: the union of overlap-edge endpoints and the ``od_mitma`` zone
registry. Statuses are REG-1 registry passthrough; overlap-side ids missing
from a registry are ``unmapped``; registry codes with no graph presence
surface as validation warnings, not silent drops. ABS node ``population``
is the audited municipal denominator allocated with municipality-normalized
overlap fractions (review ruling R3). ``overlap_population_est`` retains the
separate GEOSTAT graph-weighting quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from data.native_registry import (
    SOURCE_ABS,
    SOURCE_EDAR,
    SOURCE_MUNICIPALITY,
    STATUS_OBSERVED,
    STATUS_UNMAPPED,
    SourceRegistry,
    build_abs_registry,
    build_comarca_registry,
    build_edar_registry,
    build_municipal_registry,
)

SCHEMA = "native_graph.v2"
SOURCE_MITMA = "mitma"

#: Every od_mitma zone carries 451 in-window mobility days, so all mitma
#: context nodes are ``observed`` in the node-table vocabulary.
MITMA_NODE_STATUS = STATUS_OBSERVED

#: Columns of ``overlap_edges.parquet`` consumed by this builder.
EDGE_COLUMNS = (
    "src_type",
    "src_id",
    "dst_type",
    "dst_id",
    "overlap_pop_est",
    "frac_of_src_pop",
    "frac_of_dst_pop",
)
RELATION_VARS = ("src_idx", "dst_idx", "weight", "overlap_pop_est")

#: relation -> directed (src_type, dst_type) pairs it collects; both
#: directions of an OVL-1 pair live in one relation group.
RELATION_PAIRS: dict[str, frozenset[tuple[str, str]]] = {
    "muni_abs": frozenset({("municipality", "abs"), ("abs", "municipality")}),
    "muni_comarca": frozenset(
        {("municipality", "comarca"), ("comarca", "municipality")}
    ),
    "mitma_muni": frozenset({("mitma", "municipality"), ("municipality", "mitma")}),
    "mitma_abs": frozenset({("mitma", "abs"), ("abs", "mitma")}),
    "mitma_comarca": frozenset({("mitma", "comarca"), ("comarca", "mitma")}),
}

EDAR_RELATION_PAIRS: dict[str, frozenset[tuple[str, str]]] = {
    "muni_edar_catchment": frozenset(
        {("municipality", "edar"), ("edar", "municipality")}
    ),
    "edar_abs_catchment": frozenset({("edar", "abs"), ("abs", "edar")}),
    "edar_comarca_catchment": frozenset(
        {("edar", "comarca"), ("comarca", "edar")}
    ),
    "edar_located_in_municipality": frozenset(
        {("edar", "municipality"), ("municipality", "edar")}
    ),
    "edar_located_in_abs": frozenset({("edar", "abs"), ("abs", "edar")}),
    "edar_located_in_comarca": frozenset(
        {("edar", "comarca"), ("comarca", "edar")}
    ),
}

WEIGHT_BOUND_TOL = 1e-6  # float64 fracs equal to 1.0 up to 1 ulp
PER_DST_SUM_TOL = 1e-3  # overlapping mitma _AM zones may exceed 1 in sum
ABS_POPULATION_MAX = 2e6

OVERLAP_PARQUET = Path("data/processed/native_support/overlap_edges.parquet")
OD_MITMA_ZARR = Path("data/processed/native_support/od_mitma.zarr")
DEFAULT_OUTPUT = Path("data/processed/native_support/native_graph.zarr")
DEFAULT_ABS_CSV = Path("data/files/COVID-19__Persones_hospitalitzades.csv")
DEFAULT_DEATHS_CSV = Path(
    "data/files/Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
)
DEFAULT_MPIS_CSV = Path("data/files/mpiscatalunya.csv")
DEFAULT_POPULATION_CSV = Path("data/files/fl_population_por_municipis.csv")
DEFAULT_REFERENCE_CSV = Path("data/files/municipality_id_reference.csv")
DEFAULT_ABS_SHP = Path("data/files/geo/cartografia/ABS.shp")
DEFAULT_WASTEWATER_CSV = Path("data/files/wastewater_biomarkers_icra.csv")
DEFAULT_EDAR_CONTRIBUTION = Path("data/files/edar_muni_edges.nc")
DEFAULT_EDAR_GEOJSON = Path("data/files/geo/EDAR.geojson")
DEFAULT_EDAR_CATCHMENT = Path(
    "data/processed/native_support/edar_catchment_edges.parquet"
)
DEFAULT_EDAR_POINTS = Path(
    "data/processed/native_support/edar_point_locations.parquet"
)

#: od_mitma attrs copied into the provenance summary.
OD_ATTR_SUMMARY_KEYS = (
    "schema",
    "zone_set_derivation",
    "zone_count",
    "zone_count_shp_total",
    "zone_prefix_count",
    "zone_population_available",
    "n_dates_processed",
    "diag_mass_fraction",
    "mass_in_zone_set_fraction",
    "strata_summation_max_rel_err",
)

BUILD_POLICY: dict[str, str] = {
    "node_set": "overlap-edge endpoints + od_mitma zone registry; extra registry codes warn",
    "node_order": "node_uid lexicographic",
    "status": "registry passthrough; overlap-only ids 'unmapped'; mitma 'observed'",
    "population": "registry values; mitma: od zone_population; abs: audited muni population allocated by frac_of_src_pop",
    "overlap_population_est": "ABS-only sum of municipality->abs overlap_pop_est; graph weighting/QC, never the clinical denominator",
    "edges": "both directions per relation; weight=frac_of_dst_pop; (src_idx, dst_idx) order",
    "od_zone_ref": "od_mitma zone_id in OD coord order; joins mitma nodes to OD indices",
}


@dataclass(frozen=True)
class GraphValidationReport:
    """Counts and warnings from the frozen GRF-1 validation pass."""

    node_counts: dict[str, int]
    status_counts: dict[str, dict[str, int]]
    edge_counts: dict[str, int]
    abs_population: dict[str, float]
    warnings: tuple[str, ...]

    def to_attr(self) -> dict:
        """JSON-serializable form for zarr attrs and build summaries."""
        return {
            "node_counts": dict(self.node_counts),
            "status_counts": {k: dict(v) for k, v in self.status_counts.items()},
            "edge_counts": dict(self.edge_counts),
            "abs_population": dict(self.abs_population),
            "warnings": list(self.warnings),
        }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_relations(overlap: pd.DataFrame) -> pd.DataFrame:
    """Map every directed overlap row to its relation group.

    Raises ``ValueError`` on rows outside the five frozen relation pairs
    (e.g. abs<->comarca rows, excluded from OVL-1) and on duplicate
    directed pairs within a relation.
    """
    lookup = {
        (src, dst): name for name, pairs in RELATION_PAIRS.items() for src, dst in pairs
    }
    keys = list(zip(overlap["src_type"], overlap["dst_type"]))
    unknown = sorted({key for key in keys if key not in lookup})
    if unknown:
        raise ValueError(f"overlap rows outside the five relation pairs: {unknown}")
    out = overlap.copy()
    out["relation"] = [lookup[key] for key in keys]
    if out.duplicated(
        subset=["relation", "src_type", "src_id", "dst_type", "dst_id"]
    ).any():
        raise ValueError("duplicate directed overlap rows within a relation")
    return out


def build_node_table(
    registry: SourceRegistry,
    overlap: pd.DataFrame,
    od_zone_ids: Sequence[str],
    od_zone_populations: Sequence[float],
    edar_catchment: pd.DataFrame | None = None,
    edar_points: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Deterministic node frame (§11.3): one row per node_uid, uid-sorted.

    Nodes are the union of overlap-edge endpoints and the od zone registry
    (mitma). Population: registry values for municipality/comarca, od
    ``zone_population`` for mitma, and audited municipal population allocated
    to ABS with each municipality's ``frac_of_src_pop`` (review ruling R3).
    The raw GEOSTAT overlap sum is kept separately as
    ``overlap_population_est``.
    """
    rel = assign_relations(overlap)
    endpoints: dict[str, set[str]] = {}
    for type_col, id_col in (("src_type", "src_id"), ("dst_type", "dst_id")):
        for node_type, source_id in zip(rel[type_col], rel[id_col]):
            endpoints.setdefault(node_type, set()).add(str(source_id))
    od_ids = [str(zone) for zone in od_zone_ids]
    if len(od_ids) != len(set(od_ids)):
        raise ValueError("duplicate zone ids in the od_mitma registry")
    endpoints.setdefault(SOURCE_MITMA, set()).update(od_ids)
    for edar_table in (edar_catchment, edar_points):
        if edar_table is None:
            continue
        endpoints.setdefault(SOURCE_EDAR, set()).update(
            edar_table["edar_id"].astype(str)
        )
        for target_type, group in edar_table.groupby("target_type"):
            endpoints.setdefault(str(target_type), set()).update(
                group["target_id"].astype(str)
            )

    rows = sorted(
        (f"{node_type}:{sid}", node_type, sid)
        for node_type, ids in endpoints.items()
        for sid in ids
    )
    uids = [row[0] for row in rows]
    types = [row[1] for row in rows]
    ids = [row[2] for row in rows]

    muni_abs = rel[
        (rel["src_type"] == SOURCE_MUNICIPALITY)
        & (rel["dst_type"] == SOURCE_ABS)
    ].copy()
    frac_src = muni_abs["frac_of_src_pop"].to_numpy(dtype=np.float64)
    bad_frac = ~np.isfinite(frac_src) | (frac_src < 0) | (
        frac_src > 1 + WEIGHT_BOUND_TOL
    )
    if bad_frac.any():
        raise ValueError(
            "municipality->abs frac_of_src_pop outside [0, 1]: "
            f"{frac_src[bad_frac][:5].tolist()}"
        )
    abs_overlap_pop = muni_abs.groupby("dst_id")["overlap_pop_est"].sum()
    municipal_population = {
        rec.source_id: float(rec.population)
        for rec in registry.records_for(SOURCE_MUNICIPALITY)
        if rec.population is not None
    }
    muni_abs["_audited_municipal_population"] = muni_abs["src_id"].map(
        municipal_population
    )
    incomplete_abs = set(
        muni_abs.loc[
            muni_abs["_audited_municipal_population"].isna(), "dst_id"
        ].astype(str)
    )
    muni_abs["_allocated_audited_population"] = (
        muni_abs["_audited_municipal_population"] * muni_abs["frac_of_src_pop"]
    )
    abs_pop = muni_abs.groupby("dst_id")["_allocated_audited_population"].sum(
        min_count=1
    )
    for abs_id in incomplete_abs:
        abs_pop.loc[abs_id] = np.nan
    od_pop = {zone: float(pop) for zone, pop in zip(od_ids, od_zone_populations)}
    edar_pop: dict[str, float] = {}
    if edar_catchment is not None:
        edar_pop = {
            str(edar_id): float(values.iloc[0])
            for edar_id, values in edar_catchment.groupby("edar_id")[
                "catchment_population_est"
            ]
        }

    populations = np.full(len(uids), np.nan, dtype=np.float32)
    overlap_populations = np.full(len(uids), np.nan, dtype=np.float32)
    statuses: list[str] = []
    for k, (node_type, sid) in enumerate(zip(types, ids)):
        record = registry.get(node_type, sid)
        statuses.append(
            MITMA_NODE_STATUS
            if node_type == SOURCE_MITMA
            else STATUS_UNMAPPED
            if record is None
            else record.status
        )
        if node_type == SOURCE_ABS:
            if sid in abs_pop.index:
                populations[k] = np.float32(abs_pop[sid])
            if sid in abs_overlap_pop.index:
                overlap_populations[k] = np.float32(abs_overlap_pop[sid])
        elif node_type == SOURCE_MITMA:
            populations[k] = np.float32(od_pop[sid])
        elif node_type == SOURCE_EDAR and sid in edar_pop:
            populations[k] = np.float32(edar_pop[sid])
        elif record is not None and record.population is not None:
            populations[k] = np.float32(record.population)
    return pd.DataFrame(
        {
            "node_uid": uids,
            "node_type": types,
            "source_id": ids,
            "population": populations,
            "overlap_population_est": overlap_populations,
            "status": statuses,
        }
    )


def _relation_dataset(
    name: str, sub: pd.DataFrame, uid_index: dict[str, int]
) -> xr.Dataset:
    """Flat long edge dataset for one relation, ordered by (src_idx, dst_idx)."""
    src_idx = np.array(
        [uid_index[f"{t}:{i}"] for t, i in zip(sub["src_type"], sub["src_id"])],
        dtype=np.int32,
    )
    dst_idx = np.array(
        [uid_index[f"{t}:{i}"] for t, i in zip(sub["dst_type"], sub["dst_id"])],
        dtype=np.int32,
    )
    order = np.lexsort((dst_idx, src_idx))
    relation_pairs = RELATION_PAIRS | EDAR_RELATION_PAIRS
    return xr.Dataset(
        {
            "src_idx": ("edge", src_idx[order]),
            "dst_idx": ("edge", dst_idx[order]),
            "weight": (
                ("edge",),
                sub["frac_of_dst_pop"]
                .to_numpy(dtype=np.float64)[order]
                .astype(np.float32),
            ),
            "overlap_pop_est": (
                ("edge",),
                sub["overlap_pop_est"]
                .to_numpy(dtype=np.float64)[order]
                .astype(np.float32),
            ),
        },
        attrs={
            "relation": name,
            "directed_pairs": "|".join(
                f"{s}->{d}" for s, d in sorted(relation_pairs[name])
            ),
            "weight_definition": "frac_of_dst_pop",
            "row_order": "(src_idx, dst_idx)",
        },
    )


def _edar_relation_frames(
    catchment: pd.DataFrame | None,
    points: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Convert catchment and point tables to bidirectional graph rows."""
    columns = (
        "src_type",
        "src_id",
        "dst_type",
        "dst_id",
        "overlap_pop_est",
        "frac_of_dst_pop",
    )
    rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in EDAR_RELATION_PAIRS
    }
    if catchment is not None:
        for record in catchment.itertuples(index=False):
            base = {
                "overlap_pop_est": float(record.served_population_est),
            }
            rows[str(record.relation)].extend(
                [
                    base
                    | {
                        "src_type": str(record.target_type),
                        "src_id": str(record.target_id),
                        "dst_type": SOURCE_EDAR,
                        "dst_id": str(record.edar_id),
                        "frac_of_dst_pop": float(record.frac_of_edar_catchment),
                    },
                    base
                    | {
                        "src_type": SOURCE_EDAR,
                        "src_id": str(record.edar_id),
                        "dst_type": str(record.target_type),
                        "dst_id": str(record.target_id),
                        "frac_of_dst_pop": float(
                            record.frac_of_target_population_served
                        ),
                    },
                ]
            )
    if points is not None:
        for record in points.itertuples(index=False):
            for src_type, src_id, dst_type, dst_id in (
                (
                    SOURCE_EDAR,
                    str(record.edar_id),
                    str(record.target_type),
                    str(record.target_id),
                ),
                (
                    str(record.target_type),
                    str(record.target_id),
                    SOURCE_EDAR,
                    str(record.edar_id),
                ),
            ):
                rows[str(record.relation)].append(
                    {
                        "src_type": src_type,
                        "src_id": src_id,
                        "dst_type": dst_type,
                        "dst_id": dst_id,
                        "overlap_pop_est": np.nan,
                        "frac_of_dst_pop": 1.0,
                    }
                )
    return {
        name: pd.DataFrame(relation_rows, columns=columns)
        for name, relation_rows in rows.items()
    }


def assemble_native_graph(
    registry: SourceRegistry,
    overlap: pd.DataFrame,
    od_zone_ids: Sequence[str],
    od_zone_populations: Sequence[float],
    edar_catchment: pd.DataFrame | None = None,
    edar_points: pd.DataFrame | None = None,
) -> tuple[xr.Dataset, dict[str, xr.Dataset], GraphValidationReport]:
    """Assemble the §11.3 store in memory and validate it.

    Returns the root dataset (node table + ``od_zone_ref`` + structural
    attrs), one edge dataset per relation, and the validation report.
    """
    missing = [col for col in EDGE_COLUMNS if col not in overlap.columns]
    if missing:
        raise ValueError(f"overlap table missing columns: {missing}")
    rel = assign_relations(overlap)
    nodes = build_node_table(
        registry,
        rel,
        od_zone_ids,
        od_zone_populations,
        edar_catchment,
        edar_points,
    )
    uid_index = {uid: k for k, uid in enumerate(nodes["node_uid"])}

    root = xr.Dataset(
        {
            "node_type": ("node", nodes["node_type"].to_numpy(dtype=str)),
            "source_id": ("node", nodes["source_id"].to_numpy(dtype=str)),
            "population": ("node", nodes["population"].to_numpy()),
            "overlap_population_est": (
                "node",
                nodes["overlap_population_est"].to_numpy(),
            ),
            "status": ("node", nodes["status"].to_numpy(dtype=str)),
            "od_zone_ref": (("od_zone",), np.array([str(z) for z in od_zone_ids])),
        },
        coords={"node": nodes["node_uid"].to_numpy(dtype=str)},
        attrs={"schema": SCHEMA, "build_policy": dict(BUILD_POLICY)},
    )
    relations = {
        name: _relation_dataset(name, rel[rel["relation"] == name], uid_index)
        for name in sorted(RELATION_PAIRS)
    }
    if edar_catchment is not None or edar_points is not None:
        edar_frames = _edar_relation_frames(edar_catchment, edar_points)
        relations.update(
            {
                name: _relation_dataset(name, frame, uid_index)
                for name, frame in sorted(edar_frames.items())
            }
        )
    root.attrs["relation_names"] = sorted(relations)
    report = validate_native_graph(root, relations, registry, od_zone_ids)
    return root, relations, report


def validate_native_graph(
    root: xr.Dataset,
    relations: dict[str, xr.Dataset],
    registry: SourceRegistry,
    od_zone_ids: Sequence[str],
) -> GraphValidationReport:
    """Frozen GRF-1 validation pass (plan §12.2).

    Raises ``ValueError`` on: missing relation groups or variables,
    out-of-bounds edge indices, out-of-range weights, observed clinical
    nodes without an overlap edge (municipalities specifically without a
    membership row), mitma/od-registry misalignment, and implausible ABS
    populations. Non-fatal findings (per-destination fraction sums above 1
    from overlapping mitma zones, ABS nodes without an audited
    population, registry codes with no graph presence) become warnings.
    """
    warnings: list[str] = []
    required_root = {
        "node_type",
        "source_id",
        "population",
        "overlap_population_est",
        "status",
        "od_zone_ref",
    }
    missing_root = sorted(required_root - set(root.data_vars))
    if missing_root:
        raise ValueError(f"native graph root missing variables: {missing_root}")
    expected_relations = set(root.attrs.get("relation_names", RELATION_PAIRS))
    if set(relations) != expected_relations:
        raise ValueError(
            f"relation groups {sorted(relations)} != {sorted(expected_relations)}"
        )
    uids = np.asarray(root.coords["node"].values)
    types = np.asarray(root["node_type"].values)
    status = np.asarray(root["status"].values)
    populations = np.asarray(root["population"].values)
    n_nodes = uids.size

    for name, ds in sorted(relations.items()):
        missing = [var for var in RELATION_VARS if var not in ds]
        if missing:
            raise ValueError(f"relation {name} missing variables: {missing}")
        for var in ("src_idx", "dst_idx"):
            idx = np.asarray(ds[var].values)
            if idx.size and (idx.min() < 0 or idx.max() >= n_nodes):
                raise ValueError(
                    f"{name}/{var}: edge index out of bounds [0, {n_nodes})"
                )
        weights = np.asarray(ds["weight"].values)
        bad = ~np.isfinite(weights) | (weights < 0) | (weights > 1 + WEIGHT_BOUND_TOL)
        if bad.any():
            raise ValueError(
                f"{name}/weight outside [0, 1]: {weights[bad][:5].tolist()}"
            )
        dst_sums = np.bincount(
            np.asarray(ds["dst_idx"].values, dtype=np.int64),
            weights=weights.astype(np.float64),
            minlength=n_nodes,
        )
        over = (
            0
            if name.startswith("edar_located_in_")
            else int((dst_sums > 1 + PER_DST_SUM_TOL).sum())
        )
        if over:
            warnings.append(
                f"{name}: frac_of_dst_pop sums above 1+{PER_DST_SUM_TOL} for {over} "
                f"destination(s), max {dst_sums.max():.6f} (overlapping source geometries)"
            )

    touched = np.zeros(n_nodes, dtype=bool)
    membership_touched = np.zeros(n_nodes, dtype=bool)
    for name, ds in relations.items():
        for var in ("src_idx", "dst_idx"):
            touched[np.asarray(ds[var].values, dtype=np.int64)] = True
        if name == "muni_comarca":
            for var in ("src_idx", "dst_idx"):
                membership_touched[np.asarray(ds[var].values, dtype=np.int64)] = True

    observed = (status == STATUS_OBSERVED) & (types != SOURCE_MITMA)
    uncovered = observed & ~touched
    if uncovered.any():
        raise ValueError(
            f"observed clinical nodes without overlap edges: {uids[uncovered].tolist()}"
        )
    no_membership = observed & (types == SOURCE_MUNICIPALITY) & ~membership_touched
    if no_membership.any():
        raise ValueError(
            "observed municipalities without membership (muni_comarca) edges: "
            f"{uids[no_membership].tolist()}"
        )

    mitma_ids = {
        str(sid) for t, sid in zip(types, root["source_id"].values) if t == SOURCE_MITMA
    }
    od_ids = {str(zone) for zone in od_zone_ids}
    if mitma_ids != od_ids:
        raise ValueError(
            f"mitma nodes and od registry disagree: node-only "
            f"{sorted(mitma_ids - od_ids)}, od-only {sorted(od_ids - mitma_ids)}"
        )

    abs_pops = populations[types == SOURCE_ABS]
    finite = np.isfinite(abs_pops)
    if not finite.all():
        warnings.append(
            f"{int((~finite).sum())} ABS nodes have NaN population (no "
            "municipality->abs overlap rows)"
        )
    good = abs_pops[finite]
    if good.size and ((good <= 0).any() or (good > ABS_POPULATION_MAX).any()):
        bad_pops = good[(good <= 0) | (good > ABS_POPULATION_MAX)]
        raise ValueError(
            f"ABS audited population outside (0, {ABS_POPULATION_MAX}]: "
            f"{bad_pops[:5].tolist()}"
        )

    present = set(zip(types.tolist(), (str(s) for s in root["source_id"].values)))
    absent = [
        f"{rec.source_type}:{rec.source_id} (status {rec.status})"
        for rec in registry
        if (rec.source_type, rec.source_id) not in present
    ]
    if absent:
        warnings.append("registry codes with no graph presence: " + ", ".join(absent))

    node_counts = {str(t): int((types == t).sum()) for t in sorted(set(types.tolist()))}
    status_counts = {
        t: {
            str(s): int((status[types == t] == s).sum())
            for s in sorted(set(status[types == t].tolist()))
        }
        for t in node_counts
    }
    abs_stats: dict[str, float] = {}
    if good.size:
        abs_stats = {
            "min": float(good.min()),
            "median": float(np.median(good)),
            "max": float(good.max()),
        }
    return GraphValidationReport(
        node_counts=node_counts,
        status_counts=status_counts,
        edge_counts={n: int(ds.sizes["edge"]) for n, ds in sorted(relations.items())},
        abs_population=abs_stats,
        warnings=tuple(warnings),
    )


def assert_same_assembly(
    first: tuple[xr.Dataset, dict[str, xr.Dataset]],
    second: tuple[xr.Dataset, dict[str, xr.Dataset]],
) -> None:
    """Determinism guard: identical node order, coords, and edge indices."""
    root_a, rel_a = first
    root_b, rel_b = second
    if not np.array_equal(
        np.asarray(root_a.coords["node"].values),
        np.asarray(root_b.coords["node"].values),
    ):
        raise ValueError("node_uid ordering differs between assemblies")
    for var in ("node_type", "source_id", "population", "status", "od_zone_ref"):
        equal_nan = var == "population"  # NaN populations are designed absences
        if not np.array_equal(
            np.asarray(root_a[var].values),
            np.asarray(root_b[var].values),
            equal_nan=equal_nan,
        ):
            raise ValueError(f"root variable {var} differs between assemblies")
    for name, ds_a in rel_a.items():
        for var in RELATION_VARS:
            if not np.array_equal(
                np.asarray(ds_a[var].values),
                np.asarray(rel_b[name][var].values),
                equal_nan=var == "overlap_pop_est",
            ):
                raise ValueError(f"{name}/{var} differs between assemblies")


def write_native_graph(
    root: xr.Dataset, relations: dict[str, xr.Dataset], output: str | Path
) -> Path:
    """Write the store: root group, ``edges/<relation>`` groups, consolidated."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root.to_zarr(output, mode="w", zarr_format=3, consolidated=False)
    for name, ds in sorted(relations.items()):
        ds.to_zarr(
            output, mode="a", group=f"edges/{name}", zarr_format=3, consolidated=False
        )
    zarr.consolidate_metadata(output)
    return output


def open_native_graph(output: str | Path) -> tuple[xr.Dataset, dict[str, xr.Dataset]]:
    """Reopen a written store: (root, relations) with per-relation datasets."""
    root = xr.open_zarr(output)
    relation_names = root.attrs.get("relation_names", sorted(RELATION_PAIRS))
    relations = {
        name: xr.open_zarr(output, group=f"edges/{name}")
        for name in relation_names
    }
    return root, relations


def _load_registries(
    abs_csv: Path,
    deaths_csv: Path,
    mpis_csv: Path,
    population_csv: Path,
    reference_csv: Path,
    abs_geometry_shp: Path,
) -> SourceRegistry:
    """Build the merged REG-1 registry; ABS reconciled against geometry."""
    import geopandas  # heavy import needed only for the real build

    geometry_ids = set(
        geopandas.read_file(abs_geometry_shp, ignore_geometry=True)["CODABSa"]
    )
    registry = SourceRegistry()
    abs_registry, _ = build_abs_registry(abs_csv, geometry_ids=geometry_ids)
    comarca_registry, _ = build_comarca_registry(
        deaths_csv, mpis_csv, population_csv, reference_csv
    )
    municipal_registry, _ = build_municipal_registry(population_csv, reference_csv)
    registry.extend(abs_registry)
    registry.extend(comarca_registry)
    registry.extend(municipal_registry)
    return registry


def build(
    output: str | Path = DEFAULT_OUTPUT,
    overlap_parquet: str | Path = OVERLAP_PARQUET,
    od_mitma_zarr: str | Path = OD_MITMA_ZARR,
    abs_csv: str | Path = DEFAULT_ABS_CSV,
    deaths_csv: str | Path = DEFAULT_DEATHS_CSV,
    mpis_csv: str | Path = DEFAULT_MPIS_CSV,
    population_csv: str | Path = DEFAULT_POPULATION_CSV,
    reference_csv: str | Path = DEFAULT_REFERENCE_CSV,
    abs_geometry_shp: str | Path = DEFAULT_ABS_SHP,
    wastewater_csv: str | Path = DEFAULT_WASTEWATER_CSV,
    edar_contribution: str | Path = DEFAULT_EDAR_CONTRIBUTION,
    edar_geojson: str | Path = DEFAULT_EDAR_GEOJSON,
    edar_catchment_parquet: str | Path = DEFAULT_EDAR_CATCHMENT,
    edar_points_parquet: str | Path = DEFAULT_EDAR_POINTS,
) -> dict:
    """Assemble, validate, and write ``native_graph.zarr``; return the report.

    Determinism is enforced by assembling twice (identical node order,
    coords, and edge indices) and by re-validating the reopened store
    against the in-memory validation report.
    """
    registry = _load_registries(
        Path(abs_csv),
        Path(deaths_csv),
        Path(mpis_csv),
        Path(population_csv),
        Path(reference_csv),
        Path(abs_geometry_shp),
    )
    edar_registry, _ = build_edar_registry(
        wastewater_csv, edar_contribution, edar_geojson
    )
    registry.extend(edar_registry)
    overlap = pd.read_parquet(overlap_parquet)
    edar_catchment = pd.read_parquet(edar_catchment_parquet)
    edar_points = pd.read_parquet(edar_points_parquet)
    od = xr.open_zarr(od_mitma_zarr)
    zone_ids = [str(z) for z in od["zone_id"].values]
    zone_populations = np.asarray(od["zone_population"].values, dtype=np.float32)

    assembled = assemble_native_graph(
        registry,
        overlap,
        zone_ids,
        zone_populations,
        edar_catchment,
        edar_points,
    )
    root, relations, report = assembled
    assert_same_assembly(
        assembled[:2],
        assemble_native_graph(
            registry,
            overlap,
            zone_ids,
            zone_populations,
            edar_catchment,
            edar_points,
        )[:2],
    )

    root.attrs["provenance"] = {
        "overlap_edges": {
            "name": Path(overlap_parquet).name,
            "path": str(overlap_parquet),
            "sha256": _sha256_file(overlap_parquet),
            "rows": int(len(overlap)),
        },
        "registries": {
            "abs_csv": Path(abs_csv).name,
            "deaths_csv": Path(deaths_csv).name,
            "mpiscatalunya_csv": Path(mpis_csv).name,
            "fl_population_csv": Path(population_csv).name,
            "municipality_reference_csv": Path(reference_csv).name,
            "abs_geometry": Path(abs_geometry_shp).name,
        },
        "od_mitma": {
            "name": Path(od_mitma_zarr).name,
            "path": str(od_mitma_zarr),
            "attrs_summary": {
                k: od.attrs[k] for k in OD_ATTR_SUMMARY_KEYS if k in od.attrs
            }
            | {"n_dates_missing": len(od.attrs.get("dates_missing", []))},
        },
        "edar": {
            "wastewater_csv": Path(wastewater_csv).name,
            "contribution": Path(edar_contribution).name,
            "geojson": Path(edar_geojson).name,
            "catchment_edges": Path(edar_catchment_parquet).name,
            "point_locations": Path(edar_points_parquet).name,
        },
    }
    root.attrs["validation"] = report.to_attr()
    write_native_graph(root, relations, output)

    reopened_root, reopened_relations = open_native_graph(output)
    reopen_report = validate_native_graph(
        reopened_root, reopened_relations, registry, zone_ids
    )
    if reopen_report.to_attr() != report.to_attr():
        raise ValueError("reopened native_graph store fails its validation report")
    return report.to_attr() | {"output": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: build native_graph.zarr and print the validation summary."""
    parser = argparse.ArgumentParser(
        description="Assemble native_graph.zarr (slice GRF-1, plan section 11.3)."
    )
    flag_defaults = {
        "output": DEFAULT_OUTPUT,
        "overlap-parquet": OVERLAP_PARQUET,
        "od-mitma-zarr": OD_MITMA_ZARR,
        "abs-csv": DEFAULT_ABS_CSV,
        "deaths-csv": DEFAULT_DEATHS_CSV,
        "mpis-csv": DEFAULT_MPIS_CSV,
        "population-csv": DEFAULT_POPULATION_CSV,
        "reference-csv": DEFAULT_REFERENCE_CSV,
        "abs-geometry-shp": DEFAULT_ABS_SHP,
        "wastewater-csv": DEFAULT_WASTEWATER_CSV,
        "edar-contribution": DEFAULT_EDAR_CONTRIBUTION,
        "edar-geojson": DEFAULT_EDAR_GEOJSON,
        "edar-catchment-parquet": DEFAULT_EDAR_CATCHMENT,
        "edar-points-parquet": DEFAULT_EDAR_POINTS,
    }
    for flag, default in flag_defaults.items():
        parser.add_argument(f"--{flag}", type=Path, default=default)
    report = build(
        **{k.replace("-", "_"): v for k, v in vars(parser.parse_args(argv)).items()}
    )
    print(f"{SCHEMA} -> {report['output']}")
    print(f"nodes: {json.dumps(report['node_counts'], sort_keys=True)}")
    print(f"status: {json.dumps(report['status_counts'], sort_keys=True)}")
    print(f"edges: {json.dumps(report['edge_counts'], sort_keys=True)}")
    print(f"abs population: {json.dumps(report['abs_population'])}")
    print(f"warnings ({len(report['warnings'])}):")
    for warning in report["warnings"]:
        print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
