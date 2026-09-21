"""Native MITMA OD harmonization (slice MOB-2; plan §12.2, schema §11.3).

Harmonizes the read-only dasymetric_mob mitma_v1 daily stratified OD CSVs
(2020-02-14..2021-05-09, 451 files) into data/processed/native_support/
od_mitma.zarr: dense raw-trips float32 (date, origin, target) over the audited
Catalan MITMA zone set, plus zone registry and construction-audit attrs.
No log/threshold at rest; self-flows retained; policies recorded as attrs.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr

CATALAN_PROVINCE_PREFIXES = ("08", "17", "25", "43")
MITMA_DAILY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.daily_mobility\.v1\.recurrent\.csv\.gz$")
EXPECTED_FIRST_DATE = "2020-02-14"  # 451 consecutive daily files through:
EXPECTED_LAST_DATE = "2021-05-09"
MIN_FLOW_RAW_POLICIES = (0.0, 10.0)  # {retain,drop} self-flow × these thresholds
SELF_FLOW_POLICIES = ("retain", "drop")
REFERENCE_ZONE_COUNTS = {"prefix_verified": 442, "g1_diagnostics": 454, "transform_attr": 458}
STRATA_NOTE = (
    "trips summed over all stratification columns (activity, distance, "
    "residency, age, study_*, sex, income) per (origin, destination), "
    "matching the upstream mobility_processor aggregation"
)
DEFAULT_FOOTPRINT = Path(__file__).resolve().parents[3] / "data" / "files" / "geo" / "fl_municipios_catalonia.geojson"
DEFAULT_OUT = Path("data/processed/native_support/od_mitma.zarr")
DAILY_COLUMNS = ["date", "origin", "destination", "trips"]


@dataclass(frozen=True)
class ZoneSet:
    """Audited MITMA zone set: sorted IDs plus per-zone membership flags."""

    zone_ids: tuple[str, ...]  # sorted lexicographically → deterministic store
    from_prefix: tuple[bool, ...]  # ID[:2] in CATALAN_PROVINCE_PREFIXES
    in_footprint: tuple[bool, ...]  # geometry intersects municipal footprint
    n_shp_zones: int = 0  # total shp features (2850); 0 if unknown/injected


def zone_set_from_geodataframes(zones: gpd.GeoDataFrame, footprint: gpd.GeoDataFrame) -> ZoneSet:
    """Pure union rule: prefix match ∪ sjoin(intersects) vs footprint; sorted IDs.

    Both frames must already share a CRS (``derive_zone_set`` guarantees it).
    """
    ids = zones["ID"].astype(str).reset_index(drop=True)
    from_prefix = ids.str[:2].isin(CATALAN_PROVINCE_PREFIXES)
    prefix_ids = set(ids[from_prefix])
    footprint_ids: set[str] = set()
    if len(footprint) > 0:
        zone_gdf = gpd.GeoDataFrame({"ID": ids}, geometry=zones.geometry.values, crs=zones.crs)
        joined = gpd.sjoin(zone_gdf[["ID", "geometry"]], footprint[["geometry"]], predicate="intersects")
        footprint_ids = set(joined["ID"].astype(str))
    zone_ids = tuple(sorted(prefix_ids | footprint_ids))
    return ZoneSet(
        zone_ids=zone_ids,
        from_prefix=tuple(z in prefix_ids for z in zone_ids),
        in_footprint=tuple(z in footprint_ids for z in zone_ids),
        n_shp_zones=len(zones),
    )


def derive_zone_set(shp_path: Path | str, footprint_path: Path | str, crs: str = "EPSG:3042") -> ZoneSet:
    """I/O wrapper: read shp (ID→str) + footprint, both to_crs(crs).

    Takes paths so the real-shp path is unit-testable behind skipif; the pure
    union logic in ``zone_set_from_geodataframes`` is tested with in-memory
    polygons (no I/O).
    """
    zones = gpd.read_file(shp_path)
    zones["ID"] = zones["ID"].astype(str)
    footprint = gpd.read_file(footprint_path)
    return zone_set_from_geodataframes(zones.to_crs(crs), footprint.to_crs(crs))


def load_zone_populations(poblacion_path: Path | str, zone_ids: Sequence[str]) -> np.ndarray:
    """Pipe-separated poblacion.csv → float32 aligned to zone_ids; NaN if absent."""
    pop = pd.read_csv(
        poblacion_path,
        sep="|",
        usecols=["distrito", "poblacion"],
        dtype={"distrito": str, "poblacion": np.float64},
    ).drop_duplicates("distrito")
    mapping = dict(zip(pop["distrito"], pop["poblacion"], strict=True))
    return np.array([mapping.get(z, np.nan) for z in zone_ids], dtype=np.float32)


def find_daily_files(mitma_root: Path) -> list[tuple[str, Path]]:
    """Regex-glob *.daily_mobility.v1.recurrent.csv.gz, sorted by filename date."""
    out: list[tuple[str, Path]] = []
    for path in mitma_root.iterdir():
        match = MITMA_DAILY_RE.match(path.name)
        if match and path.is_file():
            out.append((match.group(1), path))
    out.sort(key=lambda item: item[0])
    return out


def resolve_source_layout(source_root: Path) -> dict[str, Path]:
    """Resolve {mitma_root, shp_path, poblacion_path} under a dasymetric_mob root.

    Accepts a root with or without the trailing ``data/`` level (i.e.
    ``<root>/data/mitma_v1`` or ``<root>/mitma_v1``).
    """
    data_level = source_root / "data"
    mitma_root = data_level / "mitma_v1" if (data_level / "mitma_v1").is_dir() else source_root / "mitma_v1"
    if not mitma_root.is_dir():
        raise FileNotFoundError(f"no mitma_v1 directory under {source_root}")
    v2_level = data_level if (data_level / "mitma_v2").is_dir() else source_root
    return {
        "mitma_root": mitma_root,
        "shp_path": mitma_root / "zonificacion-distritos" / "distritos_mitma.shp",
        "poblacion_path": v2_level / "mitma_v2" / "mitma_v2" / "zonificacion" / "poblacion.csv",
    }


def expected_dates(available: Sequence[str], start: str | None = None, end: str | None = None) -> list[str]:
    """Full daily range EXPECTED_FIRST_DATE..EXPECTED_LAST_DATE clipped to
    [start, end] (inclusive); the completeness audit is this list minus the
    ``available`` dates, which is what is returned (the store's
    ``dates_missing``). Expectations never depend on ``limit``.
    """
    expected = pd.date_range(EXPECTED_FIRST_DATE, EXPECTED_LAST_DATE, freq="D")
    if start is not None:
        expected = expected[expected >= start]
    if end is not None:
        expected = expected[expected <= end]
    expected_iso = [ts.strftime("%Y-%m-%d") for ts in expected]
    have = set(available)
    return [d for d in expected_iso if d not in have]


@dataclass(frozen=True)
class DailyMatrix:
    """One harmonized day: dense raw-trip matrix plus audit scalars."""

    matrix: np.ndarray  # float32 [Z, Z], zero-filled dense, RAW trips
    mass_total: float  # float64 sum of ALL raw rows (pre zone-set filter)
    mass_zone_set: float  # float64 sum after endpoint filter
    max_strata_rel_err: float  # max |float32 − float64 groupby sum| / groupby sum


def harmonize_daily_csv(path: Path, zone_index: dict[str, int]) -> DailyMatrix:
    """Sum strata of one daily MITMA CSV into a dense float32 [Z, Z] matrix.

    Reads only (date, origin, destination, trips); asserts the date column
    matches the filename date; keeps rows with BOTH endpoints in the zone set;
    groupby(origin, destination).trips.sum() in float64; scatters into a
    zero-filled float64 grid (absent pairs stay 0) with a rtol-1e-9 assertion
    against the groupby total; casts to float32. Never symmetrized.
    """
    match = MITMA_DAILY_RE.match(path.name)
    if match is None:
        raise ValueError(f"unrecognized daily mobility file name: {path.name}")
    file_date = match.group(1)
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=DAILY_COLUMNS,
        dtype={"origin": str, "destination": str, "trips": np.float64},
    )
    dates = df["date"].unique()
    if len(dates) > 1 or (len(dates) == 1 and dates[0] != file_date):
        raise ValueError(f"{path.name}: date column {dates!r} does not match filename date {file_date}")

    n_zones = len(zone_index)
    mass_total = float(df["trips"].sum())
    keys = set(zone_index)
    in_set = df["origin"].isin(keys) & df["destination"].isin(keys)
    sub = df[in_set]
    mass_zone_set = float(sub["trips"].sum())
    grouped = sub.groupby(["origin", "destination"])["trips"].sum()

    dense = np.zeros((n_zones, n_zones), dtype=np.float64)
    max_rel_err = 0.0
    if len(grouped) > 0:
        values = grouped.to_numpy()
        oi = grouped.index.get_level_values("origin").map(zone_index)
        di = grouped.index.get_level_values("destination").map(zone_index)
        dense[oi, di] = values
        assert abs(float(dense.sum()) - float(values.sum())) <= 1e-9 * abs(float(values.sum())), (
            f"{path.name}: float64 scatter total deviates from groupby total"
        )
        nonzero = values != 0.0
        if nonzero.any():
            as_float32 = dense[oi[nonzero], di[nonzero]].astype(np.float64)
            max_rel_err = float((np.abs(as_float32 - values[nonzero]) / values[nonzero]).max())
    return DailyMatrix(
        matrix=dense.astype(np.float32),
        mass_total=mass_total,
        mass_zone_set=mass_zone_set,
        max_strata_rel_err=max_rel_err,
    )


def _injected_zone_set(zone_ids: Sequence[str]) -> ZoneSet:
    """Wrap a test zone list: from_prefix from IDs, in_footprint=True."""
    ids = tuple(sorted(str(z) for z in zone_ids))
    return ZoneSet(
        zone_ids=ids,
        from_prefix=tuple(z[:2] in CATALAN_PROVINCE_PREFIXES for z in ids),
        in_footprint=tuple(True for _ in ids),
    )


def _policy_key(self_flow: str, min_flow: float) -> str:
    return f"{self_flow}_minflow{min_flow:g}"


def build_od_mitma(
    source_root: Path | str,
    out_path: Path | str,
    zones: ZoneSet | Sequence[str] | None = None,
    footprint_path: Path | str = DEFAULT_FOOTPRINT,
    population_path: Path | str | None = None,
    limit: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Full build: stream daily CSVs into a chunked zarr v3 OD store + audit.

    zones=None derives the geometry-audited zone set; a ZoneSet or a plain
    ID sequence is test injection (in_footprint=True, from_prefix from IDs).
    Streams one file at a time: template via dask zeros → per-date region
    write → audit attrs + consolidated metadata. Accumulates float64 scalars
    only; policy masks are computed on the stored float32 values, with an
    edge counted only where the stored raw trips are > 0 (structural zeros
    contribute no mass and are not edges). Returns the audit dict (same
    content as the store attrs).
    """
    source_root = Path(source_root)
    out_path = Path(out_path)
    layout = resolve_source_layout(source_root)

    if zones is None:
        zone_set = derive_zone_set(layout["shp_path"], footprint_path)
        derivation = "prefix_union_footprint_sjoin"
    elif isinstance(zones, ZoneSet):
        zone_set = zones
        derivation = "injected"
    else:
        zone_set = _injected_zone_set(zones)
        derivation = "injected"

    zone_ids = zone_set.zone_ids
    n_zones = len(zone_ids)
    if n_zones == 0:
        raise ValueError("empty zone set")
    zone_index = {z: i for i, z in enumerate(zone_ids)}

    population_path_used = ""
    if population_path is not None:
        population_path_used = str(population_path)
        populations = load_zone_populations(population_path, zone_ids)
    elif layout["poblacion_path"].is_file():
        population_path_used = str(layout["poblacion_path"])
        populations = load_zone_populations(layout["poblacion_path"], zone_ids)
    else:
        populations = np.full(n_zones, np.nan, dtype=np.float32)

    files = find_daily_files(layout["mitma_root"])
    if start is not None:
        files = [item for item in files if item[0] >= start]
    if end is not None:
        files = [item for item in files if item[0] <= end]
    processed_files = files if limit is None else files[:limit]
    processed_dates = [date for date, _ in processed_files]
    if not processed_files:
        raise ValueError(f"no daily mobility files matched under {layout['mitma_root']}")
    dates_missing = expected_dates([date for date, _ in files], start, end)

    dates_np = np.array(processed_dates, dtype="datetime64[ns]")
    id_array = np.array(zone_ids)
    prefix_arr = np.fromiter(zone_set.from_prefix, dtype=bool)
    footprint_arr = np.fromiter(zone_set.in_footprint, dtype=bool)
    template = xr.Dataset(
        {
            "mobility": (
                ("date", "origin", "target"),
                da.zeros((len(processed_dates), n_zones, n_zones), chunks=(1, n_zones, n_zones), dtype="float32"),
            ),
            "zone_population": (("zone",), populations),
            "in_footprint": (("zone",), footprint_arr),
            "from_prefix": (("zone",), prefix_arr),
        },
        coords={
            "date": dates_np,
            "origin": id_array,
            "target": id_array,
            "zone": np.arange(n_zones),
            "zone_id": (("zone",), id_array),
        },
    )
    template.to_zarr(
        out_path,
        mode="w",
        zarr_format=3,
        consolidated=False,
        encoding={"mobility": {"chunks": (1, n_zones, n_zones)}},
    )

    mass_total = 0.0
    mass_zone_set = 0.0
    diag_mass = 0.0
    stored_mass = 0.0
    max_rel_err = 0.0
    off_diag = ~np.eye(n_zones, dtype=bool)
    policy_mass = {key: 0.0 for sf in SELF_FLOW_POLICIES for key in [_policy_key(sf, mf) for mf in MIN_FLOW_RAW_POLICIES]}
    policy_edges = dict.fromkeys(policy_mass, 0)

    for i, (date, path) in enumerate(processed_files):
        daily = harmonize_daily_csv(path, zone_index)
        mat32 = daily.matrix
        xr.Dataset(
            {"mobility": (("date", "origin", "target"), mat32[np.newaxis])},
            coords={"date": np.array([date], dtype="datetime64[ns]")},
        ).to_zarr(out_path, region={"date": slice(i, i + 1)}, consolidated=False)

        f64 = mat32.astype(np.float64)
        mass_total += daily.mass_total
        mass_zone_set += daily.mass_zone_set
        diag_mass += float(np.trace(f64))
        stored_mass += float(f64.sum())
        max_rel_err = max(max_rel_err, daily.max_strata_rel_err)
        for self_flow in SELF_FLOW_POLICIES:
            for min_flow in MIN_FLOW_RAW_POLICIES:
                mask = np.ones_like(mat32, dtype=bool) if self_flow == "retain" else off_diag
                mask = mask & (mat32 > 0.0) & (mat32 >= min_flow)
                key = _policy_key(self_flow, min_flow)
                policy_mass[key] += float(f64[mask].sum())
                policy_edges[key] += int(mask.sum())

    audit: dict = {
        "schema": "od_mitma.v1",
        "zone_set_derivation": derivation,
        "source_root": str(source_root),
        "shp_path": str(layout["shp_path"]),
        "footprint_path": str(footprint_path),
        "population_path": population_path_used,
        "crs": "EPSG:3042",
        "zone_count": n_zones,
        "zone_count_shp_total": int(zone_set.n_shp_zones),
        "zone_prefix_count": int(prefix_arr.sum()),
        "zone_from_prefix_only": int((prefix_arr & ~footprint_arr).sum()),
        "zone_from_footprint_only": int((~prefix_arr & footprint_arr).sum()),
        "zone_from_both": int((prefix_arr & footprint_arr).sum()),
        "reference_zone_counts": dict(REFERENCE_ZONE_COUNTS),
        "zone_population_available": int(np.isfinite(populations).sum()),
        "n_dates_expected_full": len(pd.date_range(EXPECTED_FIRST_DATE, EXPECTED_LAST_DATE, freq="D")),
        "n_dates_expected": len(dates_missing) + len(files),
        "n_dates_processed": len(processed_files),
        "dates_missing": dates_missing,
        "limit": limit,
        "diag_mass_fraction": diag_mass / stored_mass if stored_mass > 0 else 0.0,
        "mass_in_zone_set_fraction": mass_zone_set / mass_total if mass_total > 0 else 0.0,
        "strata_summation_note": STRATA_NOTE,
        "strata_summation_max_rel_err": max_rel_err,
        "strata_summation_assertion": "passed",
    }
    for self_flow in SELF_FLOW_POLICIES:
        for min_flow in MIN_FLOW_RAW_POLICIES:
            key = _policy_key(self_flow, min_flow)
            audit[f"retained_mass_{key}"] = policy_mass[key]
            audit[f"retained_mass_fraction_{key}"] = policy_mass[key] / stored_mass if stored_mass > 0 else 0.0
            audit[f"edge_count_{key}"] = policy_edges[key]

    group = zarr.open_group(out_path, mode="a")
    group.attrs.update(audit)
    zarr.consolidate_metadata(out_path)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: build the od_mitma.zarr store and print an audit summary."""
    parser = argparse.ArgumentParser(description="Harmonize MITMA daily OD CSVs into od_mitma.zarr (slice MOB-2).")
    parser.add_argument("--source-root", required=True, help="dasymetric_mob checkout root (read-only)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output zarr path")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N daily files")
    parser.add_argument("--start", default=None, help="inclusive window start (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="inclusive window end (YYYY-MM-DD)")
    parser.add_argument("--footprint", default=str(DEFAULT_FOOTPRINT), help="municipal footprint path")
    parser.add_argument("--population-path", default=None, help="poblacion.csv override")
    args = parser.parse_args(argv)

    audit = build_od_mitma(
        source_root=args.source_root,
        out_path=args.out,
        footprint_path=args.footprint,
        population_path=args.population_path,
        limit=args.limit,
        start=args.start,
        end=args.end,
    )
    print(f"wrote {args.out}")
    print(f"zones: {audit['zone_count']} (prefix {audit['zone_prefix_count']}, shp total {audit['zone_count_shp_total']})")
    print(f"dates: {audit['n_dates_processed']} processed / {audit['n_dates_expected']} expected, missing {len(audit['dates_missing'])}")
    print(f"diag_mass_fraction: {audit['diag_mass_fraction']:.6f}")
    print(f"mass_in_zone_set_fraction: {audit['mass_in_zone_set_fraction']:.6f}")
    for self_flow in SELF_FLOW_POLICIES:
        for min_flow in MIN_FLOW_RAW_POLICIES:
            key = _policy_key(self_flow, min_flow)
            print(f"{key}: mass {audit[f'retained_mass_{key}']:.3f} edges {audit[f'edge_count_{key}']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
