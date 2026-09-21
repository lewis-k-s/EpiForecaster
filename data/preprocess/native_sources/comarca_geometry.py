"""Comarca geometry builder for the native-support refactor (GEO-1).

Primary geometry: the official IGCG v2r1 comarques shapefile (1:50k, dated
2025-07-30, EPSG:25831), read-only from the sibling ``dasymetric_mob``
checkout, reprojected to EPSG:3042 and copied into
``data/files/geo/generated/comarques_igcg_v2r1.geojson`` with stable
2-digit ``comarca_id`` strings.

Cross-check: a dissolve of the study-period municipal polygons
(``fl_municipios_catalonia.geojson``) by the ``mpiscatalunya.csv``
membership partition (plan §10.3(2)). ``reconcile`` reports ID-set
differences and per-comarca area discrepancies between the two vintages
(boundary drift 2025 vs 2020) plus membership and area closure of the
dissolve. Vintage differences are recorded, never fatal. Observed codes
from the comarca deaths CSV are reconciled as well: codes without
geometry (the ``"00"`` "No classificat" sentinel) are flagged
``unmapped``, and geometry without observed codes (``"43"``, Lluçanès,
created after the study period) is recorded.

Run as ``uv run python -m data.preprocess.native_sources.comarca_geometry``
from the repository root.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

TARGET_CRS = "EPSG:3042"
IGCG_CODE_FIELD = "CODICOMAR"
IGCG_NAME_FIELD = "NOMCOMAR"
IGCG_AREA_FIELD = "AREAC5000"
IGCG_COMPONENT_SUFFIXES = (".shp", ".dbf", ".shx", ".prj", ".cpg")

DEFAULT_IGCG_SHAPEFILE = Path(
    "/Volumes/HUBSSD/code/meprecisa/dasymetric_mob/data/"
    "IGCG_divisiones_administrativas/"
    "divisions-administratives-v2r1-comarques-50000-20250730.shp"
)
DEFAULT_MUNICIPAL_GEOJSON = Path("data/files/geo/fl_municipios_catalonia.geojson")
DEFAULT_MAPPING_CSV = Path("data/files/mpiscatalunya.csv")
DEFAULT_DEATHS_CSV = Path(
    "data/files/Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
)
DEFAULT_OUTPUT_GEOJSON = Path("data/files/geo/generated/comarques_igcg_v2r1.geojson")
DEFAULT_REPORT_PATH = Path("data/processed/native_support/comarca_geometry_report.json")

MAPPING_HEADER_ROWS = 4
MAPPING_COLUMNS = (
    "municipality_code",
    "municipality_name",
    "comarca_code",
    "comarca_name",
)


def _sha256_file(path: Path) -> str:
    """Return the hex sha256 digest of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_igcg_comarques(path: str | Path) -> gpd.GeoDataFrame:
    """Load the IGCG v2r1 comarques shapefile and normalize it.

    Returns a GeoDataFrame in ``TARGET_CRS`` with columns ``comarca_id``
    (2-digit zero-padded string), ``comarca_name``,
    ``igcg_reference_area_km2`` (the source 1:5000 reference area) and
    ``igcg_area_m2`` (recomputed in ``TARGET_CRS``), sorted by
    ``comarca_id``. The source file is opened read-only.
    """
    path = Path(path)
    gdf = gpd.read_file(path)
    for field in (IGCG_CODE_FIELD, IGCG_NAME_FIELD, IGCG_AREA_FIELD):
        if field not in gdf.columns:
            raise ValueError(f"Missing required field {field!r} in {path}")

    codes = gdf[IGCG_CODE_FIELD].astype(str).str.strip().str.zfill(2)
    if codes.isna().any() or (codes == "").any():
        raise ValueError(f"Null or empty IGCG comarca codes in {path}")
    bad = codes.str.fullmatch(r"\d{2}") ^ True
    if bad.any():
        raise ValueError(f"IGCG comarca codes must be 2-digit: {codes[bad].tolist()}")
    if codes.duplicated().any():
        duplicated = sorted(codes[codes.duplicated()].unique().tolist())
        raise ValueError(f"Duplicate IGCG comarca codes: {duplicated}")

    out = gdf[["geometry"]].copy()
    out["comarca_id"] = codes
    out["comarca_name"] = gdf[IGCG_NAME_FIELD].astype(str)
    out["igcg_reference_area_km2"] = pd.to_numeric(gdf[IGCG_AREA_FIELD], errors="coerce")
    out = out.to_crs(TARGET_CRS)
    out["igcg_area_m2"] = out.geometry.area
    return out.sort_values("comarca_id").reset_index(drop=True)


def load_municipality_comarca_mapping(path: str | Path) -> pd.DataFrame:
    """Load the municipality->comarca membership table.

    ``mpiscatalunya.csv`` has a UTF-8 BOM, a 4-line header and a trailing
    comma on every row. Returns columns ``municipality_id`` (INE 5-digit
    string, the Idescat 6-digit ``Codi`` minus its control digit),
    ``municipality_name``, ``comarca_id`` (2-digit string) and
    ``comarca_name``.
    """
    df = pd.read_csv(  # type: ignore[call-arg]
        Path(path),
        encoding="utf-8-sig",
        skiprows=MAPPING_HEADER_ROWS,
        usecols=[0, 1, 2, 3],
        names=list(MAPPING_COLUMNS),
        dtype=str,
    )
    if df["municipality_code"].isna().any():
        raise ValueError("Missing municipality codes in mapping data")
    if df["comarca_code"].isna().any():
        raise ValueError("Missing comarca codes in mapping data")
    return pd.DataFrame(
        {
            "municipality_id": df["municipality_code"]
            .str.strip()
            .str[:5]
            .str.zfill(5),
            "municipality_name": df["municipality_name"].astype(str),
            "comarca_id": df["comarca_code"].str.strip().str.zfill(2),
            "comarca_name": df["comarca_name"].astype(str),
        }
    )


@dataclass
class DissolveResult:
    """Outcome of dissolving municipal polygons into comarcas."""

    geometries: gpd.GeoDataFrame
    municipal_part_area_m2: dict[str, float]
    unmapped_municipality_ids: list[str]
    mapping_ids_without_geometry: list[str]
    area_closure_max_rel_error: float


def dissolve_municipal_comarques(
    municipalities: gpd.GeoDataFrame, mapping: pd.DataFrame
) -> DissolveResult:
    """Dissolve municipal polygons into comarcas via the membership mapping.

    ``municipalities`` must already carry a ``municipality_id`` string
    column. Polygons without a mapping entry are excluded from the
    dissolve and recorded in ``unmapped_municipality_ids``; mapping rows
    without a polygon are recorded in ``mapping_ids_without_geometry``.
    ``area_closure_max_rel_error`` is the largest
    ``|sum of member areas - dissolved area| / dissolved area`` over all
    comarcas.
    """
    if "municipality_id" not in municipalities.columns:
        raise ValueError("municipalities must carry a 'municipality_id' column")

    mun = municipalities[["municipality_id", "geometry"]].copy()
    mun["municipality_id"] = mun["municipality_id"].astype(str)
    membership = mapping[["municipality_id", "comarca_id", "comarca_name"]].copy()
    membership["municipality_id"] = membership["municipality_id"].astype(str)

    joined = mun.merge(membership, on="municipality_id", how="left")
    mapped = joined[joined["comarca_id"].notna()].copy()
    mapped["comarca_id"] = mapped["comarca_id"].astype(str)
    unmapped_ids = sorted(
        joined.loc[joined["comarca_id"].isna(), "municipality_id"].unique().tolist()
    )
    mapping_ids_without_geometry = sorted(
        set(membership["municipality_id"]) - set(mun["municipality_id"])
    )

    counts = mapped["comarca_id"].value_counts()
    part_areas = (
        mapped.assign(_area=mapped.geometry.area).groupby("comarca_id")["_area"].sum()
    )

    dissolved = mapped[["comarca_id", "comarca_name", "geometry"]].dissolve(
        by="comarca_id", as_index=False
    )
    dissolved["membership_count"] = (
        dissolved["comarca_id"].map(counts).fillna(0).astype(int)
    )
    dissolved["dissolved_area_m2"] = dissolved.geometry.area
    dissolved = dissolved.sort_values("comarca_id").reset_index(drop=True)

    dissolved_areas = dict(
        zip(dissolved["comarca_id"], dissolved["dissolved_area_m2"].astype(float))
    )
    closure_errors = [
        abs(part_areas.loc[cid] - area) / area
        for cid, area in dissolved_areas.items()
        if area > 0
    ]
    max_closure = float(max(closure_errors, default=0.0))

    return DissolveResult(
        geometries=dissolved,
        municipal_part_area_m2={cid: float(a) for cid, a in part_areas.items()},
        unmapped_municipality_ids=unmapped_ids,
        mapping_ids_without_geometry=mapping_ids_without_geometry,
        area_closure_max_rel_error=max_closure,
    )


def reconcile(
    igcg: gpd.GeoDataFrame,
    dissolved: DissolveResult,
    observed_codes: Iterable[str] | None = None,
) -> dict:
    """Build a plain-JSON reconciliation report between the two vintages.

    One row per union of IGCG and dissolved comarca ids with areas in
    km2, membership counts, ``relative_area_difference``
    ``(dissolved - igcg) / igcg`` and a ``matched`` / ``igcg_only`` /
    ``dissolve_only`` status. Data-level differences never raise; only
    structural errors do. When ``observed_codes`` is given (distinct
    comarca codes from the deaths CSV), codes without geometry are
    flagged ``unmapped`` and geometry without observed codes recorded.
    """
    igcg_names = dict(zip(igcg["comarca_id"].astype(str), igcg["comarca_name"]))
    igcg_areas = dict(
        zip(igcg["comarca_id"].astype(str), igcg["igcg_area_m2"].astype(float))
    )
    dis = dissolved.geometries
    dis_names = dict(zip(dis["comarca_id"].astype(str), dis["comarca_name"]))
    dis_areas = dict(
        zip(dis["comarca_id"].astype(str), dis["dissolved_area_m2"].astype(float))
    )
    dis_counts = dict(
        zip(
            dis["comarca_id"].astype(str),
            (int(c) for c in dis["membership_count"]),
        )
    )
    igcg_ids = set(igcg_areas)
    dis_ids = set(dis_areas)

    rows = []
    for cid in sorted(igcg_ids | dis_ids):
        ig_area = igcg_areas.get(cid)
        di_area = dis_areas.get(cid)
        if ig_area is None or ig_area == 0.0 or di_area is None:
            rel_diff = None
        else:
            rel_diff = (di_area - ig_area) / ig_area
        rows.append(
            {
                "comarca_id": cid,
                "comarca_name": igcg_names.get(cid) or dis_names.get(cid),
                "membership_count": dis_counts.get(cid),
                "dissolved_area_km2": di_area / 1e6 if di_area is not None else None,
                "igcg_area_km2": ig_area / 1e6 if ig_area is not None else None,
                "relative_area_difference": rel_diff,
                "status": (
                    "matched"
                    if cid in igcg_ids and cid in dis_ids
                    else ("igcg_only" if cid in igcg_ids else "dissolve_only")
                ),
            }
        )

    report: dict = {
        "comarcas": rows,
        "id_diffs": {
            "igcg_only": sorted(igcg_ids - dis_ids),
            "dissolve_only": sorted(dis_ids - igcg_ids),
            "both_count": len(igcg_ids & dis_ids),
        },
        "dissolve": {
            "comarca_count": len(dis_ids),
            "mapped_municipality_count": int(sum(dis_counts.values())),
            "membership_counts": {cid: dis_counts[cid] for cid in sorted(dis_counts)},
            "unmapped_municipality_ids": list(dissolved.unmapped_municipality_ids),
            "mapping_ids_without_geometry": list(
                dissolved.mapping_ids_without_geometry
            ),
            "area_closure_max_rel_error": float(dissolved.area_closure_max_rel_error),
        },
        "vintage_note": (
            "IGCG v2r1 boundaries are dated 2025-07-30 while the dissolved "
            "municipal polygons follow the study-period vintage (2020). Area "
            "and boundary differences between the two vintages are recorded "
            "here as reconciliation data, never treated as failures."
        ),
    }

    if observed_codes is not None:
        observed = {str(code).strip().zfill(2) for code in observed_codes if code is not None and str(code).strip()}
        geometry_ids = igcg_ids | dis_ids
        report["observed_codes"] = {
            "matched_codes": sorted(observed & geometry_ids),
            "codes_without_geometry": sorted(observed - geometry_ids),
            "codes_without_geometry_status": "unmapped",
            "geometry_without_observed_codes": sorted(geometry_ids - observed),
        }
    return report


def build(
    igcg_path: str | Path = DEFAULT_IGCG_SHAPEFILE,
    municipal_path: str | Path = DEFAULT_MUNICIPAL_GEOJSON,
    mapping_path: str | Path = DEFAULT_MAPPING_CSV,
    deaths_csv_path: str | Path = DEFAULT_DEATHS_CSV,
    output_geojson: str | Path = DEFAULT_OUTPUT_GEOJSON,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict:
    """Build the comarca geometry artifact and reconciliation report.

    Writes the primary IGCG comarca geometry to ``output_geojson``
    (EPSG:3042, 2-digit ``comarca_id``) and the provenance + reconciliation
    report to ``report_path``. Returns the report dict.
    """
    igcg = load_igcg_comarques(igcg_path)
    mapping = load_municipality_comarca_mapping(mapping_path)

    municipalities = gpd.read_file(municipal_path)
    municipalities["municipality_id"] = (
        municipalities["id"].astype(str).str.zfill(5)
    )
    dissolved = dissolve_municipal_comarques(municipalities, mapping)

    codes = pd.read_csv(  # type: ignore[call-arg]
        deaths_csv_path,
        usecols=["Codi Comarca"],
        dtype={"Codi Comarca": str},
    )["Codi Comarca"]
    observed_codes = sorted(codes.dropna().unique().tolist())
    report = reconcile(igcg, dissolved, observed_codes=observed_codes)

    igcg_path = Path(igcg_path)
    component_paths = [igcg_path.with_suffix(suffix) for suffix in IGCG_COMPONENT_SUFFIXES]
    components = {path.suffix: _sha256_file(path) for path in component_paths if path.exists()}
    report = {
        "provenance": {
            "igcg_shapefile": str(igcg_path),
            "igcg_component_sha256": components,
            "igcg_source_crs": "EPSG:25831",
            "target_crs": TARGET_CRS,
            "igcg_feature_count": int(len(igcg)),
            "municipal_geojson": str(municipal_path),
            "municipal_feature_count": int(len(municipalities)),
            "mapping_csv": str(mapping_path),
            "mapping_row_count": int(len(mapping)),
            "deaths_csv": str(deaths_csv_path),
            "deaths_comarca_code_count": len(observed_codes),
        },
        **report,
    }

    output_geojson = Path(output_geojson)
    report_path = Path(report_path)
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    igcg.to_file(output_geojson, driver="GeoJSON")
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return report


if __name__ == "__main__":
    build_result = build()
    print(
        f"Reconciled {len(build_result['comarcas'])} comarcas; wrote "
        f"{DEFAULT_OUTPUT_GEOJSON} and {DEFAULT_REPORT_PATH}"
    )
