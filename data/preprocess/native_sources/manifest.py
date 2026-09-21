"""Native-support manifest v2 (slice MAN-1; plan §11.3, §12.2).

``native_manifest_v2.json`` records the sha256 of every input consumed by the
native-support builders: the two native clinical CSVs (recorded as sitting
outside ``catalonia_preprocessing_inputs_v1`` — closing [D1]'s provenance
gap), the geometry inputs, the read-only ``dasymetric_mob`` inputs, a
content-hash code stamp of the producing package, the produced-artifact
metadata hashes, schema versions, and the frozen confirmation-period marker
``["2020-12-31", "2021-01-27"]`` (plan §10.3(10): evaluation code must refuse
to score that window without an explicit guard flag).

Hashing methods (each recorded in the manifest itself):

- regular input files: full per-file sha256;
- the 451 ``mitma_v1`` daily gz files: per-file sha256 aggregated into one
  digest of the sorted ``(filename, size, sha256)`` listing;
- produced zarr stores: sha256 over the sorted ``(relpath, size, sha256)``
  listing of every ``zarr.json`` metadata document (zarr v3). Chunk arrays
  are never hashed, so a data-only rewrite that leaves every metadata
  document byte-identical is not detected — by design, documented here;
- ``overlap_edges.parquet``: full file sha256 (it is a single small file).

``git rev-parse`` is unavailable to goal-loop slices, so the code-version
stamp is a content hash of the ``native_sources`` package sources.

The manifest contains no wall-clock timestamps: identical inputs and package
sources must produce byte-identical JSON (determinism contract, tested).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "native_manifest_v2"
SCHEMA_VERSION = 2
MANIFEST_NAME = "native_manifest_v2.json"

#: Frozen marker (plan §10.3(10), §11.3): [D1] used this window for
#: calibration only; scoring it requires an explicit guard flag.
CONFIRMATION_PERIOD = ["2020-12-31", "2021-01-27"]
CONFIRMATION_PERIOD_NOTE = (
    "[D1] calibration window; native evaluation must refuse to score it "
    "unless an explicit guard flag is set (plan §10.3(10))"
)

#: Read-only sibling checkout (§10.2b): consumed and hashed, never written.
DEFAULT_DASYMETRIC_ROOT = Path("/Volumes/HUBSSD/code/meprecisa/dasymetric_mob")
MITMA_DAILY_SUBDIR = Path("data/mitma_v1")
DAILY_GLOB = "????-??-??.daily_mobility.v1.recurrent.csv.gz"
EXPECTED_DAILY_FILE_COUNT = 451
DAILY_AGGREGATE_METHOD = (
    "per-file sha256 aggregated: sha256 of the sorted '(filename size sha256)' "
    "newline listing; recorded as file_count + total_bytes + aggregate_sha256"
)
ZARR_METADATA_METHOD = (
    "sha256 of the sorted '(relpath size sha256)' newline listing over every "
    "zarr.json metadata document in the store; chunk arrays are not hashed"
)
LISTING_METHOD = "sha256 of the sorted '(relpath size sha256)' newline listing"
PACKAGE_METHOD = (
    "content hash of the native_sources package sources; git rev-parse is "
    "unavailable to goal-loop slices (plan §12.0)"
)
CLINICAL_NOTE = (
    "sits outside catalonia_preprocessing_inputs_v1 (see dvc_anchor); hashed "
    "here individually per plan §11.3"
)

PRODUCED_STORES = (
    "od_mitma",
    "od_municipal",
    "native_graph",
    "obs_admissions",
    "obs_deaths",
)
PRODUCED_PARQUET = "overlap_edges"


@dataclass(frozen=True)
class InputPaths:
    """Every consumed input file (defaults = the real repo/dasymetric layout)."""

    abs_csv: Path = Path("data/files/COVID-19__Persones_hospitalitzades.csv")
    deaths_csv: Path = Path(
        "data/files/Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
    )
    inputs_v1_dvc: Path = Path("data/files/catalonia_preprocessing_inputs_v1.dvc")
    abs_shp: Path = Path("data/files/geo/cartografia/ABS.shp")
    municipal_geojson: Path = Path("data/files/geo/fl_municipios_catalonia.geojson")
    mpis_csv: Path = Path("data/files/mpiscatalunya.csv")
    population_csv: Path = Path("data/files/fl_population_por_municipis.csv")
    comarques_geojson: Path = Path(
        "data/files/geo/generated/comarques_igcg_v2r1.geojson"
    )
    reference_csv: Path = Path("data/files/municipality_id_reference.csv")
    dasymetric_root: Path = DEFAULT_DASYMETRIC_ROOT

    def daily_dir(self) -> Path:
        return self.dasymetric_root / MITMA_DAILY_SUBDIR

    def distritos_shp(self) -> Path:
        return (
            self.dasymetric_root
            / "data/mitma_v1/zonificacion-distritos/distritos_mitma.shp"
        )

    def igcg_shp(self) -> Path:
        return (
            self.dasymetric_root
            / "data/IGCG_divisiones_administrativas/divisions-administratives-v2r1-comarques-50000-20250730.shp"
        )

    def geostat_csv(self) -> Path:
        return self.dasymetric_root / "data/GEOSTAT_grid_POP_1K_2011_V2_0_1.csv"

    def grid_geom_shp(self) -> Path:
        return (
            self.dasymetric_root
            / "data/GRID_GEOM/Grid_ETRS89_LAEA_1K-ref_GEOSTAT_POP_2011_V2_0_1.shp"
        )


def sha256_file(path: str | Path) -> str:
    """Full sha256 hex digest of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_entry(path: Path, key: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"manifest input missing: {path}")
    return {
        "name": key,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _listing_digest(rows: list[tuple[str, int, str]]) -> str:
    """sha256 of the sorted ``(relpath, size, sha256)`` newline listing."""
    lines = [f"{rel}\t{size}\t{digest}" for rel, size, digest in sorted(rows)]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def shapefile_components(shp: Path) -> list[Path]:
    """All sidecar files sharing the shapefile stem, sorted (shp included)."""
    if not shp.is_file():
        raise FileNotFoundError(f"manifest input missing: {shp}")
    return sorted(p for p in shp.parent.glob(f"{shp.stem}.*") if p.is_file())


def hash_shapefile(shp: Path) -> list[dict]:
    """Per-file entries for every component of one shapefile."""
    return [_file_entry(p, p.suffix.lstrip(".")) for p in shapefile_components(shp)]


def mitma_daily_aggregate(daily_dir: Path) -> dict:
    """Aggregate manifest over the daily ``mitma_v1`` gz files (§12.2 MAN-1)."""
    files = sorted(daily_dir.glob(DAILY_GLOB))
    if not files:
        raise FileNotFoundError(f"no mitma_v1 daily files under {daily_dir}")
    rows: list[tuple[str, int, str]] = []
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        rows.append((path.name, size, sha256_file(path)))
    return {
        "method": DAILY_AGGREGATE_METHOD,
        "directory": str(daily_dir),
        "file_count": len(files),
        "expected_file_count": EXPECTED_DAILY_FILE_COUNT,
        "total_bytes": total_bytes,
        "aggregate_sha256": _listing_digest(rows),
    }


def zarr_metadata_digest(store: Path) -> dict:
    """Metadata-level hash of one zarr v3 store (chunks never hashed)."""
    docs = sorted(p for p in store.rglob("zarr.json") if p.is_file())
    if not docs:
        raise FileNotFoundError(f"no zarr.json metadata documents under {store}")
    rows = [(str(p.relative_to(store)), p.stat().st_size, sha256_file(p)) for p in docs]
    return {
        "method": ZARR_METADATA_METHOD,
        "metadata_file_count": len(docs),
        "sha256": _listing_digest(rows),
    }


def _store_schema_attr(store: Path) -> dict:
    """Schema version recorded in the store root ``zarr.json`` attrs."""
    doc = json.loads((store / "zarr.json").read_text())
    attrs = doc.get("attributes") or {}
    out: dict = {"schema": attrs.get("schema")}
    if attrs.get("schema_version") is not None:
        out["schema_version"] = attrs["schema_version"]
    return out


def code_stamp(package_dir: Path | None = None) -> dict:
    """Content hash of the producing package (no git available, §12.0)."""
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    files = sorted(p for p in root.glob("*.py") if p.is_file())
    rows = [(p.name, p.stat().st_size, sha256_file(p)) for p in files]
    return {
        "method": PACKAGE_METHOD,
        "package": str(root),
        "package_sha256": _listing_digest(rows),
        "files": {"summary": f"{len(files)} package .py files"},
    }


def build_manifest(root: str | Path, inputs: InputPaths | None = None) -> dict:
    """Assemble the §11.3 manifest dict from the real input/artifact layout.

    Raises ``FileNotFoundError`` on any missing input or produced artifact —
    the only hard-error condition; everything else in QC is warnings.
    """
    spec = inputs or InputPaths()
    root = Path(root)

    produced: dict = {
        PRODUCED_PARQUET: _file_entry(root / "overlap_edges.parquet", PRODUCED_PARQUET),
    }
    schema_versions: dict = {}
    for store in PRODUCED_STORES:
        store_dir = root / f"{store}.zarr"
        if not store_dir.is_dir():
            raise FileNotFoundError(
                f"produced native-support store missing: {store_dir}"
            )
        produced[store] = zarr_metadata_digest(store_dir)
        schema_versions[store] = _store_schema_attr(store_dir)

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "confirmation_period": list(CONFIRMATION_PERIOD),
        "confirmation_period_note": CONFIRMATION_PERIOD_NOTE,
        "code_stamp": code_stamp(),
        "inputs": {
            "clinical": {
                "note": CLINICAL_NOTE,
                "dvc_anchor": _file_entry(
                    spec.inputs_v1_dvc, "catalonia_preprocessing_inputs_v1.dvc"
                ),
                "files": [
                    _file_entry(spec.abs_csv, "hospitalizations"),
                    _file_entry(spec.deaths_csv, "deaths"),
                ],
            },
            "geometry": {
                "files": [
                    *hash_shapefile(spec.abs_shp),
                    _file_entry(
                        spec.municipal_geojson, "fl_municipios_catalonia.geojson"
                    ),
                    _file_entry(spec.mpis_csv, "mpiscatalunya.csv"),
                    _file_entry(spec.population_csv, "fl_population_por_municipis.csv"),
                    _file_entry(spec.comarques_geojson, "comarques_igcg_v2r1.geojson"),
                ],
            },
            "registry": {
                "files": [
                    _file_entry(spec.reference_csv, "municipality_id_reference.csv")
                ],
            },
            "dasymetric_readonly": {
                "root": str(spec.dasymetric_root),
                "note": "read-only sibling checkout; consumed and hashed, never written",
                "files": [
                    *hash_shapefile(spec.distritos_shp()),
                    *hash_shapefile(spec.igcg_shp()),
                    _file_entry(
                        spec.geostat_csv(), "GEOSTAT_grid_POP_1K_2011_V2_0_1.csv"
                    ),
                    *hash_shapefile(spec.grid_geom_shp()),
                ],
                "mitma_v1_daily": mitma_daily_aggregate(spec.daily_dir()),
            },
        },
        "produced": produced,
        "produced_schema_versions": schema_versions,
    }


def write_manifest(manifest: dict, path: str | Path) -> Path:
    """Serialize deterministically (sorted keys, no timestamps)."""
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
