"""Tests for native-support manifest v2 and QC (slice MAN-1; plan §12.2).

Unit tests run entirely on synthetic fixtures: tiny input files, two daily
mitma_v1-style gz files, and miniature zarr stores with hand-computable QC
numbers. The real-artifact integration test is skipif-guarded and stays
read-only (the CLI run writes the manifest/report, tests do not).
"""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.native_sources.manifest import (
    CONFIRMATION_PERIOD,
    SCHEMA,
    SCHEMA_VERSION,
    InputPaths,
    build_manifest,
    code_stamp,
    mitma_daily_aggregate,
    write_manifest,
    zarr_metadata_digest,
)
from scripts.analysis.native_support_qc import (
    abs_population_ratios,
    build_report,
    comarca_population_closure,
    format_report,
    mitma_zone_counts,
    normalize_retention,
    observed_coverage,
    od_mass_retention,
)

pytestmark = pytest.mark.epiforecaster

DAILY_NAME = "{}.daily_mobility.v1.recurrent.csv.gz"


def _touch(path: Path, content: str = "seed\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _make_inputs(
    base: Path, daily_dates: tuple[str, ...] = ("2020-03-01", "2020-03-02")
) -> InputPaths:
    """Synthetic mirror of the real input layout under ``base``."""
    files = base / "data" / "files"
    dasymetric = base / "dasy"
    spec = InputPaths(
        abs_csv=files / "COVID-19__Persones_hospitalitzades.csv",
        deaths_csv=files / "Registre_de_defuncions_per_COVID-19.csv",
        inputs_v1_dvc=files / "catalonia_preprocessing_inputs_v1.dvc",
        abs_shp=files / "geo/cartografia/ABS.shp",
        municipal_geojson=files / "geo/fl_municipios_catalonia.geojson",
        mpis_csv=files / "mpiscatalunya.csv",
        population_csv=files / "fl_population_por_municipis.csv",
        comarques_geojson=files / "geo/generated/comarques_igcg_v2r1.geojson",
        reference_csv=files / "municipality_id_reference.csv",
        dasymetric_root=dasymetric,
    )
    _touch(spec.abs_csv, "codi_abs,casos\n001,1\n")
    _touch(spec.deaths_csv, "comarca,defuncions\nc1,2\n")
    _touch(spec.inputs_v1_dvc, "md5: 0000\n")
    for suffix in ("shp", "shx", "dbf", "prj", "cpg"):
        _touch(spec.abs_shp.with_suffix(f".{suffix}"), f"abs-{suffix}\n")
    _touch(spec.municipal_geojson, "{}\n")
    _touch(spec.mpis_csv, "muni,comarca\n")
    _touch(spec.population_csv, "code,pop\n")
    _touch(spec.comarques_geojson, "{}\n")
    _touch(spec.reference_csv, "muni_code_5digit\n")
    for suffix in ("shp", "shx", "dbf", "prj", "cpg"):
        _touch(spec.distritos_shp().with_suffix(f".{suffix}"), f"distritos-{suffix}\n")
    for suffix in ("shp", "shx", "dbf", "prj", "cpg"):
        _touch(spec.igcg_shp().with_suffix(f".{suffix}"), f"igcg-{suffix}\n")
    _touch(spec.geostat_csv(), "lat,lon,pop\n")
    for suffix in ("shp", "shx", "dbf", "prj", "sbn", "sbx"):
        _touch(spec.grid_geom_shp().with_suffix(f".{suffix}"), f"grid-{suffix}\n")
    for date in daily_dates:
        daily = spec.daily_dir() / DAILY_NAME.format(date)
        daily.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(daily, "wt") as handle:
            handle.write("date\torigin\tdestination\ttrips\n")
    return spec


OD_MITMA_ATTRS = {
    "schema": "od_mitma.test.v1",
    "zone_count_shp_total": 10,
    "reference_zone_counts": {
        "prefix_verified": 442,
        "g1_diagnostics": 454,
        "transform_attr": 458,
    },
    "diag_mass_fraction": 0.5,
    "retained_mass_fraction_retain_minflow0": 1.0,
    "edge_count_retain_minflow0": 5,
    "retained_mass_fraction_drop_minflow10": 0.7,
    "edge_count_drop_minflow10": 3,
}

OD_MUNICIPAL_ATTRS = {
    "total_raw_mass": 100.0,
    "diag_mass_fraction": 0.4,
    "retained_mass_fraction": {
        "retain__min_flow_raw_0.0": 1.0,
        "drop__min_flow_raw_10.0": 0.424242,
    },
    "edge_count": {"retain__min_flow_raw_0.0": 7, "drop__min_flow_raw_10.0": 2},
}


def _write_mini_stores(root: Path) -> None:
    """Miniature produced artifacts with hand-computable QC numbers."""

    def dataset(
        attrs: dict, **variables: tuple[tuple[str, ...], np.ndarray]
    ) -> xr.Dataset:
        return xr.Dataset(
            {name: (dims, values) for name, (dims, values) in variables.items()},
            attrs=attrs,
        )

    graph = dataset(
        {},
        node_type=(
            ("node",),
            np.array(
                ["abs", "abs", "abs", "comarca", "comarca", "municipality", "mitma"]
            ),
        ),
        source_id=(("node",), np.array(["001", "002", "003", "c1", "c2", "m1", "z1"])),
        population=(
            ("node",),
            np.array(
                [1000.0, 1000.0, 500.0, 100.0, 400.0, 10.0, 50.0], dtype=np.float32
            ),
        ),
    )
    admissions = dataset(
        {},
        stratum_pop_sum=(
            ("date", "abs_id"),
            np.array([[500.0, 1500.0, 300.0], [400.0, 1400.0, 200.0]]),
        ),
    )
    admissions = admissions.assign_coords(
        abs_id=("abs_id", np.array(["001", "002", "009"], dtype=object)),
        date=("date", np.array(["2020-03-02", "2020-03-09"], dtype="datetime64[ns]")),
    )
    deaths = dataset(
        {},
        population=(("comarca_id",), np.array([100.0, 440.0])),
    ).assign_coords(comarca_id=("comarca_id", np.array(["c1", "c2"], dtype=object)))
    od_mitma = dataset(
        OD_MITMA_ATTRS,
        zone_population=(("zone",), np.array([np.nan], dtype=np.float32)),
        mobility=(("date", "origin", "target"), np.zeros((2, 1, 1), dtype=np.float32)),
    ).assign_coords(
        zone=("zone", np.array(["z1"], dtype=object)),
        date=("date", np.array(["2020-03-01", "2020-03-02"], dtype="datetime64[ns]")),
        origin=("origin", np.array(["z1"], dtype=object)),
        target=("target", np.array(["z1"], dtype=object)),
    )
    od_municipal = dataset(
        OD_MUNICIPAL_ATTRS,
        mobility=(("origin", "target"), np.zeros((2, 2), dtype=np.float16)),
    ).assign_coords(
        origin=("origin", np.array(["m1", "m2"], dtype=object)),
        target=("target", np.array(["m1", "m2"], dtype=object)),
    )

    root.mkdir(parents=True, exist_ok=True)
    graph.to_zarr(root / "native_graph.zarr", mode="w", consolidated=False)
    admissions.to_zarr(root / "obs_admissions.zarr", mode="w", consolidated=False)
    deaths.to_zarr(root / "obs_deaths.zarr", mode="w", consolidated=False)
    od_mitma.to_zarr(root / "od_mitma.zarr", mode="w", consolidated=False)
    od_municipal.to_zarr(root / "od_municipal.zarr", mode="w", consolidated=False)
    pd.DataFrame({"src_id": ["001"], "dst_id": ["m1"]}).to_parquet(
        root / "overlap_edges.parquet"
    )


def _mini_support(base: Path) -> tuple[Path, InputPaths]:
    root = base / "root"
    spec = _make_inputs(base / "inputs")
    _write_mini_stores(root)
    return root, spec


class TestManifest:
    def test_determinism(self, tmp_path):
        root, spec = _mini_support(tmp_path)
        one = build_manifest(root, spec)
        two = build_manifest(root, spec)
        assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
        path_one = write_manifest(one, tmp_path / "m1.json")
        path_two = write_manifest(two, tmp_path / "m2.json")
        assert path_one.read_bytes() == path_two.read_bytes()

    def test_schema_marker_and_code_stamp(self, tmp_path):
        root, spec = _mini_support(tmp_path)
        manifest = build_manifest(root, spec)
        assert manifest["schema"] == SCHEMA
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert (
            manifest["confirmation_period"]
            == CONFIRMATION_PERIOD
            == ["2020-12-31", "2021-01-27"]
        )
        assert manifest["confirmation_period_note"]
        stamp = manifest["code_stamp"]
        assert stamp == code_stamp()  # package stamp is environment-independent
        assert stamp["package_sha256"]

    def test_input_change_detected(self, tmp_path):
        root, spec = _mini_support(tmp_path)
        before = build_manifest(root, spec)
        clinical = {f["name"]: f for f in before["inputs"]["clinical"]["files"]}
        spec.abs_csv.write_text("codi_abs,casos\n001,999\n")
        after = build_manifest(root, spec)
        changed = {f["name"]: f for f in after["inputs"]["clinical"]["files"]}
        assert (
            changed["hospitalizations"]["sha256"]
            != clinical["hospitalizations"]["sha256"]
        )
        assert json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True)

    def test_daily_aggregate_tracks_files(self, tmp_path):
        _, spec = _mini_support(tmp_path)
        base = mitma_daily_aggregate(spec.daily_dir())
        assert base["file_count"] == 2
        assert base["expected_file_count"] == 451
        rows = []
        for path in sorted(spec.daily_dir().glob("*.gz")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((path.name, path.stat().st_size, digest))
        expected = hashlib.sha256(
            ("\n".join(f"{n}\t{s}\t{d}" for n, s, d in sorted(rows)) + "\n").encode()
        ).hexdigest()
        assert base["aggregate_sha256"] == expected
        assert base["total_bytes"] == sum(size for _, size, _ in rows)

        extra = spec.daily_dir() / DAILY_NAME.format("2020-03-03")
        with gzip.open(extra, "wt") as handle:
            handle.write("date\torigin\tdestination\ttrips\n")
        grown = mitma_daily_aggregate(spec.daily_dir())
        assert grown["file_count"] == 3
        assert grown["aggregate_sha256"] != base["aggregate_sha256"]

    def test_produced_metadata_change_detected(self, tmp_path):
        root, spec = _mini_support(tmp_path)
        before = build_manifest(root, spec)["produced"]["od_mitma"]
        ds = xr.open_zarr(root / "od_mitma.zarr", consolidated=False).load()
        ds.attrs["diag_mass_fraction"] = 0.9
        ds.to_zarr(root / "od_mitma.zarr", mode="w", consolidated=False)
        after = build_manifest(root, spec)["produced"]["od_mitma"]
        assert after["sha256"] != before["sha256"]

    def test_missing_input_is_hard_error(self, tmp_path):
        root, spec = _mini_support(tmp_path)
        spec.deaths_csv.unlink()
        with pytest.raises(FileNotFoundError, match="Registre_de_defuncions"):
            build_manifest(root, spec)

    def test_missing_produced_store_is_hard_error(self, tmp_path):
        root, spec = _mini_support(tmp_path)
        (root / "obs_deaths.zarr").rename(tmp_path / "hidden.zarr")
        with pytest.raises(FileNotFoundError, match="obs_deaths"):
            build_manifest(root, spec)

    def test_zarr_metadata_digest_shape(self, tmp_path):
        root, _ = _mini_support(tmp_path)
        digest = zarr_metadata_digest(root / "native_graph.zarr")
        docs = list((root / "native_graph.zarr").rglob("zarr.json"))
        assert digest["metadata_file_count"] == len(docs) > 0
        assert (
            digest["sha256"]
            == zarr_metadata_digest(root / "native_graph.zarr")["sha256"]
        )
        assert digest["method"]


class TestQcNumbers:
    def _nodes(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "node_type": [
                    "abs",
                    "abs",
                    "abs",
                    "comarca",
                    "comarca",
                    "municipality",
                ],
                "source_id": ["001", "002", "003", "c1", "c2", "m1"],
                "population": [1000.0, 1000.0, 500.0, 100.0, 400.0, 10.0],
            }
        )

    def _admissions(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "stratum_pop_sum": (
                    ("date", "abs_id"),
                    np.array([[500.0, 1500.0], [300.0, 200.0]]),
                )
            }
        ).assign_coords(abs_id=("abs_id", ["001", "009"]))

    def test_abs_ratios_hand_computed(self):
        check = abs_population_ratios(self._nodes(), self._admissions())
        assert check["n_abs_observed"] == 2
        assert check["n_with_graph_population"] == 1
        assert check["abs_without_graph_population"] == ["009"]
        assert check["n_ratio_above_one"] == 0
        assert check["ratio"]["n"] == 1  # only 001 has a graph population

    def test_abs_ratio_violation_warns(self):
        admissions = xr.Dataset(
            {"stratum_pop_sum": (("date", "abs_id"), np.array([[1500.0]]))}
        ).assign_coords(abs_id=("abs_id", ["001"]))
        check = abs_population_ratios(self._nodes(), admissions)
        assert check["n_ratio_above_one"] == 1
        assert check["worst_above_one"] == {"001": pytest.approx(1.5)}

    def test_comarca_closure_hand_computed(self):
        deaths = xr.Dataset(
            {"population": (("comarca_id",), np.array([100.0, 440.0, 7.0]))}
        ).assign_coords(comarca_id=("comarca_id", ["c1", "c2", "zz"]))
        closure = comarca_population_closure(self._nodes(), deaths)
        assert closure["n_store_columns"] == 3
        assert closure["n_matched"] == 2
        assert closure["unmatched_store_columns"] == ["zz"]
        assert closure["max_abs_rel_error"] == pytest.approx(0.1)
        assert closure["n_mismatch"] == 1
        assert closure["worst_mismatch"] == {"c2": pytest.approx(0.1)}

    def test_retention_passthrough_not_recomputed(self):
        flat = normalize_retention(OD_MITMA_ATTRS)
        assert flat["policies"]["retain|min_flow_raw=0"] == {
            "retained_mass_fraction": 1.0,
            "edge_count": 5,
            "retained_mass": None,
        }
        assert flat["policies"]["drop|min_flow_raw=10"]["retained_mass_fraction"] == 0.7
        dicted = normalize_retention(OD_MUNICIPAL_ATTRS)
        # odd value proves attrs are copied verbatim, never recomputed
        assert (
            dicted["policies"]["drop|min_flow_raw=10"]["retained_mass_fraction"]
            == 0.424242
        )
        assert dicted["total_raw_mass"] == 100.0
        retention = od_mass_retention(OD_MITMA_ATTRS, OD_MUNICIPAL_ATTRS)
        assert (
            retention["od_municipal"]["policies"]["retain|min_flow_raw=0"][
                "retained_mass_fraction"
            ]
            == 1.0
        )
        assert od_mass_retention(OD_MITMA_ATTRS, None)["od_municipal"] is None

    def test_zone_counts_from_attrs(self):
        od = xr.Dataset(
            {
                "zone_population": (("zone",), np.array([1.0, np.nan])),
                "mobility": (("date",), np.zeros(3)),
            }
        ).assign_coords(zone=("zone", ["z1", "z2"]))
        od.attrs = dict(OD_MITMA_ATTRS)
        counts = mitma_zone_counts(od)
        assert counts["built"] == 2
        assert counts["zone_population_nan"] == 1
        assert counts["reference_zone_counts"]["transform_attr"] == 458

    def test_coverage_counts_and_registry_gap(self):
        admissions = xr.Dataset(
            {"stratum_pop_sum": (("date", "abs_id"), np.zeros((1, 4)))}
        ).assign_coords(abs_id=("abs_id", ["001", "002", "009", "302"]))
        deaths = xr.Dataset(
            {"population": (("comarca_id",), np.array([1.0, 1.0, 1.0]))}
        ).assign_coords(comarca_id=("comarca_id", ["c1", "c2", "00"]))
        coverage = observed_coverage(
            self._nodes(),
            admissions,
            deaths,
            registry_ids={
                "abs": {"001", "002", "009", "302"},
                "comarca": {"c1", "c2", "00"},
                "municipality": {"m1", "25017", "25123"},
            },
            municipal_observed_ids={"m1"},
        )
        assert coverage["abs"]["observed_store_columns"] == 4
        assert coverage["abs"]["graph_nodes"] == 3
        assert coverage["abs"]["store_not_in_graph"] == ["009", "302"]
        assert coverage["abs"]["graph_not_observed"] == ["003"]
        assert coverage["comarca"]["observed_mapped"] == 2
        assert coverage["comarca"]["store_not_in_graph"] == ["00"]
        assert coverage["comarca"]["graph_not_observed"] == []
        assert coverage["municipality"]["observed"] == 1
        assert coverage["municipality"]["graph_nodes"] == 1
        missing = coverage["registry_codes_without_graph_presence"]
        assert missing["total"] == 5  # 302 + 009 + 00 + 25017 + 25123
        assert missing["abs"] == ["009", "302"]


class TestQcReportSynthetic:
    REGISTRY_IDS = {
        "abs": {"001", "002", "302"},
        "comarca": {"c1", "c2", "00"},
        "municipality": {"m1", "25017", "25123"},
    }

    def test_full_report(self, tmp_path):
        root, spec = _mini_support(tmp_path)
        report = build_report(
            root,
            spec,
            municipal_observed_ids={"m1"},
            registry_ids=self.REGISTRY_IDS,
        )
        assert report["errors"] == []
        assert report["manifest"]["confirmation_period"] == CONFIRMATION_PERIOD
        assert report["abs_population"]["n_ratio_above_one"] == 1
        assert report["abs_population"]["worst_above_one"] == {
            "002": pytest.approx(1.5)
        }
        assert report["comarca_population_closure"]["n_mismatch"] == 1
        assert report["comarca_population_closure"][
            "max_abs_rel_error"
        ] == pytest.approx(0.1)
        assert report["mitma_zone_counts"]["expected"]["built"] == 456
        assert any("built/report 1 != expected 456" in w for w in report["warnings"])
        assert (
            report["od_mass_retention"]["od_municipal"]["policies"][
                "drop|min_flow_raw=10"
            ]["retained_mass_fraction"]
            == 0.424242
        )
        assert (
            report["observed_coverage"]["registry_codes_without_graph_presence"][
                "total"
            ]
            == 4
        )
        assert "1/1 od_mitma zones" in report["open_items"][0]
        assert report["observed_coverage"]["municipality"]["observed"] == 1
        text = format_report(report)
        assert "native_support QC" in text
        assert "confirmation_period: ['2020-12-31', '2021-01-27']" in text

    def test_report_without_municipal_ids(self, tmp_path):
        root, spec = _mini_support(tmp_path)
        report = build_report(
            root,
            spec,
            real_with_id=tmp_path / "missing.zarr",
            registry_ids=self.REGISTRY_IDS,
        )
        assert report["observed_coverage"]["municipality"]["observed"] is None
        assert any(
            "municipal observed ids unavailable" in w for w in report["warnings"]
        )


def _real_available() -> bool:
    spec = InputPaths()
    inputs = [
        spec.abs_csv,
        spec.deaths_csv,
        spec.inputs_v1_dvc,
        spec.abs_shp,
        spec.municipal_geojson,
        spec.mpis_csv,
        spec.population_csv,
        spec.comarques_geojson,
        spec.reference_csv,
        spec.distritos_shp(),
        spec.igcg_shp(),
        spec.geostat_csv(),
        spec.grid_geom_shp(),
        spec.daily_dir(),
    ]
    root = Path("data/processed/native_support")
    stores = [
        "od_mitma",
        "od_municipal",
        "native_graph",
        "obs_admissions",
        "obs_deaths",
    ]
    return (
        root.is_dir()
        and all(Path(p).exists() for p in inputs)
        and all((root / f"{s}.zarr").is_dir() for s in stores)
        and (root / "overlap_edges.parquet").is_file()
    )


@pytest.mark.skipif(
    not _real_available(), reason="real native-support artifacts or inputs missing"
)
def test_real_manifest_and_qc_readonly():
    spec = InputPaths()
    manifest = build_manifest("data/processed/native_support", spec)
    assert manifest["schema"] == SCHEMA
    assert manifest["confirmation_period"] == CONFIRMATION_PERIOD
    daily = manifest["inputs"]["dasymetric_readonly"]["mitma_v1_daily"]
    assert daily["file_count"] == daily["expected_file_count"] == 451
    assert manifest["produced"]["overlap_edges"]["bytes"] > 0

    report = build_report("data/processed/native_support", spec, manifest=manifest)
    zones = report["mitma_zone_counts"]
    assert zones["built"] == 456 and zones["date_count"] == 451
    assert report["abs_population"]["n_abs_observed"] == 377
    assert report["comarca_population_closure"]["n_matched"] == 42
    assert report["comarca_population_closure"]["max_abs_rel_error"] == pytest.approx(
        0.0, abs=1e-12
    )
    coverage = report["observed_coverage"]
    missing = coverage["registry_codes_without_graph_presence"]
    assert missing["total"] == 4 and "302" in missing["abs"]
    assert coverage["abs"]["observed_store_columns"] == 377
    assert coverage["abs"]["graph_nodes"] == 379
    assert any("107/456" in item for item in report["open_items"])
    retention = report["od_mass_retention"]["od_mitma"]["policies"]
    assert retention["retain|min_flow_raw=0"]["retained_mass_fraction"] == 1.0
