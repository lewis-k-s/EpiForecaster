"""Tests for the municipal-projection OD store builder (slice MOB-1).

Unit tests run on a synthetic 3-day x 4-zone OD tensor with hand-computed
audit expectations; integration tests exercise the real
``data/files/mobility.zarr`` lazily (metadata + 2-day samples only) and are
skipped when the real inputs are absent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.native_sources.od_municipal import (
    build_od_municipal_store,
    compute_od_audit,
    policy_key,
)
from utils.logging import suppress_zarr_warnings

suppress_zarr_warnings()

pytestmark = pytest.mark.epiforecaster

ZONES = ["08001", "08002", "08003", "99999"]
REFERENCE_CODES = ["08001", "08002", "08003", "08005"]
FOOTPRINT_REGIONS = ["08001", "08002"]
DATES = pd.date_range("2021-03-01", periods=3)

# Asymmetric raw-trip matrices (direction must survive), diagonal self-flow
# present, zeros inside. Hand-computed audits rely on these exactly:
# per-day totals 122 / 97 / 117, diagonal sums 100 / 80 / 96, total mass 336.
RAW_DAYS = np.array(
    [
        [[10, 0, 5, 0], [1, 20, 0, 7], [0, 3, 30, 0], [2, 0, 4, 40]],
        [[8, 2, 0, 1], [0, 16, 3, 0], [4, 0, 24, 2], [0, 5, 0, 32]],
        [[12, 0, 0, 3], [2, 4, 1, 0], [0, 6, 36, 0], [1, 0, 8, 44]],
    ],
    dtype=np.float64,
)
TOTAL_MASS = 336.0
DIAG_FRACTION = float(np.mean([100 / 122, 80 / 97, 96 / 117]))
RETAINED = {  # policy key -> (retained mass fraction, retained edge count)
    policy_key("retain", 0.0): (1.0, 30),  # positive entries: 10 + 10 + 10
    policy_key("retain", 10.0): (264 / 336, 10),  # entries >= 10: 4 + 3 + 3
    policy_key("drop", 0.0): (60 / 336, 18),  # positive off-diagonal: 6 + 6 + 6
    policy_key("drop", 10.0): (0.0, 0),  # no off-diagonal entry >= 10
}


def _write_raw_store(path: Path, tensor: np.ndarray, zones: list[str]) -> Path:
    coords = np.array(zones, dtype=object)
    xr.Dataset(
        {"mobility": (("date", "origin", "target"), tensor)},
        coords={
            "date": pd.date_range("2021-03-01", periods=tensor.shape[0]),
            "origin": coords,
            "target": coords,
        },
    ).to_zarr(path, mode="w")
    return path


def _build(root, raw_store, reference_csv, cases_store):
    out = Path(root) / "od_municipal.zarr"
    build_od_municipal_store(
        source=raw_store,
        output=out,
        reference_csv=reference_csv,
        cases_store=cases_store,
    )
    return out


@pytest.fixture
def raw_store(tmp_path):
    return _write_raw_store(tmp_path / "mobility.zarr", RAW_DAYS, ZONES)


@pytest.fixture
def reference_csv(tmp_path):
    path = tmp_path / "municipality_id_reference.csv"
    pd.DataFrame({"muni_code_5digit": REFERENCE_CODES}).to_csv(path, index=False)
    return path


@pytest.fixture
def cases_store(tmp_path):
    path = tmp_path / "real_with_id.zarr"
    xr.Dataset(
        coords={"region_id": np.array(FOOTPRINT_REGIONS, dtype=object)}
    ).to_zarr(path, mode="w")
    return path


@pytest.fixture
def built_store(tmp_path, raw_store, reference_csv, cases_store):
    return _build(tmp_path / "build", raw_store, reference_csv, cases_store)


@pytest.fixture
def built(built_store):
    with xr.open_zarr(built_store) as ds:
        yield ds


class TestTransform:
    def test_log1p_float16_round_trip(self, built):
        stored = built["mobility"].values
        assert stored.dtype == np.float16
        np.testing.assert_allclose(
            stored.astype(np.float64), np.log1p(RAW_DAYS), atol=1e-2
        )
        decoded = np.expm1(stored.astype(np.float64))
        np.testing.assert_allclose(decoded, RAW_DAYS, rtol=1e-2, atol=1e-3)

    def test_no_threshold_or_self_flow_removal_at_rest(self, built):
        decoded = np.expm1(built["mobility"].values.astype(np.float64))
        # Sub-threshold raw flows (< 10) must be stored, not filtered ...
        assert decoded[0, 0, 2] == pytest.approx(5.0, abs=1e-2)
        assert decoded[2, 3, 2] == pytest.approx(8.0, abs=0.1)
        # ... and diagonal self-flows must be stored, not dropped.
        assert decoded[0, 0, 0] == pytest.approx(10.0, abs=1e-1)
        assert decoded[2, 3, 3] == pytest.approx(44.0, abs=0.5)

    def test_negative_raw_values_clipped(self, tmp_path, reference_csv, cases_store):
        tensor = np.array([[[2.0, -3.5], [0.0, 4.0]]], dtype=np.float64)
        raw = _write_raw_store(tmp_path / "neg.zarr", tensor, ["08001", "08002"])
        out = _build(tmp_path / "neg_build", raw, reference_csv, cases_store)
        stored = xr.open_zarr(out)["mobility"].values[0]
        np.testing.assert_allclose(
            np.expm1(stored.astype(np.float64)),
            [[2.0, 0.0], [0.0, 4.0]],
            atol=1e-2,
        )


class TestDirection:
    def test_origin_to_target_never_symmetrized(self, built):
        stored = built["mobility"].values.astype(np.float64)
        # raw[day0, 08001 -> 08003] = 5 while raw[day0, 08003 -> 08001] = 0.
        assert np.expm1(stored[0, 0, 2]) == pytest.approx(5.0, abs=1e-2)
        assert stored[0, 2, 0] == 0.0
        # Elementwise equality with log1p(raw) rules out any transpose/mean.
        np.testing.assert_allclose(stored, np.log1p(RAW_DAYS), atol=1e-2)


class TestAudit:
    def test_diag_mass_fraction_hand_computed(self, built):
        assert built.attrs["diag_mass_fraction"] == pytest.approx(
            DIAG_FRACTION, rel=1e-12
        )

    def test_retained_mass_and_edge_count_per_policy(self, built):
        for key, (fraction, edges) in RETAINED.items():
            assert built.attrs["retained_mass_fraction"][key] == pytest.approx(
                fraction, rel=1e-12
            )
            assert built.attrs["edge_count"][key] == edges

    def test_audit_pure_function_on_bare_array(self):
        raw = xr.DataArray(RAW_DAYS, dims=("date", "origin", "target"))
        audit = compute_od_audit(raw)
        assert audit["diag_mass_fraction"] == pytest.approx(DIAG_FRACTION, rel=1e-12)
        assert audit["total_raw_mass"] == pytest.approx(TOTAL_MASS, rel=1e-12)
        for key, (fraction, edges) in RETAINED.items():
            assert audit["retained_mass_fraction"][key] == pytest.approx(
                fraction, rel=1e-12
            )
            assert audit["edge_count"][key] == edges


class TestZoneRegistry:
    def test_zone_id_registry_variable(self, built):
        assert [str(z) for z in built["zone_id"].values] == ZONES

    def test_zone_count_reconciliation(self, built):
        rec = built.attrs["zone_count_reconciliation"]
        assert rec["od_zone_count"] == 4
        assert rec["reference_count"] == 4
        assert rec["matched"] == 3
        assert rec["od_unmatched_ids"] == ["99999"]
        assert rec["reference_unmatched_ids"] == ["08005"]

    def test_cases_footprint_mask(self, built):
        mask = built["zone_in_cases_footprint"].values
        assert mask.dtype == np.bool_
        assert mask.tolist() == [True, True, False, False]


class TestSchema:
    def test_schema_matches_contract(self, built):
        assert dict(built.sizes) == {"date": 3, "origin": 4, "target": 4, "zone": 4}
        assert built["mobility"].dims == ("date", "origin", "target")
        assert built["mobility"].dtype == np.float16
        assert set(built.data_vars) == {
            "mobility",
            "zone_id",
            "zone_in_cases_footprint",
        }
        assert [str(z) for z in built["origin"].values] == ZONES
        assert [str(z) for z in built["target"].values] == ZONES
        np.testing.assert_array_equal(built["date"].values, DATES.values)
        for key in (
            "diag_mass_fraction",
            "retained_mass_fraction",
            "edge_count",
            "zone_count_reconciliation",
            "policy_semantics",
            "date_count",
        ):
            assert key in built.attrs

    def test_idempotent_rebuild(
        self, tmp_path, raw_store, reference_csv, cases_store
    ):
        outs = [
            _build(tmp_path / f"build{i}", raw_store, reference_csv, cases_store)
            for i in (1, 2)
        ]
        first, second = (xr.open_zarr(p) for p in outs)
        assert dict(first.sizes) == dict(second.sizes)
        for name in ("origin", "target", "zone_id", "date"):
            assert [str(v) for v in first[name].values] == [
                str(v) for v in second[name].values
            ]
        assert first["mobility"].dtype == second["mobility"].dtype
        np.testing.assert_array_equal(
            first["mobility"].values, second["mobility"].values
        )
        assert (
            first["zone_in_cases_footprint"].values.tolist()
            == second["zone_in_cases_footprint"].values.tolist()
        )
        assert first.attrs == second.attrs

    def test_rebuild_same_path_overwrites(
        self, built_store, raw_store, reference_csv, cases_store
    ):
        build_od_municipal_store(
            source=raw_store,
            output=built_store,
            reference_csv=reference_csv,
            cases_store=cases_store,
        )
        with xr.open_zarr(built_store) as ds:
            assert dict(ds.sizes)["date"] == 3


REAL_SOURCE = Path("data/files/mobility.zarr")
REAL_REFERENCE = Path("data/files/municipality_id_reference.csv")
REAL_CASES = Path("data/processed/real_with_id.zarr")
REAL_FILES_PRESENT = (
    REAL_SOURCE.is_dir() and REAL_REFERENCE.is_file() and REAL_CASES.is_dir()
)


@pytest.mark.skipif(
    not REAL_FILES_PRESENT, reason="real mobility/reference/cases inputs absent"
)
class TestRealIntegration:
    def test_real_source_shape_and_coords(self):
        ds = xr.open_zarr(REAL_SOURCE)  # lazy: metadata only, no tensor load
        assert dict(ds.sizes) == {"date": 451, "origin": 963, "target": 963}
        assert str(ds["mobility"].dtype) == "float64"
        origin = [str(z) for z in ds["origin"].values]
        assert origin == [str(z) for z in ds["target"].values]
        assert all(len(z) == 5 for z in origin)

    def test_real_zone_reconciliation_and_footprint(self):
        from data.preprocess.native_sources.od_municipal import (
            cases_footprint_mask,
            load_raw_od,
            reconcile_zone_counts,
        )

        raw = load_raw_od(REAL_SOURCE)
        zones = raw["origin"].values
        rec = reconcile_zone_counts(zones, REAL_REFERENCE)
        # Verified against the real files 2026-09-21 (plan §10.2: 945/947).
        assert rec["matched"] == 945
        assert rec["od_unmatched_count"] == 18
        assert rec["reference_unmatched_count"] == 2
        mask = cases_footprint_mask(zones, REAL_CASES)
        assert mask.dtype == np.bool_
        assert int(mask.sum()) == 768

    def test_real_audit_smoke_on_two_day_sample(self):
        ds = xr.open_zarr(REAL_SOURCE)
        sample = ds["mobility"].isel(date=slice(0, 2))  # ~15 MB, not 3 GB
        audit = compute_od_audit(sample)
        assert 0.0 <= audit["diag_mass_fraction"] <= 1.0
        fractions = audit["retained_mass_fraction"]
        assert set(fractions) == {
            policy_key("retain", 0.0),
            policy_key("retain", 10.0),
            policy_key("drop", 0.0),
            policy_key("drop", 10.0),
        }
        for value in fractions.values():
            assert 0.0 <= value <= 1.0
        # Dropping self-flows or raising the threshold never retains more.
        assert (
            fractions[policy_key("drop", 0.0)] <= fractions[policy_key("retain", 0.0)]
        )
        assert (
            fractions[policy_key("drop", 10.0)]
            <= fractions[policy_key("retain", 10.0)]
        )
        assert (
            fractions[policy_key("retain", 10.0)]
            <= fractions[policy_key("retain", 0.0)] + 1e-12
        )
        assert audit["edge_count"][policy_key("drop", 10.0)] <= audit["edge_count"][
            policy_key("drop", 0.0)
        ]
