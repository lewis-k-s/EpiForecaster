"""Factorized EpiSim mobility layout in the synthetic preprocess.

The corpus stores one base edge series per mobility profile plus the per-run,
per-region NPI reduction; the loader rebuilds the dense origin-major tensor by
the ``max_endpoint_self_loop`` v1 rule. Fixtures are written as NetCDF.
"""

import json

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.pipeline import OfflinePreprocessingPipeline
from data.preprocess.processors.synthetic_processor import (
    SyntheticProcessor,
    apply_max_endpoint_self_loop,
    parse_mobility_reduction_rule,
    validate_mobility_edges,
)
from tests.test_synthetic_preprocess_chunking import (
    _make_config,
    _minimal_synthetic_dataset,
    _write_region_metadata,
)

DATES = pd.date_range("2020-01-01", periods=4, freq="D")
REGION_IDS = np.array(["r1", "r2", "r3", "r4"])
EDAR_IDS = np.array(["e1"])
# Unpadded run ids of different lengths. Twins share a profile.
RUN_IDS = np.array(["reactive_3", "baseline_3", "reactive_17"])
PROFILE_IDS = np.array([3, 3, 17], dtype=np.int32)
BASELINE_RUN = 1
PROFILES = np.array([3, 17], dtype=np.int32)
# Self-loops for r1 and r2 are inline CSV edges; the engine appends the r3 and
# r4 self-loops at the end. r4 has no off-diagonal edges.
EDGE_ORIGIN = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3], dtype=np.int32)
EDGE_DESTINATION = np.array([0, 1, 2, 0, 1, 3, 0, 1, 3, 2, 3], dtype=np.int32)
EPISIM_RULE = {
    "name": "max_endpoint_self_loop",
    "version": 1,
    "base": "mobility_base_edges",
    "profile_index": "synthetic_profile_id",
    "edge_origin": "edge_origin",
    "edge_destination": "edge_destination",
    "region_params": {"m": "npi_mobility_reduction"},
    "edge_features": {},
    "redistribution": "source_self_loop",
    "day_alignment": "npi_mobility_reduction[run, date t] applies to "
    "mobility_base_edges[profile, date t]",
    "formula": "documentation only",
}


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _reference_dense(base, reduction, origin, destination, n_regions):
    """Independent float64 loop over edges for one run: (date, origin, dest)."""
    out = np.zeros((base.shape[0], n_regions, n_regions), dtype=np.float64)
    for t in range(base.shape[0]):
        for e, (i, j) in enumerate(zip(origin, destination)):
            weight = float(base[t, e])
            if i == j:
                out[t, i, i] += weight
                continue
            kept = weight * (1.0 - max(reduction[t, i], reduction[t, j]))
            out[t, i, j] += kept
            out[t, i, i] += weight - kept
    return out


def _write_factorized_corpus(
    path,
    *,
    include_dense=True,
    dense_transposed=False,
    swap_edge_endpoints=False,
    profile_ids=PROFILE_IDS,
    rule=EPISIM_RULE,
    attrs=None,
):
    """Write a NetCDF corpus with the factorized layout.

    Returns the reference dense mobility ``(run, corpus date, origin, dest)``.
    """
    rng = np.random.default_rng(7)
    ds = _minimal_synthetic_dataset(
        run_ids=RUN_IDS,
        dates=DATES,
        region_ids=REGION_IDS,
        edar_ids=EDAR_IDS,
    )
    corpus_dates = ds["date"].values
    n_dates, n_regions = len(corpus_dates), len(REGION_IDS)

    base = rng.uniform(0.05, 1.0, size=(len(PROFILES), n_dates, len(EDGE_ORIGIN)))
    row_sums = np.zeros((len(PROFILES), n_dates, n_regions))
    for e, i in enumerate(EDGE_ORIGIN):
        row_sums[:, :, i] += base[:, :, e]
    base = (base / row_sums[:, :, EDGE_ORIGIN]).astype(np.float32)

    reduction = rng.uniform(0.0, 0.9, size=(len(RUN_IDS), n_dates, n_regions))
    reduction[BASELINE_RUN] = 0.0
    reduction = reduction.astype(np.float32)

    profile_position = {int(p): k for k, p in enumerate(PROFILES)}
    expected = np.stack(
        [
            _reference_dense(
                base[profile_position[int(p)]],
                reduction[r],
                EDGE_ORIGIN,
                EDGE_DESTINATION,
                n_regions,
            )
            for r, p in enumerate(PROFILE_IDS)
        ]
    ).astype(np.float32)

    # Mirror the EpiSim writer dims for the region-pair variables.
    ds = ds.drop_vars(["mobility_base", "origin", "target"])
    ds["mobility_base"] = (
        ("in_region_id", "out_region_id"),
        np.full((n_regions, n_regions), 1.0 / n_regions, dtype=np.float32),
    )
    if include_dense:
        dense = expected.transpose(0, 1, 3, 2) if dense_transposed else expected
        ds["mobility_time_varying"] = (
            ("run_id", "date", "in_region_id", "out_region_id"),
            dense,
        )

    origin, destination = EDGE_ORIGIN, EDGE_DESTINATION
    if swap_edge_endpoints:
        origin, destination = destination, origin
    ds = ds.assign_coords(
        profile=("profile", PROFILES),
        edge=("edge", np.arange(len(EDGE_ORIGIN), dtype=np.int32)),
    )
    ds["mobility_base_edges"] = (("profile", "date", "edge"), base)
    ds["edge_origin"] = (("edge",), origin)
    ds["edge_destination"] = (("edge",), destination)
    ds["synthetic_profile_id"] = (("run_id",), np.asarray(profile_ids, np.int32))
    ds["npi_mobility_reduction"] = (("run_id", "date", "region_id"), reduction)
    ds["synthetic_observation_density_scale"] = (
        ("run_id",),
        np.array([1.0, 0.5, 2.0], dtype=np.float32),
    )
    for name in ("latent_S_true", "latent_I_true", "latent_CH_true"):
        ds[name].attrs["role"] = "audit_only"
    ds.attrs.update(
        {
            "mobility_reduction_rule": _canonical_json(rule),
            "model_facing_targets": "sirhd_S,sirhd_I,sirhd_H,sirhd_R,sirhd_D,"
            "sirhd_death_flow",
            "audit_only_variables": "latent_S_true,latent_I_true,latent_CH_true",
            "simulation_source_commit": "b354b27",
            "reporting_profile": "sirhd_pretraining_reporting_v2",
            **(attrs or {}),
        }
    )
    ds.to_netcdf(path, engine="netcdf4")
    return expected


def _processor(tmp_path, synthetic_path):
    return SyntheticProcessor(
        _make_config(tmp_path, synthetic_path, options={"date_chunk_size": 2})
    )


@pytest.mark.epiforecaster
def test_factorized_mobility_matches_dense_layout(tmp_path):
    path = tmp_path / "raw.nc"
    expected = _write_factorized_corpus(path)
    processor = _processor(tmp_path, path)

    result = processor.process(str(path))
    mobility = result["mobility"]["mobility"]

    assert mobility.dims == ("run_id", "date", "origin", "destination")
    assert isinstance(mobility.data, da.Array)
    # One run and one date chunk per block: no dense corpus tensor in memory.
    assert mobility.data.chunks == ((1, 1, 1), (2, 2), (4,), (4,))
    np.testing.assert_allclose(mobility.values, expected[:, 1:], rtol=0, atol=1e-6)
    np.testing.assert_allclose(mobility.values.sum(axis=3), 1.0, atol=1e-5)

    provenance = result["synthetic_corpus_provenance"]
    assert provenance["mobility_layout"] == "factorized_edges"
    assert json.loads(provenance["mobility_reduction_rule"]) == EPISIM_RULE
    qa = provenance["mobility_qa"]
    assert qa["dense_parity_checked"] is True
    assert qa["max_abs_error"] <= qa["parity_atol"]
    # The fixture is asymmetric, so the transposed comparison must fail.
    assert qa["swapped_max_abs_error"] > 1e-2
    assert provenance["simulation_source_commit"] == "b354b27"
    assert provenance["reporting_profile"] == "sirhd_pretraining_reporting_v2"


@pytest.mark.epiforecaster
def test_factorized_mobility_preferred_without_dense_layout(tmp_path):
    path = tmp_path / "raw.nc"
    expected = _write_factorized_corpus(path, include_dense=False)

    result = _processor(tmp_path, path).process(str(path))

    np.testing.assert_allclose(
        result["mobility"]["mobility"].values, expected[:, 1:], rtol=0, atol=1e-6
    )
    qa = result["synthetic_corpus_provenance"]["mobility_qa"]
    assert qa["dense_parity_checked"] is False
    assert qa["max_row_sum_error"] <= qa["row_sum_atol"]


@pytest.mark.epiforecaster
def test_self_loop_receives_removed_weight():
    origin = np.array([0, 0, 0, 1, 2, 2])
    destination = np.array([0, 1, 2, 1, 2, 0])
    base = np.array([[0.5, 0.3, 0.2, 1.0, 0.6, 0.4]], dtype=np.float32)
    reduction = np.array([[0.5, 0.1, 0.25]], dtype=np.float32)

    out = apply_max_endpoint_self_loop(base, reduction, origin, destination)

    np.testing.assert_allclose(
        out[0], [0.75, 0.15, 0.1, 1.0, 0.8, 0.2], rtol=0, atol=1e-7
    )
    row_sums = np.bincount(origin, weights=out[0], minlength=3)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-7)
    off = origin != destination
    assert np.all(out >= 0)
    assert np.all(out[:, off] <= base[:, off])


@pytest.mark.epiforecaster
def test_edge_list_requires_origin_self_loop():
    with pytest.raises(ValueError, match="needs a self-loop edge"):
        validate_mobility_edges(np.array([0, 1]), np.array([1, 1]), 2)
    with pytest.raises(ValueError, match="duplicate edges"):
        validate_mobility_edges(np.array([0, 0, 0]), np.array([0, 1, 1]), 2)


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("corpus_kwargs", "message"),
    [
        ({"dense_transposed": True}, "does not match mobility_time_varying"),
        ({"swap_edge_endpoints": True}, "rows must sum to 1"),
    ],
)
def test_swapped_orientation_fails_qa(tmp_path, corpus_kwargs, message):
    path = tmp_path / "raw.nc"
    _write_factorized_corpus(path, **corpus_kwargs)

    with pytest.raises(ValueError, match=message):
        _processor(tmp_path, path).process(str(path))


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({**EPISIM_RULE, "name": "max_endpoint"}, "rule name"),
        ({**EPISIM_RULE, "version": 2}, "rule version"),
        ({**EPISIM_RULE, "version": True}, "rule version"),
        ({**EPISIM_RULE, "version": "1"}, "rule version"),
    ],
)
def test_rejects_unknown_reduction_rule(tmp_path, rule, message):
    path = tmp_path / "raw.nc"
    _write_factorized_corpus(path, rule=rule)

    with pytest.raises(ValueError, match=message):
        _processor(tmp_path, path).process(str(path))


@pytest.mark.epiforecaster
def test_reduction_rule_ignores_documentation_fields():
    rule = {**EPISIM_RULE, "formula": "anything", "day_alignment": "t + 1"}
    attrs = {"mobility_reduction_rule": _canonical_json(rule)}
    assert parse_mobility_reduction_rule(attrs) == rule
    with pytest.raises(ValueError, match="global attribute"):
        parse_mobility_reduction_rule({})


@pytest.mark.epiforecaster
def test_rejects_run_without_profile(tmp_path):
    path = tmp_path / "raw.nc"
    _write_factorized_corpus(path, profile_ids=np.array([3, -1, 17]))
    processor = _processor(tmp_path, path)

    with pytest.raises(ValueError, match=r"synthetic_profile_id is -1.*baseline_3"):
        processor.process(str(path))

    # A run without a profile is acceptable when no mobility is rebuilt for it.
    result = processor.process(str(path), run_filter=["reactive_3", "reactive_17"])
    assert list(result["mobility"]["mobility"].run_id.values) == [
        "reactive_3",
        "reactive_17",
    ]


@pytest.mark.epiforecaster
def test_rejects_profile_missing_from_base(tmp_path):
    path = tmp_path / "raw.nc"
    _write_factorized_corpus(path, profile_ids=np.array([3, 3, 99]))

    with pytest.raises(ValueError, match=r"\[99\] are not on"):
        _processor(tmp_path, path).process(str(path))


@pytest.mark.epiforecaster
def test_unpadded_run_ids_round_trip(tmp_path):
    path = tmp_path / "raw.nc"
    expected = _write_factorized_corpus(path)

    result = _processor(tmp_path, path).process(
        str(path), run_filter=["reactive_17", "baseline_3"]
    )
    mobility = result["mobility"]["mobility"]

    assert mobility.run_id.values.tolist() == ["baseline_3", "reactive_17"]
    assert result["cases"]["cases"].run_id.values.tolist() == [
        "baseline_3",
        "reactive_17",
    ]
    np.testing.assert_allclose(mobility.values, expected[1:, 1:], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(
        result["synthetic_observation_density_scale"].values, [0.5, 2.0]
    )
    np.testing.assert_array_equal(result["synthetic_profile_id"].values, [3, 17])


@pytest.mark.epiforecaster
def test_rejects_audit_only_model_target(tmp_path):
    path = tmp_path / "raw.nc"
    _write_factorized_corpus(
        path, attrs={"audit_only_variables": "latent_S_true,sirhd_I"}
    )

    with pytest.raises(ValueError, match=r"\['sirhd_I'\] as audit-only"):
        _processor(tmp_path, path).process(str(path))


@pytest.mark.epiforecaster
def test_rejects_latent_target_missing_from_model_facing_targets(tmp_path):
    path = tmp_path / "raw.nc"
    _write_factorized_corpus(
        path, attrs={"model_facing_targets": "sirhd_S,sirhd_I,sirhd_H,sirhd_R"}
    )

    with pytest.raises(ValueError, match="sirhd_D', 'sirhd_death_flow'"):
        _processor(tmp_path, path).process(str(path))


@pytest.mark.epiforecaster
@pytest.mark.integration
def test_pipeline_end_to_end_factorized_netcdf(tmp_path):
    path = tmp_path / "raw.nc"
    region_metadata_path = tmp_path / "edar_map.nc"
    expected = _write_factorized_corpus(path)
    _write_region_metadata(
        region_metadata_path, region_ids=REGION_IDS, edar_ids=EDAR_IDS
    )
    config = _make_config(tmp_path, path, options={"date_chunk_size": 2})
    config.region_metadata_file = str(region_metadata_path)
    config.continuous_storage_dtype = "float32"

    output_path = OfflinePreprocessingPipeline(config).run()

    with xr.open_dataset(output_path, engine="netcdf4") as ds:
        mobility = ds["mobility"].sel(
            run_id=RUN_IDS, origin=REGION_IDS, destination=REGION_IDS
        )
        np.testing.assert_allclose(
            mobility.values, np.log1p(expected[:, 1:]), rtol=0, atol=1e-6
        )
        np.testing.assert_array_equal(
            ds["synthetic_observation_density_scale"].sel(run_id=RUN_IDS).values,
            [1.0, 0.5, 2.0],
        )
        # Twins share a profile ID in the processed dataset.
        np.testing.assert_array_equal(
            ds["synthetic_profile_id"].sel(run_id=RUN_IDS).values, PROFILE_IDS
        )
        provenance = json.loads(ds.attrs["synthetic_corpus_provenance_json"])
    assert provenance["mobility_layout"] == "factorized_edges"
    assert provenance["mobility_qa"]["dense_parity_checked"] is True
    assert provenance["simulation_source_commit"] == "b354b27"
