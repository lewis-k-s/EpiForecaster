from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.config import PreprocessingConfig, REGION_COORD
from data.preprocess.processors.edar_processor import EDARProcessor
from data.preprocess import smoothing


def _write_dummy_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "cases": tmp_path / "cases.csv",
        "mobility": tmp_path / "mobility.nc",
        "population": tmp_path / "population.csv",
        "unused": tmp_path / "unused.csv",
    }
    paths["cases"].write_text("date,region,value\n")
    paths["population"].write_text("region_id,population\n")
    paths["unused"].write_text("dummy\n")
    paths["mobility"].write_bytes(b"\x89HDF\r\n\x1a\n")
    return paths


def _write_mapping(tmp_path: Path) -> Path:
    mapping = xr.DataArray(
        np.array([[1.0]], dtype=np.float32),
        dims=["edar_id", "home"],
        coords={"edar_id": ["EDAR1"], "home": ["REG1"]},
    )
    mapping_path = tmp_path / "edar_mapping.nc"
    mapping.to_netcdf(mapping_path)
    return mapping_path


def _write_wastewater_csv(path: Path, *, include_ld: bool) -> None:
    records = [
        {
            "id mostra": "EDAR1-2022-01-01",
            "Cabal últimes 24h(m3)": 10.0,
            "IP4(CG/L)": np.nan,
            "N1(CG/L)": 100.0,
            "N2(CG/L)": np.nan,
            "LD(CG/L)": 80.0,
        },
        {
            "id mostra": "EDAR1-2022-01-02",
            "Cabal últimes 24h(m3)": 10.0,
            "IP4(CG/L)": np.nan,
            "N1(CG/L)": 50.0,
            "N2(CG/L)": np.nan,
            "LD(CG/L)": 80.0,
        },
        {
            "id mostra": "EDAR1-2022-01-03",
            "Cabal últimes 24h(m3)": 10.0,
            "IP4(CG/L)": np.nan,
            "N1(CG/L)": 200.0,
            "N2(CG/L)": np.nan,
            "LD(CG/L)": 80.0,
        },
    ]
    df = pd.DataFrame.from_records(records)
    if not include_ld:
        df = df.drop(columns=["LD(CG/L)"])
    df.to_csv(path, index=False)


def _make_config(tmp_path: Path, wastewater_path: Path) -> PreprocessingConfig:
    paths = _write_dummy_files(tmp_path)
    return PreprocessingConfig(
        data_dir=str(tmp_path),
        cases_file=str(paths["cases"]),
        mobility_path=str(paths["mobility"]),
        wastewater_file=str(wastewater_path),
        population_file=str(paths["population"]),
        region_metadata_file=str(paths["unused"]),
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 10),
        output_path=str(tmp_path),
        dataset_name="test",
        forecast_horizon=7,
        sequence_length=1,
    )


def _expected_edar_output(
    processor: EDARProcessor,
    wastewater_file: Path,
    mapping_path: Path,
) -> xr.Dataset:
    wastewater_df = processor._load_wastewater_data(str(wastewater_file))
    selected_data = processor._select_variants(wastewater_df)
    aggregated_data = processor._remove_duplicates_and_aggregate(selected_data)
    flow_data = processor._calculate_flow_rates(aggregated_data)
    daily_data = processor._resample_to_daily(flow_data)
    daily_data = processor._apply_tobit_kalman(daily_data)

    daily_data_xr = daily_data.set_index(["date", "edar_id", "variant"])[
        "total_covid_flow"
    ].to_xarray()
    # Get censor flags if available
    if "censor_flag" in daily_data.columns:
        daily_data_xr_censor = daily_data.set_index(["date", "edar_id", "variant"])[
            "censor_flag"
        ].to_xarray()
    else:
        daily_data_xr_censor = daily_data_xr * 0  # All uncensored

    mapping = xr.open_dataarray(mapping_path).fillna(0)
    mapping = mapping.rename({"home": REGION_COORD})
    daily_data_xr = daily_data_xr.assign_coords(
        edar_id=daily_data_xr["edar_id"].astype(str)
    )
    daily_data_xr_censor = daily_data_xr_censor.assign_coords(
        edar_id=daily_data_xr_censor["edar_id"].astype(str)
    )
    mapping = mapping.assign_coords(edar_id=mapping["edar_id"].astype(str))
    daily_data_xr, mapping = xr.align(daily_data_xr, mapping, join="inner")
    daily_data_xr_censor, _ = xr.align(daily_data_xr_censor, mapping, join="inner")
    mask = daily_data_xr.notnull()
    weighted_sum = xr.dot(daily_data_xr.fillna(0), mapping, dim="edar_id")
    contribution_count = xr.dot(
        mask.astype(float), mapping.astype(bool).astype(float), dim="edar_id"
    )
    result = weighted_sum / contribution_count.where(contribution_count > 0, 1)

    outputs: dict[str, xr.DataArray] = {}
    for variant in result["variant"].values.tolist():
        variant_da = result.sel(variant=variant).drop_vars("variant")
        variant_name = f"edar_biomarker_{variant}"
        variant_da.name = variant_name
        outputs[variant_name] = variant_da

        # Mask channel: 1.0 if measured, 0.0 otherwise
        mask = xr.where(variant_da.notnull() & (variant_da > 0), 1.0, 0.0)
        outputs[f"{variant_name}_mask"] = mask

        # Censor flag channel: 0=uncensored, 1=censored, 2=missing
        variant_censor = daily_data_xr_censor.sel(variant=variant).drop_vars("variant")
        # Aggregate censor flags from EDAR sites to region using max
        variant_censor_filled = variant_censor.fillna(0)
        region_censor = variant_censor_filled.where(mapping > 0).max(dim="edar_id")
        region_censor = region_censor.fillna(0)
        outputs[f"{variant_name}_censor"] = region_censor

        # Age channel: normalized days since last measurement
        T = len(variant_da["date"])
        N = len(variant_da[REGION_COORD])
        age_data = np.zeros((T, N), dtype=np.float32)
        for n in range(N):
            last_seen = -1
            for t in range(T):
                if mask.values[t, n] > 0:
                    last_seen = t
                    age_data[t, n] = 0.0
                elif last_seen >= 0:
                    age_data[t, n] = min(t - last_seen, 14) / 14.0
                else:
                    age_data[t, n] = 1.0
        age_da = xr.DataArray(
            age_data,
            coords={
                "date": variant_da["date"].values,
                REGION_COORD: variant_da[REGION_COORD].values,
            },
            dims=["date", REGION_COORD],
        )
        outputs[f"{variant_name}_age"] = age_da

    # Add run_id dimension to match real data format
    # Real data gets run_id="real" to distinguish from synthetic runs
    result_ds = xr.Dataset(outputs)
    result_ds = result_ds.expand_dims(run_id=["real"])
    return result_ds


@pytest.mark.epiforecaster
def test_edar_processor_applies_tobit_kalman(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mapping_path = _write_mapping(tmp_path)
    wastewater_path = tmp_path / "wastewater.csv"
    _write_wastewater_csv(wastewater_path, include_ld=True)

    config = _make_config(tmp_path, wastewater_path)
    processor = EDARProcessor(config)

    # Patch fit_kalman_params in the smoothing module
    original_fit_kalman_params = smoothing.fit_kalman_params
    monkeypatch.setattr(smoothing, "fit_kalman_params", lambda series: (0.01, 0.04))

    try:
        result = processor.process(str(wastewater_path), str(mapping_path))
        expected = _expected_edar_output(processor, wastewater_path, mapping_path)
        xr.testing.assert_allclose(result, expected)
    finally:
        # Restore original function
        smoothing.fit_kalman_params = original_fit_kalman_params


@pytest.mark.epiforecaster
def test_edar_processor_skips_tobit_without_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mapping_path = _write_mapping(tmp_path)
    wastewater_path = tmp_path / "wastewater_no_ld.csv"
    _write_wastewater_csv(wastewater_path, include_ld=False)

    config = _make_config(tmp_path, wastewater_path)
    processor = EDARProcessor(config)

    # Patch fit_kalman_params in the smoothing module
    original_fit_kalman_params = smoothing.fit_kalman_params
    monkeypatch.setattr(smoothing, "fit_kalman_params", lambda series: (0.01, 0.04))

    try:
        result = processor.process(str(wastewater_path), str(mapping_path))
        expected = _expected_edar_output(processor, wastewater_path, mapping_path)
        xr.testing.assert_allclose(result, expected)
    finally:
        # Restore original function
        smoothing.fit_kalman_params = original_fit_kalman_params
