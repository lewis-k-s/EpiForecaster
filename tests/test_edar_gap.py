from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.edar_processor import EDARProcessor

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

def _write_wastewater_with_gap(path: Path) -> None:
    records = [
        {"id mostra": "EDAR1-2022-01-01", "Cabal últimes 24h(m3)": 10.0, "N1(CG/L)": 100.0, "N2(CG/L)": 0.0, "IP4(CG/L)": 0.0, "LD(CG/L)": 80.0},
        # Gap on 01-02
        {"id mostra": "EDAR1-2022-01-03", "Cabal últimes 24h(m3)": 10.0, "N1(CG/L)": 200.0, "N2(CG/L)": 0.0, "IP4(CG/L)": 0.0, "LD(CG/L)": 80.0},
    ]
    pd.DataFrame.from_records(records).to_csv(path, index=False)

@pytest.mark.epiforecaster
def test_edar_processor_marks_gap_as_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mapping_path = _write_mapping(tmp_path)
    wastewater_path = tmp_path / "wastewater_gap.csv"
    _write_wastewater_with_gap(wastewater_path)
    _write_dummy_files(tmp_path)

    config = PreprocessingConfig(
        data_dir=str(tmp_path),
        cases_file=str(tmp_path / "cases.csv"),
        mobility_path=str(tmp_path / "mobility.nc"),
        wastewater_file=str(wastewater_path),
        population_file=str(tmp_path / "population.csv"),
        region_metadata_file=str(mapping_path),
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 10), # enough range
        output_path=str(tmp_path),
        dataset_name="test",
        forecast_horizon=7,
    )
    
    processor = EDARProcessor(config)
    monkeypatch.setattr(processor, "_fit_kalman_params", lambda series: (0.01, 0.04))

    result = processor.process(str(wastewater_path), str(mapping_path))
    
    # Check Jan 2nd (the gap)
    # It should have mask=0 and age > 0
    mask = result.edar_biomarker_N1_mask.sel(run_id='real', date='2022-01-02', region_id='REG1').values
    age = result.edar_biomarker_N1_age.sel(run_id='real', date='2022-01-02', region_id='REG1').values
    
    assert mask == 0.0, "Gap should be masked"
    assert age > 0.0, "Gap should be stale"
    
    # Check Jan 1st and 3rd (measured)
    assert result.edar_biomarker_N1_mask.sel(run_id='real', date='2022-01-01', region_id='REG1').values == 1.0
    assert result.edar_biomarker_N1_mask.sel(run_id='real', date='2022-01-03', region_id='REG1').values == 1.0