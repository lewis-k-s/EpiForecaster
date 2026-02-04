"""Tests for the hospitalizations processor."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.hospitalizations_processor import (
    HospitalizationsProcessor,
)


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temp directory with sample pre-aggregated municipality hospitalization data."""
    # Hospitalization data (pre-aggregated municipality format from dasymetric_mob)
    hosp_file = tmp_path / "hospitalizations_municipality.csv"
    hosp_file.write_text(
        "setmana_epidemiologica,any,data_inici,data_final,codi_regio,nom_regio,"
        "codi_ambit,nom_ambit,sexe,grup_edat,index_socioeconomic,"
        "municipality_code,municipality_name,casos_muni,poblacio_muni\n"
        "1,2022,03/01/2022,09/01/2022,01,Barcelonès,01,Barcelonès,Home,60 a 64,2,"
        "08019,Barcelona,5.0,1000\n"
        "1,2022,03/01/2022,09/01/2022,01,Barcelonès,01,Barcelonès,Dona,60 a 64,2,"
        "08019,Barcelona,3.0,1000\n"
        "1,2022,03/01/2022,09/01/2022,11,Baix Llobregat,11,Baix Llobregat,Home,70 a 74,3,"
        "08021,Abrera,2.0,500\n"
        "2,2022,10/01/2022,16/01/2022,01,Barcelonès,01,Barcelonès,Home,60 a 64,2,"
        "08019,Barcelona,7.0,1000\n"
        "2,2022,10/01/2022,16/01/2022,11,Baix Llobregat,11,Baix Llobregat,Dona,70 a 74,3,"
        "08021,Abrera,4.0,500\n"
        "3,2022,17/01/2022,23/01/2022,21,Maresme,21,Maresme,Home,50 a 54,1,"
        "08018,Alella,3.0,200\n"
    )

    # Dummy files for config validation
    (tmp_path / "unused.csv").write_text("dummy\n")
    (tmp_path / "unused.nc").write_bytes(b"\x89HDF\r\n\x1a\n")

    return tmp_path


@pytest.fixture
def config(temp_data_dir: Path) -> PreprocessingConfig:
    """Minimal config for testing."""
    return PreprocessingConfig(
        data_dir=str(temp_data_dir),
        cases_file=str(temp_data_dir / "unused.csv"),
        mobility_path=str(temp_data_dir / "unused.nc"),
        wastewater_file=str(temp_data_dir / "unused.csv"),
        population_file=str(temp_data_dir / "unused.csv"),
        region_metadata_file=str(temp_data_dir / "unused.json"),
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 31),
        output_path=str(temp_data_dir),
        dataset_name="test",
    )


@pytest.mark.epiforecaster
def test_hospitalizations_processor_loads_data(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that processor loads hospitalization data correctly."""
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=False)

    assert "hospitalizations" in result
    assert "hospitalizations_mask" in result
    assert "hospitalizations_age" in result


@pytest.mark.epiforecaster
def test_hospitalizations_output_format(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that output has correct dimensions and coordinates."""
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=False)

    hosp_da = result["hospitalizations"]

    # Check dimensions
    assert "run_id" in hosp_da.dims
    assert "date" in hosp_da.dims
    assert "region_id" in hosp_da.dims

    # Check run_id
    assert hosp_da["run_id"].values[0] == "real"

    # Check date range
    assert hosp_da["date"].min().values == np.datetime64("2022-01-01")
    assert hosp_da["date"].max().values == np.datetime64("2022-01-31")


@pytest.mark.epiforecaster
def test_weekly_to_daily_resampling(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that weekly data is correctly resampled to daily."""
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=False)

    hosp_da = result["hospitalizations"]

    # Check we have daily data (31 days in Jan 2022)
    assert hosp_da.sizes["date"] == 31

    # Total hospitalizations should be preserved (distributed across days)
    # Week 1: 08019 = 8, 08021 = 2
    # Week 2: 08019 = 7, 08021 = 4
    # Week 3: 08018 = 3
    # Total = 24, distributed across days
    total_hosp = float(hosp_da.sum())
    assert total_hosp > 0
    # Total should be approximately 24 (sum of all casos_muni)
    assert abs(total_hosp - 24.0) < 1.0  # Allow small numerical tolerance


@pytest.mark.epiforecaster
def test_municipality_codes_preserved(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that municipality codes are preserved with leading zeros."""
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=False)

    hosp_da = result["hospitalizations"]

    # Check that municipalities from data are present
    region_ids = list(hosp_da["region_id"].values)

    # All municipalities should be present with string codes
    assert "08019" in region_ids  # Barcelona
    assert "08021" in region_ids  # Abrera
    assert "08018" in region_ids  # Alella


@pytest.mark.epiforecaster
def test_kalman_smoothing_produces_finite_values(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that Kalman smoothing produces finite values where data exists."""
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=True)

    hosp_da = result["hospitalizations"]
    mask_da = result["hospitalizations_mask"]

    # Check that finite values exist where mask is 1
    data_with_mask = hosp_da.where(mask_da == 1)
    finite_count = np.isfinite(data_with_mask).sum()
    assert finite_count > 0, "Should have some finite values where mask=1"

    # Check no infinite values anywhere
    assert not np.isinf(hosp_da).any()


@pytest.mark.epiforecaster
def test_mask_and_age_channels(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that mask and age channels are properly created."""
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=False)

    # Check mask values are 0 or 1
    mask_da = result["hospitalizations_mask"]
    valid_mask_values = mask_da.values[~np.isnan(mask_da.values)]
    assert len(valid_mask_values) > 0, "Should have some valid mask values"
    assert np.all((valid_mask_values == 0) | (valid_mask_values == 1))

    # Check age values are positive where data exists
    age_da = result["hospitalizations_age"]
    valid_age_values = age_da.values[~np.isnan(age_da.values)]
    assert len(valid_age_values) > 0, "Should have some valid age values"
    assert np.all(valid_age_values > 0)


@pytest.mark.epiforecaster
def test_integer_age_channel(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that age channel is integer and resets on week boundaries."""
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=False)

    age_da = result["hospitalizations_age"]

    # Check that age values are integers (1-7 for weekly data)
    valid_age_values = age_da.values[~np.isnan(age_da.values)]
    assert len(valid_age_values) > 0, "Should have some valid age values"

    # Check all ages are integers in range 1-7
    assert np.all(valid_age_values == valid_age_values.astype(int)), (
        "Age values should be integers"
    )
    assert np.all((valid_age_values >= 1) & (valid_age_values <= 7)), (
        "Age values should be in range 1-7"
    )

    # Check that age 1 exists (week start days)
    assert np.any(valid_age_values == 1), "Should have age=1 values (week starts)"


@pytest.mark.epiforecaster
def test_date_alignment(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that dates are properly aligned to config range."""
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=False)

    hosp_da = result["hospitalizations"]
    dates = pd.to_datetime(hosp_da["date"].values)

    # Check continuous daily dates
    expected_dates = pd.date_range(
        start=config.start_date, end=config.end_date, freq="D"
    )
    assert len(dates) == len(expected_dates)
    assert all(dates == expected_dates)


@pytest.mark.epiforecaster
def test_empty_dataset_creation(temp_data_dir: Path):
    """Test that empty dataset is created when no data in range."""
    # Create config with date range outside data
    empty_config = PreprocessingConfig(
        data_dir=str(temp_data_dir),
        cases_file=str(temp_data_dir / "unused.csv"),
        mobility_path=str(temp_data_dir / "unused.nc"),
        wastewater_file=str(temp_data_dir / "unused.csv"),
        population_file=str(temp_data_dir / "unused.csv"),
        region_metadata_file=str(temp_data_dir / "unused.json"),
        start_date=datetime(2020, 1, 1),  # Before data starts
        end_date=datetime(2020, 1, 31),
        output_path=str(temp_data_dir),
        dataset_name="test",
    )

    proc = HospitalizationsProcessor(empty_config)
    result = proc.process(temp_data_dir, apply_smoothing=False)

    # Should still have proper structure even if empty
    assert "hospitalizations" in result
    assert result["hospitalizations"].sizes["date"] == 31  # 31 days


@pytest.mark.epiforecaster
def test_demographic_aggregation(
    config: PreprocessingConfig,
    temp_data_dir: Path,
):
    """Test that demographic dimensions (sex, age) are aggregated correctly."""
    proc = HospitalizationsProcessor(config)

    # Process and check totals are preserved (distributed across days/municipalities)
    result = proc.process(temp_data_dir, apply_smoothing=False)
    total_hosp = float(result["hospitalizations"].sum())

    # Total should be approximately 24 (sum of all casos_muni in fixture)
    # Week 1: 08019 = 8, 08021 = 2
    # Week 2: 08019 = 7, 08021 = 4
    # Week 3: 08018 = 3
    # Total = 24
    assert total_hosp > 0
    assert abs(total_hosp - 24.0) < 1.0  # Allow small numerical tolerance
