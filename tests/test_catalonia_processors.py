"""Concise tests for Catalonia data processors."""

from datetime import datetime
from pathlib import Path

import pytest

from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.catalonia_cases_processor import CataloniaCasesProcessor
from data.preprocess.processors.deaths_processor import DeathsProcessor
from data.preprocess.processors.hospitalizations_processor import (
    HospitalizationsProcessor,
)
from data.preprocess.processors.municipality_mapping_processor import (
    MunicipalityMappingProcessor,
)


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temp directory with sample data files."""
    # Municipality mapping (4-line header + data)
    mapping_file = tmp_path / "mpiscatalunya.csv"
    mapping_file.write_text(
        "\n\n\nCodi,Nom,Codi comarca,Nom comarca\n"
        "08019,Barcelona,01,Barcelonès\n"
        "08021,Abrera,11,Baix Llobregat\n"
    )

    # Cases data (DD/MM/YYYY format, municipality code with leading zero)
    cases_file = (
        tmp_path / "Registre_de_casos_de_COVID-19_a_Catalunya_per_municipi_i_sexe.csv"
    )
    cases_file.write_text(
        "TipusCasData,ComarcaCodi,MunicipiCodi,SexeCodi,TipusCasDescripcio,NumCasos\n"
        "01/01/2022,01,08019,1,Positiu PCR,5\n"
        "01/01/2022,11,08021,1,Positiu TAR,3\n"
        "02/01/2022,01,08019,1,Positiu PCR,7\n"
    )

    # Deaths data (municipality level from pre-aggregated CSV)
    deaths_file = tmp_path / "deaths_municipality.csv"
    deaths_file.write_text(
        "Data defunció,municipality_code,municipality_name,defuncions_muni\n"
        "01/01/2022,08019,Barcelona,2.0\n"
        "01/01/2022,08021,Abrera,1.0\n"
        "02/01/2022,08019,Barcelona,3.0\n"
    )

    # Hospitalizations data (weekly municipality-level)
    # Week 1: Jan 3-9 (epidemiological week)
    hosp_file = tmp_path / "hospitalizations_municipality.csv"
    hosp_file.write_text(
        "setmana_epidemiologica,any,data_inici,data_final,municipality_code,municipality_name,casos_muni\n"
        "1,2022,03/01/2022,09/01/2022,08019,Barcelona,14.0\n"
        "1,2022,03/01/2022,09/01/2022,08021,Abrera,7.0\n"
        "2,2022,10/01/2022,16/01/2022,08019,Barcelona,21.0\n"
        "2,2022,10/01/2022,16/01/2022,08021,Abrera,14.0\n"
    )

    # Dummy files for config validation
    (tmp_path / "unused.csv").write_text("dummy\n")
    (tmp_path / "unused.nc").write_bytes(b"\x89HDF\r\n\x1a\n")
    (tmp_path / "unused.json").write_text("{}")

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
        end_date=datetime(2022, 1, 15),  # 2 weeks for forecast_horizon=7
        output_path=str(temp_data_dir),
        dataset_name="test",
    )


@pytest.mark.region
def test_municipality_mapping_loads(temp_data_dir: Path):
    """Test mapping processor loads data correctly."""
    proc = MunicipalityMappingProcessor(temp_data_dir)
    df = proc.load_mapping()

    assert len(df) == 2
    assert list(df.columns) == [
        "municipality_code",
        "municipality_name",
        "comarca_code",
        "comarca_name",
    ]
    assert df["municipality_code"].iloc[0] == "08019"  # Leading zero preserved


@pytest.mark.region
def test_catalonia_cases_output_format(
    config: PreprocessingConfig, temp_data_dir: Path
):
    """Test cases processor produces correct xarray format with mask/age channels."""
    proc = CataloniaCasesProcessor(config)
    cases_file = (
        temp_data_dir
        / "Registre_de_casos_de_COVID-19_a_Catalunya_per_municipi_i_sexe.csv"
    )
    result = proc.process(cases_file, apply_smoothing=False)

    assert "cases" in result
    assert "cases_mask" in result
    assert "cases_age" in result
    # Cases now has run_id dimension like other datasets
    assert result.cases.dims == ("run_id", "date", "region_id")
    assert result.cases.sizes["run_id"] == 1  # single run
    assert result.cases.sizes["date"] == 15  # full date range (reindexed)
    assert result.cases.sizes["region_id"] == 2  # 2 municipalities
    assert result.cases.sum() == 15  # 5+3+7
    # Check mask and age are present and have correct shape
    assert result.cases_mask.dims == ("run_id", "date", "region_id")
    assert result.cases_age.dims == ("run_id", "date", "region_id")


@pytest.mark.region
def test_deaths_processor_municipality_level(
    config: PreprocessingConfig, temp_data_dir: Path
):
    """Test deaths processor at municipality level from pre-aggregated data."""
    proc = DeathsProcessor(config)
    result = proc.process(temp_data_dir)

    assert "deaths" in result
    assert "deaths_mask" in result
    assert "deaths_age" in result
    assert result.deaths.dims == ("date", "region_id")
    assert result.deaths.sizes["date"] == 15  # full date range (reindexed)
    assert result.deaths.sizes["region_id"] == 2  # 2 municipalities
    # Kalman smoothing/interpolation should produce finite daily trajectories.
    assert result.deaths.notnull().all()
    assert (result.deaths >= 0).all()
    # Mask tracks true observations only (3 observed rows from fixture)
    assert float(result.deaths_mask.sum()) == 3.0
    # Age channel should include staleness beyond observation days
    assert float(result.deaths_age.max()) > 1.0


@pytest.mark.region
def test_deaths_processor_can_disable_smoothing(
    config: PreprocessingConfig, temp_data_dir: Path
):
    """Raw mode should preserve sparsity (NaNs on unobserved days)."""
    proc = DeathsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=False)
    assert result.deaths.isnull().any()


@pytest.mark.region
def test_deaths_processor_holt_damped_keeps_mask_age_semantics(
    config: PreprocessingConfig, temp_data_dir: Path
):
    """Switching to holt_damped should not change mask/age observation semantics."""
    config.smoothing.clinical_method = "holt_damped"
    proc = DeathsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=True)

    assert result.deaths.notnull().all()
    assert float(result.deaths_mask.sum()) == 3.0
    assert float(result.deaths_age.max()) > 1.0


@pytest.mark.integration
@pytest.mark.region
def test_hospitalizations_processor_weekly_data(
    config: PreprocessingConfig, temp_data_dir: Path
):
    """Integration test: Hospitalizations processor with weekly sparse data.

    This test exercises the full pipeline from weekly observations to daily
    interpolated output with Kalman smoothing. Marked as integration test
    because it tests multiple components together (resample + smooth + mask).
    """
    proc = HospitalizationsProcessor(config)
    result = proc.process(temp_data_dir, apply_smoothing=True)

    assert "hospitalizations" in result
    assert "hospitalizations_mask" in result
    assert "hospitalizations_age" in result

    # Should have run_id dimension like other datasets
    assert result.hospitalizations.dims == ("run_id", "date", "region_id")
    assert result.hospitalizations.sizes["run_id"] == 1

    # Should have daily resolution covering full config range
    assert result.hospitalizations.sizes["date"] == 15  # Jan 1-15

    # Should have 2 municipalities
    assert result.hospitalizations.sizes["region_id"] == 2

    # Values between first and second observation should be interpolated
    # Gap between Jan 3 and Jan 10 should be filled by Kalman
    gap_period = result.hospitalizations.sel(date=slice("2022-01-03", "2022-01-09"))
    assert gap_period.notnull().all()
    assert (gap_period >= 0).all()

    # Mask should track original weekly observations
    # 2 weeks × 2 municipalities = 4 observations
    # Note: mask is only 1.0 on actual observation days (week starts)
    assert float(result.hospitalizations_mask.sum()) == 4.0

    # Age channel should show staleness between weekly observations
    # Week 1 obs on Jan 3, Week 2 obs on Jan 10
    # Jan 9 should have age=7 (1 week since last obs)
    jan_9_age = result.hospitalizations_age.sel(date="2022-01-09")
    assert float(jan_9_age.max()) > 1.0
