"""Concise tests for Catalonia data processors."""

from datetime import datetime
from pathlib import Path

import pytest

from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.catalonia_cases_processor import CataloniaCasesProcessor
from data.preprocess.processors.deaths_processor import DeathsProcessor
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
    # Total deaths: 2.0 + 1.0 + 3.0 = 6.0
    assert result.deaths.sum() == 6.0
    # Check specific municipality
    assert result.deaths.sel(region_id="08019").sum() == 5.0  # 2.0 + 3.0
    assert result.deaths.sel(region_id="08021").sum() == 1.0
    # Mask tracks true observations only (3 observed rows from fixture)
    assert float(result.deaths_mask.sum()) == 3.0
    # Age channel should include staleness beyond observation days
    assert float(result.deaths_age.max()) > 1.0
