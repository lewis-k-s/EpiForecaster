"""Concise tests for Catalonia data processors."""

from datetime import datetime
from pathlib import Path

import pandas as pd
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

    # Deaths data (comarca level, DD/MM/YYYY format)
    deaths_file = (
        tmp_path
        / "Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
    )
    deaths_file.write_text(
        "Data defunció,Codi Comarca,Sexe,Nombre defuncions\n"
        "01/01/2022,01,Home,2\n"
        "01/01/2022,11,Dona,1\n"
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
    """Test cases processor produces correct xarray format."""
    proc = CataloniaCasesProcessor(config)
    result = proc.process(temp_data_dir)

    assert "cases" in result
    assert result.cases.dims == ("date", "region_id")
    assert result.cases.sizes["date"] == 15  # full date range (reindexed)
    assert result.cases.sizes["region_id"] == 2  # 2 municipalities
    assert result.cases.sum() == 15  # 5+3+7


@pytest.mark.region
def test_deaths_processor_comarca_level(
    config: PreprocessingConfig, temp_data_dir: Path
):
    """Test deaths processor at comarca level."""
    proc = DeathsProcessor(config)
    result = proc.process(temp_data_dir, allocate_to_municipalities=False)

    assert "deaths" in result
    assert result.deaths.dims == ("date", "region_id")
    assert result.deaths.sum() == 3  # 2+1


@pytest.mark.region
def test_deaths_processor_municipality_allocation(
    config: PreprocessingConfig, temp_data_dir: Path
):
    """Test deaths processor allocation to municipalities."""
    proc = DeathsProcessor(config)

    # Mock population data
    pop_df = pd.DataFrame({"region_id": ["08019", "08021"], "population": [1000, 500]})

    mapping_proc = MunicipalityMappingProcessor(temp_data_dir)
    result = proc.process(
        temp_data_dir,
        population_df=pop_df,
        allocate_to_municipalities=True,
        mapping_processor=mapping_proc,
    )

    assert "deaths" in result
    assert result.deaths.sizes["region_id"] == 2  # 2 municipalities
    # Comarca 01 has 2 deaths, allocated to 08019 (100% of comarca population)
    assert result.deaths.sel(region_id="08019").sum() == 2
