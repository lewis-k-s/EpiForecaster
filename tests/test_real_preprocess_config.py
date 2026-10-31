"""The MN5 real-data preprocessing template and its base config stay consistent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from data.preprocess import PreprocessingConfig
from scripts.workflow.run_config import (
    load_template,
    preprocessing_dataset_name,
    validate_mn5_allocation,
    validate_workflow_contract,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "workflow/config/mn5_preprocess_real.yaml"
BASE = ROOT / "configs/production_only/preprocess_mn5_real.yaml"
RAW_POINTER = ROOT / "data/files/catalonia_real_raw_inputs_v1.dvc"


def test_template_is_a_valid_preprocess_workflow() -> None:
    template = load_template(TEMPLATE)
    assert validate_workflow_contract(template) == "preprocess"
    allocation = validate_mn5_allocation(template)
    assert allocation["qos"] == "acc_debug"
    assert allocation["runtime"] <= 120
    assert preprocessing_dataset_name(template) == "real_with_id"
    assert template["preprocessing"]["expected_run_count"] == 1
    assert (ROOT / template["preprocessing"]["base_config"]) == BASE
    for target in template["dvc"]["targets"]:
        assert (ROOT / target).is_file()


@pytest.fixture
def staged_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the base config and create empty stand-ins for the staged inputs."""
    document = yaml.safe_load(BASE.read_text())
    base = tmp_path / "base.yaml"
    base.write_text(BASE.read_text())
    monkeypatch.chdir(tmp_path)
    (tmp_path / document["data_dir"]).mkdir(parents=True)
    (tmp_path / document["mobility_path"]).mkdir(parents=True)
    for key, value in document.items():
        if key.endswith("_file") and isinstance(value, str):
            (tmp_path / value).touch()
    holiday = document["temporal_covariates"]["holiday_calendar_file"]
    (tmp_path / holiday).touch()
    return base


def test_base_config_uses_current_production_defaults(staged_config: Path) -> None:
    config = PreprocessingConfig.from_file(staged_config)
    assert config.dataset_name == "real_with_id"
    assert config.hospitalizations_reporting_lag_days == 7
    assert config.continuous_storage_dtype == "float32"
    assert config.smoothing.parameter_fit_end_date is None
    assert config.synthetic_path is None


def test_base_config_inputs_come_from_the_raw_pointer(staged_config: Path) -> None:
    config = PreprocessingConfig.from_file(staged_config)
    root = "data/files/catalonia_real_raw_inputs_v1/"
    for path in (
        config.cases_file,
        config.deaths_file,
        config.hospitalizations_file,
        config.vaccination_file,
        config.wastewater_file,
        config.region_metadata_file,
        config.population_file,
    ):
        assert str(path).startswith(root)
    assert str(config.mobility_path) == "data/files/mobility.zarr"
    assert "outs:" in RAW_POINTER.read_text()
    assert "path: catalonia_real_raw_inputs_v1" in RAW_POINTER.read_text()
