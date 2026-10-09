from datetime import datetime

import numpy as np
import pandas as pd

from data.preprocess.config import PreprocessingConfig
from data.preprocess.processors.alignment_processor import AlignmentProcessor
from data.preprocess.processors.vaccination_processor import VaccinationProcessor


def _config(tmp_path) -> PreprocessingConfig:
    cases_file = tmp_path / "cases.csv"
    mobility_path = tmp_path / "mobility.zarr"
    population_file = tmp_path / "population.csv"
    vaccination_file = tmp_path / "vaccination.csv"

    cases_file.touch()
    mobility_path.mkdir()
    population_file.write_text("id,d.population\n08001,100\n08002,200\n")
    vaccination_file.write_text(
        "\n".join(
            [
                "municipi_codi,municipi,data,dosi,fabricant,no_vacunat,recompte",
                "08001,A,2021-01-01T00:00:00.000,1,BioNTech / Pfizer,,10",
                "08001,A,2021-01-01T00:00:00.000,2,BioNTech / Pfizer,,7",
                "08001,A,2021-01-02T00:00:00.000,1,No administrada,No vacunat,3",
                "08001,A,2021-01-03T00:00:00.000,1,Moderna / Lonza,,5",
                "08002,B,2021-01-02T00:00:00.000,1,BioNTech / Pfizer,,20",
            ]
        )
    )

    return PreprocessingConfig(
        data_dir=str(tmp_path),
        cases_file=str(cases_file),
        mobility_path=str(mobility_path),
        population_file=str(population_file),
        vaccination_file=str(vaccination_file),
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2021, 1, 4),
        output_path=str(tmp_path / "out"),
        dataset_name="test",
        forecast_horizon=1,
    )


def test_vaccination_processor_builds_cumulative_first_dose_rate(tmp_path):
    config = _config(tmp_path)
    ds = VaccinationProcessor(config).process(config.vaccination_file)

    rate = ds["vaccination_rate"].sel(run_id="real")
    mask = ds["vaccination_rate_mask"].sel(run_id="real")
    age = ds["vaccination_rate_age"].sel(run_id="real")

    expected_dates = pd.date_range("2021-01-01", "2021-01-04", freq="D")
    np.testing.assert_array_equal(rate["date"].values, expected_dates.values)

    assert np.isclose(float(rate.sel(date="2021-01-01", region_id="08001")), 0.10)
    assert np.isclose(float(rate.sel(date="2021-01-02", region_id="08001")), 0.10)
    assert np.isclose(float(rate.sel(date="2021-01-03", region_id="08001")), 0.15)
    assert np.isclose(float(rate.sel(date="2021-01-02", region_id="08002")), 0.10)

    assert bool(mask.sel(date="2021-01-02", region_id="08001")) is False
    assert int(age.sel(date="2021-01-02", region_id="08001")) == 2


def test_vaccination_crop_preserves_prior_doses_and_age(tmp_path):
    config = _config(tmp_path)
    config.start_date = datetime(2021, 1, 2)
    dataset = VaccinationProcessor(config).process(config.vaccination_file)
    region = dataset.sel(run_id="real", region_id="08001")
    np.testing.assert_allclose(region.vaccination_rate, [0.10, 0.15, 0.15])
    np.testing.assert_array_equal(region.vaccination_rate_mask, [False, True, False])
    np.testing.assert_array_equal(region.vaccination_rate_age, [2, 1, 2])
    aligned = AlignmentProcessor(config)._align_clinical_dataset(
        dataset,
        pd.date_range(config.start_date, config.end_date),
        value_name="vaccination_rate",
        label="Vaccination",
    )
    np.testing.assert_array_equal(
        aligned.vaccination_rate_age.sel(run_id="real", region_id="08001"),
        [2, 1, 2],
    )
