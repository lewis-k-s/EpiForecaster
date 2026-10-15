"""Fixed smoothing must not fit parameters from held-out source series."""

from types import SimpleNamespace

import pandas as pd
import pytest

from data.preprocess.config import SmoothingConfig
from data.preprocess.processors import (
    catalonia_cases_processor,
    deaths_processor,
    edar_processor,
    hospitalizations_processor,
)


@pytest.mark.parametrize(
    ("module", "processor_name", "method_name", "value_column"),
    [
        (
            catalonia_cases_processor,
            "CataloniaCasesProcessor",
            "_apply_kalman_smoothing",
            "cases",
        ),
        (
            hospitalizations_processor,
            "HospitalizationsProcessor",
            "_apply_kalman_smoothing",
            "hospitalizations",
        ),
        (deaths_processor, "DeathsProcessor", "_apply_kalman_smoothing", "deaths"),
        (edar_processor, "EDARProcessor", "_apply_tobit_kalman", "total_covid_flow"),
    ],
)
def test_fixed_smoothing_does_not_fit_source_parameters(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    processor_name: str,
    method_name: str,
    value_column: str,
) -> None:
    config = SimpleNamespace(
        smoothing=SmoothingConfig(parameter_fit_mode="fixed"),
        validation_options={"process_var": 0.05, "measurement_var": 0.5},
        censor_inflation=10.0,
    )
    dates = pd.date_range("2020-05-01", periods=4)
    data = pd.DataFrame({"date": dates, value_column: [1.0, 2.0, None, 4.0]})
    if module is edar_processor:
        data["edar_id"] = "edar-1"
        data["variant"] = "N1"
        data["limit_flow"] = 0.5
    else:
        data["municipality_code"] = "region-1"

    def reject_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fixed mode fitted held-out data")

    monkeypatch.setattr(module, "fit_kalman_params", reject_fit)
    processor = getattr(module, processor_name)(config)
    result = getattr(processor, method_name)(data)
    assert len(result) == len(data)


def test_unknown_smoothing_fit_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="parameter_fit_mode"):
        SmoothingConfig(parameter_fit_mode="unknown")


def test_fixed_spatial_preprocessing_keeps_all_wastewater_variants() -> None:
    config = SimpleNamespace(
        smoothing=SmoothingConfig(parameter_fit_mode="fixed"),
        validation_options={},
    )
    source = pd.DataFrame(
        {
            "edar_id": ["site", "site", "site"],
            "variant": ["N1", "N2", "IP4"],
            "viral_load": [1.0, 2.0, 3.0],
        }
    )
    result = edar_processor.EDARProcessor(config)._select_variants(source)
    assert set(result["variant"]) == {"N1", "N2", "IP4"}
