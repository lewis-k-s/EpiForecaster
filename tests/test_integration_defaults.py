"""Pin data-contract defaults and the cases_dim cutover."""

from copy import deepcopy

import numpy as np
import pytest

from data.epi_dataset import EpiDataset
from data.preprocess.config import PreprocessingConfig
from models.configs import ModelConfig, ObservationHeadConfig
from tests.dataset_test_helpers import (
    make_epi_config,
    make_tiny_epi_dataset,
    write_zarr_dataset,
)


def _dataset_path(tmp_path) -> str:
    periods = 15
    ds = make_tiny_epi_dataset(
        periods=periods,
        cases=np.full((1, periods, 2, 1), 10.0, dtype=np.float32),
        cases_age=np.ones((1, periods, 2), dtype=np.float32),
        hospitalizations=np.ones((1, periods, 2), dtype=np.float32),
        hospitalizations_age=np.ones((1, periods, 2), dtype=np.float32),
        deaths=np.ones((1, periods, 2), dtype=np.float32),
        deaths_age=np.ones((1, periods, 2), dtype=np.float32),
    )
    path = str(tmp_path / "d.zarr")
    write_zarr_dataset(ds, path)
    return path


@pytest.mark.epiforecaster
def test_cases_dim_default_is_nine() -> None:
    assert ModelConfig.__dataclass_fields__["cases_dim"].default == 9


@pytest.mark.epiforecaster
def test_dataset_uses_configured_cases_dim_and_rejects_mismatch(tmp_path) -> None:
    config = make_epi_config(_dataset_path(tmp_path))
    assert config.model.cases_dim == 9
    dataset = EpiDataset(config=config, target_nodes=[0, 1], context_nodes=[0, 1])
    assert dataset.cases_output_dim == config.model.cases_dim == 9

    bad = deepcopy(config)
    bad.model.cases_dim = 3
    with pytest.raises(ValueError, match="cases_dim"):
        EpiDataset(config=bad, target_nodes=[0, 1], context_nodes=[0, 1])
    assert bad.model.cases_dim == 3  # no silent rewrite

    risk = deepcopy(config)
    risk.data.use_imported_risk = True
    risk.data.mobility_lags = [1, 7]
    with pytest.raises(ValueError, match="11"):
        EpiDataset(config=risk, target_nodes=[0, 1], context_nodes=[0, 1])
    risk.model.cases_dim = 11
    assert EpiDataset(
        config=risk, target_nodes=[0, 1], context_nodes=[0, 1]
    ).cases_output_dim == 11


def test_preprocess_defaults() -> None:
    fields = PreprocessingConfig.__dataclass_fields__
    assert fields["hospitalizations_reporting_lag_days"].default == 7
    assert fields["continuous_storage_dtype"].default == "float32"


@pytest.mark.epiforecaster
def test_deaths_kernel_prior_pinned() -> None:
    config = ObservationHeadConfig()
    assert config.kernel_gamma_shape_deaths == 12.25
    assert config.kernel_gamma_scale_deaths == pytest.approx(4.0 / 7.0)
    assert config.kernel_mlp_deaths is False
    mean = config.kernel_gamma_shape_deaths * config.kernel_gamma_scale_deaths
    sd = config.kernel_gamma_shape_deaths**0.5 * config.kernel_gamma_scale_deaths
    assert (mean, sd) == (pytest.approx(7.0), pytest.approx(2.0))
