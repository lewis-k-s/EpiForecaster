from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.preprocess.config import PreprocessingConfig, REGION_COORD, TEMPORAL_COORD
from models.configs import DataConfig, EpiForecasterConfig, ModelConfig

PADDED_REAL_RUN_ID = "real                                            "


def make_epi_config(
    dataset_path: str | Path,
    *,
    model_options: dict | None = None,
    data_options: dict | None = None,
) -> EpiForecasterConfig:
    model_kwargs = {
        "type": {
            "cases": True,
            "regions": False,
            "biomarkers": True,
            "mobility": False,
        },
        "mobility_embedding_dim": 1,
        "region_embedding_dim": 1,
        "input_window_length": 3,
        "forecast_horizon": 2,
        "max_neighbors": 1,
        "gnn_depth": 1,
        "population_dim": 1,
    }
    model_kwargs.update(model_options or {})

    data_kwargs = {
        "dataset_path": str(dataset_path),
        "mobility_threshold": 0.0,
        "min_history_observed_fraction": 1.0,
        "min_horizon_observed_fraction": 1.0,
    }
    data_kwargs.update(data_options or {})
    return EpiForecasterConfig(
        model=ModelConfig(**model_kwargs),
        data=DataConfig(**data_kwargs),
    )


def make_tiny_epi_dataset(
    *,
    periods: int = 10,
    num_nodes: int = 2,
    cases: np.ndarray | None = None,
    cases_mask: np.ndarray | None = None,
    cases_age: np.ndarray | None = None,
    hospitalizations: np.ndarray | None = None,
    hospitalizations_mask: np.ndarray | None = None,
    hospitalizations_age: np.ndarray | None = None,
    deaths: np.ndarray | None = None,
    deaths_mask: np.ndarray | None = None,
    deaths_age: np.ndarray | None = None,
    biomarker_values: dict[str, np.ndarray] | None = None,
    biomarker_masks: dict[str, np.ndarray] | None = None,
    biomarker_censors: dict[str, np.ndarray] | None = None,
    biomarker_ages: dict[str, np.ndarray] | None = None,
    mobility: np.ndarray | None = None,
    mobility_time_mask: np.ndarray | None = None,
    population: np.ndarray | None = None,
    clinical_feature_dim: bool = False,
) -> xr.Dataset:
    shape = (1, periods, num_nodes)
    cases = cases if cases is not None else np.ones((*shape, 1), dtype=np.float32)
    cases_mask = (
        cases_mask if cases_mask is not None else np.ones(shape, dtype=np.float32)
    )
    cases_age = (
        cases_age if cases_age is not None else np.zeros(shape, dtype=np.float32)
    )
    clinical_shape = (*shape, 1) if clinical_feature_dim else shape
    hospitalizations = (
        hospitalizations
        if hospitalizations is not None
        else np.ones(clinical_shape, dtype=np.float32)
    )
    hospitalizations_mask = (
        hospitalizations_mask
        if hospitalizations_mask is not None
        else np.ones(shape, dtype=np.float32)
    )
    hospitalizations_age = (
        hospitalizations_age
        if hospitalizations_age is not None
        else np.zeros(shape, dtype=np.float32)
    )
    deaths = deaths if deaths is not None else np.ones(clinical_shape, dtype=np.float32)
    deaths_mask = (
        deaths_mask if deaths_mask is not None else np.ones(shape, dtype=np.float32)
    )
    deaths_age = (
        deaths_age if deaths_age is not None else np.zeros(shape, dtype=np.float32)
    )
    mobility = (
        mobility
        if mobility is not None
        else np.ones((1, periods, num_nodes, num_nodes), dtype=np.float32)
    )
    population = (
        population
        if population is not None
        else np.full(num_nodes, 1000.0, dtype=np.float32)
    )

    clinical_dims = ("run_id", TEMPORAL_COORD, REGION_COORD)
    if clinical_feature_dim:
        clinical_dims = (*clinical_dims, "feature")
    data_vars = {
        "cases": (("run_id", TEMPORAL_COORD, REGION_COORD, "feature"), cases),
        "cases_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_mask),
        "cases_age": (("run_id", TEMPORAL_COORD, REGION_COORD), cases_age),
        "hospitalizations": (clinical_dims, hospitalizations),
        "hospitalizations_mask": (
            ("run_id", TEMPORAL_COORD, REGION_COORD),
            hospitalizations_mask,
        ),
        "hospitalizations_age": (
            ("run_id", TEMPORAL_COORD, REGION_COORD),
            hospitalizations_age,
        ),
        "deaths": (clinical_dims, deaths),
        "deaths_mask": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_mask),
        "deaths_age": (("run_id", TEMPORAL_COORD, REGION_COORD), deaths_age),
        "mobility": (
            ("run_id", TEMPORAL_COORD, REGION_COORD, "region_id_to"),
            mobility,
        ),
        "population": ((REGION_COORD,), population),
    }
    zeros = np.zeros(shape, dtype=np.float32)
    biomarker_values = biomarker_values or {"N1": np.ones(shape, dtype=np.float32)}
    biomarker_masks = biomarker_masks or {}
    biomarker_censors = biomarker_censors or {}
    biomarker_ages = biomarker_ages or {}
    for variant, values in biomarker_values.items():
        prefix = f"edar_biomarker_{variant}"
        data_vars[prefix] = (("run_id", TEMPORAL_COORD, REGION_COORD), values)
        data_vars[f"{prefix}_mask"] = (
            ("run_id", TEMPORAL_COORD, REGION_COORD),
            biomarker_masks.get(variant, np.ones(shape, dtype=np.float32)),
        )
        data_vars[f"{prefix}_censor"] = (
            ("run_id", TEMPORAL_COORD, REGION_COORD),
            biomarker_censors.get(variant, zeros),
        )
        data_vars[f"{prefix}_age"] = (
            ("run_id", TEMPORAL_COORD, REGION_COORD),
            biomarker_ages.get(variant, zeros),
        )
    if mobility_time_mask is not None:
        data_vars["mobility_time_mask"] = (
            ("run_id", TEMPORAL_COORD),
            mobility_time_mask,
        )

    regions = np.arange(num_nodes, dtype=np.int64)
    return xr.Dataset(
        data_vars,
        coords={
            "run_id": [PADDED_REAL_RUN_ID],
            TEMPORAL_COORD: pd.date_range("2020-01-01", periods=periods, freq="D"),
            REGION_COORD: regions,
            "region_id_to": regions,
        },
    )


def write_zarr_dataset(dataset: xr.Dataset, path: str | Path) -> None:
    dataset.to_zarr(path, mode="w", zarr_format=2)


def make_preprocessing_config(
    tmp_path: Path,
    *,
    synthetic_path: str | Path | None = None,
    include_temporal_covariates: bool = False,
    holiday_calendar_path: str | Path | None = None,
    options: dict | None = None,
) -> PreprocessingConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    dummy = data_dir / "dummy.txt"
    dummy.write_text("ok")

    temporal_covariates = None
    if include_temporal_covariates:
        if holiday_calendar_path is None:
            holiday_calendar_path = data_dir / "holidays.csv"
            holiday_calendar_path.write_text("date\n2020-01-01\n")
        temporal_covariates = {
            "include_day_of_week": True,
            "include_holidays": True,
            "holiday_calendar_file": str(holiday_calendar_path),
        }

    config_kwargs = {
        "data_dir": str(data_dir),
        "synthetic_path": str(synthetic_path or dummy),
        "cases_file": str(dummy),
        "mobility_path": str(dummy),
        "wastewater_file": str(dummy),
        "population_file": str(dummy),
        "region_metadata_file": str(dummy),
        "start_date": datetime(2020, 1, 1),
        "end_date": datetime(2020, 1, 4),
        "forecast_horizon": 1,
        "sequence_length": 1,
        "output_path": str(tmp_path / "out"),
        "dataset_name": "synthetic_test",
        "wastewater_flow_mode": "concentration",
        "min_density_threshold": 0.0,
        "validate_alignment": False,
        "generate_alignment_report": False,
        "temporal_covariates": temporal_covariates,
    }
    config_kwargs.update(options or {})
    return PreprocessingConfig(**config_kwargs)
