"""Validate the raw synthetic EpiSim Zarr before EpiForecaster preprocessing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


REQUIRED_DIMS = {
    "run_id": 1,
    "date": 2,
    "region_id": 3,
}
REQUIRED_VARIABLE_DIMS = {
    "cases": ("run_id", "date", "region_id"),
    "hospitalizations": ("run_id", "date", "region_id"),
    "deaths": ("run_id", "date", "region_id"),
    "edar_biomarker_N1": ("run_id", "date", "edar_id"),
    "edar_biomarker_N2": ("run_id", "date", "edar_id"),
    "edar_biomarker_IP4": ("run_id", "date", "edar_id"),
    "limit_of_detection_N1": ("run_id", "date", "edar_id"),
    "limit_of_detection_N2": ("run_id", "date", "edar_id"),
    "limit_of_detection_IP4": ("run_id", "date", "edar_id"),
    "population": ("run_id", "region_id"),
    "latent_S_true": ("run_id", "date", "region_id"),
    "latent_I_true": ("run_id", "date", "region_id"),
    "latent_R_true": ("run_id", "date", "region_id"),
    "latent_D_true": ("run_id", "date", "region_id"),
    "mobility_base": ("origin", "target"),
    "mobility_kappa0": ("run_id", "date"),
}
OPTIONAL_HOSPITAL_LATENTS = ("latent_H_true", "latent_hospitalized_true")
RUN_METADATA = (
    "synthetic_cases_report_delay_mean",
    "synthetic_cases_report_rate_max",
    "synthetic_cases_report_rate_min",
    "synthetic_deaths_report_delay_mean",
    "synthetic_deaths_report_delay_std",
    "synthetic_deaths_report_rate",
    "synthetic_hosp_report_delay_mean",
    "synthetic_hosp_report_delay_std",
    "synthetic_hosp_report_rate",
    "synthetic_mobility_noise_factor",
    "synthetic_mobility_noise_sigma_D",
    "synthetic_mobility_noise_sigma_O",
    "synthetic_mobility_type",
    "synthetic_scenario_type",
    "synthetic_sparsity_level",
    "synthetic_strength",
    "synthetic_ww_noise_sigma_IP4",
    "synthetic_ww_noise_sigma_N1",
    "synthetic_ww_noise_sigma_N2",
    "synthetic_ww_transport_loss",
)
STAT_VARIABLES = (
    "cases",
    "hospitalizations",
    "deaths",
    "edar_biomarker_N1",
    "edar_biomarker_N2",
    "edar_biomarker_IP4",
    "latent_S_true",
    "latent_I_true",
    "latent_R_true",
    "latent_D_true",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _coordinate_summary(values: np.ndarray) -> dict[str, Any]:
    materialized = np.asarray(values)
    string_values = [str(value).strip() for value in materialized.tolist()]
    return {
        "count": int(materialized.size),
        "unique_count": int(len(set(string_values))),
        "first": string_values[0] if string_values else None,
        "last": string_values[-1] if string_values else None,
        "values": string_values if materialized.size <= 100 else None,
    }


def _numeric_stats(array: xr.DataArray) -> dict[str, Any]:
    values = array.astype(np.float64)
    finite = np.isfinite(values)
    total = int(values.size)
    finite_count = int(finite.sum().compute().item())
    nan_count = int(values.isnull().sum().compute().item())
    negative_count = int(((values < 0) & finite).sum().compute().item())
    if finite_count == 0:
        return {
            "total": total,
            "finite": 0,
            "nan": nan_count,
            "finite_fraction": 0.0,
            "negative": negative_count,
            "min": None,
            "max": None,
            "mean": None,
        }
    finite_values = values.where(finite)
    return {
        "total": total,
        "finite": finite_count,
        "nan": nan_count,
        "finite_fraction": finite_count / total,
        "negative": negative_count,
        "min": float(finite_values.min(skipna=True).compute().item()),
        "max": float(finite_values.max(skipna=True).compute().item()),
        "mean": float(finite_values.mean(skipna=True).compute().item()),
    }


def _per_run_observed_fraction(array: xr.DataArray) -> list[float]:
    reduce_dims = [dim for dim in array.dims if dim != "run_id"]
    fractions = np.isfinite(array).mean(dim=reduce_dims).compute().values
    return [float(value) for value in np.asarray(fractions).reshape(-1)]


def validate(path: Path) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    ds = xr.open_zarr(path, consolidated=True, chunks="auto")
    try:
        dims = {name: int(size) for name, size in ds.sizes.items()}
        for dim, minimum in REQUIRED_DIMS.items():
            if dims.get(dim, 0) < minimum:
                failures.append(f"dimension {dim!r} has size {dims.get(dim, 0)}")
        if dims.get("run_id", 0) <= 2:
            failures.append("raw corpus contains no more than two runs")

        missing_variables = [
            name for name in REQUIRED_VARIABLE_DIMS if name not in ds
        ]
        if missing_variables:
            failures.append(f"missing required variables: {missing_variables}")

        dimension_mismatches: dict[str, Any] = {}
        for name, expected_dims in REQUIRED_VARIABLE_DIMS.items():
            if name not in ds:
                continue
            actual_dims = tuple(ds[name].dims)
            if actual_dims != expected_dims:
                dimension_mismatches[name] = {
                    "expected": expected_dims,
                    "actual": actual_dims,
                }
        if dimension_mismatches:
            failures.append(f"variable dimension mismatches: {dimension_mismatches}")

        hospital_latent = next(
            (name for name in OPTIONAL_HOSPITAL_LATENTS if name in ds), None
        )
        if hospital_latent is None:
            warnings.append("no hospitalized latent is available")

        coordinates = {
            name: _coordinate_summary(ds[name].values)
            for name in ("run_id", "region_id", "origin", "target", "edar_id")
            if name in ds.coords
        }
        for name, summary in coordinates.items():
            if summary["unique_count"] != summary["count"]:
                failures.append(f"coordinate {name!r} contains duplicates")

        dates = np.asarray(ds["date"].values)
        date_ints = dates.astype("datetime64[ns]").astype(np.int64)
        if dates.size > 1 and not bool(np.all(np.diff(date_ints) > 0)):
            failures.append("date coordinate is not strictly increasing and unique")

        region_ids = [str(value).strip() for value in ds["region_id"].values]
        for coordinate_name in ("origin", "target"):
            ids = [str(value).strip() for value in ds[coordinate_name].values]
            if ids != region_ids:
                failures.append(
                    f"{coordinate_name} coordinate does not exactly match region_id"
                )

        metadata: dict[str, Any] = {}
        for name in RUN_METADATA:
            if name not in ds:
                warnings.append(f"run metadata {name!r} is absent")
                continue
            values = np.asarray(ds[name].values)
            metadata[name] = _json_value(values)
            if values.shape != (dims["run_id"],):
                failures.append(
                    f"run metadata {name!r} has shape {values.shape}, "
                    f"expected {(dims['run_id'],)}"
                )

        stats: dict[str, Any] = {}
        for name in STAT_VARIABLES:
            if name not in ds:
                continue
            stats[name] = _numeric_stats(ds[name])
            if name.startswith("latent_"):
                if stats[name]["finite_fraction"] < 1.0:
                    failures.append(f"{name} contains non-finite values")
                if stats[name]["negative"] > 0:
                    failures.append(f"{name} contains negative values")
            else:
                stats[name]["per_run_observed_fraction"] = (
                    _per_run_observed_fraction(ds[name])
                )
                if stats[name]["finite"] == 0:
                    failures.append(f"{name} has no finite observations")

        if hospital_latent is not None:
            stats[hospital_latent] = _numeric_stats(ds[hospital_latent])
            if stats[hospital_latent]["finite_fraction"] < 1.0:
                failures.append(f"{hospital_latent} contains non-finite values")
            if stats[hospital_latent]["negative"] > 0:
                failures.append(f"{hospital_latent} contains negative values")

        population_stats = _numeric_stats(ds["population"])
        if population_stats["finite_fraction"] < 1.0:
            failures.append("population contains non-finite values")
        if population_stats["min"] is None or population_stats["min"] <= 0:
            failures.append("population contains non-positive values")

        latent_names = [
            "latent_S_true",
            "latent_I_true",
            hospital_latent,
            "latent_R_true",
            "latent_D_true",
        ]
        latent_names = [name for name in latent_names if name is not None]
        population = ds["population"].astype(np.float64)
        compartment_sum = sum(ds[name].astype(np.float64) for name in latent_names)
        mass_ratio = compartment_sum / population
        mass_error = abs(mass_ratio - 1.0)
        mass_error_values = np.asarray(mass_error.compute().values)
        mass_summary = {
            "included_variables": latent_names,
            "ratio_min": float(mass_ratio.min().compute().item()),
            "ratio_max": float(mass_ratio.max().compute().item()),
            "ratio_mean": float(mass_ratio.mean().compute().item()),
            "absolute_error_mean": float(np.mean(mass_error_values)),
            "absolute_error_p99": float(np.quantile(mass_error_values, 0.99)),
        }
        if mass_summary["absolute_error_p99"] > 0.05:
            warnings.append(
                "S/I/H/R/D sum differs from population by more than 5% at p99; "
                "check whether additional mechanistic compartments explain the gap"
            )

        mobility_sample: dict[str, Any] = {}
        mobility_base = ds["mobility_base"]
        mobility_sample["base"] = _numeric_stats(mobility_base)
        mobility_sample["kappa0"] = _numeric_stats(ds["mobility_kappa0"])
        if "mobility_time_varying" in ds:
            sample_indices = {
                "run_id": sorted({0, dims["run_id"] // 2, dims["run_id"] - 1}),
                "date": sorted({0, dims["date"] // 2, dims["date"] - 1}),
                "origin": slice(0, min(32, dims["origin"])),
                "target": slice(0, min(32, dims["target"])),
            }
            mobility_sample["time_varying_sample"] = _numeric_stats(
                ds["mobility_time_varying"].isel(**sample_indices)
            )

        return {
            "schema_version": 1,
            "path": str(path),
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "warnings": warnings,
            "dimensions": dims,
            "variables": sorted(ds.data_vars),
            "coordinates": coordinates,
            "date": {
                "count": int(dates.size),
                "start": str(dates.min()),
                "end": str(dates.max()),
            },
            "run_metadata": metadata,
            "statistics": stats,
            "population": population_stats,
            "compartment_mass": mass_summary,
            "mobility": mobility_sample,
        }
    finally:
        ds.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = _json_value(validate(args.zarr.resolve()))
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
