#!/usr/bin/env python
"""Validate a processed synthetic corpus before DVC publication or training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


EXPECTED_LATENTS = (
    "latent_S_true",
    "latent_I_true",
    "latent_H_true",
    "latent_R_true",
    "latent_D_true",
)
EXPECTED_OBSERVED = ("cases", "hospitalizations", "deaths")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(store: Path) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "store": str(store),
        "failures": failures,
        "warnings": warnings,
    }

    if not store.is_dir():
        failures.append("processed Zarr store does not exist")
        report["status"] = "fail"
        return report

    metadata = store / ".zmetadata"
    if not metadata.is_file():
        failures.append("processed Zarr is missing consolidated .zmetadata")
        report["status"] = "fail"
        return report

    report["zmetadata_sha256"] = _sha256(metadata)
    ds = xr.open_zarr(store, consolidated=True, chunks="auto")
    try:
        report["sizes"] = {name: int(size) for name, size in ds.sizes.items()}
        expected_sizes = {
            "run_id": 20,
            "date": 456,
            "region_id": 945,
            "origin": 945,
            "destination": 945,
        }
        for dim, expected in expected_sizes.items():
            actual = ds.sizes.get(dim)
            if actual != expected:
                failures.append(f"{dim} size is {actual}, expected {expected}")

        run_ids = [str(value).strip() for value in ds["run_id"].values.tolist()]
        expected_run_ids = [f"{index}_Baseline" for index in range(20)]
        if set(run_ids) != set(expected_run_ids):
            failures.append("processed run IDs do not match all 20 Baseline runs")
        report["run_ids"] = run_ids

        dates = ds["date"].values
        if dates.size:
            report["date_start"] = str(dates[0])
            report["date_end"] = str(dates[-1])
        if dates.size != np.unique(dates).size:
            failures.append("processed dates are not unique")

        required = {
            *EXPECTED_OBSERVED,
            *(f"{name}_mask" for name in EXPECTED_OBSERVED),
            *EXPECTED_LATENTS,
            "mobility",
            "population",
            "temporal_covariates",
            "synthetic_sparsity_level",
            "valid_targets",
        }
        missing = sorted(required - set(ds.data_vars))
        if missing:
            failures.append(f"missing required variables: {missing}")

        variable_checks: dict[str, Any] = {}
        for name in (*EXPECTED_OBSERVED, *EXPECTED_LATENTS):
            if name not in ds:
                continue
            values = ds[name]
            finite_count = int(np.isfinite(values).sum().compute())
            negative_count = int((values < 0).sum().compute())
            variable_checks[name] = {
                "finite_count": finite_count,
                "size": int(values.size),
                "negative_count": negative_count,
            }
            if negative_count:
                failures.append(f"{name} contains {negative_count} negative values")
            if name in EXPECTED_LATENTS and finite_count != values.size:
                failures.append(f"{name} is not densely finite")

        report["variables"] = variable_checks

        if "temporal_covariates" in ds:
            covariates = [str(value) for value in ds["covariate"].values.tolist()]
            report["temporal_covariates"] = covariates
            if "lockdown_severity" not in covariates:
                failures.append("lockdown_severity temporal covariate is absent")
            else:
                lockdown = ds["temporal_covariates"].sel(covariate="lockdown_severity")
                lockdown_min = float(lockdown.min().compute())
                lockdown_max = float(lockdown.max().compute())
                report["lockdown_severity_range"] = [lockdown_min, lockdown_max]
                if lockdown_min != 0.0 or lockdown_max != 0.0:
                    failures.append(
                        "lockdown severity is nonzero despite raw mobility_kappa0=0"
                    )

        if "synthetic_sparsity_level" in ds:
            sparsity = np.asarray(ds["synthetic_sparsity_level"].compute())
            report["sparsity_range"] = [
                float(np.nanmin(sparsity)),
                float(np.nanmax(sparsity)),
            ]
            if not np.isfinite(sparsity).all():
                failures.append("synthetic_sparsity_level contains non-finite values")

        if "mobility" in ds:
            mobility = ds["mobility"]
            sample = mobility.isel(
                run_id=[0, mobility.sizes["run_id"] - 1],
                date=[0, mobility.sizes["date"] - 1],
                origin=slice(0, 32),
                destination=slice(0, 32),
            ).compute()
            report["mobility_sample"] = {
                "finite": bool(np.isfinite(sample).all()),
                "min": float(sample.min()),
                "max": float(sample.max()),
            }
            if not np.isfinite(sample).all():
                failures.append("mobility boundary sample contains non-finite values")
            if float(sample.min()) < 0:
                failures.append("mobility boundary sample contains negative values")

        report["attrs"] = {
            key: ds.attrs.get(key)
            for key in (
                "log_transformed",
                "population_norm",
                "smoothing_clinical_method",
                "wastewater_support",
            )
        }
        if ds.attrs.get("smoothing_clinical_method") != "holt_damped":
            failures.append("clinical smoothing provenance is not holt_damped")
    finally:
        ds.close()

    report["status"] = "pass" if not failures else "fail"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate(args.store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
