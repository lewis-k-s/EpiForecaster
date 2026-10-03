"""Canonical source-native wastewater observations at EDAR support.

This processor keeps concentrations at plant level. It does not broadcast or
average EDAR measurements into municipalities. Raw concentrations, detection
limits, flow, masks, and censor codes remain available for target likelihoods.
A fixed-parameter causal Tobit-Kalman history is stored separately for model
conditioning. No filtered value before the first sample is exposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from constants import EDAR_BIOMARKER_VARIANTS
from data.native_registry import STUDY_WINDOW_END, STUDY_WINDOW_START
from data.preprocess.smoothing import TobitKalmanSmoother

SCHEMA = "native_wastewater_observations.v1"
DEFAULT_PROCESS_VAR = 0.05
DEFAULT_MEASUREMENT_VAR = 0.5
DEFAULT_CENSOR_INFLATION = 4.0
DEFAULT_INPUT = Path("data/files/wastewater_biomarkers_icra.csv")
DEFAULT_OUTPUT = Path("data/processed/native_support/obs_wastewater.zarr")

_RAW_COLUMNS = {
    "id mostra": "sample_id",
    "depuradora": "plant_name",
    "LD(CG/L)": "detection_limit",
    "Cabal últimes 24h(m3)": "flow_m3_24h",
    "Pluja(mm)": "rainfall_mm",
}
_ASSAY_COLUMNS = {assay: f"{assay}(CG/L)" for assay in EDAR_BIOMARKER_VARIANTS}


def _age_since_observation(mask: np.ndarray) -> np.ndarray:
    """Days since the last sample, with NaN before the first sample."""
    age = np.full(mask.shape, np.nan, dtype=np.float32)
    last: int | None = None
    for index, observed in enumerate(mask.tolist()):
        if observed:
            last = index
            age[index] = 0.0
        elif last is not None:
            age[index] = float(index - last)
    return age


def _causal_history(
    values: np.ndarray,
    limits: np.ndarray,
    *,
    process_var: float,
    measurement_var: float,
    censor_inflation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtered ``log1p(concentration)`` and a past-observation mask."""
    history = np.full(values.shape, np.nan, dtype=np.float32)
    available = np.zeros(values.shape, dtype=bool)
    observed = np.flatnonzero(np.isfinite(values) & (values >= 0))
    if observed.size == 0:
        return history, available
    first = int(observed[0])
    smoother = TobitKalmanSmoother(
        process_var=process_var,
        measurement_var=measurement_var,
        censor_inflation=censor_inflation,
    )
    filtered_log, _flags = smoother.filter_series(values[first:], limits[first:])
    history[first:] = np.log1p(np.exp(np.asarray(filtered_log, dtype=np.float64)))
    available[first:] = True
    return history, available


def build_native_wastewater_observations(
    wastewater_csv_path: str | Path,
    *,
    start_date: str | np.datetime64 = str(STUDY_WINDOW_START),
    end_date: str | np.datetime64 = str(STUDY_WINDOW_END),
    process_var: float = DEFAULT_PROCESS_VAR,
    measurement_var: float = DEFAULT_MEASUREMENT_VAR,
    censor_inflation: float = DEFAULT_CENSOR_INFLATION,
) -> xr.Dataset:
    """Build the daily EDAR observation store from the raw ICRA CSV."""
    source = pd.read_csv(wastewater_csv_path)
    required = set(_RAW_COLUMNS) | set(_ASSAY_COLUMNS.values())
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Wastewater CSV missing columns: {missing}")

    sample = source[list(_RAW_COLUMNS) + list(_ASSAY_COLUMNS.values())].rename(
        columns=_RAW_COLUMNS
    )
    identity = (
        sample["sample_id"]
        .astype(str)
        .str.extract(r"^(?P<edar_id>[A-Za-z0-9_]+)-(?P<date>\d{4}-\d{2}-\d{2})$")
    )
    invalid = identity.isna().any(axis=1)
    if invalid.any():
        examples = sample.loc[invalid, "sample_id"].head(5).tolist()
        raise ValueError(f"Invalid EDAR sample identifiers: {examples}")
    sample["edar_id"] = identity["edar_id"].astype(str)
    sample["date"] = pd.to_datetime(identity["date"], format="%Y-%m-%d")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("Wastewater end_date precedes start_date")
    sample = sample[(sample["date"] >= start) & (sample["date"] <= end)].copy()
    if sample.empty:
        raise ValueError("No wastewater samples inside the requested date window")

    numeric_columns = [
        "detection_limit",
        "flow_m3_24h",
        "rainfall_mm",
        *_ASSAY_COLUMNS.values(),
    ]
    for column in numeric_columns:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    nonnegative = ["detection_limit", "flow_m3_24h", *_ASSAY_COLUMNS.values()]
    for column in nonnegative:
        bad = sample[column].notna() & (sample[column] < 0)
        if bad.any():
            raise ValueError(f"Wastewater column {column!r} contains negative values")

    assay_frame = sample.melt(
        id_vars=[
            "edar_id",
            "date",
            "plant_name",
            "detection_limit",
            "flow_m3_24h",
            "rainfall_mm",
        ],
        value_vars=list(_ASSAY_COLUMNS.values()),
        var_name="assay_column",
        value_name="concentration_raw",
    )
    column_to_assay = {column: assay for assay, column in _ASSAY_COLUMNS.items()}
    assay_frame["assay"] = assay_frame["assay_column"].map(column_to_assay)
    assay_daily = (
        assay_frame.groupby(["date", "edar_id", "assay"], as_index=False)
        .agg(
            concentration_raw=("concentration_raw", "mean"),
            detection_limit=("detection_limit", "max"),
        )
        .sort_values(["date", "edar_id", "assay"], kind="stable")
    )
    site_daily = (
        sample.groupby(["date", "edar_id"], as_index=False)
        .agg(
            flow_m3_24h=("flow_m3_24h", "mean"),
            rainfall_mm=("rainfall_mm", "mean"),
        )
        .sort_values(["date", "edar_id"], kind="stable")
    )

    dates = pd.date_range(start, end, freq="D").to_numpy(dtype="datetime64[ns]")
    edar_ids = sorted(sample["edar_id"].unique().astype(str).tolist())
    assays = list(EDAR_BIOMARKER_VARIANTS)
    shape = (len(dates), len(edar_ids), len(assays))
    concentration = np.full(shape, np.nan, dtype=np.float32)
    limit = np.full(shape, np.nan, dtype=np.float32)
    flow = np.full((len(dates), len(edar_ids)), np.nan, dtype=np.float32)
    rainfall = np.full_like(flow, np.nan)
    date_index = {pd.Timestamp(value): index for index, value in enumerate(dates)}
    edar_index = {edar_id: index for index, edar_id in enumerate(edar_ids)}
    assay_index = {assay: index for index, assay in enumerate(assays)}

    for row in assay_daily.itertuples(index=False):
        indices = (
            date_index[pd.Timestamp(row.date)],
            edar_index[str(row.edar_id)],
            assay_index[str(row.assay)],
        )
        concentration[indices] = np.float32(row.concentration_raw)
        limit[indices] = np.float32(row.detection_limit)
    for row in site_daily.itertuples(index=False):
        indices = date_index[pd.Timestamp(row.date)], edar_index[str(row.edar_id)]
        flow[indices] = np.float32(row.flow_m3_24h)
        rainfall[indices] = np.float32(row.rainfall_mm)

    observed_mask = np.isfinite(concentration)
    flow_mask = np.isfinite(flow)
    censor_code = np.full(shape, 2, dtype=np.uint8)
    uncensored = observed_mask & (~np.isfinite(limit) | (concentration > limit))
    censored = observed_mask & np.isfinite(limit) & (concentration <= limit)
    censor_code[uncensored] = 0
    censor_code[censored] = 1

    history = np.full(shape, np.nan, dtype=np.float32)
    history_available = np.zeros(shape, dtype=bool)
    age = np.full(shape, np.nan, dtype=np.float32)
    for edar_position in range(len(edar_ids)):
        for assay_position in range(len(assays)):
            values = concentration[:, edar_position, assay_position]
            limits = limit[:, edar_position, assay_position]
            filtered, available = _causal_history(
                values,
                limits,
                process_var=process_var,
                measurement_var=measurement_var,
                censor_inflation=censor_inflation,
            )
            history[:, edar_position, assay_position] = filtered
            history_available[:, edar_position, assay_position] = available
            age[:, edar_position, assay_position] = _age_since_observation(
                observed_mask[:, edar_position, assay_position]
            )

    plant_names = (
        sample.groupby("edar_id")["plant_name"]
        .agg(
            lambda values: sorted(set(map(str, values.dropna())))[0]
            if values.notna().any()
            else ""
        )
        .reindex(edar_ids)
        .to_numpy(dtype=str)
    )
    return xr.Dataset(
        {
            "concentration_raw": (
                ("date", "edar_id", "assay"),
                concentration,
            ),
            "detection_limit": (("date", "edar_id", "assay"), limit),
            "observed_mask": (("date", "edar_id", "assay"), observed_mask),
            "censor_code": (("date", "edar_id", "assay"), censor_code),
            "history_log1p": (("date", "edar_id", "assay"), history),
            "history_available_mask": (
                ("date", "edar_id", "assay"),
                history_available,
            ),
            "history_age_days": (("date", "edar_id", "assay"), age),
            "flow_m3_24h": (("date", "edar_id"), flow),
            "flow_mask": (("date", "edar_id"), flow_mask),
            "rainfall_mm": (("date", "edar_id"), rainfall),
            "plant_name": (("edar_id",), plant_names),
        },
        coords={"date": dates, "edar_id": edar_ids, "assay": assays},
        attrs={
            "schema": SCHEMA,
            "target_support": "edar",
            "target_unit": "copies_per_litre",
            "history_transform": "fixed_parameter_causal_tobit_kalman_then_log1p",
            "process_var": float(process_var),
            "measurement_var": float(measurement_var),
            "censor_inflation": float(censor_inflation),
            "availability_date_status": "not_provided_by_source",
            "sample_date_coordinate": "date",
            "censor_code_definition": "0=uncensored,1=left_censored_at_detection_limit,2=missing",
        },
    )


def write_native_wastewater_observations(
    dataset: xr.Dataset, output_path: str | Path
) -> None:
    """Write the validated EDAR store as zarr v3."""
    if dataset.attrs.get("schema") != SCHEMA:
        raise ValueError(f"Expected wastewater schema {SCHEMA!r}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(output, mode="w", zarr_format=3)


def rebuild_native_wastewater_histories(dataset: xr.Dataset) -> xr.Dataset:
    """Repair derived histories from preserved raw observations and parameters."""
    if dataset.attrs.get("schema") != SCHEMA:
        raise ValueError(f"Expected wastewater schema {SCHEMA!r}")
    parameters = {
        name: float(dataset.attrs[name])
        for name in ("process_var", "measurement_var", "censor_inflation")
    }
    if any(not np.isfinite(value) or value <= 0 for value in parameters.values()):
        raise ValueError("Wastewater filter parameters must be finite and positive")
    dimensions = ("date", "edar_id", "assay")
    for name in ("concentration_raw", "detection_limit", "history_log1p"):
        if dataset[name].dims != dimensions:
            raise ValueError(f"Unexpected wastewater dimensions for {name}")
    repaired = dataset.copy(deep=True)
    history = np.full(dataset["history_log1p"].shape, np.nan, dtype=np.float32)
    for plant in range(dataset.sizes["edar_id"]):
        for assay in range(dataset.sizes["assay"]):
            filtered, available = _causal_history(
                dataset["concentration_raw"].values[:, plant, assay],
                dataset["detection_limit"].values[:, plant, assay],
                **parameters,
            )
            if not np.array_equal(
                available, dataset["history_available_mask"].values[:, plant, assay]
            ):
                raise ValueError("History repair would change the availability mask")
            history[:, plant, assay] = filtered
    repaired["history_log1p"].data = history
    return repaired


def qa_native_wastewater_history_repair(
    parent: xr.Dataset, repaired: xr.Dataset
) -> dict:
    """Check preservation and prefix causality, including poisoned future values."""
    xr.testing.assert_identical(
        parent.drop_vars("history_log1p"), repaired.drop_vars("history_log1p")
    )
    expected = rebuild_native_wastewater_histories(parent)
    xr.testing.assert_identical(expected, repaired)
    parameters = {
        name: float(parent.attrs[name])
        for name in ("process_var", "measurement_var", "censor_inflation")
    }
    changed_series = []
    checks = 0
    for plant in range(parent.sizes["edar_id"]):
        for assay in range(parent.sizes["assay"]):
            values = parent["concentration_raw"].values[:, plant, assay]
            limits = parent["detection_limit"].values[:, plant, assay]
            history = repaired["history_log1p"].values[:, plant, assay]
            previous = parent["history_log1p"].values[:, plant, assay]
            equal = np.isclose(history, previous, rtol=1e-6, atol=1e-6, equal_nan=True)
            if not equal.all():
                changed_series.append(
                    {
                        "edar_id": str(parent["edar_id"].values[plant]),
                        "assay": str(parent["assay"].values[assay]),
                        "changed_days": int((~equal).sum()),
                        "max_absolute_change": float(
                            np.nanmax(np.abs(history - previous))
                        ),
                    }
                )
            # Check shared calendar cuts and each series' initialization boundary.
            cuts = {len(values) // 4, len(values) // 2, 3 * len(values) // 4}
            for mask in (np.isfinite(values), np.isfinite(values) & (values > 0)):
                rows = np.flatnonzero(mask)
                if rows.size:
                    cuts.update((int(rows[0]), int(rows[0]) + 1))
            for stop in sorted(cuts):
                if not 0 < stop < len(values):
                    continue
                prefix, _ = _causal_history(values[:stop], limits[:stop], **parameters)
                poisoned = values.copy()
                poisoned[stop:][np.isfinite(poisoned[stop:])] = 1e20
                poison_history, _ = _causal_history(poisoned, limits, **parameters)
                if not (
                    np.array_equal(history[:stop], prefix, equal_nan=True)
                    and np.array_equal(
                        history[:stop], poison_history[:stop], equal_nan=True
                    )
                ):
                    raise ValueError(
                        "Wastewater histories depend on future observations"
                    )
                checks += 1
    return {
        "qa_passed": True,
        "dimensions": dict(repaired.sizes),
        "preserved_variables": sorted(parent.data_vars.keys() - {"history_log1p"}),
        "coordinates_and_attributes_preserved": True,
        "parameters": parameters,
        "changed_series": changed_series,
        "changed_series_count": len(changed_series),
        "causality_checks": checks,
        "future_poison_value": 1e20,
        "causality_policy": "calendar quartiles and first-sample/first-positive boundaries",
    }


def main(argv: list[str] | None = None) -> int:
    """CLI for the canonical source-native wastewater observation store."""
    parser = argparse.ArgumentParser(
        description="Build EDAR-level native wastewater observations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default=str(STUDY_WINDOW_START))
    parser.add_argument("--end-date", default=str(STUDY_WINDOW_END))
    args = parser.parse_args(argv)
    dataset = build_native_wastewater_observations(
        args.input, start_date=args.start_date, end_date=args.end_date
    )
    write_native_wastewater_observations(dataset, args.output)
    summary = {
        "output": str(args.output),
        "schema": SCHEMA,
        "sizes": dict(dataset.sizes),
        "observations": int(dataset["observed_mask"].sum()),
        "censored": int((dataset["censor_code"] == 1).sum()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
