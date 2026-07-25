#!/usr/bin/env python
"""Audit cross-validation split representativity beyond mobility structure."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD


CLINICAL_SERIES = ("cases", "hospitalizations", "deaths")
WASTEWATER_COMPONENTS = (
    "edar_biomarker_N1",
    "edar_biomarker_N2",
    "edar_biomarker_IP4",
)
ALL_TARGET_SERIES = (*CLINICAL_SERIES, "wastewater")
ANALYSIS_CRS = "EPSG:3035"


@dataclass(frozen=True)
class TargetSeries:
    """Canonical target values and observation masks on log1p scale."""

    values: np.ndarray
    mask: np.ndarray
    raw_unit: str


@dataclass(frozen=True)
class MetricDefinition:
    """Region-level metric included in the balance audit."""

    column: str
    label: str
    unit: str
    transform: str
    additive: bool = False
    availability: bool = False


def _select_run(dataset: xr.Dataset, run_id: str) -> xr.Dataset:
    if "run_id" in dataset.dims:
        return dataset.sel(run_id=run_id)
    return dataset


def _load_target_values_and_mask(
    dataset: xr.Dataset,
    variable: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        dataset[variable].transpose(TEMPORAL_COORD, REGION_COORD).values,
        dtype=np.float64,
    )
    mask_name = f"{variable}_mask"
    if mask_name in dataset:
        mask = np.asarray(
            dataset[mask_name].transpose(TEMPORAL_COORD, REGION_COORD).values,
            dtype=bool,
        )
    else:
        mask = np.isfinite(values)
    mask &= np.isfinite(values) & (values >= 0.0)
    return values, mask


def combine_wastewater_target(dataset: xr.Dataset) -> TargetSeries:
    """Reproduce the model's masked mean across wastewater variants."""
    component_values: list[np.ndarray] = []
    component_masks: list[np.ndarray] = []
    for variable in WASTEWATER_COMPONENTS:
        if variable not in dataset:
            continue
        values, mask = _load_target_values_and_mask(dataset, variable)
        component_values.append(values)
        component_masks.append(mask)

    if not component_values:
        shape = (
            int(dataset.sizes[TEMPORAL_COORD]),
            int(dataset.sizes[REGION_COORD]),
        )
        return TargetSeries(
            values=np.full(shape, np.nan, dtype=np.float64),
            mask=np.zeros(shape, dtype=bool),
            raw_unit="wastewater signal",
        )

    stacked_values = np.stack(component_values, axis=0)
    stacked_masks = np.stack(component_masks, axis=0)
    valid_count = stacked_masks.sum(axis=0)
    combined_values = np.divide(
        np.where(stacked_masks, stacked_values, 0.0).sum(axis=0),
        np.where(valid_count > 0, valid_count, 1),
    )
    combined_mask = valid_count > 0
    combined_values = np.where(combined_mask, combined_values, np.nan)
    return TargetSeries(
        values=combined_values,
        mask=combined_mask,
        raw_unit="wastewater signal",
    )


def load_target_series(dataset: xr.Dataset) -> dict[str, TargetSeries]:
    """Load the four canonical supervision targets used by the model."""
    targets: dict[str, TargetSeries] = {}
    for variable in CLINICAL_SERIES:
        values, mask = _load_target_values_and_mask(dataset, variable)
        targets[variable] = TargetSeries(
            values=values,
            mask=mask,
            raw_unit="daily events per 100k",
        )
    targets["wastewater"] = combine_wastewater_target(dataset)
    return targets


def _load_region_area(
    geojson_path: Path,
    *,
    region_ids: list[str],
) -> np.ndarray:
    regions = gpd.read_file(geojson_path)
    if "id" not in regions:
        raise ValueError("GeoJSON must contain an 'id' property")
    regions["region_id"] = regions["id"].astype(str)
    regions = regions.set_index("region_id")
    missing = sorted(set(region_ids) - set(regions.index))
    if missing:
        raise ValueError(f"Missing geometries for region IDs: {missing[:10]}")
    ordered = regions.loc[region_ids].to_crs(ANALYSIS_CRS)
    area_km2 = ordered.geometry.area.to_numpy(dtype=np.float64) / 1_000_000.0
    if np.any(~np.isfinite(area_km2)) or np.any(area_km2 <= 0.0):
        raise ValueError("All valid regions must have positive finite area")
    return area_km2


def build_region_features(
    *,
    dataset: xr.Dataset,
    targets: dict[str, TargetSeries],
    geojson_path: Path,
) -> tuple[pd.DataFrame, list[MetricDefinition]]:
    """Build one representativity feature row per valid target region."""
    valid_mask = np.asarray(dataset["valid_targets"].values, dtype=bool)
    all_region_ids = np.asarray(dataset[REGION_COORD].values).astype(str)
    region_ids = all_region_ids[valid_mask].tolist()
    population = np.asarray(
        dataset["population"].values,
        dtype=np.float64,
    )[valid_mask]
    area_km2 = _load_region_area(geojson_path, region_ids=region_ids)

    features = pd.DataFrame(
        {
            "region_id": region_ids,
            "population": population,
            "area_km2": area_km2,
            "population_density_per_km2": population / area_km2,
        }
    )
    metric_definitions = [
        MetricDefinition(
            column="population",
            label="Population",
            unit="people",
            transform="log1p",
            additive=True,
        ),
        MetricDefinition(
            column="population_density_per_km2",
            label="Population density",
            unit="people per km2",
            transform="log1p",
        ),
    ]

    for series_name, target in targets.items():
        values = target.values[:, valid_mask]
        mask = target.mask[:, valid_mask]
        raw_values = np.expm1(np.where(mask, values, 0.0))
        observed_count = mask.sum(axis=0, dtype=np.int64)
        total_raw = raw_values.sum(axis=0, dtype=np.float64)
        mean_raw = np.divide(
            total_raw,
            observed_count,
            out=np.full_like(total_raw, np.nan),
            where=observed_count > 0,
        )
        availability = observed_count / mask.shape[0]

        features[f"{series_name}_observed_days"] = observed_count
        features[f"{series_name}_availability_fraction"] = availability
        features[f"{series_name}_mean_observed_raw"] = mean_raw
        features[f"{series_name}_total_observed_raw"] = total_raw

        metric_definitions.extend(
            [
                MetricDefinition(
                    column=f"{series_name}_availability_fraction",
                    label=f"{series_name.title()} availability",
                    unit="fraction of region-days",
                    transform="identity",
                    availability=True,
                ),
                MetricDefinition(
                    column=f"{series_name}_mean_observed_raw",
                    label=f"{series_name.title()} mean observed burden",
                    unit=target.raw_unit,
                    transform="log1p",
                ),
            ]
        )

        if series_name in CLINICAL_SERIES:
            count_equivalent = total_raw * population / 100_000.0
            column = f"{series_name}_total_observed_count_equivalent"
            features[column] = count_equivalent
            metric_definitions.append(
                MetricDefinition(
                    column=column,
                    label=f"{series_name.title()} observed load",
                    unit="count-equivalent events",
                    transform="log1p",
                    additive=True,
                )
            )
        else:
            metric_definitions.append(
                MetricDefinition(
                    column="wastewater_total_observed_raw",
                    label="Wastewater observed load",
                    unit="summed wastewater signal",
                    transform="log1p",
                    additive=True,
                )
            )

    return features, metric_definitions


def _transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values
    if transform == "log1p":
        return np.log1p(values)
    raise ValueError(f"Unknown transform: {transform}")


def standardized_mean_difference(
    sample: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return mean difference in reference standard-deviation units."""
    sample = sample[np.isfinite(sample)]
    reference = reference[np.isfinite(reference)]
    if sample.size == 0 or reference.size < 2:
        return np.nan
    scale = float(np.std(reference, ddof=1))
    difference = float(np.mean(sample) - np.mean(reference))
    if scale == 0.0:
        return 0.0 if difference == 0.0 else np.nan
    return difference / scale


def ks_statistic(sample: np.ndarray, reference: np.ndarray) -> float:
    """Return the two-sample Kolmogorov-Smirnov distance."""
    sample = np.sort(sample[np.isfinite(sample)])
    reference = np.sort(reference[np.isfinite(reference)])
    if sample.size == 0 or reference.size == 0:
        return np.nan
    support = np.sort(np.unique(np.concatenate([sample, reference])))
    sample_cdf = np.searchsorted(sample, support, side="right") / sample.size
    reference_cdf = np.searchsorted(reference, support, side="right") / reference.size
    return float(np.max(np.abs(sample_cdf - reference_cdf)))


def _validate_membership(
    membership: pd.DataFrame,
    *,
    region_ids: set[str],
) -> None:
    expected_roles = {"train", "validation", "test"}
    if set(membership["role"]) != expected_roles:
        raise ValueError(
            f"Expected roles {sorted(expected_roles)}, "
            f"found {sorted(membership['role'].unique())}"
        )
    duplicates = membership.duplicated(["split_id", "region_id"], keep=False)
    if duplicates.any():
        raise ValueError("Each region must have exactly one role per split")
    for split_id, split in membership.groupby("split_id", sort=False):
        observed = set(split["region_id"])
        if observed != region_ids:
            raise ValueError(f"{split_id} does not partition the valid-target universe")


def build_fold_role_summary(
    *,
    membership: pd.DataFrame,
    region_features: pd.DataFrame,
    metric_definitions: list[MetricDefinition],
) -> pd.DataFrame:
    """Summarize region distributions for every fold role."""
    joined = membership.merge(
        region_features,
        on="region_id",
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, object]] = []
    for (split_id, role), group in joined.groupby(
        ["split_id", "role"],
        sort=False,
    ):
        for metric in metric_definitions:
            values = group[metric.column].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            records.append(
                {
                    "split_id": split_id,
                    "role": role,
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "unit": metric.unit,
                    "region_count": len(group),
                    "finite_region_count": len(finite),
                    "mean": float(np.mean(finite)) if finite.size else np.nan,
                    "median": float(np.median(finite)) if finite.size else np.nan,
                    "p10": (float(np.quantile(finite, 0.1)) if finite.size else np.nan),
                    "p90": (float(np.quantile(finite, 0.9)) if finite.size else np.nan),
                    "sum": float(np.sum(finite)) if finite.size else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def build_test_fold_balance(
    *,
    membership: pd.DataFrame,
    region_features: pd.DataFrame,
    metric_definitions: list[MetricDefinition],
    role: str = "test",
) -> pd.DataFrame:
    """Compare one role in every split with the valid-target universe."""
    reference_by_region = region_features.set_index("region_id")
    valid_region_count = len(reference_by_region)
    records: list[dict[str, object]] = []
    role_membership = membership[membership["role"] == role]
    for split_id, split in role_membership.groupby("split_id", sort=False):
        sample = reference_by_region.loc[split["region_id"]]
        expected_share = len(sample) / valid_region_count
        for metric in metric_definitions:
            sample_raw = sample[metric.column].to_numpy(dtype=np.float64)
            reference_raw = reference_by_region[metric.column].to_numpy(
                dtype=np.float64
            )
            sample_values = _transform(sample_raw, metric.transform)
            reference_values = _transform(reference_raw, metric.transform)
            reference_mean = float(np.nanmean(reference_raw))
            sample_mean = float(np.nanmean(sample_raw))
            relative_mean_difference = (
                sample_mean / reference_mean - 1.0 if reference_mean != 0.0 else np.nan
            )

            total_share = np.nan
            share_ratio = np.nan
            if metric.additive:
                reference_total = float(np.nansum(reference_raw))
                total_share = (
                    float(np.nansum(sample_raw)) / reference_total
                    if reference_total != 0.0
                    else np.nan
                )
                share_ratio = (
                    total_share / expected_share if expected_share > 0.0 else np.nan
                )

            records.append(
                {
                    "split_id": split_id,
                    "role": role,
                    "role_region_count": len(sample),
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "unit": metric.unit,
                    "analysis_transform": metric.transform,
                    "test_mean": sample_mean,
                    "universe_mean": reference_mean,
                    "relative_mean_difference": relative_mean_difference,
                    "standardized_mean_difference": standardized_mean_difference(
                        sample_values,
                        reference_values,
                    ),
                    "ks_statistic": ks_statistic(
                        sample_values,
                        reference_values,
                    ),
                    "expected_share_by_region_count": expected_share,
                    "test_total_share": total_share,
                    "share_ratio_to_region_count": share_ratio,
                    "availability_difference_pp": (
                        100.0 * (sample_mean - reference_mean)
                        if metric.availability
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def build_metric_summary(test_balance: pd.DataFrame) -> pd.DataFrame:
    """Collapse fold-level diagnostics to worst-case metric balance."""
    records: list[dict[str, object]] = []
    for metric, group in test_balance.groupby("metric", sort=False):
        first = group.iloc[0]
        share_ratios = group["share_ratio_to_region_count"].dropna()
        availability_differences = group["availability_difference_pp"].dropna()
        records.append(
            {
                "metric": metric,
                "metric_label": first["metric_label"],
                "unit": first["unit"],
                "analysis_transform": first["analysis_transform"],
                "mean_abs_standardized_difference": float(
                    group["standardized_mean_difference"].abs().mean()
                ),
                "max_abs_standardized_difference": float(
                    group["standardized_mean_difference"].abs().max()
                ),
                "max_ks_statistic": float(group["ks_statistic"].max()),
                "min_relative_mean_difference": float(
                    group["relative_mean_difference"].min()
                ),
                "max_relative_mean_difference": float(
                    group["relative_mean_difference"].max()
                ),
                "min_share_ratio_to_region_count": (
                    float(share_ratios.min()) if not share_ratios.empty else np.nan
                ),
                "max_share_ratio_to_region_count": (
                    float(share_ratios.max()) if not share_ratios.empty else np.nan
                ),
                "max_abs_availability_difference_pp": (
                    float(availability_differences.abs().max())
                    if not availability_differences.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_temporal_target_summary(
    *,
    membership: pd.DataFrame,
    dataset: xr.Dataset,
    targets: dict[str, TargetSeries],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize daily target availability and observed magnitude by fold role."""
    region_ids = np.asarray(dataset[REGION_COORD].values).astype(str)
    region_index = {region_id: index for index, region_id in enumerate(region_ids)}
    dates = pd.DatetimeIndex(dataset[TEMPORAL_COORD].values)
    valid_ids = sorted(set(membership["region_id"]))
    valid_indices = np.asarray(
        [region_index[region_id] for region_id in valid_ids],
        dtype=np.int64,
    )

    universe_daily = {}
    for series_name, target in targets.items():
        universe_daily[series_name] = target.mask[:, valid_indices].mean(axis=1)

    records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for (split_id, role), group in membership.groupby(
        ["split_id", "role"],
        sort=False,
    ):
        indices = np.asarray(
            [region_index[region_id] for region_id in group["region_id"]],
            dtype=np.int64,
        )
        for series_name, target in targets.items():
            mask = target.mask[:, indices]
            values = target.values[:, indices]
            raw_values = np.expm1(np.where(mask, values, 0.0))
            observed_count = mask.sum(axis=1, dtype=np.int64)
            total_raw = raw_values.sum(axis=1, dtype=np.float64)
            mean_observed_raw = np.divide(
                total_raw,
                observed_count,
                out=np.full_like(total_raw, np.nan),
                where=observed_count > 0,
            )
            daily_availability = observed_count / len(indices)

            records.extend(
                {
                    "split_id": split_id,
                    "role": role,
                    "date": date,
                    "series": series_name,
                    "region_count": len(indices),
                    "observed_region_count": int(count),
                    "availability_fraction": availability,
                    "mean_observed_raw": mean_raw,
                    "total_observed_raw": total,
                }
                for date, count, availability, mean_raw, total in zip(
                    dates,
                    observed_count,
                    daily_availability,
                    mean_observed_raw,
                    total_raw,
                    strict=True,
                )
            )

            weekly = (
                pd.DataFrame(
                    {
                        "role": daily_availability,
                        "universe": universe_daily[series_name],
                    },
                    index=dates,
                )
                .resample("W-SUN")
                .mean()
            )
            summary_records.append(
                {
                    "split_id": split_id,
                    "role": role,
                    "series": series_name,
                    "availability_fraction": float(mask.mean()),
                    "universe_availability_fraction": float(
                        universe_daily[series_name].mean()
                    ),
                    "availability_difference_pp": float(
                        100.0 * (mask.mean() - universe_daily[series_name].mean())
                    ),
                    "max_abs_weekly_availability_difference_pp": float(
                        100.0 * (weekly["role"] - weekly["universe"]).abs().max()
                    ),
                }
            )
    return (
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(summary_records),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", default="real")
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    membership = pd.read_csv(args.membership, dtype={"region_id": str})
    membership = membership[membership["strategy"] == args.strategy].copy()
    if membership.empty:
        raise ValueError(f"Strategy not found in membership table: {args.strategy}")

    dataset = xr.open_zarr(args.dataset, consolidated=True)
    try:
        selected = _select_run(dataset, args.run_id)
        targets = load_target_series(selected)
        region_features, metric_definitions = build_region_features(
            dataset=selected,
            targets=targets,
            geojson_path=args.geojson,
        )
        _validate_membership(
            membership,
            region_ids=set(region_features["region_id"]),
        )
        fold_role_summary = build_fold_role_summary(
            membership=membership,
            region_features=region_features,
            metric_definitions=metric_definitions,
        )
        fold_role_balance = pd.concat(
            [
                build_test_fold_balance(
                    membership=membership,
                    region_features=region_features,
                    metric_definitions=metric_definitions,
                    role=role,
                )
                for role in ("train", "validation", "test")
            ],
            ignore_index=True,
        )
        test_fold_balance = fold_role_balance[
            fold_role_balance["role"] == "test"
        ].copy()
        metric_summary = build_metric_summary(test_fold_balance)
        temporal_targets, temporal_availability = build_temporal_target_summary(
            membership=membership,
            dataset=selected,
            targets=targets,
        )
        date_start = str(pd.Timestamp(selected[TEMPORAL_COORD].values[0]).date())
        date_end = str(pd.Timestamp(selected[TEMPORAL_COORD].values[-1]).date())
    finally:
        dataset.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    region_features.to_csv(args.output_dir / "region_features.csv", index=False)
    fold_role_summary.to_csv(
        args.output_dir / "fold_role_summary.csv",
        index=False,
    )
    fold_role_balance.to_csv(
        args.output_dir / "fold_role_balance.csv",
        index=False,
    )
    test_fold_balance.to_csv(
        args.output_dir / "test_fold_balance.csv",
        index=False,
    )
    metric_summary.to_csv(args.output_dir / "metric_summary.csv", index=False)
    temporal_targets.to_csv(
        args.output_dir / "temporal_target_summary.csv",
        index=False,
    )
    temporal_availability.to_csv(
        args.output_dir / "temporal_availability_summary.csv",
        index=False,
    )

    summary = {
        "schema_version": 1,
        "strategy": args.strategy,
        "dataset": str(args.dataset),
        "run_id": args.run_id,
        "membership": str(args.membership),
        "geojson": str(args.geojson),
        "analysis_crs": ANALYSIS_CRS,
        "date_start": date_start,
        "date_end": date_end,
        "valid_region_count": int(region_features["region_id"].nunique()),
        "split_count": int(membership["split_id"].nunique()),
        "target_series": list(ALL_TARGET_SERIES),
        "clinical_value_definition": (
            "expm1(canonical log1p per-100k target); observed masks only"
        ),
        "wastewater_value_definition": (
            "expm1(masked mean of available canonical log1p N1/N2/IP4 targets)"
        ),
        "population_density_definition": (
            f"population divided by polygon area after reprojection to {ANALYSIS_CRS}"
        ),
        "standardized_difference_definition": (
            "(test-fold mean - valid-universe mean) / valid-universe SD; "
            "log1p scale for skewed size and burden metrics"
        ),
        "share_ratio_definition": (
            "test-fold share of additive total divided by its share of valid regions"
        ),
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(metric_summary.to_string(index=False))


if __name__ == "__main__":
    main()
