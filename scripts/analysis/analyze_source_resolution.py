#!/usr/bin/env python3
"""Static source-resolution diagnostics for the full-coverage loss campaign.

The analysis keeps trained forecasts fixed. It recovers the exact geographic
operators used to project ABS hospitalizations and comarca deaths onto
municipalities, then asks two questions:

1. Does mobility change hospitalization point accuracy when fold-safe municipal
   forecasts are mapped forward to the native ABS support?
2. Does the known mobility benefit for municipal cases weaken when the case
   labels are artificially passed through the hospitalization or death
   geographic operators?

Native-source probabilistic scores are deliberately not constructed: marginal
municipality distributions do not identify their cross-municipality dependence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

WIS_INTERVAL_LEVELS = np.asarray(
    (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98),
    dtype=float,
)
IDENTITY_COLUMNS = ("condition", "fold", "seed")
PAIR_COLUMNS = ("family", "objective", "fold", "seed")
GRANULAR_COLUMNS = (
    "target",
    "region_id",
    "target_date",
    "observed",
    "abs_error",
    "wis",
    "distribution_loc",
    "distribution_scale",
    "distribution_family",
    "distribution_df",
    "interval_low_50",
    "interval_high_50",
    "interval_low_80",
    "interval_high_80",
    "interval_low_95",
    "interval_high_95",
)


@dataclass(frozen=True)
class ForecastIdentity:
    condition: str
    fold: int
    seed: int
    path: Path
    evidence_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose geographic support mismatch without retraining."
    )
    parser.add_argument(
        "--reuse-manifest",
        type=Path,
        default=Path(
            "outputs/loss_research/loss-type-full-coverage-v2/"
            "reuse_evidence_v1/manifest.json"
        ),
    )
    parser.add_argument(
        "--trained-root",
        type=Path,
        default=Path(
            "outputs/loss_research/loss-type-full-coverage-v2/"
            "source_resolution_input/r14_training"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "outputs/loss_research/loss-type-research-v1/real_loss_fit20200930_v1.nc"
        ),
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=Path("configs/crossval/representative_10fold_mobility_refined.json"),
    )
    parser.add_argument(
        "--cases-source",
        type=Path,
        default=Path(
            "outputs/loss_research/loss-type-research-v1/raw_inputs/"
            "Registre_de_casos_de_COVID-19_a_Catalunya_per_municipi_i_sexe.csv"
        ),
    )
    parser.add_argument(
        "--hospitalizations-source",
        type=Path,
        default=Path(
            "../dasymetric_mob/data/raw/COVID-19__Persones_hospitalitzades.csv"
        ),
    )
    parser.add_argument(
        "--deaths-source",
        type=Path,
        default=Path(
            "../dasymetric_mob/data/raw/"
            "Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
        ),
    )
    parser.add_argument(
        "--abs-mapping",
        type=Path,
        default=Path("../dasymetric_mob/data/output/abs_to_municipality_mapping.csv"),
    )
    parser.add_argument(
        "--comarca-mapping",
        type=Path,
        default=Path(
            "../dasymetric_mob/data/output/comarca_to_municipality_mapping.csv"
        ),
    )
    parser.add_argument(
        "--campaign-hospitalizations",
        type=Path,
        default=Path(
            "outputs/loss_research/loss-type-research-v1/raw_inputs/"
            "hospitalizations_municipality.csv"
        ),
    )
    parser.add_argument(
        "--campaign-deaths",
        type=Path,
        default=Path(
            "outputs/loss_research/loss-type-research-v1/raw_inputs/"
            "deaths_municipality.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/loss_research/loss-type-full-coverage-v2/"
            "source_resolution_analysis"
        ),
    )
    parser.add_argument("--hospitalization-lag-days", type=int, default=7)
    parser.add_argument("--source-coverage-threshold", type=float, default=0.99)
    parser.add_argument("--expected-identities", type=int, default=160)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_code(values: pd.Series, width: int) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(width)
    )


def load_mapping(path: Path, source_col: str, source_width: int) -> pd.DataFrame:
    mapping = pd.read_csv(
        path,
        dtype={"municipality_code": "string", source_col: "string"},
    )
    required = {
        "municipality_code",
        source_col,
        "weight",
        "overlap_population",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    mapping = mapping.copy()
    mapping["municipality_code"] = normalize_code(mapping["municipality_code"], 5)
    mapping[source_col] = normalize_code(mapping[source_col], source_width)
    mapping["forward_weight"] = pd.to_numeric(mapping["weight"], errors="raise")
    mapping["overlap_population"] = pd.to_numeric(
        mapping["overlap_population"], errors="raise"
    )
    source_denominator = mapping.groupby(source_col)["overlap_population"].transform(
        "sum"
    )
    if (source_denominator <= 0).any():
        raise ValueError(f"{path} has a source with non-positive mapped population")
    mapping["source_weight"] = mapping["overlap_population"] / source_denominator

    municipality_sums = mapping.groupby("municipality_code")["forward_weight"].sum()
    source_sums = mapping.groupby(source_col)["source_weight"].sum()
    if not np.allclose(municipality_sums, 1.0, atol=1e-8):
        raise ValueError(f"{path} forward weights do not sum to one by municipality")
    if not np.allclose(source_sums, 1.0, atol=1e-8):
        raise ValueError(f"{path} source weights do not sum to one by source")
    return mapping


def load_forecast_identities(
    reuse_manifest: Path,
    trained_root: Path,
    expected_identities: int,
) -> list[ForecastIdentity]:
    payload = json.loads(reuse_manifest.read_text())
    identities: list[ForecastIdentity] = []
    for record in payload["records"]:
        granular_paths = [
            item["path"]
            for item in record["evidence_files"]
            if str(item.get("path", "")).endswith("/test_granular.parquet")
        ]
        if len(granular_paths) != 1:
            raise ValueError("Each reuse record must identify one granular artifact")
        identities.append(
            ForecastIdentity(
                condition=str(record["condition"]),
                fold=int(record["fold"]),
                seed=int(record["seed"]),
                path=reuse_manifest.parent / granular_paths[0],
                evidence_kind="reused",
            )
        )

    for identity_path in trained_root.rglob("campaign_identity.json"):
        record = json.loads(identity_path.read_text())
        identities.append(
            ForecastIdentity(
                condition=str(record["condition"]),
                fold=int(record["fold"]),
                seed=int(record["seed"]),
                path=identity_path.parent / "test_granular.parquet",
                evidence_kind="trained",
            )
        )

    keys = [(item.condition, item.fold, item.seed) for item in identities]
    if len(keys) != expected_identities or len(set(keys)) != expected_identities:
        raise ValueError(
            f"Expected {expected_identities} unique identities, found "
            f"{len(keys)} records and {len(set(keys))} unique keys"
        )
    missing = [str(item.path) for item in identities if not item.path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing granular artifacts: {missing[:5]}")
    return identities


def load_granular(identities: list[ForecastIdentity]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for identity in identities:
        frame = pd.read_parquet(identity.path, columns=list(GRANULAR_COLUMNS))
        frames.append(
            frame.assign(
                condition=identity.condition,
                fold=identity.fold,
                seed=identity.seed,
                evidence_kind=identity.evidence_kind,
            )
        )
    granular = pd.concat(frames, ignore_index=True)
    granular["region_id"] = normalize_code(granular["region_id"], 5)
    granular["target_date"] = pd.to_datetime(granular["target_date"])
    if granular.empty:
        raise ValueError("No granular forecasts were loaded")
    return granular


def inverse_clinical(values: pd.Series, population: pd.Series) -> np.ndarray:
    return (
        np.expm1(values.to_numpy(dtype=float)) * population.to_numpy(dtype=float) / 1e5
    )


def load_hospital_source(path: Path, lag_days: int) -> pd.DataFrame:
    source = pd.read_csv(
        path,
        usecols=["data_inici", "codi_abs", "casos"],
        dtype={"codi_abs": "string"},
    )
    source["abs_code"] = normalize_code(source["codi_abs"], 3)
    source["target_date"] = pd.to_datetime(
        source["data_inici"], format="%d/%m/%Y", errors="raise"
    ) + pd.Timedelta(days=lag_days)
    source["observed_count"] = pd.to_numeric(source["casos"], errors="raise")
    return (
        source.groupby(["abs_code", "target_date"], as_index=False)["observed_count"]
        .sum()
        .sort_values(["abs_code", "target_date"])
    )


def load_death_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(
        path,
        usecols=["Data defunció", "Codi Comarca", "Nombre defuncions"],
        dtype={"Codi Comarca": "string"},
    )
    source["comarca_code"] = normalize_code(source["Codi Comarca"], 2)
    source["target_date"] = pd.to_datetime(
        source["Data defunció"], format="%d/%m/%Y", errors="raise"
    )
    source["observed_count"] = pd.to_numeric(
        source["Nombre defuncions"], errors="raise"
    )
    return (
        source.groupby(["comarca_code", "target_date"], as_index=False)[
            "observed_count"
        ]
        .sum()
        .sort_values(["comarca_code", "target_date"])
    )


def source_to_municipality(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
    source_col: str,
) -> pd.DataFrame:
    projected = source.merge(
        mapping[[source_col, "municipality_code", "source_weight"]],
        on=source_col,
        how="inner",
        validate="many_to_many",
    )
    projected["projected_count"] = (
        projected["observed_count"] * projected["source_weight"]
    )
    return projected.groupby(["target_date", "municipality_code"], as_index=False)[
        "projected_count"
    ].sum()


def dataset_observed_counts(
    dataset: xr.Dataset,
    value_name: str,
    mask_name: str,
) -> pd.DataFrame:
    values = dataset[value_name]
    masks = dataset[mask_name]
    if "run_id" in values.dims:
        values = values.sel(run_id="real")
    if "run_id" in masks.dims:
        masks = masks.sel(run_id="real")

    dates = pd.to_datetime(dataset["date"].values)
    regions = dataset["region_id"].values.astype(str)
    population = np.asarray(dataset["population"].values, dtype=float)
    frame = pd.DataFrame(
        {
            "target_date": np.repeat(dates, len(regions)),
            "municipality_code": np.tile(regions, len(dates)),
            "transformed": np.asarray(values.values).reshape(-1),
            "observed_mask": np.asarray(masks.values).reshape(-1),
            "population": np.tile(population, len(dates)),
        }
    )
    frame = frame[frame["observed_mask"] > 0].copy()
    frame["dataset_count"] = inverse_clinical(frame["transformed"], frame["population"])
    return frame


def validate_source_operator(
    target: str,
    dataset_counts: pd.DataFrame,
    projected_counts: pd.DataFrame,
) -> dict[str, Any]:
    compared = dataset_counts.merge(
        projected_counts,
        on=["target_date", "municipality_code"],
        how="left",
        validate="one_to_one",
    )
    compared["projected_count"] = compared["projected_count"].fillna(0.0)
    difference = (compared["dataset_count"] - compared["projected_count"]).abs()
    return {
        "target": target,
        "observed_rows": int(len(compared)),
        "matched_positive_projected_rows": int((compared["projected_count"] > 0).sum()),
        "max_abs_count_difference": float(difference.max()),
        "mean_abs_count_difference": float(difference.mean()),
        "p99_abs_count_difference": float(difference.quantile(0.99)),
    }


def load_region_folds(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text())
    region_folds = {
        str(region_id): int(fold["fold_index"])
        for fold in payload["folds"]
        for region_id in fold["region_ids"]
    }
    if len(region_folds) != sum(len(fold["region_ids"]) for fold in payload["folds"]):
        raise ValueError("Fold manifest contains duplicate region IDs")
    return region_folds


def build_source_support_audit(
    mapping: pd.DataFrame,
    source_col: str,
    target: str,
    valid_ids: set[str],
    region_folds: dict[str, int],
    coverage_threshold: float,
) -> pd.DataFrame:
    valid = mapping[mapping["municipality_code"].isin(valid_ids)].copy()
    valid["fold"] = valid["municipality_code"].map(region_folds)
    if valid["fold"].isna().any():
        raise ValueError(f"{target}: valid municipality missing from fold manifest")

    rows: list[dict[str, Any]] = []
    for source_id, group in valid.groupby(source_col, sort=True):
        fold_coverage = (
            group.groupby("fold")["source_weight"].sum().sort_values(ascending=False)
        )
        contributing_folds = sorted(
            int(value)
            for value in group.loc[group["source_weight"] > 1e-12, "fold"].unique()
        )
        valid_coverage = float(group["source_weight"].sum())
        rows.append(
            {
                "target": target,
                "source_id": source_id,
                "valid_coverage": valid_coverage,
                "valid_municipalities": int(group["municipality_code"].nunique()),
                "contributing_folds": len(contributing_folds),
                "fold_ids": ",".join(str(value) for value in contributing_folds),
                "dominant_fold": int(fold_coverage.index[0]),
                "dominant_fold_valid_coverage": float(fold_coverage.iloc[0]),
                "dominant_fold_share": float(fold_coverage.iloc[0] / valid_coverage),
                "coverage_eligible": valid_coverage >= coverage_threshold,
                "fold_safe_eligible": (
                    valid_coverage >= coverage_threshold
                    and len(contributing_folds) == 1
                ),
            }
        )
    return pd.DataFrame(rows)


def allocation_shape(weights: pd.Series) -> dict[str, float]:
    """Summarize how one source-zone signal is allocated to municipalities."""
    values = weights.to_numpy(dtype=float)
    if len(values) == 0 or not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("Allocation weights must be non-empty and sum to one")

    effective_municipalities = float(1.0 / np.square(values).sum())
    normalized_entropy = 1.0
    if len(values) > 1:
        normalized_entropy = float(
            -(values * np.log(values)).sum() / np.log(len(values))
        )

    allocation_weight_skewness = np.nan
    if len(values) >= 3 and not np.allclose(values, values[0]):
        allocation_weight_skewness = float(stats.skew(values, bias=False))

    return {
        "mapped_municipalities": float(len(values)),
        "effective_municipalities": effective_municipalities,
        "effective_fraction": effective_municipalities / len(values),
        "top_municipality_share": float(values.max()),
        "normalized_entropy": normalized_entropy,
        "allocation_weight_skewness": allocation_weight_skewness,
    }


def build_source_transformation_metrics(
    mapping: pd.DataFrame,
    source_col: str,
    target: str,
    source_resolution: str,
    valid_ids: set[str],
    region_folds: dict[str, int],
    coverage_threshold: float,
) -> pd.DataFrame:
    """Describe source-to-municipality breadth and allocation concentration."""
    rows: list[dict[str, Any]] = []
    mapping_municipalities = int(mapping["municipality_code"].nunique())
    for source_id, group in mapping.groupby(source_col, sort=True):
        full_shape = allocation_shape(group["source_weight"])
        valid = group[group["municipality_code"].isin(valid_ids)].copy()
        valid_coverage = float(valid["source_weight"].sum())
        valid_shape = {
            key: np.nan
            for key in (
                "mapped_municipalities",
                "effective_municipalities",
                "effective_fraction",
                "top_municipality_share",
                "normalized_entropy",
                "allocation_weight_skewness",
            )
        }
        contributing_folds: list[int] = []
        if valid_coverage > 0:
            valid_shape = allocation_shape(valid["source_weight"] / valid_coverage)
            contributing_folds = sorted(
                int(value)
                for value in valid["municipality_code"]
                .map(region_folds)
                .dropna()
                .unique()
            )

        rows.append(
            {
                "target": target,
                "source_resolution": source_resolution,
                "source_id": source_id,
                "mapping_municipalities": mapping_municipalities,
                **{f"full_{key}": value for key, value in full_shape.items()},
                "valid_coverage": valid_coverage,
                **{f"valid_{key}": value for key, value in valid_shape.items()},
                "contributing_folds": len(contributing_folds),
                "coverage_eligible": valid_coverage >= coverage_threshold,
                "fold_safe_eligible": (
                    valid_coverage >= coverage_threshold
                    and len(contributing_folds) == 1
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_source_transformations(
    metrics: pd.DataFrame,
    model_municipalities: int,
) -> pd.DataFrame:
    """Create one compact, report-ready row for each source resolution."""
    rows: list[dict[str, Any]] = []
    for (target, source_resolution), group in metrics.groupby(
        ["target", "source_resolution"], sort=False
    ):
        skewness = group["full_allocation_weight_skewness"].dropna()
        rows.append(
            {
                "target": target,
                "source_resolution": source_resolution,
                "source_zones": int(len(group)),
                "mapping_municipalities": int(group["mapping_municipalities"].iloc[0]),
                "mapping_links": int(group["full_mapped_municipalities"].sum()),
                "model_municipalities": model_municipalities,
                "sources_touching_model_support": int(
                    (group["valid_coverage"] > 0).sum()
                ),
                "coverage_eligible_sources": int(group["coverage_eligible"].sum()),
                "fold_safe_sources": int(group["fold_safe_eligible"].sum()),
                "median_mapped_municipalities": float(
                    group["full_mapped_municipalities"].median()
                ),
                "p10_mapped_municipalities": float(
                    group["full_mapped_municipalities"].quantile(0.1)
                ),
                "p90_mapped_municipalities": float(
                    group["full_mapped_municipalities"].quantile(0.9)
                ),
                "median_effective_municipalities": float(
                    group["full_effective_municipalities"].median()
                ),
                "p10_effective_municipalities": float(
                    group["full_effective_municipalities"].quantile(0.1)
                ),
                "p90_effective_municipalities": float(
                    group["full_effective_municipalities"].quantile(0.9)
                ),
                "median_effective_fraction": float(
                    group["full_effective_fraction"].median()
                ),
                "median_top_municipality_share": float(
                    group["full_top_municipality_share"].median()
                ),
                "skewness_defined_sources": int(len(skewness)),
                "median_allocation_weight_skewness": float(skewness.median()),
                "p10_allocation_weight_skewness": float(skewness.quantile(0.1)),
                "p90_allocation_weight_skewness": float(skewness.quantile(0.9)),
            }
        )
    return pd.DataFrame(rows)


def summarize_source_coverage_thresholds(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Count source zones retained over the full valid-mass threshold curve."""
    rows: list[dict[str, Any]] = []
    thresholds = np.linspace(0.0, 1.0, 101)
    for (target, source_resolution), group in metrics.groupby(
        ["target", "source_resolution"], sort=False
    ):
        touching = group[group["valid_coverage"] > 0]
        for threshold in thresholds:
            retained = touching[touching["valid_coverage"] >= threshold - 1e-12]
            rows.append(
                {
                    "target": target,
                    "source_resolution": source_resolution,
                    "coverage_threshold": float(threshold),
                    "touching_sources": int(len(touching)),
                    "retained_sources": int(len(retained)),
                    "retained_fraction": float(len(retained) / len(touching)),
                }
            )
    return pd.DataFrame(rows)


def summarize_scoring_coverage(
    transformation_summary: pd.DataFrame,
    hospital_source_forecasts: pd.DataFrame,
    death_source_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Report the source-zone counts that reach native scoring."""
    summaries = transformation_summary.set_index("target")
    specifications = (
        (
            "hospitalizations",
            "ABS",
            "single-fold",
            int(summaries.loc["hospitalizations", "fold_safe_sources"]),
            int(hospital_source_forecasts["abs_code"].nunique()),
            int(
                hospital_source_forecasts[["abs_code", "target_date"]]
                .drop_duplicates()
                .shape[0]
            ),
        ),
        (
            "deaths",
            "comarca",
            "cross-fitted",
            int(summaries.loc["deaths", "coverage_eligible_sources"]),
            int(death_source_forecasts["comarca_code"].nunique()),
            int(
                death_source_forecasts[["comarca_code", "target_date"]]
                .drop_duplicates()
                .shape[0]
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    for (
        target,
        source_resolution,
        scoring_strategy,
        scoring_eligible_sources,
        evaluated_sources,
        evaluated_source_dates,
    ) in specifications:
        summary = summaries.loc[target]
        rows.append(
            {
                "target": target,
                "source_resolution": source_resolution,
                "scoring_strategy": scoring_strategy,
                "mapping_sources": int(summary["source_zones"]),
                "sources_touching_model_support": int(
                    summary["sources_touching_model_support"]
                ),
                "coverage_eligible_sources": int(summary["coverage_eligible_sources"]),
                "scoring_eligible_sources": scoring_eligible_sources,
                "evaluated_sources": evaluated_sources,
                "evaluated_source_dates": evaluated_source_dates,
                "evaluated_fraction_of_mapping": float(
                    evaluated_sources / summary["source_zones"]
                ),
                "evaluated_fraction_of_touching": float(
                    evaluated_sources / summary["sources_touching_model_support"]
                ),
            }
        )
    return pd.DataFrame(rows)


def add_wis_components(
    frame: pd.DataFrame,
    target_col: str,
    prefix: str,
) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    target = out[target_col].to_numpy(dtype=float)
    loc = out["distribution_loc"].to_numpy(dtype=float)
    scale = out["distribution_scale"].to_numpy(dtype=float)
    signed_error = target - loc
    absolute_error = np.abs(signed_error)
    dispersion = np.zeros(len(out), dtype=float)
    underprediction = np.zeros(len(out), dtype=float)
    overprediction = np.zeros(len(out), dtype=float)

    probabilities = (1 + WIS_INTERVAL_LEVELS) / 2
    for family in out["distribution_family"].unique():
        mask = out["distribution_family"].eq(family).to_numpy()
        indices = np.flatnonzero(mask)
        df_values = out.loc[mask, "distribution_df"].unique()
        if len(df_values) != 1:
            raise ValueError(f"{family} rows mix distribution_df values")
        df = float(df_values[0])
        if family == "normal":
            factors = stats.norm.ppf(probabilities)
        elif family == "student_t":
            factors = stats.t.ppf(probabilities, df)
        else:
            raise ValueError(f"Unsupported distribution family: {family}")
        half_widths = scale[indices, None] * factors[None, :]
        misses = np.maximum(absolute_error[indices, None] - half_widths, 0.0)
        dispersion[indices] = (half_widths * (1 - WIS_INTERVAL_LEVELS)[None, :]).sum(
            axis=1
        )
        underprediction[indices] = np.where(
            signed_error[indices, None] > 0, misses, 0.0
        ).sum(axis=1)
        overprediction[indices] = np.where(
            signed_error[indices, None] < 0, misses, 0.0
        ).sum(axis=1)

    denominator = len(WIS_INTERVAL_LEVELS) + 0.5
    component_columns = {
        f"{prefix}_wis_median": 0.5 * absolute_error / denominator,
        f"{prefix}_wis_dispersion": dispersion / denominator,
        f"{prefix}_wis_under": underprediction / denominator,
        f"{prefix}_wis_over": overprediction / denominator,
    }
    for column, values in component_columns.items():
        out[column] = values
    out[f"{prefix}_wis"] = out[list(component_columns)].sum(axis=1)
    out[f"{prefix}_abs_error"] = absolute_error
    for level in (50, 80, 95):
        out[f"{prefix}_covered_{level}"] = (
            (target >= out[f"interval_low_{level}"].to_numpy(dtype=float))
            & (target <= out[f"interval_high_{level}"].to_numpy(dtype=float))
        ).astype(float)
    return out


def split_condition(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parts = out["condition"].str.split(":", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("Condition names must be family:objective:mobility")
    out[["family", "objective", "mobility"]] = parts
    return out


def paired_summary(
    identity: pd.DataFrame,
    metric_columns: list[str],
    extra_index: list[str] | None = None,
    pair_unit_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra_index = extra_index or []
    pair_unit_columns = pair_unit_columns or ["fold", "seed"]
    indexed = ["family", "objective", *extra_index, *pair_unit_columns]
    pivot = identity.pivot(
        index=indexed,
        columns="mobility",
        values=metric_columns,
    )
    required = {"mobility_off", "mobility_on"}
    if set(pivot.columns.get_level_values("mobility")) != required:
        raise ValueError("Mobility pairing is incomplete")

    pair_rows: list[dict[str, Any]] = []
    for index, row in pivot.iterrows():
        if not isinstance(index, tuple):
            index = (index,)
        record = dict(zip(indexed, index, strict=True))
        for metric in metric_columns:
            off = float(row[(metric, "mobility_off")])
            on = float(row[(metric, "mobility_on")])
            record[f"{metric}_off"] = off
            record[f"{metric}_on"] = on
            record[f"{metric}_gain"] = off - on
        pair_rows.append(record)
    pairs = pd.DataFrame(pair_rows)

    summary_index = ["family", "objective", *extra_index]
    summary_rows: list[dict[str, Any]] = []
    for index, group in pairs.groupby(summary_index, sort=True):
        if not isinstance(index, tuple):
            index = (index,)
        record = dict(zip(summary_index, index, strict=True))
        record["pairs"] = int(len(group))
        for metric in metric_columns:
            for suffix in ("off", "on", "gain"):
                column = f"{metric}_{suffix}"
                record[column] = float(group[column].mean())
            record[f"{metric}_improved_pairs"] = int(
                (group[f"{metric}_gain"] > 0).sum()
            )
        summary_rows.append(record)
    return pairs, pd.DataFrame(summary_rows)


def load_case_counts(
    path: Path,
    dates: pd.DatetimeIndex,
    region_ids: np.ndarray,
) -> pd.DataFrame:
    cases = pd.read_csv(
        path,
        usecols=["TipusCasData", "MunicipiCodi", "NumCasos"],
        dtype={"MunicipiCodi": "string"},
    )
    cases["target_date"] = pd.to_datetime(
        cases["TipusCasData"], format="%d/%m/%Y", errors="raise"
    )
    cases["municipality_code"] = normalize_code(cases["MunicipiCodi"], 5)
    cases["native_count"] = pd.to_numeric(cases["NumCasos"], errors="raise")
    daily = cases.groupby(["target_date", "municipality_code"], as_index=False)[
        "native_count"
    ].sum()
    grid = pd.MultiIndex.from_product(
        [dates, region_ids.astype(str)],
        names=["target_date", "municipality_code"],
    ).to_frame(index=False)
    return grid.merge(
        daily,
        on=["target_date", "municipality_code"],
        how="left",
        validate="one_to_one",
    ).fillna({"native_count": 0.0})


def project_case_labels(
    cases: pd.DataFrame,
    mapping: pd.DataFrame,
    source_col: str,
) -> pd.DataFrame:
    forward = cases.merge(
        mapping[["municipality_code", source_col, "forward_weight"]],
        on="municipality_code",
        how="inner",
        validate="many_to_many",
    )
    forward["source_count"] = forward["native_count"] * forward["forward_weight"]
    source = forward.groupby(["target_date", source_col], as_index=False)[
        "source_count"
    ].sum()
    projected = source.merge(
        mapping[[source_col, "municipality_code", "source_weight"]],
        on=source_col,
        how="inner",
        validate="many_to_many",
    )
    projected["projected_count"] = (
        projected["source_count"] * projected["source_weight"]
    )
    return projected.groupby(["target_date", "municipality_code"], as_index=False)[
        "projected_count"
    ].sum()


def analyze_case_projection_controls(
    granular: pd.DataFrame,
    dataset: xr.Dataset,
    cases_source: Path,
    abs_mapping: pd.DataFrame,
    comarca_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cases_forecasts = granular[granular["target"].eq("cases")].copy()
    dates = pd.DatetimeIndex(sorted(cases_forecasts["target_date"].unique()))
    region_ids = dataset["region_id"].values.astype(str)
    population = pd.Series(
        np.asarray(dataset["population"].values, dtype=float),
        index=region_ids,
    )
    cases = load_case_counts(cases_source, dates, region_ids)
    abs_projected = project_case_labels(cases, abs_mapping, "abs_code")
    comarca_projected = project_case_labels(cases, comarca_mapping, "comarca_code")

    cases_forecasts["population"] = cases_forecasts["region_id"].map(population)
    cases_forecasts = cases_forecasts.merge(
        abs_projected.rename(
            columns={
                "municipality_code": "region_id",
                "projected_count": "abs_projected_count",
            }
        ),
        on=["target_date", "region_id"],
        how="left",
        validate="many_to_one",
    )
    cases_forecasts = cases_forecasts.merge(
        comarca_projected.rename(
            columns={
                "municipality_code": "region_id",
                "projected_count": "comarca_projected_count",
            }
        ),
        on=["target_date", "region_id"],
        how="left",
        validate="many_to_one",
    )
    for source in ("abs", "comarca"):
        count_col = f"{source}_projected_count"
        target_col = f"{source}_projection_target"
        cases_forecasts[count_col] = cases_forecasts[count_col].fillna(0.0)
        cases_forecasts[target_col] = np.log1p(
            cases_forecasts[count_col] * 1e5 / cases_forecasts["population"]
        )

    scored = add_wis_components(cases_forecasts, "observed", "native")
    scored = add_wis_components(scored, "abs_projection_target", "abs_projection")
    scored = add_wis_components(
        scored, "comarca_projection_target", "comarca_projection"
    )
    reconstruction_error = float((scored["native_wis"] - scored["wis"]).abs().max())
    if reconstruction_error > 1e-5:
        raise ValueError(
            f"Canonical WIS reconstruction error is too large: {reconstruction_error}"
        )

    metric_columns: list[str] = []
    for support in ("native", "abs_projection", "comarca_projection"):
        metric_columns.extend(
            [
                f"{support}_wis",
                f"{support}_wis_median",
                f"{support}_wis_dispersion",
                f"{support}_wis_under",
                f"{support}_wis_over",
                f"{support}_abs_error",
                f"{support}_covered_50",
                f"{support}_covered_80",
                f"{support}_covered_95",
            ]
        )
    identity = (
        scored.groupby(list(IDENTITY_COLUMNS), as_index=False)[metric_columns]
        .mean()
        .pipe(split_condition)
    )
    pairs, wide_summary = paired_summary(identity, metric_columns)

    summary_rows: list[dict[str, Any]] = []
    for _, row in wide_summary.iterrows():
        for support in ("native", "abs_projection", "comarca_projection"):
            record: dict[str, Any] = {
                "family": row["family"],
                "objective": row["objective"],
                "support": support,
                "pairs": int(row["pairs"]),
            }
            for metric in (
                "wis",
                "wis_median",
                "wis_dispersion",
                "wis_under",
                "wis_over",
                "abs_error",
                "covered_50",
                "covered_80",
                "covered_95",
            ):
                source_metric = f"{support}_{metric}"
                for suffix in ("off", "on", "gain"):
                    record[f"{metric}_{suffix}"] = float(
                        row[f"{source_metric}_{suffix}"]
                    )
                record[f"{metric}_improved_pairs"] = int(
                    row[f"{source_metric}_improved_pairs"]
                )
            summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    totals = {
        "native_case_count": float(cases["native_count"].sum()),
        "abs_projected_count_retained_in_dataset_regions": float(
            abs_projected[abs_projected["municipality_code"].isin(set(region_ids))][
                "projected_count"
            ].sum()
        ),
        "comarca_projected_count_retained_in_dataset_regions": float(
            comarca_projected[
                comarca_projected["municipality_code"].isin(set(region_ids))
            ]["projected_count"].sum()
        ),
        "native_wis_reconstruction_max_abs_error": reconstruction_error,
    }
    return identity, pairs, summary, totals


def analyze_head_wis_decomposition(
    granular: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = add_wis_components(granular, "observed", "native")
    reconstruction_error = float((scored["native_wis"] - scored["wis"]).abs().max())
    if reconstruction_error > 1e-5:
        raise ValueError(
            f"Head WIS reconstruction error is too large: {reconstruction_error}"
        )
    metrics = [
        "native_wis",
        "native_wis_median",
        "native_wis_dispersion",
        "native_wis_under",
        "native_wis_over",
        "native_abs_error",
        "native_covered_50",
        "native_covered_80",
        "native_covered_95",
    ]
    identity = (
        scored.groupby([*IDENTITY_COLUMNS, "target"], as_index=False)[metrics]
        .mean()
        .pipe(split_condition)
    )
    return paired_summary(identity, metrics, extra_index=["target"])


def analyze_hospital_native_source(
    granular: pd.DataFrame,
    dataset: xr.Dataset,
    mapping: pd.DataFrame,
    source_observations: pd.DataFrame,
    support_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = support_audit[support_audit["fold_safe_eligible"]].copy()
    eligible = eligible.rename(
        columns={"source_id": "abs_code", "dominant_fold": "source_fold"}
    )
    hospital = granular[granular["target"].eq("hospitalizations")].copy()
    population = pd.Series(
        np.asarray(dataset["population"].values, dtype=float),
        index=dataset["region_id"].values.astype(str),
    )
    hospital["population"] = hospital["region_id"].map(population)
    hospital["prediction_count"] = (
        np.expm1(hospital["distribution_loc"]) * hospital["population"] / 1e5
    )

    contributions = hospital.merge(
        mapping[["municipality_code", "abs_code", "forward_weight"]],
        left_on="region_id",
        right_on="municipality_code",
        how="inner",
        validate="many_to_many",
    ).merge(
        eligible[["abs_code", "source_fold", "valid_coverage"]],
        on="abs_code",
        how="inner",
        validate="many_to_one",
    )
    contributions = contributions[
        contributions["fold"].eq(contributions["source_fold"])
    ].copy()
    contributions["predicted_source_contribution"] = (
        contributions["prediction_count"] * contributions["forward_weight"]
    )
    source_predictions = contributions.groupby(
        [
            "condition",
            "fold",
            "seed",
            "abs_code",
            "target_date",
            "valid_coverage",
        ],
        as_index=False,
    )["predicted_source_contribution"].sum()
    source_predictions = source_predictions.rename(
        columns={"predicted_source_contribution": "prediction_count"}
    )

    population_frame = (
        population.rename("municipality_population")
        .rename_axis("municipality_code")
        .reset_index()
    )
    source_population = mapping.merge(
        population_frame,
        on="municipality_code",
        how="inner",
        validate="many_to_one",
    )
    source_population["population_contribution"] = (
        source_population["forward_weight"]
        * source_population["municipality_population"]
    )
    source_population = (
        source_population.groupby("abs_code", as_index=False)["population_contribution"]
        .sum()
        .rename(columns={"population_contribution": "source_population"})
    )

    evaluated = source_predictions.merge(
        source_observations,
        on=["abs_code", "target_date"],
        how="inner",
        validate="many_to_one",
    ).merge(
        source_population,
        on="abs_code",
        how="left",
        validate="many_to_one",
    )
    evaluated["raw_abs_error"] = (
        evaluated["prediction_count"] - evaluated["observed_count"]
    ).abs()
    evaluated["raw_bias"] = evaluated["prediction_count"] - evaluated["observed_count"]
    evaluated["log_count_abs_error"] = (
        np.log1p(evaluated["prediction_count"]) - np.log1p(evaluated["observed_count"])
    ).abs()
    evaluated["log_count_bias"] = np.log1p(evaluated["prediction_count"]) - np.log1p(
        evaluated["observed_count"]
    )
    evaluated["rate_prediction"] = np.log1p(
        evaluated["prediction_count"] * 1e5 / evaluated["source_population"]
    )
    evaluated["rate_observed"] = np.log1p(
        evaluated["observed_count"] * 1e5 / evaluated["source_population"]
    )
    evaluated["rate_abs_error"] = (
        evaluated["rate_prediction"] - evaluated["rate_observed"]
    ).abs()
    evaluated["rate_bias"] = evaluated["rate_prediction"] - evaluated["rate_observed"]
    evaluated["absolute_percentage_error_plus_one"] = evaluated["raw_abs_error"] / (
        evaluated["observed_count"] + 1.0
    )

    identity = (
        evaluated.groupby(list(IDENTITY_COLUMNS), as_index=False)
        .agg(
            raw_mae=("raw_abs_error", "mean"),
            raw_bias=("raw_bias", "mean"),
            log_count_mae=("log_count_abs_error", "mean"),
            log_count_bias=("log_count_bias", "mean"),
            rate_mae=("rate_abs_error", "mean"),
            rate_bias=("rate_bias", "mean"),
            ape_plus_one=("absolute_percentage_error_plus_one", "mean"),
            source_dates=("abs_code", "size"),
            sources=("abs_code", "nunique"),
        )
        .pipe(split_condition)
    )
    metrics = [
        "raw_mae",
        "raw_bias",
        "log_count_mae",
        "log_count_bias",
        "rate_mae",
        "rate_bias",
        "ape_plus_one",
    ]
    pairs, summary = paired_summary(identity, metrics)

    municipal = (
        hospital.groupby(list(IDENTITY_COLUMNS), as_index=False)
        .agg(municipal_wis=("wis", "mean"), municipal_mae=("abs_error", "mean"))
        .pipe(split_condition)
    )
    _, municipal_summary = paired_summary(municipal, ["municipal_wis", "municipal_mae"])
    summary = summary.merge(
        municipal_summary[
            [
                "family",
                "objective",
                "municipal_wis_off",
                "municipal_wis_on",
                "municipal_wis_gain",
                "municipal_mae_off",
                "municipal_mae_on",
                "municipal_mae_gain",
            ]
        ],
        on=["family", "objective"],
        how="left",
        validate="one_to_one",
    )
    for metric in ("rate_mae", "municipal_wis"):
        summary[f"{metric}_relative_gain_pct"] = (
            100 * summary[f"{metric}_gain"] / summary[f"{metric}_off"]
        )
    return evaluated, identity, pairs, summary


def analyze_death_cross_fitted_native_source(
    granular: pd.DataFrame,
    dataset: xr.Dataset,
    mapping: pd.DataFrame,
    source_observations: pd.DataFrame,
    support_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score stitched out-of-fold death forecasts at native comarca support.

    Each municipal prediction remains out of fold, but a comarca can combine
    municipalities predicted by several fold-specific models. The resulting
    score is therefore a cross-fitted point diagnostic, not a prediction from
    one fitted model or a source-level probabilistic score.
    """
    eligible = support_audit[support_audit["coverage_eligible"]].copy()
    eligible = eligible.rename(columns={"source_id": "comarca_code"})
    if eligible.empty:
        raise ValueError("No comarca meets the configured source coverage threshold")

    deaths = granular[granular["target"].eq("deaths")].copy()
    held_out_counts = deaths.groupby(
        ["condition", "seed", "region_id", "target_date"], sort=False
    ).size()
    if not held_out_counts.eq(1).all():
        raise ValueError("Death forecasts are not unique cross-fitted municipal rows")

    population = pd.Series(
        np.asarray(dataset["population"].values, dtype=float),
        index=dataset["region_id"].values.astype(str),
    )
    deaths["population"] = deaths["region_id"].map(population)
    deaths["prediction_count"] = (
        np.expm1(deaths["distribution_loc"]) * deaths["population"] / 1e5
    )

    contributions = deaths.merge(
        mapping[["municipality_code", "comarca_code", "forward_weight"]],
        left_on="region_id",
        right_on="municipality_code",
        how="inner",
        validate="many_to_many",
    ).merge(
        eligible[["comarca_code", "valid_coverage", "contributing_folds"]],
        on="comarca_code",
        how="inner",
        validate="many_to_one",
    )
    contributions["predicted_source_contribution"] = (
        contributions["prediction_count"] * contributions["forward_weight"]
    )
    source_predictions = contributions.groupby(
        [
            "condition",
            "seed",
            "comarca_code",
            "target_date",
            "valid_coverage",
            "contributing_folds",
        ],
        as_index=False,
    )["predicted_source_contribution"].sum()
    source_predictions = source_predictions.rename(
        columns={"predicted_source_contribution": "prediction_count"}
    )

    population_frame = (
        population.rename("municipality_population")
        .rename_axis("municipality_code")
        .reset_index()
    )
    source_population = mapping.merge(
        population_frame,
        on="municipality_code",
        how="inner",
        validate="many_to_one",
    )
    source_population["population_contribution"] = (
        source_population["forward_weight"]
        * source_population["municipality_population"]
    )
    source_population = (
        source_population.groupby("comarca_code", as_index=False)[
            "population_contribution"
        ]
        .sum()
        .rename(columns={"population_contribution": "source_population"})
    )

    evaluated = source_predictions.merge(
        source_observations,
        on=["comarca_code", "target_date"],
        how="inner",
        validate="many_to_one",
    ).merge(
        source_population,
        on="comarca_code",
        how="left",
        validate="many_to_one",
    )
    if evaluated.empty:
        raise ValueError("No cross-fitted death forecasts overlap source observations")

    evaluated["raw_abs_error"] = (
        evaluated["prediction_count"] - evaluated["observed_count"]
    ).abs()
    evaluated["raw_bias"] = evaluated["prediction_count"] - evaluated["observed_count"]
    evaluated["log_count_abs_error"] = (
        np.log1p(evaluated["prediction_count"]) - np.log1p(evaluated["observed_count"])
    ).abs()
    evaluated["log_count_bias"] = np.log1p(evaluated["prediction_count"]) - np.log1p(
        evaluated["observed_count"]
    )
    evaluated["rate_prediction"] = np.log1p(
        evaluated["prediction_count"] * 1e5 / evaluated["source_population"]
    )
    evaluated["rate_observed"] = np.log1p(
        evaluated["observed_count"] * 1e5 / evaluated["source_population"]
    )
    evaluated["rate_abs_error"] = (
        evaluated["rate_prediction"] - evaluated["rate_observed"]
    ).abs()
    evaluated["rate_bias"] = evaluated["rate_prediction"] - evaluated["rate_observed"]
    evaluated["absolute_percentage_error_plus_one"] = evaluated["raw_abs_error"] / (
        evaluated["observed_count"] + 1.0
    )

    identity = (
        evaluated.groupby(["condition", "seed"], as_index=False)
        .agg(
            raw_mae=("raw_abs_error", "mean"),
            raw_bias=("raw_bias", "mean"),
            log_count_mae=("log_count_abs_error", "mean"),
            log_count_bias=("log_count_bias", "mean"),
            rate_mae=("rate_abs_error", "mean"),
            rate_bias=("rate_bias", "mean"),
            ape_plus_one=("absolute_percentage_error_plus_one", "mean"),
            source_dates=("comarca_code", "size"),
            sources=("comarca_code", "nunique"),
            contributing_folds=("contributing_folds", "max"),
        )
        .pipe(split_condition)
    )
    metrics = [
        "raw_mae",
        "raw_bias",
        "log_count_mae",
        "log_count_bias",
        "rate_mae",
        "rate_bias",
        "ape_plus_one",
    ]
    pairs, summary = paired_summary(
        identity,
        metrics,
        pair_unit_columns=["seed"],
    )

    municipal = (
        deaths.groupby(["condition", "seed"], as_index=False)
        .agg(municipal_wis=("wis", "mean"), municipal_mae=("abs_error", "mean"))
        .pipe(split_condition)
    )
    _, municipal_summary = paired_summary(
        municipal,
        ["municipal_wis", "municipal_mae"],
        pair_unit_columns=["seed"],
    )
    summary = summary.merge(
        municipal_summary[
            [
                "family",
                "objective",
                "municipal_wis_off",
                "municipal_wis_on",
                "municipal_wis_gain",
                "municipal_mae_off",
                "municipal_mae_on",
                "municipal_mae_gain",
            ]
        ],
        on=["family", "objective"],
        how="left",
        validate="one_to_one",
    )
    for metric in ("rate_mae", "municipal_wis"):
        summary[f"{metric}_relative_gain_pct"] = (
            100 * summary[f"{metric}_gain"] / summary[f"{metric}_off"]
        )
    return evaluated, identity, pairs, summary


def plot_dashboard(
    case_summary: pd.DataFrame,
    hospital_summary: pd.DataFrame,
    death_summary: pd.DataFrame,
    support_audit: pd.DataFrame,
    head_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = ["Normal CRPS", "Normal NLL", "Student-t CRPS", "Student-t NLL"]
    keys = [
        ("normal", "crps"),
        ("normal", "nll"),
        ("student_t", "crps"),
        ("student_t", "nll"),
    ]
    colors = {
        "native": "#167d8d",
        "abs_projection": "#e3a62f",
        "comarca_projection": "#b95454",
    }
    support_labels = {
        "native": "Native municipal cases",
        "abs_projection": "Cases projected via ABS",
        "comarca_projection": "Cases projected via comarca",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axis = axes[0, 0]
    x = np.arange(len(keys))
    width = 0.24
    for offset, support in zip((-1, 0, 1), colors, strict=True):
        values = [
            float(
                case_summary[
                    case_summary["family"].eq(family)
                    & case_summary["objective"].eq(objective)
                    & case_summary["support"].eq(support)
                ]["wis_gain"].iloc[0]
            )
            for family, objective in keys
        ]
        axis.bar(
            x + offset * width,
            values,
            width,
            color=colors[support],
            label=support_labels[support],
        )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel("Mobility WIS gain (off − on)")
    axis.set_title("A. Case benefit collapses after artificial projection")
    axis.legend(frameon=False, fontsize=9)

    axis = axes[0, 1]
    hospital_municipal = []
    hospital_native_source = []
    death_municipal = []
    death_native_source = []
    for family, objective in keys:
        hospital_row = hospital_summary[
            hospital_summary["family"].eq(family)
            & hospital_summary["objective"].eq(objective)
        ].iloc[0]
        death_row = death_summary[
            death_summary["family"].eq(family)
            & death_summary["objective"].eq(objective)
        ].iloc[0]
        hospital_municipal.append(
            float(hospital_row["municipal_wis_relative_gain_pct"])
        )
        hospital_native_source.append(float(hospital_row["rate_mae_relative_gain_pct"]))
        death_municipal.append(float(death_row["municipal_wis_relative_gain_pct"]))
        death_native_source.append(float(death_row["rate_mae_relative_gain_pct"]))
    axis.bar(
        x - 0.27,
        hospital_municipal,
        0.18,
        color="#b95454",
        label="Hospital municipal WIS",
    )
    axis.bar(
        x - 0.09,
        hospital_native_source,
        0.18,
        color="#167d8d",
        label="Hospital native ABS MAE",
    )
    axis.bar(
        x + 0.09,
        death_municipal,
        0.18,
        color="#e3a62f",
        label="Death municipal WIS",
    )
    axis.bar(
        x + 0.27,
        death_native_source,
        0.18,
        color="#263f5e",
        label="Death cross-fitted comarca MAE",
    )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel("Relative mobility gain (%)")
    axis.set_title("B. Source scoring strengthens hospitals more than deaths")
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    for target, color, label in (
        ("hospitalizations", "#167d8d", "ABS hospitalizations"),
        ("deaths", "#b95454", "Comarca deaths"),
    ):
        values = support_audit.loc[support_audit["target"].eq(target), "valid_coverage"]
        axis.hist(values, bins=bins, alpha=0.6, color=color, label=label)
    axis.axvline(0.99, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel("Source mass covered by 161 held-out target municipalities")
    axis.set_ylabel("Native sources")
    axis.set_title("C. Death source totals lack complete held-out support")
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1, 1]
    selected = head_summary[
        head_summary["target"].isin(["deaths", "hospitalizations"])
    ].copy()
    selected["config"] = (
        selected["family"].replace({"student_t": "Student-t", "normal": "Normal"})
        + " "
        + selected["objective"].str.upper()
    )
    selected["row_label"] = (
        selected["target"].replace({"hospitalizations": "Hospital", "deaths": "Deaths"})
        + " · "
        + selected["config"]
    )
    selected = selected.set_index("row_label")
    ordered_rows = [
        f"{target} · {label}" for target in ("Hospital", "Deaths") for label in labels
    ]
    selected = selected.reindex(ordered_rows)
    components = [
        ("native_wis_median_gain", "Location", "#2c7fb8"),
        ("native_wis_dispersion_gain", "Dispersion", "#7fcdbb"),
        ("native_wis_under_gain", "Underprediction", "#f0ad4e"),
        ("native_wis_over_gain", "Overprediction", "#d95f5f"),
    ]
    positive_base = np.zeros(len(selected))
    negative_base = np.zeros(len(selected))
    y = np.arange(len(selected))
    for column, label, color in components:
        values = selected[column].to_numpy(dtype=float)
        starts = np.where(values >= 0, positive_base, negative_base)
        axis.barh(y, values, left=starts, color=color, label=label)
        positive_base += np.where(values >= 0, values, 0)
        negative_base += np.where(values < 0, values, 0)
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_yticks(y, selected.index)
    axis.invert_yaxis()
    axis.set_xlabel("Contribution to mobility WIS gain")
    axis.set_title("D. Loss interactions include location and calibration")
    axis.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle(
        "Static source-resolution diagnostic — calibration stage only",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def main() -> int:
    args = parse_args()
    if not 0 < args.source_coverage_threshold <= 1:
        raise ValueError("source coverage threshold must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    abs_mapping = load_mapping(args.abs_mapping, "abs_code", 3)
    comarca_mapping = load_mapping(args.comarca_mapping, "comarca_code", 2)
    identities = load_forecast_identities(
        args.reuse_manifest,
        args.trained_root,
        args.expected_identities,
    )
    granular = load_granular(identities)
    dataset = xr.open_dataset(args.dataset)
    try:
        valid_mask = np.asarray(
            dataset["valid_targets"].sel(run_id="real").values
        ).astype(bool)
        valid_ids = set(dataset["region_id"].values[valid_mask].astype(str))
        region_folds = load_region_folds(args.fold_manifest)
        if valid_ids != set(region_folds):
            raise ValueError("Dataset valid targets do not match the fold manifest")
        if set(granular["region_id"].unique()) != valid_ids:
            raise ValueError("Granular region support does not match valid targets")

        hospital_source = load_hospital_source(
            args.hospitalizations_source,
            args.hospitalization_lag_days,
        )
        death_source = load_death_source(args.deaths_source)
        hospital_projected = source_to_municipality(
            hospital_source, abs_mapping, "abs_code"
        )
        death_projected = source_to_municipality(
            death_source, comarca_mapping, "comarca_code"
        )
        operator_validation = pd.DataFrame(
            [
                validate_source_operator(
                    "hospitalizations",
                    dataset_observed_counts(
                        dataset, "hospitalizations", "hospitalizations_mask"
                    ),
                    hospital_projected,
                ),
                validate_source_operator(
                    "deaths",
                    dataset_observed_counts(dataset, "deaths", "deaths_mask"),
                    death_projected,
                ),
            ]
        )

        support_audit = pd.concat(
            [
                build_source_support_audit(
                    abs_mapping,
                    "abs_code",
                    "hospitalizations",
                    valid_ids,
                    region_folds,
                    args.source_coverage_threshold,
                ),
                build_source_support_audit(
                    comarca_mapping,
                    "comarca_code",
                    "deaths",
                    valid_ids,
                    region_folds,
                    args.source_coverage_threshold,
                ),
            ],
            ignore_index=True,
        )
        transformation_metrics = pd.concat(
            [
                build_source_transformation_metrics(
                    abs_mapping,
                    "abs_code",
                    "hospitalizations",
                    "ABS",
                    valid_ids,
                    region_folds,
                    args.source_coverage_threshold,
                ),
                build_source_transformation_metrics(
                    comarca_mapping,
                    "comarca_code",
                    "deaths",
                    "comarca",
                    valid_ids,
                    region_folds,
                    args.source_coverage_threshold,
                ),
            ],
            ignore_index=True,
        )
        transformation_summary = summarize_source_transformations(
            transformation_metrics,
            model_municipalities=len(valid_ids),
        )
        coverage_threshold_summary = summarize_source_coverage_thresholds(
            transformation_metrics
        )

        case_identity, case_pairs, case_summary, case_totals = (
            analyze_case_projection_controls(
                granular,
                dataset,
                args.cases_source,
                abs_mapping,
                comarca_mapping,
            )
        )
        head_pairs, head_summary = analyze_head_wis_decomposition(granular)
        (
            hospital_source_forecasts,
            hospital_identity,
            hospital_pairs,
            hospital_summary,
        ) = analyze_hospital_native_source(
            granular,
            dataset,
            abs_mapping,
            hospital_source,
            support_audit[support_audit["target"].eq("hospitalizations")],
        )
        (
            death_source_forecasts,
            death_identity,
            death_pairs,
            death_summary,
        ) = analyze_death_cross_fitted_native_source(
            granular,
            dataset,
            comarca_mapping,
            death_source,
            support_audit[support_audit["target"].eq("deaths")],
        )
        scoring_coverage_summary = summarize_scoring_coverage(
            transformation_summary,
            hospital_source_forecasts,
            death_source_forecasts,
        )
    finally:
        dataset.close()

    output_tables = {
        "source_operator_validation.csv": operator_validation,
        "source_support_audit.csv": support_audit,
        "source_transformation_metrics.csv": transformation_metrics,
        "source_transformation_summary.csv": transformation_summary,
        "source_coverage_threshold_summary.csv": coverage_threshold_summary,
        "source_scoring_coverage_summary.csv": scoring_coverage_summary,
        "cases_projection_identity_scores.csv": case_identity,
        "cases_projection_paired_deltas.csv": case_pairs,
        "cases_projection_mobility_summary.csv": case_summary,
        "head_wis_decomposition_paired.csv": head_pairs,
        "head_wis_decomposition_summary.csv": head_summary,
        "hospital_native_source_forecasts.csv": hospital_source_forecasts,
        "hospital_native_source_identity_metrics.csv": hospital_identity,
        "hospital_native_source_paired_deltas.csv": hospital_pairs,
        "hospital_native_source_summary.csv": hospital_summary,
        "death_cross_fitted_source_forecasts.csv": death_source_forecasts,
        "death_cross_fitted_source_identity_metrics.csv": death_identity,
        "death_cross_fitted_source_paired_deltas.csv": death_pairs,
        "death_cross_fitted_source_summary.csv": death_summary,
    }
    for filename, frame in output_tables.items():
        frame.to_csv(args.output_dir / filename, index=False)

    plot_dashboard(
        case_summary,
        hospital_summary,
        death_summary,
        support_audit,
        head_summary,
        args.output_dir / "source_resolution_diagnostic.png",
    )

    case_native = case_summary[case_summary["support"].eq("native")]
    case_abs = case_summary[case_summary["support"].eq("abs_projection")]
    case_comarca = case_summary[case_summary["support"].eq("comarca_projection")]
    hospital_eligible = support_audit[
        support_audit["target"].eq("hospitalizations")
        & support_audit["fold_safe_eligible"]
    ]
    death_eligible = support_audit[
        support_audit["target"].eq("deaths") & support_audit["fold_safe_eligible"]
    ]
    source_date_count = int(
        hospital_source_forecasts[["abs_code", "target_date"]]
        .drop_duplicates()
        .shape[0]
    )
    death_source_date_count = int(
        death_source_forecasts[["comarca_code", "target_date"]]
        .drop_duplicates()
        .shape[0]
    )
    summary = {
        "schema_version": 1,
        "scope": {
            "campaign": "loss_type_full_coverage_v2",
            "stage": "calibration",
            "confirmation_consumed": False,
            "forecast_identities": len(identities),
            "reused_identities": sum(
                item.evidence_kind == "reused" for item in identities
            ),
            "trained_identities": sum(
                item.evidence_kind == "trained" for item in identities
            ),
            "valid_target_municipalities": len(valid_ids),
            "forecast_target_date_start": str(granular["target_date"].min().date()),
            "forecast_target_date_end": str(granular["target_date"].max().date()),
        },
        "provenance": {
            "dataset": {
                "path": str(args.dataset),
                "sha256": sha256_file(args.dataset),
            },
            "cases_source": {
                "path": str(args.cases_source),
                "sha256": sha256_file(args.cases_source),
            },
            "hospitalizations_source": {
                "path": str(args.hospitalizations_source),
                "sha256": sha256_file(args.hospitalizations_source),
                "campaign_manifested": False,
            },
            "deaths_source": {
                "path": str(args.deaths_source),
                "sha256": sha256_file(args.deaths_source),
                "campaign_manifested": False,
            },
            "abs_mapping": {
                "path": str(args.abs_mapping),
                "sha256": sha256_file(args.abs_mapping),
            },
            "comarca_mapping": {
                "path": str(args.comarca_mapping),
                "sha256": sha256_file(args.comarca_mapping),
            },
            "campaign_hospitalizations_sha256": sha256_file(
                args.campaign_hospitalizations
            ),
            "campaign_deaths_sha256": sha256_file(args.campaign_deaths),
        },
        "operator_validation": operator_validation.to_dict("records"),
        "source_support": {
            "coverage_threshold": args.source_coverage_threshold,
            "hospital_fold_safe_sources": int(len(hospital_eligible)),
            "hospital_evaluated_sources": int(
                hospital_source_forecasts["abs_code"].nunique()
            ),
            "hospital_evaluated_source_dates": source_date_count,
            "hospital_fold_seed_pairs_per_configuration": int(
                hospital_summary["pairs"].min()
            ),
            "death_fold_safe_sources": int(len(death_eligible)),
            "death_single_fold_source_scoring_feasible": bool(len(death_eligible) > 0),
            "death_cross_fitted_complete_sources": int(
                death_source_forecasts["comarca_code"].nunique()
            ),
            "death_cross_fitted_source_dates": death_source_date_count,
            "death_cross_fitted_seed_pairs_per_configuration": int(
                death_summary["pairs"].min()
            ),
        },
        "source_transformation": transformation_summary.to_dict("records"),
        "source_scoring_coverage": scoring_coverage_summary.to_dict("records"),
        "case_projection_control": {
            "native_positive_configurations": int((case_native["wis_gain"] > 0).sum()),
            "abs_projection_positive_configurations": int(
                (case_abs["wis_gain"] > 0).sum()
            ),
            "comarca_projection_positive_configurations": int(
                (case_comarca["wis_gain"] > 0).sum()
            ),
            "mean_native_wis_gain": float(case_native["wis_gain"].mean()),
            "mean_abs_projection_wis_gain": float(case_abs["wis_gain"].mean()),
            "mean_comarca_projection_wis_gain": float(case_comarca["wis_gain"].mean()),
            **case_totals,
        },
        "hospital_native_source": {
            "municipal_wis_positive_configurations": int(
                (hospital_summary["municipal_wis_gain"] > 0).sum()
            ),
            "native_source_rate_mae_positive_configurations": int(
                (hospital_summary["rate_mae_gain"] > 0).sum()
            ),
            "mean_municipal_wis_gain": float(
                hospital_summary["municipal_wis_gain"].mean()
            ),
            "mean_native_source_rate_mae_gain": float(
                hospital_summary["rate_mae_gain"].mean()
            ),
            "mean_native_source_relative_gain_pct": float(
                hospital_summary["rate_mae_relative_gain_pct"].mean()
            ),
        },
        "death_cross_fitted_native_source": {
            "municipal_wis_positive_configurations": int(
                (death_summary["municipal_wis_gain"] > 0).sum()
            ),
            "native_source_rate_mae_positive_configurations": int(
                (death_summary["rate_mae_gain"] > 0).sum()
            ),
            "mean_municipal_wis_gain": float(
                death_summary["municipal_wis_gain"].mean()
            ),
            "mean_native_source_rate_mae_gain": float(
                death_summary["rate_mae_gain"].mean()
            ),
            "mean_native_source_relative_gain_pct": float(
                death_summary["rate_mae_relative_gain_pct"].mean()
            ),
        },
        "decision": {
            "support_mismatch_signal": "positive",
            "reason": (
                "Artificial geographic projection reverses the case mobility WIS "
                "benefit in most or all configurations, while fold-safe native ABS "
                "point rescoring favors mobility in all four configurations. "
                "Cross-fitted native-comarca death scoring improves three of four "
                "configurations, but support is limited to one complete comarca and "
                "the effect is not robust across objectives."
            ),
            "retraining_conclusion": (
                "Evidence justifies designing native-source supervision, but this "
                "static diagnostic does not estimate the effect of retraining."
            ),
        },
        "limitations": [
            "Native-source point scores aggregate conditional medians; no source-level probabilistic score is claimed.",
            "Hospital source analysis uses only sources with at least the configured valid-target coverage and exactly one contributing held-out fold.",
            "Death source scoring stitches out-of-fold municipal forecasts from multiple fold-specific models; it is not a prediction from one fitted model.",
            "Only one comarca has at least the configured valid-target coverage, so the death source comparison is a narrow sensitivity check.",
            "The native hospital and death raw source files are external provenance inputs, not members of the immutable campaign raw-input manifest.",
            "Shared sources, spatial dependence, and four hospital target dates make all effects descriptive rather than inferential.",
        ],
    }
    (args.output_dir / "source_resolution_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(json_ready(summary["decision"]), indent=2))
    print(f"Wrote {len(output_tables) + 2} artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
