"""Fail-closed probabilistic evidence tables for complete ablation campaigns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evaluation.granular_export import GRANULAR_KEY_COLUMNS
from scripts.hpo.ablation_worker import AblationCampaign

REQUIRED_HEADS = ("wastewater", "hospitalizations", "cases", "deaths")
PROBABILISTIC_METRICS = ("wis", "crps", "nll", "abs_error")
INTERVAL_COLUMNS = ("covered_50", "covered_80", "covered_95")
WIDTH_COLUMNS = (
    ("interval_low_50", "interval_high_50"),
    ("interval_low_80", "interval_high_80"),
    ("interval_low_95", "interval_high_95"),
)
SUPPORT_COLUMNS = [*GRANULAR_KEY_COLUMNS, "region_id", "target_date"]


def _load_campaign_document(path: Path) -> dict[str, Any]:
    document = json.loads(json.dumps(yaml.safe_load(path.read_text())))
    if not isinstance(document, dict):
        raise ValueError(f"Campaign document must be a mapping: {path}")
    return document


def campaign_analysis_spec(campaign: AblationCampaign) -> dict[str, Any] | None:
    """Return a validated optional probabilistic-analysis declaration."""
    raw = _load_campaign_document(campaign.path)
    analysis = raw.get("analysis")
    if analysis is None:
        return None
    if not isinstance(analysis, dict):
        raise ValueError("campaign.analysis must be a mapping")
    if analysis.get("kind") != "probabilistic_full_coverage":
        raise ValueError("campaign.analysis.kind must be probabilistic_full_coverage")
    heads = tuple(analysis.get("required_heads", []))
    if heads != REQUIRED_HEADS:
        raise ValueError(
            "probabilistic campaign analysis must declare ordered required_heads "
            f"{list(REQUIRED_HEADS)}"
        )
    if analysis.get("granular_filename") != "test_granular.parquet":
        raise ValueError(
            "probabilistic campaign analysis must consume test_granular.parquet"
        )
    temporal = analysis.get("temporal_evaluation")
    if not isinstance(temporal, dict):
        raise ValueError("probabilistic campaign analysis needs temporal_evaluation")
    design = analysis.get("evaluation_design", "hybrid")
    if design not in {"hybrid", "temporal", "spatial"}:
        raise ValueError("invalid probabilistic evaluation design")
    expected_role = "all_dates" if design == "spatial" else "calibration"
    expected_confirmation = "included" if design == "spatial" else "preserved"
    if temporal.get("confirmation") != expected_confirmation:
        raise ValueError("probabilistic campaign has invalid confirmation policy")
    if temporal.get("scored_role") != expected_role:
        raise ValueError("probabilistic campaign has invalid scored role")
    if temporal.get("calibration_control") != "C0":
        raise ValueError("probabilistic campaign must score unrefitted C0 scales")
    mobility = analysis.get("mobility_context")
    if (
        not isinstance(mobility, dict)
        or mobility.get("kind") != "incoming_flow_raw_tertiles"
    ):
        raise ValueError(
            "probabilistic campaign needs predeclared incoming_flow_raw_tertiles"
        )
    return analysis


def _read_granular(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"missing canonical granular artifact: {path}")
    frame = pd.read_parquet(path)
    required = {
        *SUPPORT_COLUMNS,
        "observed",
        "observed_mask",
        "observation_weight",
        *PROBABILISTIC_METRICS,
        *INTERVAL_COLUMNS,
        *(column for pair in WIDTH_COLUMNS for column in pair),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{path}: granular artifact missing columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{path}: granular artifact is empty")
    if frame.duplicated(SUPPORT_COLUMNS).any():
        raise ValueError(f"{path}: duplicated canonical granular support rows")
    frame = frame.copy()
    if not frame["target"].isin(REQUIRED_HEADS).all():
        unexpected = sorted(set(frame["target"]) - set(REQUIRED_HEADS))
        raise ValueError(f"{path}: unexpected granular targets: {unexpected}")
    if set(frame["target"]) != set(REQUIRED_HEADS):
        missing_heads = sorted(set(REQUIRED_HEADS) - set(frame["target"]))
        raise ValueError(f"{path}: missing required granular heads: {missing_heads}")
    for column in ["observed_mask", "observation_weight", *PROBABILISTIC_METRICS]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: non-finite {column}")
        frame[column] = values
    if (
        not (frame["observed_mask"] > 0).all()
        or not (frame["observation_weight"] > 0).all()
    ):
        raise ValueError(f"{path}: granular output contains unobserved support")
    if (frame["wis"] < 0).any() or (frame["crps"] < 0).any():
        raise ValueError(f"{path}: WIS and CRPS must be nonnegative")
    for column in INTERVAL_COLUMNS:
        values = frame[column]
        if not values.isin([True, False, 0, 1]).all():
            raise ValueError(f"{path}: {column} must be Boolean")
        frame[column] = values.astype(bool)
    for low, high in WIDTH_COLUMNS:
        widths = pd.to_numeric(frame[high], errors="coerce") - pd.to_numeric(
            frame[low], errors="coerce"
        )
        if not np.isfinite(widths.to_numpy(dtype=float)).all() or (widths < 0).any():
            raise ValueError(f"{path}: invalid interval width")
        frame[f"width_{high.rsplit('_', 1)[-1]}"] = widths
    frame["region_id"] = frame["region_id"].astype(str)
    frame["target_date"] = frame["target_date"].astype(str)
    return frame


def _manifest_regions(campaign: AblationCampaign) -> dict[int, set[str]]:
    if campaign.split_manifest is None:
        raise ValueError(
            "probabilistic full-coverage analysis requires a split manifest"
        )
    payload = json.loads(campaign.split_manifest.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, list):
        raise ValueError("split manifest folds must be a list")
    regions: dict[int, set[str]] = {}
    for item in folds:
        if not isinstance(item, dict):
            raise ValueError("split manifest fold must be a mapping")
        fold = item.get("fold_index")
        values = item.get("region_ids")
        if not isinstance(fold, int) or not isinstance(values, list):
            raise ValueError("split manifest fold lacks fold_index or region_ids")
        value_set = {str(value) for value in values}
        if not value_set or len(value_set) != len(values):
            raise ValueError(f"split manifest fold {fold} has invalid region IDs")
        regions[fold] = value_set
    if tuple(sorted(regions)) != campaign.fold_indices:
        raise ValueError("campaign folds do not match persisted manifest folds")
    all_regions = set().union(*regions.values())
    if sum(len(value) for value in regions.values()) != len(all_regions):
        raise ValueError(
            "persisted manifest does not provide exact-once region coverage"
        )
    return regions


def _expected_regions(
    campaign: AblationCampaign, analysis: Mapping[str, Any]
) -> dict[int, set[str]]:
    if analysis.get("evaluation_design") != "temporal":
        return _manifest_regions(campaign)
    if campaign.fold_indices:
        raise ValueError("temporal campaign must not have spatial folds")
    manifest_path = analysis.get("region_manifest")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError("temporal campaign needs a complete region manifest")
    path = Path(manifest_path)
    if not path.is_absolute():
        path = campaign.path.resolve().parents[2] / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("temporal region manifest has no folds")
    region_lists = [item.get("region_ids") for item in folds]
    if any(not isinstance(values, list) or not values for values in region_lists):
        raise ValueError("temporal region manifest has invalid region IDs")
    all_regions = [str(region) for values in region_lists for region in values]
    if len(all_regions) != len(set(all_regions)):
        raise ValueError("temporal region manifest repeats a region")
    # A negative sentinel keeps the temporal unit distinct from spatial fold 0.
    return {-1: set(all_regions)}


def _run_frame(
    run: Any,
    *,
    granular_filename: str,
    expected_regions: set[str],
) -> pd.DataFrame:
    if run.seed is None:
        raise ValueError("probabilistic campaign run has no seed")
    frame = _read_granular(run.granular_path or run.run_dir / granular_filename)
    observed_regions = set(frame["region_id"])
    if observed_regions != expected_regions:
        missing = sorted(expected_regions - observed_regions)
        unexpected = sorted(observed_regions - expected_regions)
        raise ValueError(
            f"{run.run_dir}: held-out regional coverage mismatch; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    frame["condition"] = run.ablation
    frame["seed"] = run.seed
    frame["fold"] = -1 if run.fold is None else run.fold
    frame["run_id"] = run.run_dir.name
    return frame


def _per_run_head_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = frame.groupby(
        ["condition", "seed", "fold", "run_id", "target"], sort=True
    )[list(PROBABILISTIC_METRICS)].mean()
    return metrics.reset_index()


def _equal_head_metrics(per_head: pd.DataFrame) -> pd.DataFrame:
    return (
        per_head.groupby(["condition", "seed", "fold", "run_id"], sort=True)[
            list(PROBABILISTIC_METRICS)
        ]
        .mean()
        .rename(columns={"wis": "mean_target_wis"})
        .reset_index()
    )


def _paired_metrics(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    keys: list[str],
    condition: str,
) -> pd.DataFrame:
    merged = baseline.merge(
        candidate,
        on=keys,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            f"{condition}: baseline/candidate support is not exactly matched"
        )
    observed_baseline = "observed_baseline"
    observed_candidate = "observed_candidate"
    if (
        observed_baseline in merged
        and observed_candidate in merged
        and not np.array_equal(
            merged[observed_baseline].to_numpy(),
            merged[observed_candidate].to_numpy(),
        )
    ):
        raise ValueError(f"{condition}: paired observed values differ")
    for metric in PROBABILISTIC_METRICS:
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
        )
    merged["condition"] = condition
    return merged.drop(columns="_merge")


def _regional_summary(paired_regions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for condition, group in paired_regions.groupby("condition", sort=True):
        values = group["mean_target_wis_delta"].to_numpy(dtype=float)
        improved = values < -1e-12
        worsened = values > 1e-12
        records.append(
            {
                "condition": condition,
                "region_seed_pairs": len(group),
                "mean_target_wis_delta_mean": float(values.mean()),
                "mean_target_wis_delta_q05": float(np.quantile(values, 0.05)),
                "mean_target_wis_delta_q25": float(np.quantile(values, 0.25)),
                "mean_target_wis_delta_median": float(np.quantile(values, 0.5)),
                "mean_target_wis_delta_q75": float(np.quantile(values, 0.75)),
                "mean_target_wis_delta_q95": float(np.quantile(values, 0.95)),
                "improved_regions": int(improved.sum()),
                "unchanged_regions": int((~improved & ~worsened).sum()),
                "worsened_regions": int(worsened.sum()),
                "improved_fraction": float(improved.mean()),
                "unchanged_fraction": float((~improved & ~worsened).mean()),
                "worsened_fraction": float(worsened.mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def _mobility_context(
    regions: set[str],
    analysis: Mapping[str, Any],
) -> pd.DataFrame:
    context = analysis["mobility_context"]
    path = Path(str(context["path"]))
    if not path.is_absolute():
        # Deployed runs execute against a staged source tree whose layout
        # matches the repository root, while local analyses resolve relative
        # to the working directory. Prefer whichever candidate exists.
        source_root_candidate = Path(__file__).resolve().parents[2] / path
        if not path.exists() and source_root_candidate.exists():
            path = source_root_candidate
    if not path.exists():
        raise ValueError(f"mobility context data does not exist: {path}")

    import xarray as xr

    dataset = xr.open_zarr(path)
    variable = str(context.get("variable", "mobility"))
    if variable not in dataset:
        raise ValueError(f"mobility context variable is absent: {variable}")
    start = str(context["date_start"])
    end = str(context["date_end"])
    values = dataset[variable].sel(date=slice(start, end), target=sorted(regions))
    # The declared context path is the raw OD file; preprocessing applies
    # log1p only when building the processed training dataset.
    incoming = values.sum(dim="origin").mean(dim="date").to_pandas()
    incoming.index = incoming.index.astype(str)
    incoming = incoming.reindex(sorted(regions))
    if incoming.isna().any() or not np.isfinite(incoming.to_numpy(dtype=float)).all():
        raise ValueError("mobility context has missing or non-finite regions")
    boundaries = np.quantile(incoming.to_numpy(dtype=float), [1 / 3, 2 / 3])
    labels = np.select(
        [
            incoming.to_numpy(dtype=float) <= boundaries[0],
            incoming.to_numpy(dtype=float) <= boundaries[1],
        ],
        ["low", "middle"],
        default="high",
    )
    return pd.DataFrame(
        {
            "region_id": incoming.index,
            "mean_incoming_flow_raw": incoming.to_numpy(dtype=float),
            "mobility_context": labels,
        }
    )


def build_probabilistic_campaign_outputs(
    campaign: AblationCampaign,
    ablation_runs: Mapping[str, list[Any]],
) -> dict[str, pd.DataFrame]:
    """Validate full support, then return canonical probabilistic evidence tables."""
    analysis = campaign_analysis_spec(campaign)
    if analysis is None:
        return {}
    regions_by_fold = _expected_regions(campaign, analysis)
    expected_conditions = {condition.name for condition in campaign.conditions}
    if set(ablation_runs) != expected_conditions:
        raise ValueError("campaign runs do not contain exactly the declared conditions")

    frames: list[pd.DataFrame] = []
    coverage_records: list[dict[str, Any]] = []
    for condition, runs in sorted(ablation_runs.items()):
        for run in runs:
            fold = -1 if run.fold is None and not campaign.fold_indices else run.fold
            if fold not in regions_by_fold:
                raise ValueError(
                    f"{run.run_dir}: fold is outside the declared manifest"
                )
            frame = _run_frame(
                run,
                granular_filename=str(analysis["granular_filename"]),
                expected_regions=regions_by_fold[fold],
            )
            frames.append(frame)
            coverage_records.append(
                {
                    "condition": condition,
                    "seed": run.seed,
                    "fold": fold,
                    "expected_region_count": len(regions_by_fold[fold]),
                    "observed_region_count": frame["region_id"].nunique(),
                    "observed_rows": len(frame),
                    "required_heads_present": ";".join(
                        sorted(frame["target"].unique())
                    ),
                    "valid": True,
                }
            )
    granular = pd.concat(frames, ignore_index=True)
    coverage = pd.DataFrame.from_records(coverage_records).sort_values(
        ["condition", "seed", "fold"]
    )
    expected_all_regions = set().union(*regions_by_fold.values())
    for (condition, seed), group in coverage.groupby(["condition", "seed"], sort=True):
        folds = tuple(group["fold"].tolist())
        expected_folds = campaign.fold_indices or (-1,)
        if folds != expected_folds:
            raise ValueError(
                f"{condition}/seed={seed}: missing or reordered campaign folds"
            )
        regions = set(
            granular.loc[
                (granular["condition"] == condition) & (granular["seed"] == seed),
                "region_id",
            ]
        )
        if regions != expected_all_regions:
            raise ValueError(
                f"{condition}/seed={seed}: does not cover all valid regions"
            )

    per_head = _per_run_head_metrics(granular)
    scorecard = (
        _equal_head_metrics(per_head)
        .groupby("condition", sort=True)[
            ["mean_target_wis", "crps", "nll", "abs_error"]
        ]
        .agg(["mean", "std"])
    )
    scorecard.columns = ["_".join(column) for column in scorecard.columns]
    scorecard = scorecard.reset_index()
    head_scorecard = per_head.groupby(["condition", "target"], sort=True)[
        list(PROBABILISTIC_METRICS)
    ].agg(["mean", "std"])
    head_scorecard.columns = ["_".join(column) for column in head_scorecard.columns]
    head_scorecard = head_scorecard.reset_index()

    baseline_name = campaign.baseline_condition
    baseline = granular[granular["condition"] == baseline_name]
    pairwise: list[pd.DataFrame] = []
    region_pairs: list[pd.DataFrame] = []
    horizon_pairs: list[pd.DataFrame] = []
    for condition in sorted(expected_conditions - {baseline_name}):
        candidate = granular[granular["condition"] == condition]
        keys = ["seed", "fold", *SUPPORT_COLUMNS]
        pairwise.append(
            _paired_metrics(baseline, candidate, keys=keys, condition=condition)
        )
        # Per-region effects retain the complete head set before equal-head reduction.
        base_region = (
            baseline.groupby(["seed", "fold", "region_id", "target"], sort=True)[
                list(PROBABILISTIC_METRICS)
            ]
            .mean()
            .reset_index()
        )
        candidate_region = (
            candidate.groupby(["seed", "fold", "region_id", "target"], sort=True)[
                list(PROBABILISTIC_METRICS)
            ]
            .mean()
            .reset_index()
        )
        joined_region = _paired_metrics(
            base_region,
            candidate_region,
            keys=["seed", "fold", "region_id", "target"],
            condition=condition,
        )
        equal_region = (
            joined_region.groupby(
                ["condition", "seed", "fold", "region_id"], sort=True
            )[[f"{metric}_delta" for metric in PROBABILISTIC_METRICS]]
            .mean()
            .rename(columns={"wis_delta": "mean_target_wis_delta"})
            .reset_index()
        )
        region_pairs.append(equal_region)
        base_horizon = (
            baseline.groupby(["seed", "fold", "target", "horizon"], sort=True)[
                list(PROBABILISTIC_METRICS)
            ]
            .mean()
            .reset_index()
        )
        candidate_horizon = (
            candidate.groupby(["seed", "fold", "target", "horizon"], sort=True)[
                list(PROBABILISTIC_METRICS)
            ]
            .mean()
            .reset_index()
        )
        horizon_pairs.append(
            _paired_metrics(
                base_horizon,
                candidate_horizon,
                keys=["seed", "fold", "target", "horizon"],
                condition=condition,
            )
        )

    paired_rows = pd.concat(pairwise, ignore_index=True)
    temporal_blocks = analysis.get("temporal_blocks")
    block_deltas = None
    if temporal_blocks is not None:
        if analysis.get("evaluation_design") != "temporal":
            raise ValueError("temporal blocks require a temporal campaign")
        block_labels = []
        for index, block in enumerate(temporal_blocks):
            start, end = str(block["label_start"]), str(block["label_end"])
            block_labels.append((index, start, end))
        paired_rows["temporal_block"] = -1
        for index, start, end in block_labels:
            in_block = paired_rows["target_date"].between(start, end)
            if (paired_rows.loc[in_block, "temporal_block"] >= 0).any():
                raise ValueError("temporal blocks overlap")
            paired_rows.loc[in_block, "temporal_block"] = index
        if (paired_rows["temporal_block"] < 0).any():
            raise ValueError("scored target date is outside temporal blocks")
        observed_blocks = set(paired_rows["temporal_block"])
        if observed_blocks != set(range(len(block_labels))):
            raise ValueError("temporal evaluation lacks a declared block")
        block_deltas = (
            paired_rows.groupby(
                ["condition", "seed", "temporal_block", "target", "horizon"],
                sort=True,
            )[[f"{metric}_delta" for metric in PROBABILISTIC_METRICS]]
            .mean()
            .reset_index()
        )
    paired_fold_seed = (
        paired_rows.groupby(["condition", "fold", "seed", "target"], sort=True)[
            [f"{metric}_delta" for metric in PROBABILISTIC_METRICS]
        ]
        .mean()
        .reset_index()
    )
    paired_regions = pd.concat(region_pairs, ignore_index=True)
    paired_horizons = pd.concat(horizon_pairs, ignore_index=True)
    regional_summary = _regional_summary(paired_regions)
    context = _mobility_context(expected_all_regions, analysis)
    paired_regions = paired_regions.merge(
        context, on="region_id", how="left", validate="many_to_one"
    )
    if paired_regions["mobility_context"].isna().any():
        raise ValueError("regional mobility context is incomplete")
    mobility_summary = (
        paired_regions.groupby(["condition", "mobility_context"], sort=True)[
            "mean_target_wis_delta"
        ]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "region_seed_pairs"})
    )
    interval = (
        granular.groupby(["condition", "target"], sort=True)[
            [*INTERVAL_COLUMNS, "width_50", "width_80", "width_95"]
        ]
        .mean()
        .reset_index()
    )
    invalid_audit = (
        granular.groupby(["condition", "seed", "fold"], sort=True)
        .agg(
            granular_rows=("target", "size"),
            regions=("region_id", "nunique"),
            targets=("target", "nunique"),
            horizons=("horizon", "nunique"),
            target_dates=("target_date", "nunique"),
        )
        .reset_index()
    )
    invalid_audit["invalid_prediction_count"] = 0
    invalid_audit["source_measurements_independent"] = False
    invalid_audit["source_dependence_treatment"] = (
        "descriptive_only_no_independent_row_inference"
    )
    outputs = {
        "campaign_completeness_manifest": coverage,
        "condition_scorecard": scorecard,
        "condition_head_scorecard": head_scorecard,
        "paired_fold_seed_deltas": paired_fold_seed,
        "paired_region_deltas": paired_regions,
        "paired_head_horizon_deltas": paired_horizons,
        "regional_effect_distribution": regional_summary,
        "mobility_context_summary": mobility_summary,
        "calibration_interval_diagnostics": interval,
        "invalid_prediction_support_audit": invalid_audit,
    }
    if block_deltas is not None:
        outputs["paired_temporal_block_horizon_deltas"] = block_deltas
    return outputs
