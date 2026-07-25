#!/usr/bin/env python
"""Measure and refine region splits for mobility-network context retention."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.dataset_factory import split_nodes_by_ratio
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import EpiForecasterConfig


BASE_STRATEGY = "Representative 10-fold"
REFINED_STRATEGY = "Refined representative 10-fold"


@dataclass(frozen=True)
class MobilityNetwork:
    """Complete directed mean-flow network plus episodic high-flow mass."""

    region_ids: list[str]
    mean_flow: np.ndarray
    high_flow_event_mass: np.ndarray
    high_flow_event_threshold_raw: float


@dataclass(frozen=True)
class SplitDefinition:
    strategy: str
    split_id: str
    train_region_ids: list[str]
    validation_region_ids: list[str]
    test_region_ids: list[str]


@dataclass(frozen=True)
class SplitMobilityMetrics:
    strategy: str
    split_id: str
    test_region_count: int
    mean_incoming_flow_retained: float
    p10_incoming_flow_retained: float
    minimum_incoming_flow_retained: float
    minimum_retention_region_id: str
    mean_incoming_edges_retained: float
    test_internal_flow_fraction: float
    high_flow_event_internal_fraction: float
    boundary_flow_fraction: float
    training_flow_retained_fraction: float


@dataclass(frozen=True)
class RefinementSettings:
    """Hard constraints and objective weights for deterministic pair swaps."""

    min_retained_flow: float = 0.70
    ideal_retained_flow: float = 0.75
    high_flow_event_weight: float = 0.25
    worst_fold_weight: float = 0.50
    ideal_shortfall_weight: float = 10.0
    balance_count_slack: float = 0.0
    max_swaps: int = 100
    improvement_tolerance: float = 1.0e-12


@dataclass(frozen=True)
class AssignmentState:
    score: float
    internal_flow_by_fold: np.ndarray
    high_flow_internal_by_fold: np.ndarray
    internal_incoming_flow: np.ndarray
    retained_incoming_flow: np.ndarray
    balance_counts: np.ndarray


@dataclass(frozen=True)
class RefinementResult:
    folds: list[list[str]]
    history: pd.DataFrame
    balance_audit: pd.DataFrame
    initial_state: AssignmentState
    final_state: AssignmentState
    outlier_internal_flow_ceiling: float


def _select_run(dataset: xr.Dataset, run_id: str) -> xr.Dataset:
    if "run_id" in dataset.dims:
        return dataset.sel(run_id=run_id)
    return dataset


def load_valid_mobility(
    dataset_path: Path,
    *,
    run_id: str,
    high_flow_event_quantile: float,
) -> MobilityNetwork:
    """Load the complete directed valid-target network in raw trip units."""
    if not 0.0 < high_flow_event_quantile < 1.0:
        raise ValueError("high_flow_event_quantile must be between zero and one")

    dataset = xr.open_zarr(dataset_path, consolidated=True)
    try:
        selected = _select_run(dataset, run_id)
        valid_mask = np.asarray(selected["valid_targets"].values, dtype=bool)
        all_region_ids = np.asarray(selected[REGION_COORD].values).astype(str)
        valid_region_ids = all_region_ids[valid_mask].tolist()

        mobility = selected["mobility"].sel(
            origin=valid_region_ids,
            destination=valid_region_ids,
        )
        mobility = mobility.transpose(TEMPORAL_COORD, "origin", "destination")
        log_mobility = np.asarray(mobility.values, dtype=np.float64)
        if "mobility_time_mask" in selected:
            time_mask = np.asarray(
                selected["mobility_time_mask"].values,
                dtype=bool,
            )
            log_mobility = np.where(time_mask[:, None, None], log_mobility, 0.0)
    finally:
        dataset.close()

    raw_mobility = np.expm1(
        np.nan_to_num(log_mobility, nan=0.0, posinf=0.0, neginf=0.0)
    )
    diagonal = np.arange(len(valid_region_ids))
    raw_mobility[:, diagonal, diagonal] = 0.0

    positive_values = raw_mobility[raw_mobility > 0.0]
    if positive_values.size == 0:
        high_flow_threshold = 0.0
        high_flow_event_mass = np.zeros(
            raw_mobility.shape[1:],
            dtype=np.float64,
        )
    else:
        high_flow_threshold = float(
            np.quantile(positive_values, high_flow_event_quantile)
        )
        high_flow_event_mass = np.where(
            raw_mobility >= high_flow_threshold,
            raw_mobility,
            0.0,
        ).mean(axis=0, dtype=np.float64)

    return MobilityNetwork(
        region_ids=valid_region_ids,
        mean_flow=raw_mobility.mean(axis=0, dtype=np.float64),
        high_flow_event_mass=high_flow_event_mass,
        high_flow_event_threshold_raw=high_flow_threshold,
    )


def build_independent_seed_splits(
    *,
    config_path: Path,
    dataset_path: Path,
    run_id: str,
    seeds: list[int],
) -> list[SplitDefinition]:
    overrides = [
        f"data.dataset_path={dataset_path}",
        f"data.run_id={run_id}",
        "training.split_strategy=node",
        "training.node_split_strategy=stratified",
    ]
    config = EpiForecasterConfig.load(str(config_path), overrides=overrides)
    dataset = _select_run(
        xr.open_zarr(dataset_path, consolidated=True),
        run_id,
    )
    try:
        region_ids = np.asarray(dataset[REGION_COORD].values).astype(str)
        splits: list[SplitDefinition] = []
        for seed in seeds:
            config.training.split_seed = seed
            train_nodes, validation_nodes, test_nodes = split_nodes_by_ratio(config)
            splits.append(
                SplitDefinition(
                    strategy="Independent stratified seeds",
                    split_id=f"Seed {seed}",
                    train_region_ids=region_ids[train_nodes].tolist(),
                    validation_region_ids=region_ids[validation_nodes].tolist(),
                    test_region_ids=region_ids[test_nodes].tolist(),
                )
            )
    finally:
        dataset.close()
    return splits


def load_manifest_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("folds"), list):
        raise ValueError(f"Manifest does not contain a fold list: {path}")
    return payload


def folds_from_manifest_payload(payload: dict[str, object]) -> list[list[str]]:
    entries = sorted(
        payload["folds"],
        key=lambda entry: int(entry["fold_index"]),
    )
    return [[str(region_id) for region_id in entry["region_ids"]] for entry in entries]


def build_manifest_splits(
    folds: list[list[str]],
    *,
    strategy: str,
) -> list[SplitDefinition]:
    """Build full-coverage splits, using the next fold for validation."""
    splits: list[SplitDefinition] = []
    for fold_index, test_fold in enumerate(folds):
        validation_index = (fold_index + 1) % len(folds)
        validation_fold = folds[validation_index]
        train_folds = [
            region_id
            for index, fold in enumerate(folds)
            if index not in {fold_index, validation_index}
            for region_id in fold
        ]
        splits.append(
            SplitDefinition(
                strategy=strategy,
                split_id=f"Fold {fold_index}",
                train_region_ids=train_folds,
                validation_region_ids=list(validation_fold),
                test_region_ids=list(test_fold),
            )
        )
    return splits


def load_manifest_splits(path: Path) -> list[SplitDefinition]:
    return build_manifest_splits(
        folds_from_manifest_payload(load_manifest_payload(path)),
        strategy=BASE_STRATEGY,
    )


def _split_indices(
    split: SplitDefinition,
    region_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    region_index = {region_id: index for index, region_id in enumerate(region_ids)}

    def indices(values: list[str]) -> np.ndarray:
        missing = sorted(set(values) - set(region_index))
        if missing:
            raise ValueError(f"Split contains unknown region IDs: {missing[:5]}")
        return np.asarray([region_index[value] for value in values], dtype=np.int64)

    return (
        indices(split.train_region_ids),
        indices(split.validation_region_ids),
        indices(split.test_region_ids),
    )


def build_target_retention_table(
    *,
    splits: list[SplitDefinition],
    network: MobilityNetwork,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for split in splits:
        train, validation, test = _split_indices(split, network.region_ids)
        context = np.concatenate([train, validation])
        incoming_flow = network.mean_flow[:, test].sum(axis=0, dtype=np.float64)
        retained_flow = network.mean_flow[np.ix_(context, test)].sum(
            axis=0,
            dtype=np.float64,
        )
        retained_fraction = np.divide(
            retained_flow,
            incoming_flow,
            out=np.ones_like(retained_flow),
            where=incoming_flow > 0.0,
        )
        records.extend(
            {
                "strategy": split.strategy,
                "split_id": split.split_id,
                "region_id": network.region_ids[target],
                "incoming_flow_raw": incoming,
                "retained_incoming_flow_raw": retained,
                "incoming_flow_retained_fraction": fraction,
            }
            for target, incoming, retained, fraction in zip(
                test,
                incoming_flow,
                retained_flow,
                retained_fraction,
                strict=True,
            )
        )
    return pd.DataFrame.from_records(records)


def compute_split_mobility_metrics(
    *,
    split: SplitDefinition,
    network: MobilityNetwork,
) -> SplitMobilityMetrics:
    """Compute full-network test context and training-graph retention."""
    train, validation, test = _split_indices(split, network.region_ids)
    context = np.concatenate([train, validation])
    adjacency = network.mean_flow > 0.0

    incoming_flow = network.mean_flow[:, test].sum(axis=0, dtype=np.float64)
    retained_incoming_flow = network.mean_flow[np.ix_(context, test)].sum(
        axis=0,
        dtype=np.float64,
    )
    incoming_flow_fraction = np.divide(
        retained_incoming_flow,
        incoming_flow,
        out=np.ones_like(retained_incoming_flow),
        where=incoming_flow > 0.0,
    )

    incoming_edges = adjacency[:, test].sum(axis=0, dtype=np.float64)
    retained_incoming_edges = adjacency[np.ix_(context, test)].sum(
        axis=0,
        dtype=np.float64,
    )
    incoming_edge_fraction = np.divide(
        retained_incoming_edges,
        incoming_edges,
        out=np.ones_like(retained_incoming_edges),
        where=incoming_edges > 0.0,
    )

    total_flow = float(network.mean_flow.sum(dtype=np.float64))
    total_high_flow = float(network.high_flow_event_mass.sum(dtype=np.float64))
    test_internal_flow = float(
        network.mean_flow[np.ix_(test, test)].sum(dtype=np.float64)
    )
    test_internal_high_flow = float(
        network.high_flow_event_mass[np.ix_(test, test)].sum(dtype=np.float64)
    )
    boundary_flow = float(
        network.mean_flow[np.ix_(test, context)].sum(dtype=np.float64)
        + network.mean_flow[np.ix_(context, test)].sum(dtype=np.float64)
    )
    training_flow = float(network.mean_flow[np.ix_(train, train)].sum(dtype=np.float64))
    minimum_index = int(np.argmin(incoming_flow_fraction))

    return SplitMobilityMetrics(
        strategy=split.strategy,
        split_id=split.split_id,
        test_region_count=len(test),
        mean_incoming_flow_retained=float(incoming_flow_fraction.mean()),
        p10_incoming_flow_retained=float(np.quantile(incoming_flow_fraction, 0.1)),
        minimum_incoming_flow_retained=float(incoming_flow_fraction[minimum_index]),
        minimum_retention_region_id=network.region_ids[test[minimum_index]],
        mean_incoming_edges_retained=float(incoming_edge_fraction.mean()),
        test_internal_flow_fraction=test_internal_flow / total_flow,
        high_flow_event_internal_fraction=(
            test_internal_high_flow / total_high_flow if total_high_flow > 0.0 else 0.0
        ),
        boundary_flow_fraction=boundary_flow / total_flow,
        training_flow_retained_fraction=training_flow / total_flow,
    )


def build_membership_table(splits: list[SplitDefinition]) -> pd.DataFrame:
    records = [
        {
            "strategy": split.strategy,
            "split_id": split.split_id,
            "region_id": region_id,
            "role": role,
        }
        for split in splits
        for role, region_ids in (
            ("train", split.train_region_ids),
            ("validation", split.validation_region_ids),
            ("test", split.test_region_ids),
        )
        for region_id in region_ids
    ]
    return pd.DataFrame.from_records(records)


def build_strategy_summary(
    metrics: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    valid_region_count: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    test_membership = membership[membership["role"] == "test"]
    for strategy, strategy_metrics in metrics.groupby("strategy", sort=False):
        strategy_tests = test_membership[test_membership["strategy"] == strategy]
        unique_regions = int(strategy_tests["region_id"].nunique())
        test_assignments = int(len(strategy_tests))
        records.append(
            {
                "strategy": strategy,
                "split_count": int(strategy_metrics["split_id"].nunique()),
                "valid_region_count": valid_region_count,
                "unique_test_regions": unique_regions,
                "test_region_coverage": unique_regions / valid_region_count,
                "repeated_test_assignments": test_assignments - unique_regions,
                "mean_incoming_flow_retained": strategy_metrics[
                    "mean_incoming_flow_retained"
                ].mean(),
                "worst_target_incoming_flow_retained": strategy_metrics[
                    "minimum_incoming_flow_retained"
                ].min(),
                "mean_test_internal_flow_fraction": strategy_metrics[
                    "test_internal_flow_fraction"
                ].mean(),
                "mean_high_flow_event_internal_fraction": strategy_metrics[
                    "high_flow_event_internal_fraction"
                ].mean(),
                "mean_training_flow_retained_fraction": strategy_metrics[
                    "training_flow_retained_fraction"
                ].mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def _quantile_feature_frame(
    values: np.ndarray,
    *,
    prefix: str,
    num_bins: int,
    index: pd.Index,
) -> pd.DataFrame:
    ranked_values = pd.Series(values, index=index).rank(method="first")
    bins = pd.qcut(
        ranked_values,
        q=num_bins,
        labels=False,
    ).astype(int)
    return pd.DataFrame(
        {
            f"{prefix}_q{bin_index}": (bins == bin_index).astype(np.float64)
            for bin_index in sorted(bins.unique())
        },
        index=index,
    )


def load_balance_features(
    dataset_path: Path,
    *,
    run_id: str,
    network: MobilityNetwork,
    geojson_path: Path,
) -> pd.DataFrame:
    """Reconstruct the representative manifest's balancing feature families."""
    import geopandas as gpd
    from libpysal.weights import Queen

    dataset = xr.open_zarr(dataset_path, consolidated=True)
    try:
        selected = _select_run(dataset, run_id)
        selected = selected.sel(region_id=network.region_ids)
        index = pd.Index(network.region_ids, name="region_id")

        population = np.log1p(
            np.asarray(selected["population"].values, dtype=np.float64)
        )
        wastewater_source = np.asarray(
            selected["edar_has_source"].values,
            dtype=bool,
        )
        cases_coverage = np.asarray(
            selected["cases_mask"].values,
            dtype=np.float64,
        ).mean(axis=0)
        hospitalizations_coverage = np.asarray(
            selected["hospitalizations_mask"].values,
            dtype=np.float64,
        ).mean(axis=0)
        deaths_coverage = np.asarray(
            selected["deaths_mask"].values,
            dtype=np.float64,
        ).mean(axis=0)
        biomarker_coverage = np.stack(
            [
                np.asarray(
                    selected[f"edar_biomarker_{variant}_mask"].values,
                    dtype=np.float64,
                )
                for variant in ("N1", "N2", "IP4")
            ]
        ).mean(axis=(0, 1))
    finally:
        dataset.close()

    regions = gpd.read_file(geojson_path)
    regions["id"] = regions["id"].astype(str)
    regions = regions.set_index("id").loc[network.region_ids].copy()
    if regions.geometry.is_empty.any():
        raise ValueError("GeoJSON contains empty valid-target geometries")
    centroids = regions.geometry.centroid
    queen = Queen.from_dataframe(regions, use_index=True)
    queen_degree = np.asarray(
        [len(queen.neighbors[region_id]) for region_id in network.region_ids],
        dtype=np.float64,
    )

    adjacency = network.mean_flow > 0.0
    continuous_features = [
        ("population", population, 5),
        ("longitude", centroids.x.to_numpy(dtype=np.float64), 5),
        ("latitude", centroids.y.to_numpy(dtype=np.float64), 5),
        ("queen_degree", queen_degree, 5),
        ("mobility_in_degree", adjacency.sum(axis=0), 5),
        ("mobility_out_degree", adjacency.sum(axis=1), 5),
        (
            "mobility_in_mass",
            np.log1p(network.mean_flow.sum(axis=0)),
            5,
        ),
        (
            "mobility_out_mass",
            np.log1p(network.mean_flow.sum(axis=1)),
            5,
        ),
        ("cases_coverage", cases_coverage, 5),
        ("hospitalizations_coverage", hospitalizations_coverage, 3),
        ("deaths_coverage", deaths_coverage, 5),
        ("biomarker_coverage", biomarker_coverage, 4),
    ]
    frames = [
        pd.DataFrame(
            {
                "wastewater_source": wastewater_source.astype(np.float64),
                "no_wastewater_source": (~wastewater_source).astype(np.float64),
            },
            index=index,
        )
    ]
    frames.extend(
        _quantile_feature_frame(
            values,
            prefix=name,
            num_bins=num_bins,
            index=index,
        )
        for name, values, num_bins in continuous_features
    )
    return pd.concat(frames, axis=1)


def _fold_assignments(
    folds: list[list[str]],
    region_ids: list[str],
) -> np.ndarray:
    region_index = {region_id: index for index, region_id in enumerate(region_ids)}
    assignments = np.full(len(region_ids), -1, dtype=np.int64)
    for fold_index, fold in enumerate(folds):
        for region_id in fold:
            if region_id not in region_index:
                raise ValueError(f"Fold contains unknown region ID {region_id!r}")
            node_index = region_index[region_id]
            if assignments[node_index] >= 0:
                raise ValueError(f"Region {region_id!r} appears in multiple folds")
            assignments[node_index] = fold_index
    missing = [region_ids[index] for index in np.flatnonzero(assignments < 0).tolist()]
    if missing:
        raise ValueError(f"Folds do not cover valid regions: {missing[:5]}")
    return assignments


def _fold_feature_counts(
    assignments: np.ndarray,
    *,
    num_folds: int,
    balance_matrix: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [balance_matrix[assignments == fold].sum(axis=0) for fold in range(num_folds)]
    )


def _score_state(
    *,
    internal_flow_by_fold: np.ndarray,
    high_flow_internal_by_fold: np.ndarray,
    retained_incoming_flow: np.ndarray,
    total_flow: float,
    total_high_flow: float,
    settings: RefinementSettings,
) -> float:
    internal_fraction = float(internal_flow_by_fold.sum()) / total_flow
    high_flow_fraction = (
        float(high_flow_internal_by_fold.sum()) / total_high_flow
        if total_high_flow > 0.0
        else 0.0
    )
    worst_fold_fraction = float(internal_flow_by_fold.max()) / total_flow
    shortfall = np.maximum(
        0.0,
        settings.ideal_retained_flow - retained_incoming_flow,
    )
    return (
        internal_fraction
        + settings.high_flow_event_weight * high_flow_fraction
        + settings.worst_fold_weight * worst_fold_fraction
        + settings.ideal_shortfall_weight * float(np.square(shortfall).mean())
    )


def _build_assignment_state(
    *,
    assignments: np.ndarray,
    network: MobilityNetwork,
    balance_matrix: np.ndarray,
    num_folds: int,
    settings: RefinementSettings,
) -> AssignmentState:
    internal_flow_by_fold = np.asarray(
        [
            network.mean_flow[np.ix_(assignments == fold, assignments == fold)].sum(
                dtype=np.float64
            )
            for fold in range(num_folds)
        ]
    )
    high_flow_internal_by_fold = np.asarray(
        [
            network.high_flow_event_mass[
                np.ix_(assignments == fold, assignments == fold)
            ].sum(dtype=np.float64)
            for fold in range(num_folds)
        ]
    )
    same_fold = assignments[:, None] == assignments[None, :]
    internal_incoming_flow = np.where(
        same_fold,
        network.mean_flow,
        0.0,
    ).sum(axis=0, dtype=np.float64)
    total_incoming_flow = network.mean_flow.sum(axis=0, dtype=np.float64)
    retained_incoming_flow = np.divide(
        total_incoming_flow - internal_incoming_flow,
        total_incoming_flow,
        out=np.ones_like(total_incoming_flow),
        where=total_incoming_flow > 0.0,
    )
    balance_counts = _fold_feature_counts(
        assignments,
        num_folds=num_folds,
        balance_matrix=balance_matrix,
    )
    return AssignmentState(
        score=_score_state(
            internal_flow_by_fold=internal_flow_by_fold,
            high_flow_internal_by_fold=high_flow_internal_by_fold,
            retained_incoming_flow=retained_incoming_flow,
            total_flow=float(network.mean_flow.sum(dtype=np.float64)),
            total_high_flow=float(network.high_flow_event_mass.sum(dtype=np.float64)),
            settings=settings,
        ),
        internal_flow_by_fold=internal_flow_by_fold,
        high_flow_internal_by_fold=high_flow_internal_by_fold,
        internal_incoming_flow=internal_incoming_flow,
        retained_incoming_flow=retained_incoming_flow,
        balance_counts=balance_counts,
    )


def _candidate_state(
    *,
    assignments: np.ndarray,
    state: AssignmentState,
    first_node: int,
    second_node: int,
    network: MobilityNetwork,
    balance_matrix: np.ndarray,
    settings: RefinementSettings,
) -> AssignmentState:
    first_fold = int(assignments[first_node])
    second_fold = int(assignments[second_node])
    first_members = np.flatnonzero(assignments == first_fold)
    first_members = first_members[first_members != first_node]
    second_members = np.flatnonzero(assignments == second_fold)
    second_members = second_members[second_members != second_node]

    internal_flow_by_fold = state.internal_flow_by_fold.copy()
    high_flow_internal_by_fold = state.high_flow_internal_by_fold.copy()
    for matrix, fold_values in (
        (network.mean_flow, internal_flow_by_fold),
        (network.high_flow_event_mass, high_flow_internal_by_fold),
    ):
        fold_values[first_fold] += (
            matrix[second_node, first_members].sum()
            + matrix[first_members, second_node].sum()
            - matrix[first_node, first_members].sum()
            - matrix[first_members, first_node].sum()
        )
        fold_values[second_fold] += (
            matrix[first_node, second_members].sum()
            + matrix[second_members, first_node].sum()
            - matrix[second_node, second_members].sum()
            - matrix[second_members, second_node].sum()
        )

    internal_incoming_flow = state.internal_incoming_flow.copy()
    internal_incoming_flow[first_members] += (
        network.mean_flow[second_node, first_members]
        - network.mean_flow[first_node, first_members]
    )
    internal_incoming_flow[second_members] += (
        network.mean_flow[first_node, second_members]
        - network.mean_flow[second_node, second_members]
    )
    internal_incoming_flow[first_node] = network.mean_flow[
        second_members,
        first_node,
    ].sum(dtype=np.float64)
    internal_incoming_flow[second_node] = network.mean_flow[
        first_members,
        second_node,
    ].sum(dtype=np.float64)

    total_incoming_flow = network.mean_flow.sum(axis=0, dtype=np.float64)
    retained_incoming_flow = np.divide(
        total_incoming_flow - internal_incoming_flow,
        total_incoming_flow,
        out=np.ones_like(total_incoming_flow),
        where=total_incoming_flow > 0.0,
    )
    balance_counts = state.balance_counts.copy()
    balance_counts[first_fold] += (
        balance_matrix[second_node] - balance_matrix[first_node]
    )
    balance_counts[second_fold] += (
        balance_matrix[first_node] - balance_matrix[second_node]
    )

    return AssignmentState(
        score=_score_state(
            internal_flow_by_fold=internal_flow_by_fold,
            high_flow_internal_by_fold=high_flow_internal_by_fold,
            retained_incoming_flow=retained_incoming_flow,
            total_flow=float(network.mean_flow.sum(dtype=np.float64)),
            total_high_flow=float(network.high_flow_event_mass.sum(dtype=np.float64)),
            settings=settings,
        ),
        internal_flow_by_fold=internal_flow_by_fold,
        high_flow_internal_by_fold=high_flow_internal_by_fold,
        internal_incoming_flow=internal_incoming_flow,
        retained_incoming_flow=retained_incoming_flow,
        balance_counts=balance_counts,
    )


def _balance_reference(
    *,
    initial_counts: np.ndarray,
    fold_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    global_proportion = initial_counts.sum(axis=0) / fold_sizes.sum()
    ideal_counts = fold_sizes[:, None] * global_proportion[None, :]
    max_deviation = np.abs(initial_counts - ideal_counts).max(axis=0)
    total_squared_deviation = float(np.square(initial_counts - ideal_counts).sum())
    return ideal_counts, max_deviation, total_squared_deviation


def _balance_is_valid(
    counts: np.ndarray,
    *,
    ideal_counts: np.ndarray,
    baseline_max_deviation: np.ndarray,
    baseline_total_squared_deviation: float,
    settings: RefinementSettings,
) -> bool:
    deviation = counts - ideal_counts
    if np.any(
        np.abs(deviation)
        > baseline_max_deviation + settings.balance_count_slack + 1.0e-9
    ):
        return False
    return (
        float(np.square(deviation).sum())
        <= baseline_total_squared_deviation + settings.balance_count_slack + 1.0e-9
    )


def _candidate_pairs(
    assignments: np.ndarray,
    *,
    focus_folds: set[int] | None,
) -> list[tuple[int, int]]:
    num_folds = int(assignments.max()) + 1
    pairs: list[tuple[int, int]] = []
    for first_fold in range(num_folds):
        for second_fold in range(first_fold + 1, num_folds):
            if (
                focus_folds is not None
                and first_fold not in focus_folds
                and second_fold not in focus_folds
            ):
                continue
            pairs.extend(
                (int(first_node), int(second_node))
                for first_node in np.flatnonzero(assignments == first_fold)
                for second_node in np.flatnonzero(assignments == second_fold)
            )
    return pairs


def _history_record(
    *,
    iteration: int,
    state: AssignmentState,
    network: MobilityNetwork,
    settings: RefinementSettings,
    first_region_id: str | None,
    second_region_id: str | None,
) -> dict[str, object]:
    total_flow = float(network.mean_flow.sum(dtype=np.float64))
    total_high_flow = float(network.high_flow_event_mass.sum(dtype=np.float64))
    return {
        "iteration": iteration,
        "first_region_id": first_region_id,
        "second_region_id": second_region_id,
        "objective_score": state.score,
        "total_test_internal_flow_fraction": (
            float(state.internal_flow_by_fold.sum()) / total_flow
        ),
        "worst_fold_internal_flow_fraction": (
            float(state.internal_flow_by_fold.max()) / total_flow
        ),
        "high_flow_event_internal_fraction": (
            float(state.high_flow_internal_by_fold.sum()) / total_high_flow
            if total_high_flow > 0.0
            else 0.0
        ),
        "minimum_target_retained_flow": float(state.retained_incoming_flow.min()),
        "targets_below_ideal": int(
            np.count_nonzero(
                state.retained_incoming_flow < settings.ideal_retained_flow
            )
        ),
    }


def refine_manifest_folds(
    *,
    folds: list[list[str]],
    network: MobilityNetwork,
    balance_features: pd.DataFrame,
    settings: RefinementSettings,
) -> RefinementResult:
    """Refine fold membership with deterministic, balance-safe pair swaps."""
    if settings.min_retained_flow > settings.ideal_retained_flow:
        raise ValueError("min_retained_flow cannot exceed ideal_retained_flow")
    if settings.max_swaps < 0:
        raise ValueError("max_swaps must be non-negative")
    if list(balance_features.index.astype(str)) != network.region_ids:
        raise ValueError("Balance features must follow mobility region order")

    assignments = _fold_assignments(folds, network.region_ids)
    num_folds = len(folds)
    fold_sizes = np.bincount(assignments, minlength=num_folds)
    balance_matrix = balance_features.to_numpy(dtype=np.float64)
    initial_state = _build_assignment_state(
        assignments=assignments,
        network=network,
        balance_matrix=balance_matrix,
        num_folds=num_folds,
        settings=settings,
    )
    if float(initial_state.retained_incoming_flow.min()) < settings.min_retained_flow:
        raise ValueError(
            "Base manifest violates the required retained-flow floor: "
            f"{initial_state.retained_incoming_flow.min():.6f} "
            f"< {settings.min_retained_flow:.6f}"
        )

    ideal_counts, baseline_max_deviation, baseline_total_squared_deviation = (
        _balance_reference(
            initial_counts=initial_state.balance_counts,
            fold_sizes=fold_sizes,
        )
    )
    state = initial_state
    total_flow = float(network.mean_flow.sum(dtype=np.float64))
    if num_folds > 1:
        outlier_internal_flow_ceiling = float(
            np.partition(initial_state.internal_flow_by_fold, -2)[-2] / total_flow
        )
    else:
        outlier_internal_flow_ceiling = float(
            initial_state.internal_flow_by_fold[0] / total_flow
        )
    history = [
        _history_record(
            iteration=0,
            state=state,
            network=network,
            settings=settings,
            first_region_id=None,
            second_region_id=None,
        )
    ]

    for iteration in range(1, settings.max_swaps + 1):
        outlier_fold = int(np.argmax(state.internal_flow_by_fold))
        worst_target_fold = int(
            assignments[int(np.argmin(state.retained_incoming_flow))]
        )
        best: tuple[float, int, int, AssignmentState] | None = None

        for focus_folds in ({outlier_fold, worst_target_fold}, None):
            for first_node, second_node in _candidate_pairs(
                assignments,
                focus_folds=focus_folds,
            ):
                candidate = _candidate_state(
                    assignments=assignments,
                    state=state,
                    first_node=first_node,
                    second_node=second_node,
                    network=network,
                    balance_matrix=balance_matrix,
                    settings=settings,
                )
                if (
                    float(candidate.retained_incoming_flow.min())
                    < settings.min_retained_flow - 1.0e-12
                ):
                    continue
                if not _balance_is_valid(
                    candidate.balance_counts,
                    ideal_counts=ideal_counts,
                    baseline_max_deviation=baseline_max_deviation,
                    baseline_total_squared_deviation=(baseline_total_squared_deviation),
                    settings=settings,
                ):
                    continue
                ordering = (
                    candidate.score,
                    first_node,
                    second_node,
                    candidate,
                )
                if best is None or ordering[:3] < best[:3]:
                    best = ordering

            if (
                best is not None
                and best[0] < state.score - settings.improvement_tolerance
            ):
                break

        if best is None or best[0] >= state.score - settings.improvement_tolerance:
            break

        _, first_node, second_node, state = best
        first_region_id = network.region_ids[first_node]
        second_region_id = network.region_ids[second_node]
        assignments[first_node], assignments[second_node] = (
            assignments[second_node],
            assignments[first_node],
        )
        history.append(
            _history_record(
                iteration=iteration,
                state=state,
                network=network,
                settings=settings,
                first_region_id=first_region_id,
                second_region_id=second_region_id,
            )
        )
        if (
            float(state.retained_incoming_flow.min()) >= settings.ideal_retained_flow
            and float(state.internal_flow_by_fold.max()) / total_flow
            <= outlier_internal_flow_ceiling
        ):
            break

    verified_state = _build_assignment_state(
        assignments=assignments,
        network=network,
        balance_matrix=balance_matrix,
        num_folds=num_folds,
        settings=settings,
    )
    for field_name in (
        "internal_flow_by_fold",
        "high_flow_internal_by_fold",
        "internal_incoming_flow",
        "retained_incoming_flow",
        "balance_counts",
    ):
        np.testing.assert_allclose(
            getattr(state, field_name),
            getattr(verified_state, field_name),
            err_msg=f"Incremental swap state drifted for {field_name}",
        )
    np.testing.assert_allclose(
        state.score,
        verified_state.score,
        err_msg="Incremental swap objective drifted",
    )
    state = verified_state

    refined_folds = [
        [
            network.region_ids[index]
            for index in np.flatnonzero(assignments == fold_index)
        ]
        for fold_index in range(num_folds)
    ]
    initial_deviation = initial_state.balance_counts - ideal_counts
    final_deviation = state.balance_counts - ideal_counts
    balance_audit = pd.DataFrame(
        {
            "feature": balance_features.columns,
            "baseline_max_abs_count_deviation": np.abs(initial_deviation).max(axis=0),
            "refined_max_abs_count_deviation": np.abs(final_deviation).max(axis=0),
            "baseline_squared_count_deviation": np.square(initial_deviation).sum(
                axis=0
            ),
            "refined_squared_count_deviation": np.square(final_deviation).sum(axis=0),
        }
    )
    balance_audit["within_baseline_envelope"] = (
        balance_audit["refined_max_abs_count_deviation"]
        <= balance_audit["baseline_max_abs_count_deviation"]
        + settings.balance_count_slack
        + 1.0e-9
    )

    return RefinementResult(
        folds=refined_folds,
        history=pd.DataFrame.from_records(history),
        balance_audit=balance_audit,
        initial_state=initial_state,
        final_state=state,
        outlier_internal_flow_ceiling=outlier_internal_flow_ceiling,
    )


def build_refined_manifest(
    *,
    base_payload: dict[str, object],
    base_manifest_path: Path,
    result: RefinementResult,
    settings: RefinementSettings,
    network: MobilityNetwork,
) -> dict[str, object]:
    payload = copy.deepcopy(base_payload)
    payload["folds"] = [
        {"fold_index": fold_index, "region_ids": sorted(region_ids)}
        for fold_index, region_ids in enumerate(result.folds)
    ]
    payload["balance_features"] = result.balance_audit["feature"].tolist()
    payload["diagnostics"] = {
        "refinement": {
            "base_manifest": str(base_manifest_path),
            "base_manifest_sha256": hashlib.sha256(
                base_manifest_path.read_bytes()
            ).hexdigest(),
            "swap_count": int(len(result.history) - 1),
            "initial_objective": result.initial_state.score,
            "final_objective": result.final_state.score,
            "initial_minimum_target_retained_flow": float(
                result.initial_state.retained_incoming_flow.min()
            ),
            "final_minimum_target_retained_flow": float(
                result.final_state.retained_incoming_flow.min()
            ),
            "initial_worst_fold_internal_flow_fraction": float(
                result.initial_state.internal_flow_by_fold.max()
                / network.mean_flow.sum()
            ),
            "final_worst_fold_internal_flow_fraction": float(
                result.final_state.internal_flow_by_fold.max() / network.mean_flow.sum()
            ),
            "outlier_internal_flow_ceiling": (result.outlier_internal_flow_ceiling),
            "baseline_balance_squared_count_deviation": float(
                result.balance_audit["baseline_squared_count_deviation"].sum()
            ),
            "final_balance_squared_count_deviation": float(
                result.balance_audit["refined_squared_count_deviation"].sum()
            ),
            "high_flow_event_threshold_raw": (network.high_flow_event_threshold_raw),
            "settings": asdict(settings),
        }
    }
    payload["strategy"] = "representative-10fold-mobility-refined"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-coverage-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="real")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--geojson",
        type=Path,
        default=Path("data/files/geo/fl_municipios_catalonia.geojson"),
    )
    parser.add_argument("--minimum-retained-flow", type=float, default=0.70)
    parser.add_argument("--ideal-retained-flow", type=float, default=0.75)
    parser.add_argument("--high-flow-event-quantile", type=float, default=0.995)
    parser.add_argument("--high-flow-event-weight", type=float, default=0.25)
    parser.add_argument("--worst-fold-weight", type=float, default=0.50)
    parser.add_argument("--ideal-shortfall-weight", type=float, default=10.0)
    parser.add_argument("--balance-count-slack", type=float, default=0.0)
    parser.add_argument("--max-swaps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")

    settings = RefinementSettings(
        min_retained_flow=args.minimum_retained_flow,
        ideal_retained_flow=args.ideal_retained_flow,
        high_flow_event_weight=args.high_flow_event_weight,
        worst_fold_weight=args.worst_fold_weight,
        ideal_shortfall_weight=args.ideal_shortfall_weight,
        balance_count_slack=args.balance_count_slack,
        max_swaps=args.max_swaps,
    )
    network = load_valid_mobility(
        args.dataset,
        run_id=args.run_id,
        high_flow_event_quantile=args.high_flow_event_quantile,
    )
    base_payload = load_manifest_payload(args.full_coverage_manifest)
    base_folds = folds_from_manifest_payload(base_payload)
    balance_features = load_balance_features(
        args.dataset,
        run_id=args.run_id,
        network=network,
        geojson_path=args.geojson,
    )
    refinement = refine_manifest_folds(
        folds=base_folds,
        network=network,
        balance_features=balance_features,
        settings=settings,
    )

    splits = build_independent_seed_splits(
        config_path=args.config,
        dataset_path=args.dataset,
        run_id=args.run_id,
        seeds=seeds,
    )
    splits.extend(build_manifest_splits(base_folds, strategy=BASE_STRATEGY))
    splits.extend(
        build_manifest_splits(
            refinement.folds,
            strategy=REFINED_STRATEGY,
        )
    )

    metrics = pd.DataFrame.from_records(
        [
            asdict(
                compute_split_mobility_metrics(
                    split=split,
                    network=network,
                )
            )
            for split in splits
        ]
    )
    membership = build_membership_table(splits)
    target_retention = build_target_retention_table(
        splits=splits,
        network=network,
    )
    summary = build_strategy_summary(
        metrics,
        membership,
        valid_region_count=len(network.region_ids),
    )
    refined_manifest = build_refined_manifest(
        base_payload=base_payload,
        base_manifest_path=args.full_coverage_manifest,
        result=refinement,
        settings=settings,
        network=network,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "split_mobility_metrics.csv", index=False)
    membership.to_csv(args.output_dir / "split_membership.csv", index=False)
    target_retention.to_csv(
        args.output_dir / "target_mobility_retention.csv",
        index=False,
    )
    summary.to_csv(args.output_dir / "strategy_summary.csv", index=False)
    refinement.balance_audit.to_csv(
        args.output_dir / "balance_audit.csv",
        index=False,
    )
    refinement.history.to_csv(
        args.output_dir / "optimization_history.csv",
        index=False,
    )
    refined_manifest_path = (
        args.output_dir / "refined_representative_10fold_manifest.json"
    )
    refined_manifest_path.write_text(
        json.dumps(refined_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": str(args.dataset),
                "run_id": args.run_id,
                "base_manifest": str(args.full_coverage_manifest),
                "base_manifest_sha256": hashlib.sha256(
                    args.full_coverage_manifest.read_bytes()
                ).hexdigest(),
                "refined_manifest": str(refined_manifest_path),
                "complete_directed_network": True,
                "top_k_filtering": False,
                "high_flow_event_quantile": args.high_flow_event_quantile,
                "high_flow_event_threshold_raw": (
                    network.high_flow_event_threshold_raw
                ),
                "settings": asdict(settings),
                "swap_count": int(len(refinement.history) - 1),
                "outlier_internal_flow_ceiling": (
                    refinement.outlier_internal_flow_ceiling
                ),
                "all_targets_meet_minimum_retained_flow": bool(
                    refinement.final_state.retained_incoming_flow.min()
                    >= settings.min_retained_flow
                ),
                "all_targets_meet_ideal_retained_flow": bool(
                    refinement.final_state.retained_incoming_flow.min()
                    >= settings.ideal_retained_flow
                ),
                "balance_squared_count_deviation_not_increased": bool(
                    refinement.balance_audit["refined_squared_count_deviation"].sum()
                    <= refinement.balance_audit[
                        "baseline_squared_count_deviation"
                    ].sum()
                    + settings.balance_count_slack
                    + 1.0e-9
                ),
                "all_balance_features_within_baseline_envelope": bool(
                    refinement.balance_audit["within_baseline_envelope"].all()
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"\nRefined manifest: {refined_manifest_path}")


if __name__ == "__main__":
    main()
