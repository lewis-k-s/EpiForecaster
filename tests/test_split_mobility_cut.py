from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.analyze_split_mobility_cut import (
    MobilityNetwork,
    RefinementSettings,
    SplitDefinition,
    build_membership_table,
    build_strategy_summary,
    compute_split_mobility_metrics,
    refine_manifest_folds,
)


def _network(mobility: np.ndarray) -> MobilityNetwork:
    return MobilityNetwork(
        region_ids=[chr(ord("a") + index) for index in range(len(mobility))],
        mean_flow=mobility.astype(np.float64),
        high_flow_event_mass=mobility.astype(np.float64),
        high_flow_event_threshold_raw=5.0,
    )


def test_compute_split_mobility_metrics_tracks_unavailable_test_flow() -> None:
    mobility = np.array(
        [
            [0.0, 5.0, 1.0, 0.0],
            [4.0, 0.0, 3.0, 0.0],
            [1.0, 2.0, 0.0, 7.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )
    split = SplitDefinition(
        strategy="example",
        split_id="fold",
        train_region_ids=["a", "d"],
        validation_region_ids=[],
        test_region_ids=["b", "c"],
    )

    metrics = compute_split_mobility_metrics(
        split=split,
        network=_network(mobility),
    )

    np.testing.assert_allclose(
        metrics.mean_incoming_flow_retained,
        ((5.0 / 7.0) + (7.0 / 10.0)) / 2.0,
    )
    np.testing.assert_allclose(
        metrics.test_internal_flow_fraction,
        (3.0 + 2.0) / mobility.sum(),
    )
    np.testing.assert_allclose(
        metrics.training_flow_retained_fraction,
        0.0,
    )
    assert metrics.minimum_retention_region_id == "c"


def test_strategy_summary_counts_unique_and_repeated_test_regions() -> None:
    splits = [
        SplitDefinition("seeds", "one", ["c"], [], ["a", "b"]),
        SplitDefinition("seeds", "two", ["b"], [], ["a", "c"]),
    ]
    membership = build_membership_table(splits)
    metrics = pd.DataFrame.from_records(
        [
            {
                "strategy": "seeds",
                "split_id": "one",
                "mean_incoming_flow_retained": 0.8,
                "minimum_incoming_flow_retained": 0.5,
                "test_internal_flow_fraction": 0.1,
                "high_flow_event_internal_fraction": 0.05,
                "training_flow_retained_fraction": 0.6,
            },
            {
                "strategy": "seeds",
                "split_id": "two",
                "mean_incoming_flow_retained": 0.9,
                "minimum_incoming_flow_retained": 0.7,
                "test_internal_flow_fraction": 0.2,
                "high_flow_event_internal_fraction": 0.1,
                "training_flow_retained_fraction": 0.8,
            },
        ]
    )

    summary = build_strategy_summary(
        metrics,
        membership,
        valid_region_count=3,
    ).iloc[0]

    assert summary["unique_test_regions"] == 3
    assert summary["repeated_test_assignments"] == 1
    assert summary["test_region_coverage"] == 1.0
    assert summary["worst_target_incoming_flow_retained"] == 0.5


def test_refinement_reduces_complete_directed_internal_flow_deterministically() -> None:
    mobility = np.ones((6, 6), dtype=np.float64)
    np.fill_diagonal(mobility, 0.0)
    for first, second in ((0, 1), (2, 3), (4, 5)):
        mobility[first, second] = 10.0
        mobility[second, first] = 10.0
    network = _network(mobility)
    folds = [["a", "b"], ["c", "d"], ["e", "f"]]
    balance = pd.DataFrame(
        {"constant": np.ones(6)},
        index=pd.Index(network.region_ids, name="region_id"),
    )
    settings = RefinementSettings(
        min_retained_flow=0.0,
        ideal_retained_flow=0.75,
        max_swaps=10,
    )

    first = refine_manifest_folds(
        folds=folds,
        network=network,
        balance_features=balance,
        settings=settings,
    )
    second = refine_manifest_folds(
        folds=folds,
        network=network,
        balance_features=balance,
        settings=settings,
    )

    assert first.folds == second.folds
    assert first.final_state.score < first.initial_state.score
    assert (
        first.final_state.internal_flow_by_fold.sum()
        < first.initial_state.internal_flow_by_fold.sum()
    )
    assert first.balance_audit["within_baseline_envelope"].all()
    assert all(len(fold) == 2 for fold in first.folds)
    assert sorted(region for fold in first.folds for region in fold) == sorted(
        network.region_ids
    )


def test_refinement_rejects_base_manifest_below_hard_retention_floor() -> None:
    mobility = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(mobility, 0.0)
    mobility[0, 1] = 20.0
    mobility[1, 0] = 20.0
    network = _network(mobility)
    balance = pd.DataFrame(
        {"constant": np.ones(4)},
        index=pd.Index(network.region_ids, name="region_id"),
    )

    with pytest.raises(ValueError, match="retained-flow floor"):
        refine_manifest_folds(
            folds=[["a", "b"], ["c", "d"]],
            network=network,
            balance_features=balance,
            settings=RefinementSettings(
                min_retained_flow=0.70,
                ideal_retained_flow=0.75,
                max_swaps=2,
            ),
        )
