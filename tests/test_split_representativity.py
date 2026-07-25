import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.analysis.analyze_split_representativity import (
    MetricDefinition,
    build_test_fold_balance,
    combine_wastewater_target,
    ks_statistic,
    standardized_mean_difference,
)


def test_combine_wastewater_target_matches_masked_variant_mean():
    dataset = xr.Dataset(
        {
            "edar_biomarker_N1": (
                ("date", "region_id"),
                np.array([[1.0, 4.0], [3.0, 6.0]]),
            ),
            "edar_biomarker_N1_mask": (
                ("date", "region_id"),
                np.array([[True, False], [True, False]]),
            ),
            "edar_biomarker_N2": (
                ("date", "region_id"),
                np.array([[3.0, 8.0], [5.0, 10.0]]),
            ),
            "edar_biomarker_N2_mask": (
                ("date", "region_id"),
                np.array([[True, True], [False, False]]),
            ),
        },
        coords={"date": [0, 1], "region_id": ["a", "b"]},
    )

    target = combine_wastewater_target(dataset)

    np.testing.assert_allclose(target.values[0], [2.0, 8.0])
    assert target.values[1, 0] == 3.0
    assert np.isnan(target.values[1, 1])
    np.testing.assert_array_equal(
        target.mask,
        np.array([[True, True], [True, False]]),
    )


def test_distribution_statistics_handle_identical_and_shifted_samples():
    reference = np.array([0.0, 1.0, 2.0, 3.0])

    assert standardized_mean_difference(reference, reference) == pytest.approx(0.0)
    assert ks_statistic(reference, reference) == pytest.approx(0.0)
    assert standardized_mean_difference(reference + 1.0, reference) > 0.0
    assert ks_statistic(np.array([2.0, 3.0]), reference) == pytest.approx(0.5)


def test_fold_balance_reports_additive_share_and_availability_delta():
    membership = pd.DataFrame(
        {
            "split_id": ["Fold 0"] * 4,
            "region_id": ["a", "b", "c", "d"],
            "role": ["test", "test", "train", "validation"],
        }
    )
    region_features = pd.DataFrame(
        {
            "region_id": ["a", "b", "c", "d"],
            "population": [10.0, 20.0, 30.0, 40.0],
            "cases_availability_fraction": [0.5, 0.5, 1.0, 1.0],
        }
    )
    definitions = [
        MetricDefinition(
            column="population",
            label="Population",
            unit="people",
            transform="log1p",
            additive=True,
        ),
        MetricDefinition(
            column="cases_availability_fraction",
            label="Cases availability",
            unit="fraction",
            transform="identity",
            availability=True,
        ),
    ]

    result = build_test_fold_balance(
        membership=membership,
        region_features=region_features,
        metric_definitions=definitions,
    ).set_index("metric")

    assert result.loc["population", "test_total_share"] == pytest.approx(0.3)
    assert result.loc["population", "share_ratio_to_region_count"] == pytest.approx(0.6)
    assert result.loc[
        "cases_availability_fraction", "availability_difference_pp"
    ] == pytest.approx(-25.0)
