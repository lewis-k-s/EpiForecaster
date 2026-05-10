import numpy as np
import pandas as pd

from dataviz.mobility_ego_lockdown_analysis import (
    build_ego_mask,
    classify_lockdown,
    compute_valid_history_mask,
    compute_window_starts,
    vectorized_window_regression,
)


def test_build_ego_mask_applies_time_mask_and_forces_self() -> None:
    mobility = np.array(
        [
            [[0.0, 3.0], [1.0, 0.0]],
            [[5.0, 0.0], [0.0, 7.0]],
        ],
        dtype=np.float32,
    )

    mask = build_ego_mask(
        mobility,
        mobility_threshold=2.0,
        mobility_time_mask=np.array([True, False]),
    )

    expected = np.array(
        [
            [[True, True], [False, True]],
            [[True, False], [False, True]],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected)


def test_compute_window_starts_uses_history_window_only() -> None:
    starts = compute_window_starts(num_timesteps=10, window_length=4, window_stride=2)
    np.testing.assert_array_equal(starts, np.array([0, 2, 4, 6]))


def test_compute_valid_history_mask_uses_missing_permit() -> None:
    cases_mask = np.array(
        [
            [True, True],
            [True, False],
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )
    starts = np.array([0, 1])

    valid = compute_valid_history_mask(
        cases_mask,
        starts,
        window_length=3,
        missing_permit=1,
    )

    expected = np.array([[True, True], [True, True]], dtype=bool)
    np.testing.assert_array_equal(valid, expected)


def test_classify_lockdown_by_center_date() -> None:
    status, period = classify_lockdown(pd.Timestamp("2020-03-20"))
    assert status == "during_lockdown"
    assert period == "Spain-wide lockdown"

    status, period = classify_lockdown(pd.Timestamp("2020-07-01"))
    assert status == "outside_lockdown"
    assert period == "outside_lockdown"


def test_vectorized_window_regression_matches_loop() -> None:
    cases = np.array(
        [
            [1.0, 2.0, 5.0],
            [2.0, 3.0, 7.0],
            [3.0, 4.0, 9.0],
            [4.0, 5.0, 11.0],
        ],
        dtype=np.float32,
    )
    cases_mask = np.ones_like(cases, dtype=bool)
    population = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    ego_mask = np.array(
        [
            [[True, False, True], [True, True, False], [False, True, True]],
            [[True, False, True], [True, True, False], [False, True, True]],
            [[True, False, True], [True, True, False], [False, True, True]],
            [[True, False, True], [True, True, False], [False, True, True]],
        ],
        dtype=bool,
    )

    stats = vectorized_window_regression(cases, cases_mask, population, ego_mask)

    valid_pop = population > 0
    weights = population[valid_pop] / population[valid_pop].sum()
    global_trend = (cases[:, valid_pop] * weights[None, :]).sum(axis=1)

    expected_slopes = []
    expected_r2 = []
    for target_idx in range(cases.shape[1]):
        target_trend = []
        for t in range(cases.shape[0]):
            neighbors = ego_mask[t, :, target_idx]
            target_trend.append(cases[t, neighbors].mean())
        x = np.asarray(target_trend)
        y = global_trend
        dx = x - x.mean()
        dy = y - y.mean()
        slope = np.sum(dx * dy) / np.sum(dx * dx)
        pred = y.mean() + slope * dx
        r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum(dy**2)
        expected_slopes.append(slope)
        expected_r2.append(r2)

    np.testing.assert_allclose(stats["slope"], expected_slopes, rtol=1e-6)
    np.testing.assert_allclose(stats["r2"], expected_r2, rtol=1e-6)
    np.testing.assert_array_equal(stats["n_timepoints"], np.array([4, 4, 4]))
