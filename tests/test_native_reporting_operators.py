"""Tests for the weekly ABS reporting operators (plan §11.5, slice ROP-1).

All expected values are hand-computed; the canonical fixture is a 14-step
trajectory with daily values 1..14 observed from each of the seven origin
weekdays (0=Monday .. 6=Sunday).
"""

import math

import pytest
import torch

from models.reporting_operators import (
    weekly_interval_sum,
    weekly_log1p_per100k_location,
)

pytestmark = pytest.mark.epiforecaster


def test_fourteen_step_series_each_origin_weekday() -> None:
    daily = torch.arange(1, 15, dtype=torch.float32).unsqueeze(0).expand(7, 14)
    dow = torch.arange(7)

    sums, mask = weekly_interval_sum(daily, dow)

    assert sums.shape == (7, 2)
    assert mask.shape == (7, 2)
    expected_sums = torch.tensor(
        [
            [70.0, 0.0],
            [63.0, 0.0],
            [56.0, 0.0],
            [49.0, 0.0],
            [42.0, 0.0],
            [35.0, 0.0],
            [28.0, 77.0],
        ]
    )
    assert torch.equal(sums, expected_sums)
    expected_mask = torch.tensor(
        [[True, False]] * 6 + [[True, True]], dtype=torch.bool
    )
    assert torch.equal(mask, expected_mask)
    # Monday origin: week 0 collects days 6..12 (values 7..13); the leading
    # days 0..5 (21) and the trailing day 13 (14) fall outside every week.
    assert sums[0].sum().item() == 70
    assert sums[0].sum().item() + 21 + 14 == daily[0].sum().item() == 105


def test_day_to_week_membership_each_origin_weekday() -> None:
    daily = torch.eye(14)  # row b = one-hot forecast on local day b
    membership = {
        0: [None] * 6 + [0] * 7 + [None],
        1: [None] * 5 + [0] * 7 + [None] * 2,
        2: [None] * 4 + [0] * 7 + [None] * 3,
        3: [None] * 3 + [0] * 7 + [None] * 4,
        4: [None] * 2 + [0] * 7 + [None] * 5,
        5: [None] + [0] * 7 + [None] * 6,
        6: [0] * 7 + [1] * 7,
    }

    for origin_dow in range(7):
        sums, _ = weekly_interval_sum(
            daily, torch.full((14,), origin_dow, dtype=torch.long)
        )
        expected = torch.zeros(14, 2)
        for day in range(14):
            week = membership[origin_dow][day]
            if week is not None:
                expected[day, week] = 1.0
        assert torch.equal(sums, expected)


def test_short_and_boundary_horizons() -> None:
    # H=7, Sunday origin: exactly one complete Mon-Sun week, days sum to 28.
    week = torch.arange(1, 8, dtype=torch.float32).unsqueeze(0)
    sums, mask = weekly_interval_sum(week, torch.tensor([6]))
    assert sums.shape == (1, 1)
    assert torch.equal(sums, torch.tensor([[28.0]]))
    assert torch.equal(mask, torch.tensor([[True]]))

    # H=7, Monday origin: the only candidate week is cut by the horizon.
    sums, mask = weekly_interval_sum(week, torch.tensor([0]))
    assert torch.equal(sums, torch.zeros(1, 1))
    assert torch.equal(mask, torch.tensor([[False]]))

    # H=6: no complete Mon-Sun week fits for any origin weekday.
    six_days = torch.arange(1, 7, dtype=torch.float32).unsqueeze(0)
    sums, mask = weekly_interval_sum(six_days, torch.tensor([3]))
    assert sums.shape == (1, 1)
    assert sums.item() == 0.0
    assert not mask.any()

    # H=15: W = ceil(15/7) = 3; the third column is masked for every origin.
    fifteen = torch.arange(1, 16, dtype=torch.float32).unsqueeze(0).expand(7, 15)
    sums, mask = weekly_interval_sum(fifteen, torch.arange(7))
    assert sums.shape == (7, 3)
    assert not mask[:, 2].any()
    assert mask[:, 0].all()

    # W = ceil(H/7) across horizons.
    for horizon, expected_weeks in ((1, 1), (7, 1), (13, 2), (14, 2), (15, 3), (28, 4)):
        sums, _ = weekly_interval_sum(
            torch.zeros(1, horizon), torch.zeros(1, dtype=torch.long)
        )
        assert sums.shape[1] == expected_weeks


def test_nowcast_column_excluded() -> None:
    base = torch.arange(1, 15, dtype=torch.float32).unsqueeze(0)
    dow = torch.tensor([6])
    base_sums, base_mask = weekly_interval_sum(base, dow)

    junk = torch.cat([torch.full((1, 1), 1e6), base], dim=1)
    sums, mask = weekly_interval_sum(junk, dow, horizon=14)
    assert torch.equal(sums, base_sums)
    assert torch.equal(mask, base_mask)

    nan_lead = torch.cat([torch.full((1, 1), float("nan")), base], dim=1)
    sums_nan, _ = weekly_interval_sum(nan_lead, dow, horizon=14)
    assert torch.isfinite(sums_nan).all()
    assert torch.equal(sums_nan, base_sums)

    # The [B, H] form is unchanged when horizon is given explicitly.
    sums_h, mask_h = weekly_interval_sum(base, dow, horizon=14)
    assert torch.equal(sums_h, base_sums)
    assert torch.equal(mask_h, base_mask)


def test_horizon_and_input_validation() -> None:
    daily = torch.zeros(2, 14)
    dow = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError):
        weekly_interval_sum(torch.zeros(2, 10), dow, horizon=14)
    with pytest.raises(ValueError):
        weekly_interval_sum(daily, dow, horizon=0)
    with pytest.raises(ValueError):
        weekly_interval_sum(daily, dow, horizon=-3)
    with pytest.raises(ValueError):
        weekly_interval_sum(daily, dow, horizon=True)
    with pytest.raises(ValueError):
        weekly_interval_sum(daily.unsqueeze(0), dow)
    with pytest.raises(ValueError):
        weekly_interval_sum(daily.long(), dow)
    with pytest.raises(ValueError):
        weekly_interval_sum(daily, dow.float())
    with pytest.raises(ValueError):
        weekly_interval_sum(daily, torch.full((2,), 7, dtype=torch.long))
    with pytest.raises(ValueError):
        weekly_interval_sum(daily, torch.full((2,), -1, dtype=torch.long))
    with pytest.raises(ValueError):
        weekly_interval_sum(daily, torch.zeros(3, dtype=torch.long))


def test_sum_before_transform_contrast() -> None:
    daily = torch.tensor(
        [[1e-4, 2e-4, 3e-4, 2e-4, 1e-4, 5e-5, 1e-4]], dtype=torch.float32
    )
    sums, mask = weekly_interval_sum(daily, torch.tensor([6]))
    assert torch.equal(mask, torch.tensor([[True]]))
    location = weekly_log1p_per100k_location(sums)

    # R17: fraction-space flow needs only the per-100k factor.
    assert math.isclose(location.item(), math.log1p(105.0), rel_tol=1e-6)
    reference = torch.log1p(torch.tensor(1.05e-3 * 1e5))
    assert location.item() == reference.item()

    # Transform-per-day-then-sum is a different, wrong operator.
    per_day = torch.log1p(daily * 1e5).sum()
    assert abs(location.item() - per_day.item()) > 10
    assert not torch.allclose(location, per_day)


def test_fraction_location_does_not_depend_on_population() -> None:
    """R17: an identical fraction-space flow has one reporting location."""
    weekly_sums = torch.tensor([[0.001, 0.002], [0.003, 0.0]])
    expected = torch.log1p(weekly_sums * 1e5)
    location = weekly_log1p_per100k_location(weekly_sums)
    assert torch.equal(location, expected)
    assert location[0, 0].item() == location.new_tensor(math.log1p(100.0)).item()


def test_location_input_validation() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        weekly_log1p_per100k_location(torch.ones(2))
    with pytest.raises(ValueError, match="floating"):
        weekly_log1p_per100k_location(torch.ones(1, 2, dtype=torch.long))


def test_dtype_and_determinism() -> None:
    daily = torch.arange(1, 15, dtype=torch.float32).unsqueeze(0)
    dow = torch.tensor([6])

    sums, mask = weekly_interval_sum(daily, dow)
    assert sums.dtype == torch.float32
    assert mask.dtype == torch.bool
    location = weekly_log1p_per100k_location(sums)
    assert location.dtype == torch.float32

    sums_again, mask_again = weekly_interval_sum(daily, dow)
    location_again = weekly_log1p_per100k_location(sums_again)
    assert torch.equal(sums, sums_again)
    assert torch.equal(mask, mask_again)
    assert torch.equal(location, location_again)


def test_batch_rows_independent_and_gradient_flow() -> None:
    base = torch.arange(1, 15, dtype=torch.float32).unsqueeze(0)
    single_sums, single_mask = weekly_interval_sum(base, torch.tensor([6]))
    batch_sums, batch_mask = weekly_interval_sum(
        base.expand(4, 14), torch.full((4,), 6, dtype=torch.long)
    )
    assert torch.equal(batch_sums, single_sums.expand(4, 2))
    assert torch.equal(batch_mask, single_mask.expand(4, 2))

    # One-hot rows keep s_w in {0, 1}, making the diagonal gradient explicit:
    # d loc / d f_t = 1e5 / (1 + s_w * 1e5) when day t sits in a complete
    # week (its value is the row's only contribution to s_w), exactly 0 for
    # leading/trailing partial days (the completeness mask zeroes them).
    gain = 1e5 / (1.0 + 1e5)
    for origin_dow in (0, 6):
        daily = torch.eye(14, requires_grad=True)
        sums, _ = weekly_interval_sum(daily, torch.full((14,), origin_dow))
        location = weekly_log1p_per100k_location(sums)
        location.sum().backward()

        for day in range(14):
            grad = daily.grad[day, day]
            in_complete_week = (origin_dow == 0 and 6 <= day <= 12) or origin_dow == 6
            if in_complete_week:
                assert torch.allclose(grad, torch.tensor(gain))
            else:
                assert grad.item() == 0.0
