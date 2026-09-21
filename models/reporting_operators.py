"""Weekly reporting-interval operators for native-support heads.

Implements the frozen reporting-operator semantics of
``plan_heterogeneous_graph_refactor.md`` §11.5 (slice ROP-1): complete
Monday–Sunday epi weeks fully contained in the open forecast interval
``(origin, origin + H]`` are aggregated *before* any nonlinear transform,
then mapped to the weekly ``log1p(per-100k)`` location used by the weekly
ABS hospital-admissions head. Torch-only; no I/O.
"""

from __future__ import annotations

import torch

_DAYS_PER_WEEK = 7
_PER_100K = 100_000.0


def weekly_interval_sum(
    daily: torch.Tensor,
    origin_dow: torch.Tensor,
    horizon: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum a daily forecast trajectory over complete Mon–Sun epi weeks.

    The forecast covers daily steps ``1..H`` after the origin, i.e. the open
    interval ``(origin, origin + H]``. Column ``t`` of the (post-nowcast-drop)
    trajectory holds the value for day ``origin + 1 + t``. A week ``w`` is the
    Monday–Sunday window starting at the local index of the ``w``-th Monday
    inside the interval; it is reported only when its Sunday also lies inside
    the interval. Leading and trailing partial weeks are never summed.

    Args:
        daily: Daily trajectory ``[B, H]`` (no nowcast column) or
            ``[B, H + 1]`` (a ``t=0`` nowcast column first, dropped when
            ``horizon`` is given). Floating point.
        origin_dow: Weekday of the forecast origin per batch row ``[B]``,
            integer-coded ``0=Monday .. 6=Sunday``.
        horizon: Optional explicit forecast horizon ``H``. When given, the
            input width must be ``H`` or ``H + 1`` (nowcast form). When
            ``None``, the input is read as ``[B, H]`` as-is.

    Returns:
        Tuple of:
            - ``sums [B, W]`` with ``W = ceil(H / 7)``; week sums where the
              week is complete, else 0. Differentiable w.r.t. ``daily``.
            - ``week_mask [B, W]`` bool; True where the week is a complete
              Mon–Sun window inside ``(origin, origin + H]``.

    Raises:
        ValueError: If shapes, dtypes, ``horizon``, or ``origin_dow`` values
            violate the contract above.
    """
    if daily.ndim != 2:
        raise ValueError(f"daily must be rank-2 [B, T], got shape {tuple(daily.shape)}")
    if not daily.dtype.is_floating_point:
        raise ValueError(f"daily must have a floating dtype, got {daily.dtype}")

    if horizon is None:
        forecast = daily
    else:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError(f"horizon must be a positive int or None, got {horizon!r}")
        steps = daily.shape[1]
        if steps == horizon + 1:
            forecast = daily[:, 1:]  # drop the t=0 nowcast column
        elif steps == horizon:
            forecast = daily
        else:
            raise ValueError(
                f"daily has {steps} steps; expected the forecast [B, {horizon}] "
                f"or the nowcast-inclusive [B, {horizon + 1}]"
            )

    batch_size, num_steps = forecast.shape
    if origin_dow.ndim != 1 or origin_dow.shape[0] != batch_size:
        raise ValueError(
            f"origin_dow must be rank-1 [{batch_size}] matching daily's batch, "
            f"got shape {tuple(origin_dow.shape)}"
        )
    if origin_dow.dtype.is_floating_point:
        raise ValueError(f"origin_dow must be an integer tensor, got {origin_dow.dtype}")
    dow = origin_dow.long()
    if bool(((dow < 0) | (dow > _DAYS_PER_WEEK - 1)).any()):
        raise ValueError("origin_dow entries must be weekdays 0=Monday..6=Sunday")

    num_weeks = -(-num_steps // _DAYS_PER_WEEK)  # ceil(H / 7)

    # Local index of the first Monday at-or-after day origin+1, whose weekday
    # is (origin_dow + 1) mod 7.
    first_monday = torch.remainder(-dow - 1, _DAYS_PER_WEEK)  # [B]

    day_offsets = torch.arange(num_steps, device=forecast.device)  # [H]
    rel = day_offsets.unsqueeze(0) - first_monday.unsqueeze(1)  # [B, H]
    week_of_day = torch.div(rel, _DAYS_PER_WEEK, rounding_mode="floor")  # [B, H]

    # Day t belongs to week w = week_of_day[t] when t >= first_monday; that
    # week is complete iff its Sunday (start + 6) is inside the interval.
    week_start_of_day = first_monday.unsqueeze(1) + _DAYS_PER_WEEK * week_of_day
    day_in_complete_week = (week_of_day >= 0) & (
        week_start_of_day <= num_steps - _DAYS_PER_WEEK
    )

    contributions = forecast * day_in_complete_week.to(forecast.dtype)
    sums = torch.zeros(
        batch_size, num_weeks, dtype=forecast.dtype, device=forecast.device
    )
    sums = sums.scatter_add(1, week_of_day.clamp(min=0), contributions)

    week_offsets = torch.arange(num_weeks, device=forecast.device)  # [W]
    week_starts = first_monday.unsqueeze(1) + _DAYS_PER_WEEK * week_offsets.unsqueeze(0)
    week_mask = week_starts <= num_steps - _DAYS_PER_WEEK  # [B, W]

    return sums, week_mask


def weekly_log1p_per100k_location(
    weekly_sums: torch.Tensor,
    population: torch.Tensor,
) -> torch.Tensor:
    """Mechanistic weekly location in log1p(per-100k) space (plan §11.5).

    Computes ``log1p(weekly_sums * population * 1e5)``: the weekly sum of the
    fraction-space daily flow becomes a per-100k count via the source
    population for that epi week, then the single transform is applied. The
    aggregation happens in :func:`weekly_interval_sum` before this transform
    (§6: preserve units before aggregation, transform afterwards); applying
    this transform per day and summing is a different, wrong operator.

    Args:
        weekly_sums: Weekly sums ``[B, W]`` from :func:`weekly_interval_sum`.
        population: Source population, either per source ``[B]`` (broadcast
            over weeks) or per (source, week) ``[B, W]`` — the audited
            denominator for that epi week (for ABS, the native weekly
            population assigned at ``native_graph`` assembly, not the
            case-only CSV ``poblacio``, per plan §10.3(4)).

    Returns:
        Weekly locations ``[B, W]`` in the dtype of ``weekly_sums``.

    Raises:
        ValueError: If shapes are inconsistent or any population entry is not
            positive and finite.
    """
    if weekly_sums.ndim != 2:
        raise ValueError(
            f"weekly_sums must be rank-2 [B, W], got shape {tuple(weekly_sums.shape)}"
        )
    batch_size, num_weeks = weekly_sums.shape

    if population.ndim == 1:
        if population.shape[0] != batch_size:
            raise ValueError(
                f"population must have length {batch_size} to broadcast over "
                f"weeks, got shape {tuple(population.shape)}"
            )
        pop = population.reshape(batch_size, 1)
    elif population.ndim == 2:
        if population.shape != weekly_sums.shape:
            raise ValueError(
                f"population must match weekly_sums shape {batch_size, num_weeks}, "
                f"got shape {tuple(population.shape)}"
            )
        pop = population
    else:
        raise ValueError(
            f"population must be [B] or [B, W], got shape {tuple(population.shape)}"
        )

    pop = pop.to(weekly_sums.dtype)
    if not bool(torch.isfinite(pop).all()) or not bool((pop > 0).all()):
        raise ValueError("population entries must be positive and finite")

    return torch.log1p(weekly_sums * pop * _PER_100K)
