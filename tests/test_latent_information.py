"""Latent-information indicators: within-window coupling of latent, observation, forecast."""

import math

import pytest
import torch

from evaluation.latent_information import (
    MAX_LAG_DAYS,
    TorchLatentInformationAccumulator,
)

pytestmark = pytest.mark.epiforecaster

WINDOWS = 240
HORIZON = 28
NODES = 24


def _latent_fraction(log_per_100k: torch.Tensor) -> torch.Tensor:
    """Population fraction whose log1p(per-100k) equals ``log_per_100k``."""
    return torch.expm1(log_per_100k) / 1e5


def _random_walks(generator: torch.Generator, length: int = HORIZON) -> torch.Tensor:
    steps = torch.randn(WINDOWS, length, generator=generator, dtype=torch.float64)
    # Base 10 keeps log1p(per-100k) positive, so the fraction is non-negative.
    return 10.0 + 0.3 * torch.cumsum(steps, dim=1)


def _accumulate(
    *,
    head: str,
    latent_log: torch.Tensor,
    observed: torch.Tensor,
    prediction: torch.Tensor,
    residual: torch.Tensor,
    weights: torch.Tensor,
    batches: int = 3,
) -> dict[str, float | int]:
    accumulator = TorchLatentInformationAccumulator()
    nodes = torch.arange(WINDOWS) % NODES
    for rows in torch.arange(WINDOWS).chunk(batches):
        accumulator.update(
            head=head,
            latent=_latent_fraction(latent_log[rows]),
            observed=observed[rows],
            prediction=prediction[rows],
            residual=residual[rows],
            weights=weights[rows],
            nodes=nodes[rows],
        )
    return accumulator.finalize()


def test_coupled_latent_tracks_observation_above_permutation_null() -> None:
    generator = torch.Generator().manual_seed(1)
    latent = _random_walks(generator)
    noise = torch.randn(WINDOWS, HORIZON, generator=generator, dtype=torch.float64)
    observed = 2.0 + latent + 0.2 * noise
    # The forecast follows the latent mechanistically; the residual is a
    # per-window constant, which within-window centring removes entirely.
    offset = torch.randn(WINDOWS, 1, generator=generator, dtype=torch.float64)
    prediction = latent + offset
    residual = offset.expand(-1, HORIZON).clone()

    metrics = _accumulate(
        head="cases",
        latent_log=latent,
        observed=observed,
        prediction=prediction,
        residual=residual,
        weights=torch.ones(WINDOWS, HORIZON),
    )

    assert metrics["latent_info_cases_window_count"] == WINDOWS
    assert metrics["latent_info_cases_node_count"] == NODES
    assert metrics["latent_info_cases_r2_lat_obs"] > 0.8
    assert (
        metrics["latent_info_cases_r2_lat_obs"]
        > 10 * metrics["latent_info_cases_r2_lat_obs_null95"]
    )
    assert (
        metrics["latent_info_cases_mi_lat_obs"]
        > metrics["latent_info_cases_mi_lat_obs_null95"]
    )
    assert metrics["latent_info_cases_te_lat_obs"] > 0.0
    assert metrics["latent_info_cases_r2_lat_pred"] == pytest.approx(1.0)
    assert metrics["latent_info_cases_resid_share"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["latent_info_cases_best_lag_lat_obs"] == 0.0
    lo = metrics["latent_info_cases_r2_lat_obs_q05"]
    hi = metrics["latent_info_cases_r2_lat_obs_q95"]
    assert lo <= metrics["latent_info_cases_r2_lat_obs"] <= hi


def test_independent_latent_stays_within_permutation_null() -> None:
    generator = torch.Generator().manual_seed(2)
    latent = _random_walks(generator)
    observed = _random_walks(generator)

    metrics = _accumulate(
        head="deaths",
        latent_log=latent,
        observed=observed,
        prediction=observed.clone(),
        residual=observed.clone(),
        weights=torch.ones(WINDOWS, HORIZON),
    )

    assert (
        metrics["latent_info_deaths_r2_lat_obs"]
        < 2 * metrics["latent_info_deaths_r2_lat_obs_null95"]
    )
    # A forecast carried entirely by the residual has a residual share of 1.
    assert metrics["latent_info_deaths_resid_share"] == pytest.approx(1.0)
    assert metrics["latent_info_deaths_mi_pred_obs"] > 3.0


def test_lag_scan_recovers_latent_lead() -> None:
    lead = 4
    generator = torch.Generator().manual_seed(3)
    path = _random_walks(generator, length=HORIZON + lead)
    latent = path[:, lead:]
    # The observation on day h equals the latent on day h - lead.
    observed = path[:, :HORIZON]

    metrics = _accumulate(
        head="cases",
        latent_log=latent,
        observed=observed,
        prediction=observed.clone(),
        residual=torch.zeros(WINDOWS, HORIZON, dtype=torch.float64),
        weights=torch.ones(WINDOWS, HORIZON),
    )

    assert metrics["latent_info_cases_best_lag_lat_obs"] == float(lead)
    assert metrics[f"latent_info_cases_lag_r2_lat_obs_p{lead}"] == pytest.approx(
        metrics["latent_info_cases_best_lag_r2_lat_obs"]
    )
    assert (
        metrics[f"latent_info_cases_lag_r2_lat_obs_p{lead}"]
        > metrics["latent_info_cases_lag_r2_lat_obs_p0"]
    )


def test_weekly_head_uses_weekly_pairs_and_lags() -> None:
    generator = torch.Generator().manual_seed(4)
    latent = _random_walks(generator)
    observed = latent + 0.1 * torch.randn(
        WINDOWS, HORIZON, generator=generator, dtype=torch.float64
    )
    weights = torch.zeros(WINDOWS, HORIZON)
    weights[:, 3::7] = 1.0

    metrics = _accumulate(
        head="hosp",
        latent_log=latent,
        observed=observed,
        prediction=observed.clone(),
        residual=torch.zeros(WINDOWS, HORIZON, dtype=torch.float64),
        weights=weights,
    )

    lag_keys = sorted(
        key.removeprefix("latent_info_hosp_lag_r2_lat_obs_")
        for key in metrics
        if key.startswith("latent_info_hosp_lag_r2_lat_obs_")
    )
    assert lag_keys == sorted(["m21", "m14", "m7", "p0", "p7", "p14", "p21"])
    assert MAX_LAG_DAYS["hosp"] == 21
    assert metrics["latent_info_hosp_observation_count"] == WINDOWS * 4
    # Weekly changes of a latent that the observation tracks agree in sign.
    assert metrics["latent_info_hosp_sign_lat_obs"] > 0.9
    assert math.isfinite(metrics["latent_info_hosp_te_lat_obs"])


def test_windows_without_two_supervised_days_are_excluded() -> None:
    generator = torch.Generator().manual_seed(5)
    latent = _random_walks(generator)
    weights = torch.ones(WINDOWS, HORIZON)
    weights[:40] = 0.0
    weights[40:60, 1:] = 0.0  # one supervised day only
    observed = latent.clone()
    observed[60:80, 5] = float("nan")  # one non-finite day leaves 27

    metrics = _accumulate(
        head="ww",
        latent_log=latent,
        observed=observed,
        prediction=latent.clone(),
        residual=torch.zeros(WINDOWS, HORIZON, dtype=torch.float64),
        weights=weights,
    )

    assert metrics["latent_info_ww_window_count"] == WINDOWS - 60
    assert metrics["latent_info_ww_observation_count"] == (WINDOWS - 60) * HORIZON - 20
    assert metrics["latent_info_ww_r2_lat_obs"] == pytest.approx(1.0)


def test_weekly_blocks_separate_day_of_week_residual_from_dynamics() -> None:
    generator = torch.Generator().manual_seed(6)
    latent = _random_walks(generator)
    cycle = torch.tensor([0.0, 0.0, -1.5, -1.5, 0.0, 0.0, 0.0], dtype=torch.float64)
    weekday = cycle.repeat(HORIZON // 7).expand(WINDOWS, -1)
    observed = latent + weekday
    # The residual reproduces the reporting cycle; the mechanistic path the latent.
    prediction = latent + weekday

    metrics = _accumulate(
        head="cases",
        latent_log=latent,
        observed=observed,
        prediction=prediction,
        residual=weekday.clone(),
        weights=torch.ones(WINDOWS, HORIZON),
    )

    assert metrics["latent_info_cases_resid_share"] > 0.5
    assert metrics["latent_info_cases_resid_share_weekly"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert metrics["latent_info_cases_r2_lat_obs_weekly"] == pytest.approx(1.0)
    assert metrics["latent_info_cases_r2_lat_obs"] < 0.5
    assert "latent_info_hosp_resid_share_weekly" not in metrics


def test_heads_without_updates_report_zero_windows_only() -> None:
    metrics = TorchLatentInformationAccumulator().finalize()

    assert metrics == {
        "latent_info_hosp_window_count": 0,
        "latent_info_ww_window_count": 0,
        "latent_info_cases_window_count": 0,
        "latent_info_deaths_window_count": 0,
    }


def test_update_rejects_mismatched_shapes_and_unknown_heads() -> None:
    accumulator = TorchLatentInformationAccumulator()
    series = torch.zeros(2, HORIZON)
    with pytest.raises(ValueError, match="Unknown latent-information head"):
        accumulator.update(
            head="icu",
            latent=series,
            observed=series,
            prediction=series,
            residual=series,
            weights=series,
            nodes=torch.zeros(2, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="one \\[B, H\\] shape"):
        accumulator.update(
            head="cases",
            latent=series,
            observed=torch.zeros(2, HORIZON - 1),
            prediction=series,
            residual=series,
            weights=series,
            nodes=torch.zeros(2, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="nodes must have shape"):
        accumulator.update(
            head="cases",
            latent=series,
            observed=series,
            prediction=series,
            residual=series,
            weights=series,
            nodes=torch.zeros(3, dtype=torch.long),
        )
