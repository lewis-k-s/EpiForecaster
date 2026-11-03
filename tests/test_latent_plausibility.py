"""Definitions pinned for the ground-truth-free latent plausibility indicators."""

from __future__ import annotations

import math

import pytest
import torch

from evaluation.pathway_metrics import (
    LATENT_PLAUSIBILITY_I_COLLAPSE_THRESHOLD,
    TorchLatentPlausibilityAccumulator,
)

CPU = torch.device("cpu")
PREFIX = "latent_plausibility_"
QUANTILES = ("q05", "q25", "median", "q75", "q95")
STATE_FAMILIES = (
    "i_origin",
    "i_peak",
    "s_depletion_origin",
    "forecast_attack",
    "rt_origin",
    "rt_horizon_mean",
)
HEADS = ("cases", "hosp", "deaths")
EXPECTED_KEYS = (
    {f"{PREFIX}window_count", f"{PREFIX}i_collapse_fraction"}
    | {f"{PREFIX}{family}_{q}" for family in STATE_FAMILIES for q in QUANTILES}
    | {
        f"{PREFIX}ascertainment_{head}_{suffix}"
        for head in HEADS
        for suffix in (
            *QUANTILES,
            "pooled",
            "window_count",
            "zero_latent_window_fraction",
        )
    }
)


def _outputs(
    *,
    S: torch.Tensor,
    I: torch.Tensor,  # noqa: E741
    beta: torch.Tensor | float = 0.3,
    gamma: torch.Tensor | float = 0.1,
    mortality: torch.Tensor | float = 0.01,
    hospitalization: torch.Tensor | float = 0.05,
) -> dict[str, torch.Tensor]:
    """Model outputs for ``B`` windows; ``S``/``I`` are ``[B, H+1]``."""
    rows, steps = S.shape[0], S.shape[1] - 1

    def rate(value: torch.Tensor | float) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.full((rows, steps), float(value))

    return {
        "S_trajectory": S,
        "I_trajectory": I,
        "beta_t": rate(beta),
        "gamma_t": rate(gamma),
        "mortality_t": rate(mortality),
        "hospitalization_rate_t": rate(hospitalization),
        # Must never enter R_t: only the direct I -> D mortality does.
        "hospital_mortality_t": torch.full((rows, steps), 100.0),
    }


def _state_windows(i_origin: list[float]) -> dict[str, torch.Tensor]:
    count = len(i_origin)
    I = torch.tensor(i_origin).unsqueeze(1).repeat(1, 3)  # noqa: E741
    return _outputs(S=torch.full((count, 3), 0.9), I=I)


def _accumulator() -> TorchLatentPlausibilityAccumulator:
    return TorchLatentPlausibilityAccumulator(device=CPU)


def test_empty_accumulator_is_nan_with_zero_counts() -> None:
    metrics = _accumulator().finalize()

    assert set(metrics) == EXPECTED_KEYS
    assert metrics[f"{PREFIX}window_count"] == 0
    for head in HEADS:
        assert metrics[f"{PREFIX}ascertainment_{head}_window_count"] == 0
    count_keys = {key for key in EXPECTED_KEYS if key.endswith("window_count")}
    assert all(
        math.isnan(value) for key, value in metrics.items() if key not in count_keys
    )


def test_quantiles_are_exact_linear_interpolation_over_windows() -> None:
    i_origin = [0.01 * k for k in range(11)]
    accumulator = _accumulator()
    accumulator.update(model_outputs=_state_windows(i_origin), ascertainment={})
    metrics = accumulator.finalize()

    assert metrics[f"{PREFIX}window_count"] == 11
    expected = {"q05": 0.005, "q25": 0.025, "median": 0.05, "q75": 0.075, "q95": 0.095}
    for label, value in expected.items():
        assert metrics[f"{PREFIX}i_origin_{label}"] == pytest.approx(value, abs=1e-7)


def test_streaming_updates_match_a_single_combined_update() -> None:
    values = [0.02, 0.4, 0.1, 0.07, 0.3, 0.2]
    combined = _accumulator()
    combined.update(model_outputs=_state_windows(values), ascertainment={})
    streamed = _accumulator()
    streamed.update(model_outputs=_state_windows(values[:4]), ascertainment={})
    streamed.update(model_outputs=_state_windows(values[4:]), ascertainment={})

    assert streamed.finalize() == combined.finalize()


def test_reused_output_buffers_do_not_change_accumulated_windows() -> None:
    # A CUDA-graph forward overwrites its output buffers on the next batch.
    reused = _state_windows([0.02, 0.04])
    reference = _accumulator()
    reference.update(model_outputs=_state_windows([0.02, 0.04]), ascertainment={})
    accumulator = _accumulator()
    accumulator.update(model_outputs=reused, ascertainment={})
    for tensor in reused.values():
        tensor.fill_(0.5)

    assert accumulator.finalize() == reference.finalize()


def test_state_indicators_use_origin_peak_depletion_and_attack_definitions() -> None:
    accumulator = _accumulator()
    accumulator.update(
        model_outputs=_outputs(
            S=torch.tensor([[0.99, 0.90, 0.80]]),
            I=torch.tensor([[0.01, 0.07, 0.03]]),
        ),
        ascertainment={},
    )
    metrics = accumulator.finalize()

    for label in QUANTILES:
        assert metrics[f"{PREFIX}i_origin_{label}"] == pytest.approx(0.01)
        assert metrics[f"{PREFIX}i_peak_{label}"] == pytest.approx(0.07)
        assert metrics[f"{PREFIX}s_depletion_origin_{label}"] == pytest.approx(0.01)
        assert metrics[f"{PREFIX}forecast_attack_{label}"] == pytest.approx(0.19)


def test_i_collapse_threshold_is_strict_and_uses_peak_not_origin() -> None:
    threshold = LATENT_PLAUSIBILITY_I_COLLAPSE_THRESHOLD
    assert threshold == 1.0e-6
    peaks = torch.tensor(
        [
            1.0e-8,  # collapsed
            0.999e-6,  # collapsed, just below the threshold
            1.0e-6,  # exactly the threshold: not collapsed
            1.0e-3,  # healthy
        ]
    )
    # The origin of the healthy-peak window is tiny: peak decides, not origin.
    I = torch.zeros(4, 3)  # noqa: E741
    I[:, 1] = peaks
    I[3, 0] = 1.0e-9
    accumulator = _accumulator()
    accumulator.update(
        model_outputs=_outputs(S=torch.full((4, 3), 0.9), I=I), ascertainment={}
    )
    metrics = accumulator.finalize()

    assert metrics[f"{PREFIX}i_collapse_fraction"] == pytest.approx(2.0 / 4.0)


def test_non_finite_state_windows_are_excluded_from_quantiles_and_collapse() -> None:
    I = torch.tensor(  # noqa: E741
        [
            [0.1, 0.2, 0.3],
            [float("nan"), 0.2, 0.3],
            [1.0e-9, 1.0e-9, 1.0e-9],
            [0.1, float("inf"), 0.1],
        ]
    )
    accumulator = _accumulator()
    accumulator.update(
        model_outputs=_outputs(S=torch.full((4, 3), 0.9), I=I), ascertainment={}
    )
    metrics = accumulator.finalize()

    assert metrics[f"{PREFIX}window_count"] == 4
    # Finite origins: 0.1, 1e-9, 0.1 (the inf window has finite origin).
    assert metrics[f"{PREFIX}i_origin_median"] == pytest.approx(0.1)
    # Finite peaks: 0.3 and 1e-9 only; one of the two is collapsed.
    assert metrics[f"{PREFIX}i_peak_q95"] == pytest.approx(0.3 - 0.05 * 0.3, rel=1e-4)
    assert metrics[f"{PREFIX}i_collapse_fraction"] == pytest.approx(0.5)


def test_rt_uses_direct_mortality_and_excludes_non_positive_denominators() -> None:
    steps = 2
    # Window 0: R_0 = 0.4*0.5/(0.1+0.09+0.01) = 1.0; R_1 = 0.6*0.4/(0.1+0.08+0.02) = 1.2.
    # Window 1: zero denominator at h=0 (excluded), R_1 = 0.5*0.5/0.2 = 1.25.
    # Window 2: zero denominator at every step (no valid value).
    beta = torch.tensor([[0.4, 0.6], [0.3, 0.5], [0.3, 0.5]])
    gamma = torch.tensor([[0.1, 0.1], [0.0, 0.1], [0.0, 0.0]])
    mortality = torch.tensor([[0.01, 0.02], [0.0, 0.05], [0.0, 0.0]])
    hospitalization = torch.tensor([[0.09, 0.08], [0.0, 0.05], [0.0, 0.0]])
    S = torch.tensor([[0.5, 0.4, 0.3], [0.5, 0.5, 0.4], [0.5, 0.5, 0.4]])
    assert S.shape == (3, steps + 1)
    accumulator = _accumulator()
    accumulator.update(
        model_outputs=_outputs(
            S=S,
            I=torch.full((3, steps + 1), 0.01),
            beta=beta,
            gamma=gamma,
            mortality=mortality,
            hospitalization=hospitalization,
        ),
        ascertainment={},
    )
    metrics = accumulator.finalize()

    # Only window 0 has a finite R at the origin.
    for label in QUANTILES:
        assert metrics[f"{PREFIX}rt_origin_{label}"] == pytest.approx(1.0)
    # Horizon means: window 0 -> 1.1, window 1 -> 1.25 (valid step only), window 2 NaN.
    assert metrics[f"{PREFIX}rt_horizon_mean_median"] == pytest.approx(1.175)
    assert metrics[f"{PREFIX}rt_horizon_mean_q05"] == pytest.approx(1.1 + 0.05 * 0.15)
    assert metrics[f"{PREFIX}rt_horizon_mean_q95"] == pytest.approx(1.1 + 0.95 * 0.15)


def _ascertainment_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cases-style ``(latent_flow, target, weights)`` over four windows.

    A: weights [1, 1, 0]; the weight-0 day carries a huge target and flow that
       must be ignored. observed 10 + 30 = 40, latent 10 + 30 = 40 -> 1.0.
    B: weights [1, 0, 1]; observed 20 + 60 = 80, latent 10 + 10 = 20 -> 4.0.
    C: all weights zero and NaN targets: not an observed window.
    D: weights [1, 1, 1], zero latent flow: observed 15, latent 0.
    """
    nan = float("nan")
    latent_flow = torch.tensor(
        [
            [1.0e-4, 3.0e-4, 5.0e-1],
            [1.0e-4, 7.0e-1, 1.0e-4],
            [2.0e-4, 2.0e-4, 2.0e-4],
            [0.0, 0.0, 0.0],
        ]
    )
    target = torch.log1p(
        torch.tensor(
            [
                [10.0, 30.0, 999.0],
                [20.0, 0.0, 60.0],
                [nan, nan, nan],
                [5.0, 5.0, 5.0],
            ]
        )
    )
    weights = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    return latent_flow, target, weights


def test_ascertainment_ratio_excludes_zero_weight_days_and_windows() -> None:
    latent_flow, target, weights = _ascertainment_batch()
    accumulator = _accumulator()
    accumulator.update(
        model_outputs=_state_windows([0.01] * 4),
        ascertainment={"cases": (latent_flow, target, weights)},
    )
    metrics = accumulator.finalize()
    key = f"{PREFIX}ascertainment_cases_"

    # Window ratios are 1.0 and 4.0; D (zero latent) and C (no weight) are out.
    assert metrics[f"{key}window_count"] == 2
    assert metrics[f"{key}median"] == pytest.approx(2.5, rel=1e-5)
    assert metrics[f"{key}q05"] == pytest.approx(1.0 + 0.05 * 3.0, rel=1e-5)
    assert metrics[f"{key}q95"] == pytest.approx(1.0 + 0.95 * 3.0, rel=1e-5)


def test_ascertainment_pooled_differs_from_per_window_and_counts_zero_latent() -> None:
    latent_flow, target, weights = _ascertainment_batch()
    accumulator = _accumulator()
    accumulator.update(
        model_outputs=_state_windows([0.01] * 4),
        ascertainment={"cases": (latent_flow, target, weights)},
    )
    metrics = accumulator.finalize()
    key = f"{PREFIX}ascertainment_cases_"

    # Pooled = (40 + 80 + 15) / (40 + 20 + 0); window median is 2.5, mean 2.5.
    assert metrics[f"{key}pooled"] == pytest.approx(135.0 / 60.0, rel=1e-5)
    assert metrics[f"{key}pooled"] != pytest.approx(metrics[f"{key}median"], rel=1e-3)
    # Windows A, B, D carry weight with finite sums; only D has no latent flow.
    assert metrics[f"{key}zero_latent_window_fraction"] == pytest.approx(1.0 / 3.0)


def test_ascertainment_heads_are_independent_and_unsupervised_heads_stay_empty() -> (
    None
):
    latent_flow, target, weights = _ascertainment_batch()
    accumulator = _accumulator()
    accumulator.update(
        model_outputs=_state_windows([0.01] * 4),
        ascertainment={
            "hosp": (latent_flow * 2.0, target, weights),
            "deaths": None,
        },
    )
    metrics = accumulator.finalize()

    # Doubling the latent flow halves every ratio.
    assert metrics[f"{PREFIX}ascertainment_hosp_median"] == pytest.approx(
        1.25, rel=1e-5
    )
    for head in ("cases", "deaths"):
        assert metrics[f"{PREFIX}ascertainment_{head}_window_count"] == 0
        assert math.isnan(metrics[f"{PREFIX}ascertainment_{head}_pooled"])
        assert math.isnan(metrics[f"{PREFIX}ascertainment_{head}_median"])
        assert math.isnan(
            metrics[f"{PREFIX}ascertainment_{head}_zero_latent_window_fraction"]
        )


def test_ascertainment_non_finite_latent_on_a_supervised_day_is_excluded() -> None:
    latent_flow = torch.tensor([[1.0e-4, float("nan")], [1.0e-4, 1.0e-4]])
    target = torch.log1p(torch.tensor([[10.0, 10.0], [10.0, 10.0]]))
    weights = torch.ones(2, 2)
    accumulator = _accumulator()
    accumulator.update(
        model_outputs=_state_windows([0.01] * 2),
        ascertainment={"deaths": (latent_flow, target, weights)},
    )
    metrics = accumulator.finalize()
    key = f"{PREFIX}ascertainment_deaths_"

    assert metrics[f"{key}window_count"] == 1
    assert metrics[f"{key}median"] == pytest.approx(1.0, rel=1e-5)
    assert metrics[f"{key}pooled"] == pytest.approx(1.0, rel=1e-5)
    assert metrics[f"{key}zero_latent_window_fraction"] == pytest.approx(0.0)


def test_ascertainment_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        _accumulator().update(
            model_outputs=_state_windows([0.01]),
            ascertainment={
                "cases": (torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1, 3))
            },
        )
