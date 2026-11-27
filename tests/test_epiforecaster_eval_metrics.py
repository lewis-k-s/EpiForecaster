from __future__ import annotations

import math

import pytest
import torch
from types import SimpleNamespace

from evaluation.losses import JointInferenceLoss
from evaluation.metrics import TorchDistributionMetricAccumulator
from evaluation.epiforecaster_eval import evaluate_loader


class _DummyLoader:
    def __init__(self, batches: list):
        self._batches = batches
        self.dataset = None

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward_batch(self, batch_data, region_embeddings=None, **kwargs):  # noqa: ANN001
        return batch_data.model_outputs, batch_data.targets_dict

    def forward_inputs(self, batch_data, region_embeddings=None, **kwargs):  # noqa: ANN001
        return {"batch_data": batch_data}

    def forward(self, batch_data):  # noqa: ANN001
        return batch_data.model_outputs

    @staticmethod
    def batch_targets(batch_data):  # noqa: ANN001
        return batch_data.targets_dict


def test_distribution_metric_accumulator_preserves_negative_nll() -> None:
    accumulator = TorchDistributionMetricAccumulator(device=torch.device("cpu"))

    batch_metrics = accumulator.update(
        loc=torch.zeros((1, 1)),
        scale=torch.full((1, 1), 0.1),
        targets=torch.zeros((1, 1)),
        observed_mask=torch.ones((1, 1)),
    )
    summary = accumulator.finalize()

    expected = (
        torch.log(torch.tensor(0.1)).item()
        + 0.5 * torch.log(torch.tensor(2.0 * torch.pi)).item()
    )
    assert batch_metrics.nll.item() == pytest.approx(expected)
    assert summary.nll == pytest.approx(expected)


def test_distribution_metric_accumulator_clips_intervals_at_zero() -> None:
    accumulator = TorchDistributionMetricAccumulator(device=torch.device("cpu"))

    batch_metrics = accumulator.update(
        loc=torch.tensor([[0.0, 5.0]]),
        scale=torch.ones((1, 2)),
        targets=torch.tensor([[0.0, 5.0]]),
        observed_mask=torch.ones((1, 2)),
    )
    summary = accumulator.finalize()

    half_width = 1.959964
    low, high = batch_metrics.intervals["95"]
    assert low[0].tolist() == pytest.approx([0.0, 5.0 - half_width])
    assert high[0].tolist() == pytest.approx([half_width, 5.0 + half_width])
    assert summary.interval_width_by_level["95"] == pytest.approx(1.5 * half_width)
    assert summary.coverage_by_level["95"] == 1.0
    # A zero forecast of an observed zero pays only for the half above zero.
    assert batch_metrics.wis[0, 0].item() == pytest.approx(
        0.5 * batch_metrics.wis[0, 1].item()
    )


def _make_batch(*, with_residuals: bool = False) -> SimpleNamespace:
    batch = SimpleNamespace(
        target_node=torch.tensor([0, 1], dtype=torch.long),
        model_outputs={
            "pred_hosp": torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32),
            "pred_ww": torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
            "pred_cases": torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
            "pred_deaths": torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            "physics_residual": torch.zeros((2, 2), dtype=torch.float32),
            "S_trajectory": torch.tensor([[0.9, 0.85, 0.8], [0.95, 0.9, 0.85]]),
            "I_trajectory": torch.tensor([[0.01, 0.02, 0.03], [0.01, 0.01, 0.01]]),
            "beta_t": torch.full((2, 2), 0.3),
            "gamma_t": torch.full((2, 2), 0.1),
            "mortality_t": torch.full((2, 2), 0.01),
            "hospitalization_rate_t": torch.full((2, 2), 0.05),
            "infection_flow": torch.tensor([[1.0e-4, 2.0e-4], [3.0e-4, 4.0e-4]]),
            "hospitalization_flow": torch.tensor([[1.0e-5, 1.0e-5], [2.0e-5, 2.0e-5]]),
            "death_flow": torch.full((2, 2), 1.0e-6),
        },
        targets_dict={
            "hosp": torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32),
            "ww": torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
            "cases": torch.tensor([[2.0, 4.0], [8.0, 6.0]], dtype=torch.float32),
            "deaths": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
            "hosp_mask": torch.ones((2, 2), dtype=torch.float32),
            "ww_mask": torch.ones((2, 2), dtype=torch.float32),
            "cases_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
            "deaths_mask": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        },
        mob_batch=None,
    )
    if with_residuals:
        for head in ("hosp", "ww", "cases", "deaths"):
            batch.model_outputs[f"residual_{head}"] = torch.zeros((2, 2))
    batch.to = lambda device, **_: batch
    return batch


def _evaluate(batch: SimpleNamespace, **kwargs):  # noqa: ANN202
    return evaluate_loader(
        model=_DummyModel(),
        loader=_DummyLoader([batch]),
        criterion=JointInferenceLoss(obs_weight_sum=4.0, latent_total_weight=0.0),
        horizon=2,
        device=torch.device("cpu"),
        **kwargs,
    )


def test_latent_information_indicators_follow_their_flag():
    _loss, disabled, _ = _evaluate(_make_batch(with_residuals=True))
    _loss, enabled, _ = _evaluate(
        _make_batch(with_residuals=True), latent_information=True
    )

    assert not any(key.startswith("latent_info_") for key in disabled)
    # Every head has two windows; cases and deaths each keep only the window
    # with two supervised days.
    assert enabled["latent_info_hosp_window_count"] == 2
    assert enabled["latent_info_ww_window_count"] == 2
    assert enabled["latent_info_cases_window_count"] == 1
    assert enabled["latent_info_deaths_window_count"] == 0
    assert enabled["latent_info_hosp_observation_count"] == 4


def test_latent_information_uses_stored_true_latents_of_the_relevant_source():
    batch = _make_batch(with_residuals=True)
    # [B, H + 1] state target; index 0 is the forecast origin.
    batch.targets_dict["I_target"] = torch.tensor(
        [[0.01, 0.02, 0.04], [0.02, 0.01, 0.03]]
    )
    _loss, metrics, _ = _evaluate(batch, latent_information=True)

    # Wastewater and cases read I, which has a stored truth; the admission flow
    # behind hospitalizations has none.
    assert metrics["latent_info_ww_true_window_count"] == 2
    assert metrics["latent_info_ww_true_observation_count"] == 4
    assert metrics["latent_info_cases_true_window_count"] == 1
    assert not any(key.startswith("latent_info_hosp_true") for key in metrics)
    # A two-day horizon holds no complete week.
    assert metrics["latent_info_hosp_week_total_window_count"] == 0


def test_evaluate_loader_emits_cases_and_deaths_metrics():
    _loss, metrics, _node_mae = _evaluate(_make_batch())

    expected_joint_obs_loss = (
        metrics["loss_ww_weighted"]
        + metrics["loss_hosp_weighted"]
        + metrics["loss_cases_weighted"]
        + metrics["loss_deaths_weighted"]
    )
    assert metrics["mae"] == pytest.approx(expected_joint_obs_loss, rel=1e-5)
    assert "rmse" not in metrics
    assert "r2" not in metrics
    assert metrics["mae_hosp_log1p_per_100k"] == 0.0
    assert metrics["observed_count_cases"] == 3
    assert metrics["observed_count_deaths"] == 2
    assert metrics["mae_cases_log1p_per_100k"] > 0.0
    assert _node_mae["hospitalizations"] == {0: 0.0, 1: 0.0}

    # Latent plausibility: every window counts and each head divides the
    # observed per-100k counts (expm1 of the log1p targets) on weighted days by
    # the aligned latent flow per 100k.
    prefix = "latent_plausibility_"
    assert metrics[f"{prefix}window_count"] == 2
    assert metrics[f"{prefix}i_origin_median"] == pytest.approx(0.01)
    assert metrics[f"{prefix}i_peak_q95"] == pytest.approx(0.01 + 0.95 * 0.02)
    assert metrics[f"{prefix}forecast_attack_median"] == pytest.approx(0.1)
    assert metrics[f"{prefix}rt_origin_q05"] == pytest.approx(
        (0.3 * 0.9 + 0.05 * 0.3 * (0.95 - 0.9)) / 0.16, rel=1e-4
    )
    expm1 = math.expm1
    cases_ratios = [expm1(2.0) / 10.0, (expm1(8.0) + expm1(6.0)) / 70.0]
    assert metrics[f"{prefix}ascertainment_cases_window_count"] == 2
    assert metrics[f"{prefix}ascertainment_cases_median"] == pytest.approx(
        sum(cases_ratios) / 2.0, rel=1e-4
    )
    assert metrics[f"{prefix}ascertainment_cases_pooled"] == pytest.approx(
        (expm1(2.0) + expm1(8.0) + expm1(6.0)) / 80.0, rel=1e-4
    )
    assert metrics[f"{prefix}ascertainment_hosp_pooled"] == pytest.approx(
        (expm1(1.0) + 2 * expm1(2.0) + expm1(3.0)) / 6.0, rel=1e-4
    )
    assert metrics[f"{prefix}ascertainment_deaths_pooled"] == pytest.approx(
        expm1(1.0) / 0.1, rel=1e-4
    )
    assert metrics[f"{prefix}ascertainment_deaths_zero_latent_window_fraction"] == 0.0
