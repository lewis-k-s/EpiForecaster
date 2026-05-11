from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from evaluation.losses import JointInferenceLoss
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


def test_evaluate_loader_emits_cases_and_deaths_metrics():
    model = _DummyModel()

    batch = SimpleNamespace(
        target_node=torch.tensor([0, 1], dtype=torch.long),
        model_outputs={
            "pred_hosp": torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32),
            "pred_ww": torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
            "pred_cases": torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
            "pred_deaths": torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            "physics_residual": torch.zeros((2, 2), dtype=torch.float32),
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
    batch.to = lambda device, **_: batch

    loader = _DummyLoader([batch])
    criterion = JointInferenceLoss(
        obs_weight_sum=4.0,
        w_sird_supervision=0.0,
    )
    _loss, metrics, _node_mae = evaluate_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        horizon=2,
        device=torch.device("cpu"),
    )

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
    assert "mae_cases_log1p_per_100k" in metrics
    assert "mae_deaths_log1p_per_100k" in metrics
    assert metrics["observed_count_cases"] == 3
    assert metrics["observed_count_deaths"] == 2
    assert metrics["mae_cases_log1p_per_100k"] > 0.0
    assert _node_mae["hospitalizations"] == {0: 0.0, 1: 0.0}
