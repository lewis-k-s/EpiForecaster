import pytest
import torch

from evaluation.epiforecaster_eval import JointInferenceLoss
from models.sir_rollforward import SIRRollForward


def test_physics_residual_not_trivially_zero_when_constraints_bind():
    """Regression test: residual should reflect constrained-vs-expected mismatch."""
    module = SIRRollForward()

    # This setup forces the raw Euler step to violate constraints (S goes negative),
    # so the constrained step differs materially from the unconstrained derivative.
    beta_t = torch.tensor([[50.0]])
    gamma_t = torch.tensor([[0.2]])
    mortality_t = torch.tensor([[0.0]])
    population = torch.tensor([1.0])
    S0 = torch.tensor([0.1])
    I0 = torch.tensor([0.9])
    R0 = torch.tensor([0.0])

    outputs = module(beta_t, gamma_t, mortality_t, S0, I0, R0, population)
    residual = outputs["physics_residual"]

    assert residual.shape == (1, 1)
    assert residual.item() > 1e-3


def test_joint_inference_sir_loss_zero_when_all_observation_masks_missing():
    """Regression test: SIR loss must not apply where no observations exist."""
    loss_fn = JointInferenceLoss(
        w_ww=0.0, w_hosp=0.0, w_cases=0.0, w_deaths=0.0, w_sir=1.0
    )

    model_outputs = {
        "pred_ww": torch.zeros(2, 3),
        "pred_hosp": torch.zeros(2, 3),
        "pred_cases": torch.zeros(2, 3),
        "pred_deaths": torch.zeros(2, 3),
        "physics_residual": torch.ones(2, 3),
    }
    targets = {
        "ww": None,
        "hosp": None,
        "cases": None,
        "deaths": None,
        "ww_mask": torch.zeros(2, 3),
        "hosp_mask": torch.zeros(2, 3),
        "cases_mask": torch.zeros(2, 3),
        "deaths_mask": torch.zeros(2, 3),
    }

    loss = loss_fn(model_outputs, targets)
    assert loss.item() == 0.0


def test_joint_inference_sir_loss_respects_observation_mask():
    """Regression test: SIR loss should average only masked-in timesteps."""
    loss_fn = JointInferenceLoss(
        w_ww=0.0, w_hosp=0.0, w_cases=0.0, w_deaths=0.0, w_sir=1.0
    )

    residual = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model_outputs = {
        "pred_ww": torch.zeros_like(residual),
        "pred_hosp": torch.zeros_like(residual),
        "pred_cases": torch.zeros_like(residual),
        "pred_deaths": torch.zeros_like(residual),
        "physics_residual": residual,
    }
    targets = {
        "ww": None,
        "hosp": None,
        "cases": None,
        "deaths": None,
        "ww_mask": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "hosp_mask": torch.zeros_like(residual),
        "cases_mask": torch.zeros_like(residual),
        "deaths_mask": torch.zeros_like(residual),
    }

    # physics_residual is already squared error, and target is 0 in loss implementation.
    # Expected: mean over selected positions -> (1^2 + 5^2) / 2 = 13.
    loss = loss_fn(model_outputs, targets)
    assert torch.isclose(loss, torch.tensor(13.0))


def test_joint_inference_loss_all_masked_keeps_grad_graph():
    """Regression test: all-masked batches must still allow backward()."""
    loss_fn = JointInferenceLoss(
        w_ww=1.0, w_hosp=1.0, w_cases=1.0, w_deaths=1.0, w_sir=0.0
    )

    pred_ww = torch.randn(2, 3, requires_grad=True)
    pred_hosp = torch.randn(2, 3, requires_grad=True)
    pred_cases = torch.randn(2, 3, requires_grad=True)
    pred_deaths = torch.randn(2, 3, requires_grad=True)
    model_outputs = {
        "pred_ww": pred_ww,
        "pred_hosp": pred_hosp,
        "pred_cases": pred_cases,
        "pred_deaths": pred_deaths,
        "physics_residual": torch.zeros(2, 3),
    }
    targets = {
        "ww": torch.full((2, 3), float("nan")),
        "hosp": torch.full((2, 3), float("nan")),
        "cases": torch.full((2, 3), float("nan")),
        "deaths": torch.full((2, 3), float("nan")),
        "ww_mask": torch.zeros(2, 3),
        "hosp_mask": torch.zeros(2, 3),
        "cases_mask": torch.zeros(2, 3),
        "deaths_mask": torch.zeros(2, 3),
    }

    loss = loss_fn(model_outputs, targets)
    assert loss.requires_grad
    loss.backward()
    assert pred_ww.grad is not None
    assert pred_hosp.grad is not None
    assert pred_cases.grad is not None
    assert pred_deaths.grad is not None
    assert torch.all(pred_ww.grad == 0)
    assert torch.all(pred_hosp.grad == 0)
    assert torch.all(pred_cases.grad == 0)
    assert torch.all(pred_deaths.grad == 0)


def test_joint_inference_loss_ignores_nonfinite_targets_under_mask():
    """Regression test: NaN/Inf targets outside valid mask must not contaminate loss."""
    loss_fn = JointInferenceLoss(
        w_ww=1.0, w_hosp=0.0, w_cases=0.0, w_deaths=0.0, w_sir=0.0
    )

    pred_ww = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    model_outputs = {
        "pred_ww": pred_ww,
        "pred_hosp": torch.zeros_like(pred_ww),
        "pred_cases": torch.zeros_like(pred_ww),
        "pred_deaths": torch.zeros_like(pred_ww),
        "physics_residual": torch.zeros_like(pred_ww),
    }
    targets = {
        "ww": torch.tensor([[1.0, float("nan"), float("inf")]]),
        "hosp": None,
        "cases": None,
        "deaths": None,
        "ww_mask": torch.tensor([[1.0, 0.0, 0.0]]),
        "hosp_mask": None,
        "cases_mask": None,
        "deaths_mask": None,
    }

    loss = loss_fn(model_outputs, targets)
    assert torch.isfinite(loss)
    # Only position 0 is observed (mask=1.0) with pred=1.0, target=1.0 → MSE=0
    assert torch.isclose(loss, torch.tensor(0.0))


@pytest.mark.device
def test_joint_inference_loss_cross_device(accelerator_device):
    """Test that JointInferenceLoss works when model outputs are on accelerator and targets on CPU.

    This simulates the real training scenario where:
    - Model outputs come from forward pass on GPU/MPS
    - Targets come from DataLoader on CPU

    Regression test for device mismatch bugs in loss computation.
    """
    loss_fn = JointInferenceLoss(
        w_ww=1.0,
        w_hosp=1.0,
        w_cases=1.0,
        w_deaths=0.0,
        w_sir=0.0,
    )

    # Model outputs on accelerator (simulates GPU forward pass)
    model_outputs = {
        "pred_ww": torch.ones(2, 3, device=accelerator_device),
        "pred_hosp": torch.ones(2, 3, device=accelerator_device),
        "pred_cases": torch.ones(2, 3, device=accelerator_device),
        "pred_deaths": torch.zeros(2, 3, device=accelerator_device),
        "physics_residual": torch.zeros(2, 3, device=accelerator_device),
    }

    # Targets on CPU (simulates DataLoader output)
    targets = {
        "ww": torch.zeros(2, 3),  # CPU
        "hosp": torch.zeros(2, 3),  # CPU
        "cases": torch.zeros(2, 3),  # CPU
        "deaths": None,
        "ww_mask": torch.ones(2, 3),  # CPU
        "hosp_mask": torch.ones(2, 3),  # CPU
        "cases_mask": torch.ones(2, 3),  # CPU
        "deaths_mask": None,
    }

    # This should NOT raise RuntimeError about device mismatch
    loss = loss_fn(model_outputs, targets)

    # Verify loss is computed correctly (3 heads × MSE of 1.0 each)
    assert torch.isfinite(loss)
    assert loss.item() == 3.0  # w_ww=1.0 + w_hosp=1.0 + w_cases=1.0, each MSE=1.0


def test_joint_inference_loss_zero_weight_head_not_poisoned_by_nonfinite_output():
    """Disabled heads should not poison total loss through zero-initialization anchors."""
    loss_fn = JointInferenceLoss(
        w_ww=0.0,
        w_hosp=1.0,
        w_cases=0.0,
        w_deaths=0.0,
        w_sir=0.0,
    )

    model_outputs = {
        "pred_ww": torch.tensor([[float("inf"), 1.0]]),
        "pred_hosp": torch.tensor([[1.0, 1.0]]),
        "pred_cases": torch.zeros(1, 2),
        "pred_deaths": torch.zeros(1, 2),
        "physics_residual": torch.zeros(1, 2),
    }
    targets = {
        "ww": torch.zeros(1, 2),
        "hosp": torch.zeros(1, 2),
        "cases": None,
        "deaths": None,
        "ww_mask": torch.tensor([[0.0, 0.0]]),
        "hosp_mask": torch.tensor([[1.0, 1.0]]),
        "cases_mask": None,
        "deaths_mask": None,
    }

    loss = loss_fn(model_outputs, targets)
    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(1.0))
