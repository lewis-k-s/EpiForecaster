import pytest
import torch
from types import SimpleNamespace

from evaluation.losses import JointInferenceLoss, get_loss_from_config
from models.configs import LossConfig
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
    loss_fn = JointInferenceLoss(w_sir=1.0)

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


def test_get_loss_from_config_default_is_joint_inference() -> None:
    criterion = get_loss_from_config(None)
    assert isinstance(criterion, JointInferenceLoss)


def test_get_loss_from_config_rejects_non_joint_name() -> None:
    loss_cfg = LossConfig(name="joint_inference")
    loss_cfg.name = "mse"
    with pytest.raises(ValueError, match="Only training.loss.name='joint_inference'"):
        _ = get_loss_from_config(loss_cfg)


def test_joint_inference_sir_loss_respects_observation_mask():
    """Regression test: SIR loss should average only masked-in timesteps."""
    loss_fn = JointInferenceLoss(w_sir=1.0)

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
    loss_fn = JointInferenceLoss(w_sir=0.0)

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
    loss_fn = JointInferenceLoss(w_sir=0.0)

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
        obs_weight_sum=3.0,
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
    assert loss.item() == 3.0


def test_joint_inference_loss_zero_weight_head_not_poisoned_by_nonfinite_output():
    """Disabled heads should not poison total loss through zero-initialization anchors."""
    loss_fn = JointInferenceLoss(
        obs_weight_sum=1.0,
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


def test_joint_inference_obs_active_mask_from_masks() -> None:
    loss_fn = JointInferenceLoss(w_sir=0.0)
    model_outputs = {
        "pred_ww": torch.zeros(1, 2),
        "pred_hosp": torch.zeros(1, 2),
        "pred_cases": torch.zeros(1, 2),
        "pred_deaths": torch.zeros(1, 2),
        "physics_residual": torch.zeros(1, 2),
    }
    targets = {
        "ww": torch.zeros(1, 1),
        "hosp": torch.zeros(1, 1),
        "cases": torch.zeros(1, 1),
        "deaths": torch.zeros(1, 1),
        "ww_mask": torch.tensor([[1.0]]),
        "hosp_mask": torch.tensor([[0.0]]),
        "cases_mask": torch.tensor([[1.0]]),
        "deaths_mask": torch.tensor([[0.0]]),
    }

    components = loss_fn.compute_components(model_outputs, targets)
    assert components["obs_active_mask"].dtype == torch.bool
    assert components["obs_active_mask"].tolist() == [True, False, True, False]


def test_joint_inference_compute_components_train_matches_default() -> None:
    loss_fn = JointInferenceLoss(w_sir=0.2, w_continuity=0.0)
    model_outputs = {
        "pred_ww": torch.tensor([[0.2, 0.4]]),
        "pred_hosp": torch.tensor([[0.1, 0.3]]),
        "pred_cases": torch.tensor([[0.5, 0.7]]),
        "pred_deaths": torch.tensor([[0.6]]),
        "physics_residual": torch.tensor([[0.25]]),
    }
    targets = {
        "ww": torch.tensor([[0.4]]),
        "hosp": torch.tensor([[0.3]]),
        "cases": torch.tensor([[0.7]]),
        "deaths": torch.tensor([[0.6]]),
        "ww_mask": torch.tensor([[1.0]]),
        "hosp_mask": torch.tensor([[1.0]]),
        "cases_mask": torch.tensor([[1.0]]),
        "deaths_mask": torch.tensor([[1.0]]),
    }

    components_default = loss_fn.compute_components(model_outputs, targets)
    components_train = loss_fn.compute_components_train(model_outputs, targets)

    assert torch.allclose(
        components_default["obs_active_mask"].float(),
        components_train["obs_active_mask"].float(),
    )
    assert torch.allclose(components_default["total"], components_train["total"])


def test_joint_inference_obs_active_mask_respects_min_observed() -> None:
    loss_fn = JointInferenceLoss(
        w_sir=0.0,
        ww_min_observed=2,
        disable_hosp=True,
        disable_cases=True,
        disable_deaths=True,
    )
    model_outputs = {
        "pred_ww": torch.zeros(1, 3),
        "pred_hosp": torch.zeros(1, 2),
        "pred_cases": torch.zeros(1, 2),
        "pred_deaths": torch.zeros(1, 2),
        "physics_residual": torch.zeros(1, 2),
    }
    targets = {
        "ww": torch.zeros(1, 2),
        "hosp": None,
        "cases": None,
        "deaths": None,
        "ww_mask": torch.tensor([[1.0, 0.0]]),  # only one observed point (< min_observed)
        "hosp_mask": None,
        "cases_mask": None,
        "deaths_mask": None,
    }
    components = loss_fn.compute_components(model_outputs, targets)
    assert components["obs_active_mask"].tolist() == [False, False, False, False]
    assert torch.isclose(components["ww"], torch.tensor(0.0))


def test_joint_inference_n_eff_scaling_applies_to_head_loss() -> None:
    loss_fn = JointInferenceLoss(
        obs_weight_sum=1.0,
        w_sir=0.0,
        disable_hosp=True,
        disable_cases=True,
        disable_deaths=True,
        obs_n_eff_power=1.0,
        ww_n_eff_reference=4.0,
    )
    model_outputs = {
        # Includes nowcast at t=0; supervised horizon has 4 points of value 1.0
        "pred_ww": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]),
        "pred_hosp": torch.zeros(1, 2),
        "pred_cases": torch.zeros(1, 2),
        "pred_deaths": torch.zeros(1, 2),
        "physics_residual": torch.zeros(1, 2),
    }
    targets = {
        "ww": torch.zeros(1, 4),
        "hosp": None,
        "cases": None,
        "deaths": None,
        # n_eff = 2.0 (two contributing points), base MSE=1.0, scale=(2/4)^1=0.5
        "ww_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        "hosp_mask": None,
        "cases_mask": None,
        "deaths_mask": None,
    }
    components = loss_fn.compute_components(model_outputs, targets)
    assert components["obs_active_mask"].tolist() == [True, False, False, False]
    assert torch.isclose(components["ww_n_eff"], torch.tensor(2.0))
    assert torch.isclose(components["ww"], torch.tensor(0.5))
    assert torch.isclose(components["total"], torch.tensor(0.5))


def test_joint_inference_shared_supervision_matches_obs_active_mask() -> None:
    loss_fn = JointInferenceLoss(w_sir=0.0, hosp_min_observed=2)
    model_outputs = {
        "pred_ww": torch.zeros(1, 3),
        "pred_hosp": torch.zeros(1, 3),
        "pred_cases": torch.zeros(1, 3),
        "pred_deaths": torch.zeros(1, 2),
        "physics_residual": torch.zeros(1, 2),
    }
    targets = {
        "ww": torch.zeros(1, 2),
        "hosp": torch.zeros(1, 2),
        "cases": torch.zeros(1, 2),
        "deaths": torch.zeros(1, 2),
        "ww_mask": torch.tensor([[1.0, 0.0]]),
        "hosp_mask": torch.tensor([[1.0, 0.0]]),  # below hosp_min_observed
        "cases_mask": torch.tensor([[1.0, 1.0]]),
        "deaths_mask": torch.tensor([[0.0, 0.0]]),
    }

    supervision = loss_fn.compute_observation_supervision(
        targets,
        device=torch.device("cpu"),
    )
    expected_active = torch.stack(
        [
            supervision["ww"]["active"],
            supervision["hosp"]["active"],
            supervision["cases"]["active"],
            supervision["deaths"]["active"],
        ]
    ).to(dtype=torch.bool)
    components = loss_fn.compute_components(model_outputs, targets)
    assert torch.equal(components["obs_active_mask"], expected_active)


def test_continuity_uses_only_active_heads() -> None:
    loss_fn = JointInferenceLoss(
        w_sir=0.0,
        w_continuity=1.0,
        disable_ww=True,
    )
    model_outputs = {
        "pred_ww": torch.zeros(1, 2),
        "pred_hosp": torch.tensor([[1.0, 0.0]]),
        "pred_cases": torch.tensor([[10.0, 0.0]]),
        "pred_deaths": torch.tensor([[20.0]]),
        "physics_residual": torch.zeros(1, 1),
    }
    targets = {
        "ww": None,
        "hosp": torch.tensor([[0.0]]),
        "cases": torch.tensor([[0.0]]),
        "deaths": torch.tensor([[0.0]]),
        "ww_mask": None,
        "hosp_mask": torch.tensor([[1.0]]),
        "cases_mask": torch.tensor([[0.0]]),
        "deaths_mask": torch.tensor([[0.0]]),
    }
    batch_data = SimpleNamespace(
        hosp_hist=torch.tensor([[[0.0, 0.0, 0.0], [3.0, 1.0, 0.0]]]),
        cases_hist=torch.tensor([[[0.0, 0.0, 0.0], [1000.0, 1.0, 0.0]]]),
        deaths_hist=torch.tensor([[[0.0, 0.0, 0.0], [2000.0, 1.0, 0.0]]]),
    )

    components = loss_fn.compute_components(
        model_outputs=model_outputs,
        targets=targets,
        batch_data=batch_data,
    )

    assert components["obs_active_mask"].tolist() == [False, True, False, False]
    assert torch.isclose(components["continuity"], torch.tensor(4.0))
