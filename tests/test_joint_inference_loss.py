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
    assert torch.isclose(loss, torch.tensor(0.0))


def test_joint_inference_loss_components_sum_to_total():
    """Regression test: weighted components must sum exactly to total loss."""
    loss_fn = JointInferenceLoss(
        w_ww=2.0, w_hosp=3.0, w_cases=1.0, w_deaths=0.5, w_sir=4.0
    )

    model_outputs = {
        "pred_ww": torch.tensor([[1.0, 2.0]]),
        "pred_hosp": torch.tensor([[1.0, 3.0]]),
        "pred_cases": torch.tensor([[2.0, 4.0]]),
        "pred_deaths": torch.tensor([[0.5, 1.0]]),
        "physics_residual": torch.tensor([[2.0, 4.0]]),
    }
    targets = {
        "ww": torch.tensor([[0.0, 0.0]]),
        "hosp": torch.tensor([[0.0, 0.0]]),
        "cases": torch.tensor([[0.0, 0.0]]),
        "deaths": torch.tensor([[0.0, 0.0]]),
        "ww_mask": torch.ones(1, 2),
        "hosp_mask": torch.ones(1, 2),
        "cases_mask": torch.ones(1, 2),
        "deaths_mask": torch.ones(1, 2),
    }

    comps = loss_fn.compute_components(model_outputs, targets)
    total_from_parts = (
        comps["ww_weighted"]
        + comps["hosp_weighted"]
        + comps["cases_weighted"]
        + comps["deaths_weighted"]
        + comps["sir_weighted"]
    )
    assert torch.isclose(comps["total"], total_from_parts)


def test_joint_inference_cases_loss_with_mask():
    """Test that cases loss respects masks and computes correctly."""
    loss_fn = JointInferenceLoss(
        w_ww=0.0,
        w_hosp=0.0,
        w_cases=1.0,
        w_deaths=0.0,
        w_sir=0.0,
        cases_imputed_weight=0.0,
    )

    model_outputs = {
        "pred_ww": torch.zeros(2, 3),
        "pred_hosp": torch.zeros(2, 3),
        "pred_cases": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "pred_deaths": torch.zeros(2, 3),
        "physics_residual": torch.zeros(2, 3),
    }
    targets = {
        "ww": None,
        "hosp": None,
        "cases": torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]]),
        "deaths": None,
        "ww_mask": None,
        "hosp_mask": None,
        "cases_mask": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        "deaths_mask": None,
    }

    # Expected: positions (0,0)=0, (0,2)=0, (1,1)=0 are masked in -> loss = 0
    loss = loss_fn(model_outputs, targets)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_joint_inference_deaths_loss_can_be_disabled():
    """Test that deaths loss can be disabled (weighted=0) but is still reported."""
    loss_fn = JointInferenceLoss(w_deaths=0.0)  # Explicitly disable deaths loss

    model_outputs = {
        "pred_ww": torch.zeros(2, 3),
        "pred_hosp": torch.zeros(2, 3),
        "pred_cases": torch.zeros(2, 3),
        "pred_deaths": torch.tensor([[100.0, 200.0, 300.0]]),  # Large error
        "physics_residual": torch.zeros(2, 3),
    }
    targets = {
        "ww": None,
        "hosp": None,
        "cases": None,
        "deaths": torch.zeros(1, 3),
        "ww_mask": None,
        "hosp_mask": None,
        "cases_mask": None,
        "deaths_mask": torch.ones(1, 3),
    }

    # Deaths weighted loss should be 0 because w_deaths=0
    # But raw deaths loss should be computed for reporting
    comps = loss_fn.compute_components(model_outputs, targets)
    assert torch.isclose(comps["deaths_weighted"], torch.tensor(0.0))
    assert comps["deaths"] > 0.0


def test_joint_inference_imputed_targets_have_lower_weight():
    """Imputed days (mask=0) should contribute with reduced configurable weight."""
    loss_fn = JointInferenceLoss(
        w_ww=0.0,
        w_hosp=1.0,
        w_cases=0.0,
        w_deaths=0.0,
        w_sir=0.0,
        hosp_imputed_weight=0.25,
    )

    model_outputs = {
        "pred_ww": torch.zeros(1, 2),
        "pred_hosp": torch.tensor([[2.0, 2.0]]),
        "pred_cases": torch.zeros(1, 2),
        "pred_deaths": torch.zeros(1, 2),
        "physics_residual": torch.zeros(1, 2),
    }
    targets = {
        "ww": None,
        "hosp": torch.tensor([[0.0, 0.0]]),
        "cases": None,
        "deaths": None,
        # First day observed, second imputed.
        "ww_mask": None,
        "hosp_mask": torch.tensor([[1.0, 0.0]]),
        "cases_mask": None,
        "deaths_mask": None,
    }

    # Weighted MSE = ((4*1.0) + (4*0.25)) / (1.0 + 0.25) = 4.0
    comps = loss_fn.compute_components(model_outputs, targets)
    assert torch.isclose(comps["hosp"], torch.tensor(4.0))


def test_joint_inference_ww_imputed_targets_have_lower_weight():
    """WW imputed timesteps should use the same weighted-mask logic."""
    loss_fn = JointInferenceLoss(
        w_ww=1.0,
        w_hosp=0.0,
        w_cases=0.0,
        w_deaths=0.0,
        w_sir=0.0,
        ww_imputed_weight=0.25,
    )

    model_outputs = {
        "pred_ww": torch.tensor([[2.0, 2.0]]),
        "pred_hosp": torch.zeros(1, 2),
        "pred_cases": torch.zeros(1, 2),
        "pred_deaths": torch.zeros(1, 2),
        "physics_residual": torch.zeros(1, 2),
    }
    targets = {
        "ww": torch.tensor([[0.0, 0.0]]),
        "hosp": None,
        "cases": None,
        "deaths": None,
        "ww_mask": torch.tensor([[1.0, 0.0]]),
        "hosp_mask": None,
        "cases_mask": None,
        "deaths_mask": None,
    }

    # ((4*1.0) + (4*0.25)) / (1.0 + 0.25) = 4.0
    comps = loss_fn.compute_components(model_outputs, targets)
    assert torch.isclose(comps["ww"], torch.tensor(4.0))


def test_joint_inference_loss_applies_min_observed_gate_per_sample():
    """Samples below min observed count should contribute zero target loss."""
    loss_fn = JointInferenceLoss(
        w_ww=1.0,
        w_hosp=0.0,
        w_cases=0.0,
        w_deaths=0.0,
        w_sir=0.0,
        ww_min_observed=2,
    )

    model_outputs = {
        "pred_ww": torch.tensor([[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]),
        "pred_hosp": torch.zeros(2, 3),
        "pred_cases": torch.zeros(2, 3),
        "pred_deaths": torch.zeros(2, 3),
        "physics_residual": torch.zeros(2, 3),
    }
    targets = {
        "ww": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "hosp": None,
        "cases": None,
        "deaths": None,
        # First sample has only 1 observed point (should be gated out).
        # Second sample has 2 observed points (should be kept).
        "ww_mask": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        "hosp_mask": None,
        "cases_mask": None,
        "deaths_mask": None,
    }

    comps = loss_fn.compute_components(model_outputs, targets)
    # Only second sample contributes: squared errors are [1,1] on observed points.
    assert torch.isclose(comps["ww"], torch.tensor(1.0))


def test_joint_inference_loss_ignores_masked_nonfinite_predictions():
    """Masked-out non-finite predictions should not contaminate loss."""
    loss_fn = JointInferenceLoss(
        w_ww=1.0,
        w_hosp=0.0,
        w_cases=0.0,
        w_deaths=0.0,
        w_sir=0.0,
        ww_imputed_weight=0.0,
    )

    model_outputs = {
        "pred_ww": torch.tensor([[1.0, float("inf")]]),
        "pred_hosp": torch.zeros(1, 2),
        "pred_cases": torch.zeros(1, 2),
        "pred_deaths": torch.zeros(1, 2),
        "physics_residual": torch.zeros(1, 2),
    }
    targets = {
        "ww": torch.tensor([[0.0, 0.0]]),
        "hosp": None,
        "cases": None,
        "deaths": None,
        "ww_mask": torch.tensor([[1.0, 0.0]]),
        "hosp_mask": None,
        "cases_mask": None,
        "deaths_mask": None,
    }

    loss = loss_fn(model_outputs, targets)
    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(1.0))


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
