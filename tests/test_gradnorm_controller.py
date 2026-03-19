from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from evaluation.losses import JointInferenceLoss
from training.epiforecaster_trainer import EpiForecasterTrainer
from training.gradnorm import GradNormController
from training.grad_utils import compute_gradient_norms_and_clip


def test_gradnorm_weights_positive_and_sum_invariant() -> None:
    controller = GradNormController(obs_weight_sum=0.95, min_weight=1e-3)
    active = torch.tensor([True, True, False, True])
    weights = controller.weights(active)
    assert torch.all(weights[active] > 0)
    assert torch.allclose(weights[~active], torch.zeros_like(weights[~active]))
    assert torch.isclose(weights.sum(), torch.tensor(0.95), atol=1e-6)


def test_gradnorm_l0_capture_and_state_roundtrip() -> None:
    controller = GradNormController(warmup_steps=0)
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    active = torch.tensor([True, False, True, False])
    updated = controller.maybe_init_l0(losses, step=0, active_mask=active)
    assert updated
    assert controller.l0_initialized.tolist() == [True, False, True, False]
    assert torch.isclose(controller.l0[0], torch.tensor(1.0))
    assert torch.isclose(controller.l0[2], torch.tensor(3.0))

    state = controller.state_dict()
    clone = GradNormController(warmup_steps=0)
    clone.load_state_dict(state)
    assert torch.equal(clone.l0_initialized, controller.l0_initialized)
    assert torch.allclose(clone.l0, controller.l0)


def test_gradnorm_terms_are_finite_with_probe_and_ema() -> None:
    controller = GradNormController(warmup_steps=0, ema_decay=0.9)
    probe = torch.tensor([0.5, -0.5], dtype=torch.float32, requires_grad=True)

    losses = torch.stack(
        [
            (probe[0] - 1.0).pow(2) + 1.0,
            (probe[0] + 2.0).pow(2) + 1.0,
            (probe[1] - 0.5).pow(2) + 1.0,
            (probe[1] + 1.0).pow(2) + 1.0,
        ]
    )
    active = torch.tensor([True, True, True, True])
    controller.maybe_init_l0(losses.detach(), step=0, active_mask=active)
    terms = controller.compute_gradnorm_terms(
        losses,
        probe=probe,
        active_mask=active,
    )

    assert torch.isfinite(terms["gradnorm_loss"])
    assert torch.all(torch.isfinite(terms["grad_norms"][active]))
    assert torch.all(torch.isfinite(terms["target_grad_norms"][active]))
    assert torch.all(torch.isfinite(terms["ema_losses"][active]))
    assert torch.all(torch.isfinite(terms["ema_grad_norms"][active]))


class _StepModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

    def forward_batch(  # noqa: ANN001
        self,
        *,
        batch_data,
        region_embeddings=None,
        **kwargs,
    ):
        s = self.shared
        obs_context = torch.stack([s, s * 2.0]).reshape(1, 1, 2)
        c0 = obs_context[..., 0].reshape(())
        c1 = obs_context[..., 1].reshape(())
        model_outputs = {
            "pred_ww": torch.stack([c0, c0 + 0.1]).reshape(1, 2),
            "pred_hosp": torch.stack([c1 + 0.2, c1 + 0.4]).reshape(1, 2),
            "pred_cases": torch.stack([c0 + 0.3, c0 + 0.7]).reshape(1, 2),
            "pred_deaths": torch.stack([c1 + 0.5]).reshape(1, 1),
            "physics_residual": torch.stack([s * 0.1]).reshape(1, 1),
            "obs_context": obs_context,
        }
        targets = {
            "ww": torch.tensor([[0.1]], dtype=torch.float32),
            "hosp": torch.tensor([[0.2]], dtype=torch.float32),
            "cases": torch.tensor([[0.3]], dtype=torch.float32),
            "deaths": torch.tensor([[0.4]], dtype=torch.float32),
            "ww_mask": torch.ones((1, 1), dtype=torch.float32),
            "hosp_mask": torch.ones((1, 1), dtype=torch.float32),
            "cases_mask": torch.ones((1, 1), dtype=torch.float32),
            "deaths_mask": torch.ones((1, 1), dtype=torch.float32),
        }
        return model_outputs, targets


class _ToyDelayKernel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float32))


class _ToyObservationHead(torch.nn.Module):
    def __init__(self, *, include_delay_kernel: bool) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        if include_delay_kernel:
            self.delay_kernel = _ToyDelayKernel()
        self.residual_proj = torch.nn.Linear(2, 1)


class _ToyGradNormModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mobility_gnn = torch.nn.Linear(3, 2)
        self.backbone = torch.nn.Module()
        self.backbone.beta_projection = torch.nn.Linear(2, 1)
        self.backbone.encoder_projection = torch.nn.Linear(2, 2)
        self.ww_head = _ToyObservationHead(include_delay_kernel=False)
        self.hosp_head = _ToyObservationHead(include_delay_kernel=False)
        self.cases_head = _ToyObservationHead(include_delay_kernel=True)
        self.deaths_head = _ToyObservationHead(include_delay_kernel=True)
        self.other_block = torch.nn.Linear(2, 2)


def _make_gradnorm_stub_trainer(*, update_every: int) -> EpiForecasterTrainer:
    trainer = object.__new__(EpiForecasterTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = _StepModel()
    trainer.region_embeddings = None
    trainer.precision_policy = SimpleNamespace(
        autocast_dtype=torch.bfloat16,
        autocast_enabled=False,
    )
    trainer.criterion = JointInferenceLoss(obs_weight_sum=0.95, w_sir=0.0)
    trainer.gradnorm_controller = GradNormController(
        warmup_steps=0,
        update_every=update_every,
        obs_weight_sum=0.95,
        ema_decay=0.9,
    )
    trainer.gradnorm_optimizer = torch.optim.Adam(
        trainer.gradnorm_controller.parameters(),
        lr=1e-2,
    )
    trainer._gradnorm_probe = "obs_context"
    trainer._gradnorm_cached_weights = trainer.gradnorm_controller.weights(
        torch.tensor([True, True, True, True])
    ).detach()
    trainer._gradnorm_last_active_mask = torch.tensor([True, True, True, True])
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(gradient_clip_value=5.0),
    )
    return trainer


def test_gradient_norm_group_init_assigns_observation_heads() -> None:
    trainer = object.__new__(EpiForecasterTrainer)
    trainer.model = _ToyGradNormModel()

    EpiForecasterTrainer._init_gradient_norm_groups(trainer)

    name_by_id = {id(param): name for name, param in trainer.model.named_parameters()}
    grouped_names = {
        group: [name_by_id[id(param)] for param in params]
        for group, params in trainer._grad_norm_groups.items()
    }

    assert grouped_names["observation_head_ww"] == [
        "ww_head.scale",
        "ww_head.alpha",
        "ww_head.residual_proj.weight",
        "ww_head.residual_proj.bias",
    ]
    assert grouped_names["observation_head_hosp"] == [
        "hosp_head.scale",
        "hosp_head.alpha",
        "hosp_head.residual_proj.weight",
        "hosp_head.residual_proj.bias",
    ]
    assert grouped_names["observation_head_cases"] == [
        "cases_head.scale",
        "cases_head.alpha",
        "cases_head.delay_kernel.kernel",
        "cases_head.residual_proj.weight",
        "cases_head.residual_proj.bias",
    ]
    assert grouped_names["observation_head_deaths"] == [
        "deaths_head.scale",
        "deaths_head.alpha",
        "deaths_head.delay_kernel.kernel",
        "deaths_head.residual_proj.weight",
        "deaths_head.residual_proj.bias",
    ]
    assert all("_head." not in name for name in grouped_names["other"])


def test_compute_gradient_norms_logs_distinct_observation_heads() -> None:
    trainer = object.__new__(EpiForecasterTrainer)
    trainer.model = _ToyGradNormModel()
    EpiForecasterTrainer._init_gradient_norm_groups(trainer)

    grad_scales = {
        "mobility_gnn": 1.5,
        "sird": 2.0,
        "backbone": 2.5,
        "other": 3.0,
        "observation_head_ww": 4.0,
        "observation_head_hosp": 5.0,
        "observation_head_cases": 6.0,
        "observation_head_deaths": 7.0,
    }
    for group_name, params in trainer._grad_norm_groups.items():
        fill_value = grad_scales[group_name]
        for param in params:
            param.grad = torch.full_like(param, fill_value)

    _, norms = compute_gradient_norms_and_clip(
        grad_norm_groups=trainer._grad_norm_groups,
        model=trainer.model,
        device=torch.device("cpu"),
        step=0,
        frequency=1,
        clip_value=1.0e9,
    )

    expected = {}
    for group_name, params in trainer._grad_norm_groups.items():
        sq_sum = sum(float(param.grad.pow(2).sum().item()) for param in params)
        expected[group_name] = sq_sum**0.5

    assert norms["gradnorm_mobility_gnn"] == pytest.approx(expected["mobility_gnn"])
    assert norms["gradnorm_sird_physics"] == pytest.approx(expected["sird"])
    assert norms["gradnorm_backbone_encoder"] == pytest.approx(expected["backbone"])
    assert norms["gradnorm_other"] == pytest.approx(expected["other"])
    assert norms["gradnorm_obs_ww"] == pytest.approx(expected["observation_head_ww"])
    assert norms["gradnorm_obs_hosp"] == pytest.approx(
        expected["observation_head_hosp"]
    )
    assert norms["gradnorm_obs_cases"] == pytest.approx(
        expected["observation_head_cases"]
    )
    assert norms["gradnorm_obs_deaths"] == pytest.approx(
        expected["observation_head_deaths"]
    )

    assert norms["gradnorm_obs_ww"] != norms["gradnorm_obs_hosp"]
    assert norms["gradnorm_obs_hosp"] != norms["gradnorm_obs_cases"]
    assert norms["gradnorm_obs_cases"] != norms["gradnorm_obs_deaths"]


def test_trainer_adaptive_step_and_sidecar_updates_weights() -> None:
    trainer = _make_gradnorm_stub_trainer(update_every=1)
    trainer.global_step = 0

    initial = trainer.gradnorm_controller.log_weights.detach().clone()
    loss = EpiForecasterTrainer._training_step_impl_adaptive(
        trainer,
        {},
        trainer._gradnorm_cached_weights,
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(),
        max_norm=trainer.config.training.gradient_clip_value,
        foreach=True,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(grad_norm)

    trainer.model.zero_grad(set_to_none=True)
    sidecar = EpiForecasterTrainer._gradnorm_sidecar_update(trainer, {})
    assert sidecar["gradnorm_sidecar_ran"] > 0
    assert not torch.allclose(
        trainer.gradnorm_controller.log_weights.detach(),
        initial,
    )


def test_gradnorm_sidecar_respects_update_cadence() -> None:
    trainer = _make_gradnorm_stub_trainer(update_every=16)
    trainer.global_step = 3
    initial = trainer.gradnorm_controller.log_weights.detach().clone()

    sidecar = EpiForecasterTrainer._gradnorm_sidecar_update(trainer, {})

    # Sidecar returns empty dict when cadence doesn't match (no sidecar metrics logged)
    assert sidecar.get("gradnorm_sidecar_ran", 0) == 0
    assert torch.allclose(trainer.gradnorm_controller.log_weights.detach(), initial)


def test_gradnorm_sidecar_keeps_model_grads_cleared() -> None:
    trainer = _make_gradnorm_stub_trainer(update_every=1)
    trainer.global_step = 0

    _ = EpiForecasterTrainer._gradnorm_sidecar_update(trainer, {})

    for param in trainer.model.parameters():
        assert param.grad is None


def test_gradnorm_sidecar_keeps_global_cached_weights_when_head_inactive() -> None:
    trainer = _make_gradnorm_stub_trainer(update_every=1)
    trainer.global_step = 0

    def forward_batch_with_missing_ww(  # noqa: ANN001
        *,
        batch_data,
        region_embeddings=None,
        **kwargs,
    ):
        s = trainer.model.shared
        obs_context = torch.stack([s, s * 2.0]).reshape(1, 1, 2)
        c0 = obs_context[..., 0].reshape(())
        c1 = obs_context[..., 1].reshape(())
        model_outputs = {
            "pred_ww": torch.stack([c0, c0 + 0.1]).reshape(1, 2),
            "pred_hosp": torch.stack([c1 + 0.2, c1 + 0.4]).reshape(1, 2),
            "pred_cases": torch.stack([c0 + 0.3, c0 + 0.7]).reshape(1, 2),
            "pred_deaths": torch.stack([c1 + 0.5]).reshape(1, 1),
            "physics_residual": torch.stack([s * 0.1]).reshape(1, 1),
            "obs_context": obs_context,
        }
        targets = {
            "ww": torch.tensor([[0.1]], dtype=torch.float32),
            "hosp": torch.tensor([[0.2]], dtype=torch.float32),
            "cases": torch.tensor([[0.3]], dtype=torch.float32),
            "deaths": torch.tensor([[0.4]], dtype=torch.float32),
            "ww_mask": torch.zeros((1, 1), dtype=torch.float32),
            "hosp_mask": torch.ones((1, 1), dtype=torch.float32),
            "cases_mask": torch.ones((1, 1), dtype=torch.float32),
            "deaths_mask": torch.ones((1, 1), dtype=torch.float32),
        }
        return model_outputs, targets

    trainer.model.forward_batch = forward_batch_with_missing_ww  # type: ignore[method-assign]

    sidecar = EpiForecasterTrainer._gradnorm_sidecar_update(trainer, {})

    assert sidecar["gradnorm_sidecar_ran"] > 0
    assert trainer._gradnorm_last_active_mask.tolist() == [False, True, True, True]
    assert trainer._gradnorm_cached_weights[0] > 0
