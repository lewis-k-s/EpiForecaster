"""Focused tests for the observation-residual log-location budget schedule.

Covers configuration validation/resolution, head-level bound enforcement,
model propagation, trainer checkpoint/resume contracts, and evaluation
restoration.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from models.configs import (
    EpiForecasterConfig,
    ModelVariant,
    ObservationHeadConfig,
    ResidualScheduleConfig,
    ResidualSchedulePhaseConfig,
    SIRPhysicsConfig,
    resolve_residual_schedule_phase,
)
from models.epiforecaster import EpiForecaster
from models.observation_heads import (
    ClinicalObservationHead,
    WastewaterObservationHead,
)

pytestmark = pytest.mark.epiforecaster


def _basic_model_config() -> dict[str, Any]:
    return {
        "variant_type": ModelVariant(
            cases=True, mobility=False, regions=False, biomarkers=False
        ),
        "sir_physics": SIRPhysicsConfig(),
        "observation_heads": ObservationHeadConfig(),
        "temporal_input_dim": 9,
        "forecast_horizon": 7,
        "sequence_length": 14,
        "head_d_model": 16,
    }


class TestResidualScheduleConfig:
    """Configuration validation and phase resolution."""

    def test_default_is_disabled_and_empty(self):
        schedule = ResidualScheduleConfig()
        assert schedule.enabled is False
        assert schedule.schedule == []
        assert resolve_residual_schedule_phase(schedule, 0) == (None, None)

    def test_disabled_with_schedule_is_rejected(self):
        with pytest.raises(ValueError, match="must be empty when"):
            ResidualScheduleConfig(
                enabled=False,
                schedule=[ResidualSchedulePhaseConfig(0, 1, 0.0)],
            )

    def test_enabled_with_empty_schedule_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty when"):
            ResidualScheduleConfig(enabled=True, schedule=[])

    @pytest.mark.parametrize(
        "phases",
        [
            [(1, 3, 0.1)],  # first phase does not start at 0
            [(0, 0, 0.1)],  # empty epoch range
            [(-1, 1, 0.1)],  # negative start
            [(0, 1, -0.1)],
            [(0, 1, 1.5)],
            [(0, 2, 0.1), (1, 3, 0.1)],  # overlap
            [(0, 1, 0.1), (2, 3, 0.1)],  # gap
            [(0, 2, 0.1), (1, 2, 0.1)],  # out of order
        ],
    )
    def test_invalid_schedules_are_rejected(self, phases):
        with pytest.raises(ValueError):
            ResidualScheduleConfig(
                enabled=True,
                schedule=[
                    ResidualSchedulePhaseConfig(
                        start_epoch=start,
                        end_epoch=end,
                        max_log_location_fraction=fraction,
                    )
                    for start, end, fraction in phases
                ],
            )

    def test_nan_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            ResidualScheduleConfig(
                enabled=True,
                schedule=[ResidualSchedulePhaseConfig(0, 1, float("nan"))],
            )

    def test_dict_hydration_and_phase_resolution(self):
        schedule = ResidualScheduleConfig(
            enabled=True,
            schedule=[
                {
                    "start_epoch": 0,
                    "end_epoch": 1,
                    "max_log_location_fraction": 0.0,
                },
                {
                    "start_epoch": 1,
                    "end_epoch": 3,
                    "max_log_location_fraction": 0.1,
                },
            ],
        )
        assert all(
            isinstance(phase, ResidualSchedulePhaseConfig)
            for phase in schedule.schedule
        )
        assert resolve_residual_schedule_phase(schedule, 0)[0] == 0
        assert resolve_residual_schedule_phase(schedule, 1)[0] == 1
        assert resolve_residual_schedule_phase(schedule, 2)[0] == 1
        # The final contiguous phase persists after its exclusive end epoch.
        index, phase = resolve_residual_schedule_phase(schedule, 99)
        assert (
            index == 1 and phase is not None and phase.max_log_location_fraction == 0.1
        )

    def test_ambiguous_legacy_fraction_field_is_rejected(self):
        with pytest.raises(TypeError, match="max_fraction"):
            ResidualScheduleConfig(
                enabled=True,
                schedule=[
                    {
                        "start_epoch": 0,
                        "end_epoch": 1,
                        "max_fraction": 0.1,
                    }
                ],
            )

    def test_gap_between_phases_is_rejected(self):
        with pytest.raises(ValueError, match="contiguous"):
            ResidualScheduleConfig(
                enabled=True,
                schedule=[
                    ResidualSchedulePhaseConfig(0, 1, 0.0),
                    ResidualSchedulePhaseConfig(2, 4, 0.1),
                ],
            )

    def test_out_of_order_phases_are_rejected(self):
        with pytest.raises(ValueError, match="contiguous"):
            ResidualScheduleConfig(
                enabled=True,
                schedule=[
                    ResidualSchedulePhaseConfig(0, 2, 0.0),
                    ResidualSchedulePhaseConfig(1, 3, 0.1),
                ],
            )

    def test_non_integer_epochs_are_rejected(self):
        with pytest.raises(ValueError, match="integer epoch"):
            ResidualSchedulePhaseConfig(0.5, 2, 0.1)

    def test_negative_epoch_is_rejected(self):
        schedule = ResidualScheduleConfig(
            enabled=True,
            schedule=[ResidualSchedulePhaseConfig(0, 1, 0.0)],
        )
        with pytest.raises(ValueError, match="non-negative"):
            resolve_residual_schedule_phase(schedule, -1)

    def test_yaml_round_trip_preserves_schedule(self):
        cfg = EpiForecasterConfig.load(
            Path("configs/train_epifor_smoke.yaml"),
            overrides=[
                "training.residual_schedule.enabled=true",
                (
                    "training.residual_schedule.schedule=[{start_epoch: 0, "
                    "end_epoch: 1, max_log_location_fraction: 0.0}, "
                    "{start_epoch: 1, end_epoch: 3, "
                    "max_log_location_fraction: 0.1}]"
                ),
            ],
        )
        assert cfg.training.residual_schedule.enabled is True
        rebuilt = EpiForecasterConfig.from_dict(cfg.to_dict())
        assert rebuilt.training.residual_schedule == cfg.training.residual_schedule


_HEAD_FACTORIES = (
    lambda: ClinicalObservationHead(
        residual_dim=8,
        delta_forecasting=False,
        anchor_mode="disabled",
    ),
    lambda: WastewaterObservationHead(
        residual_dim=8,
        delta_forecasting=False,
        anchor_mode="disabled",
    ),
)


def _forward_kwargs(head, trajectory, context):
    kwargs: dict[str, Any] = {"I_trajectory": trajectory, "obs_context": context}
    if isinstance(head, WastewaterObservationHead):
        kwargs["population"] = torch.ones(trajectory.shape)
    return kwargs


class TestHeadResidualBound:
    """Runtime bound enforcement inside observation heads."""

    @pytest.mark.parametrize("make_head", _HEAD_FACTORIES)
    def test_none_restores_unbounded_legacy_residual(self, make_head):
        torch.manual_seed(0)
        head = make_head()
        context = torch.randn(2, 5, 8)
        trajectory = torch.rand(2, 5)
        kwargs = _forward_kwargs(head, trajectory, context)

        legacy = head.forward_components(**kwargs)
        head.set_residual_max_log_location_fraction(0.1)
        head.set_residual_max_log_location_fraction(None)
        restored = head.forward_components(**kwargs)
        assert torch.allclose(restored["residual_loc"], legacy["residual_loc"])
        assert torch.allclose(restored["loc"], legacy["loc"])

    @pytest.mark.parametrize("make_head", _HEAD_FACTORIES)
    def test_zero_budget_is_exact_lockout_with_live_mechanistic_path(self, make_head):
        torch.manual_seed(0)
        head = make_head()
        context = torch.randn(2, 5, 8)
        trajectory = torch.rand(2, 5, requires_grad=True)
        kwargs = _forward_kwargs(head, trajectory, context)

        head.set_residual_max_log_location_fraction(0.0)
        components = head.forward_components(**kwargs)

        # The final prediction contribution is exactly zero.
        assert torch.equal(
            components["residual_loc"], torch.zeros_like(components["loc"])
        )
        assert torch.equal(components["loc"], components["mechanistic_loc"])

        components["loc"].sum().backward()
        # The mechanistic path stays fully live...
        assert trajectory.grad is not None
        assert torch.isfinite(trajectory.grad).all()
        assert trajectory.grad.abs().sum() > 0
        # ...while residual parameters receive exactly zero gradient.
        assert head.residual_proj.weight.grad is not None
        assert float(head.residual_proj.weight.grad.abs().sum()) == 0.0
        if isinstance(head.alpha, torch.nn.Parameter):
            assert head.alpha.grad is not None
            assert float(head.alpha.grad.abs().sum()) == 0.0

    @pytest.mark.parametrize("make_head", _HEAD_FACTORIES)
    @pytest.mark.parametrize("fraction", [0.05, 0.1, 0.2])
    def test_bounded_residual_never_exceeds_relative_budget(self, make_head, fraction):
        torch.manual_seed(1)
        head = make_head()
        context = torch.randn(4, 12, 8) * 5.0
        trajectory = torch.rand(4, 12)
        kwargs = _forward_kwargs(head, trajectory, context)

        head.set_residual_max_log_location_fraction(fraction)
        components = head.forward_components(**kwargs)

        limit = fraction * components["mechanistic_loc"].abs()
        violation = components["residual_loc"].abs() > limit + 1e-6
        assert not violation.any(), violation.sum()
        # The composition stays exact.
        assert torch.allclose(
            components["loc"],
            components["mechanistic_loc"] + components["residual_loc"],
        )

    @pytest.mark.parametrize("make_head", _HEAD_FACTORIES)
    def test_bound_holds_under_delta_anchoring(self, make_head):
        # Anchored delta forecasting is the deployed configuration; the bound
        # must hold on the final anchored contribution, not just raw space.
        torch.manual_seed(5)
        head = make_head()
        head.delta_forecasting = True
        head.anchor_mode = "last_valid_step"
        context = torch.randn(3, 10, 8)
        trajectory = torch.rand(3, 10)
        kwargs = _forward_kwargs(head, trajectory, context)
        kwargs["last_observed"] = torch.rand(3)
        kwargs["last_observed_mask"] = torch.ones(3)

        head.set_residual_max_log_location_fraction(0.1)
        components = head.forward_components(**kwargs)
        limit = 0.1 * components["mechanistic_loc"].abs()
        assert not (components["residual_loc"].abs() > limit + 1e-6).any()
        assert torch.allclose(
            components["loc"],
            components["mechanistic_loc"] + components["residual_loc"],
        )

        head.set_residual_max_log_location_fraction(0.0)
        locked = head.forward_components(**kwargs)
        assert torch.equal(locked["residual_loc"], torch.zeros_like(locked["loc"]))
        assert torch.equal(locked["loc"], locked["mechanistic_loc"])

    @pytest.mark.parametrize("make_head", _HEAD_FACTORIES)
    def test_bound_matches_limit_tanh_formula(self, make_head):
        torch.manual_seed(2)
        head = make_head()
        context = torch.randn(2, 6, 8)
        trajectory = torch.rand(2, 6)
        kwargs = _forward_kwargs(head, trajectory, context)

        head.set_residual_max_log_location_fraction(0.1)
        components = head.forward_components(**kwargs)

        head.set_residual_max_log_location_fraction(None)
        raw = head.forward_components(**kwargs)["residual_loc"]
        mechanistic = components["mechanistic_loc"]
        limit = 0.1 * mechanistic.abs()
        expected = limit * torch.tanh(raw / limit.clamp_min(1e-30))
        assert torch.allclose(
            components["residual_loc"], expected, atol=1e-6, rtol=1e-5
        )

    @pytest.mark.parametrize("make_head", _HEAD_FACTORIES)
    def test_invalid_budget_values_are_rejected(self, make_head):
        head = make_head()
        for bad in (-0.1, 1.5, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="finite value in \\[0, 1\\]"):
                head.set_residual_max_log_location_fraction(bad)

    @pytest.mark.parametrize("make_head", _HEAD_FACTORIES)
    def test_static_disable_survives_enabled_checkpoint_load(self, make_head):
        torch.manual_seed(6)
        enabled = make_head()
        with torch.no_grad():
            enabled.alpha.fill_(0.7)
        disabled = make_head()
        disabled._residual_location_enabled = False

        context = torch.randn(2, 5, 8)
        trajectory = torch.rand(2, 5)
        kwargs = _forward_kwargs(disabled, trajectory, context)

        # Simulate loading an enabled checkpoint's state into a disabled head.
        disabled.load_state_dict(enabled.state_dict())
        assert float(disabled.alpha.detach().abs().sum()) > 0.0
        for fraction in (None, 0.2):
            disabled.set_residual_max_log_location_fraction(fraction)
            components = disabled.forward_components(**kwargs)
            assert torch.equal(
                components["residual_loc"], torch.zeros_like(components["loc"])
            )
            assert torch.equal(components["loc"], components["mechanistic_loc"])

    @pytest.mark.parametrize("make_head", _HEAD_FACTORIES)
    def test_budget_updates_take_effect_without_recreation(self, make_head):
        torch.manual_seed(4)
        head = make_head()
        context = torch.randn(2, 5, 8)
        trajectory = torch.rand(2, 5)
        kwargs = _forward_kwargs(head, trajectory, context)

        norms = []
        for fraction in (0.0, 0.1, None):
            head.set_residual_max_log_location_fraction(fraction)
            residual = head.forward_components(**kwargs)["residual_loc"]
            norms.append(float(residual.abs().max()))
        assert norms[0] == 0.0
        assert 0.0 < norms[1] < norms[2]


class TestModelPropagation:
    """EpiForecaster-level propagation."""

    def test_set_observation_residual_max_log_location_fraction_reaches_all_heads(
        self,
    ):
        model = EpiForecaster(**_basic_model_config())
        heads = (
            model.ww_head,
            model.hosp_head,
            model.cases_head,
            model.deaths_head,
        )
        model.set_observation_residual_max_log_location_fraction(0.1)
        for head in heads:
            assert float(
                head._residual_max_log_location_fraction.item()
            ) == pytest.approx(0.1)
        model.set_observation_residual_max_log_location_fraction(None)
        for head in heads:
            assert float(head._residual_max_log_location_fraction.item()) < 0.0


class TestEvaluationLoaderBudgetRestore:
    @staticmethod
    def _reference_config() -> dict[str, Any]:
        # The loader builds from the checkpoint config; align every
        # architecture-affecting field with the payload below.
        config = _basic_model_config()
        config["head_d_model"] = 128
        config["head_num_layers"] = 3
        return config

    def _checkpoint(
        self,
        tmp_path,
        enabled,
        state=None,
        *,
        include_schedule: bool = True,
    ):
        reference = EpiForecaster(**self._reference_config())
        path = tmp_path / "ckpt.pt"
        payload = {
            "epoch": 1,
            "model_state_dict": reference.state_dict(),
            "config": {
                "model": {
                    "type": {
                        "cases": True,
                        "regions": False,
                        "biomarkers": False,
                        "mobility": False,
                    },
                    "mobility_embedding_dim": 1,
                    "region_embedding_dim": 1,
                    "input_window_length": 14,
                    "forecast_horizon": 7,
                    "max_neighbors": 1,
                    "head_num_layers": 3,
                },
                "data": {},
                "training": {
                    "residual_schedule": {
                        "enabled": enabled,
                        "schedule": (
                            [
                                {
                                    "start_epoch": 0,
                                    "end_epoch": 1,
                                    "max_log_location_fraction": 0.0,
                                },
                                {
                                    "start_epoch": 1,
                                    "end_epoch": 3,
                                    "max_log_location_fraction": 0.1,
                                },
                            ]
                            if enabled
                            else []
                        ),
                    }
                },
                "output": {},
            },
        }
        if not include_schedule:
            del payload["config"]["training"]["residual_schedule"]
        if state is not None:
            payload["residual_schedule_state"] = state
        torch.save(payload, path)
        return path

    def test_enabled_checkpoint_without_state_fails_closed(self, tmp_path):
        from evaluation import loaders

        path = self._checkpoint(tmp_path, enabled=True)
        with pytest.raises(ValueError, match="no residual_schedule_state"):
            loaders.load_model_from_checkpoint(path)

    def test_legacy_checkpoint_without_schedule_stays_unrestricted(self, tmp_path):
        from evaluation import loaders

        path = self._checkpoint(
            tmp_path,
            enabled=False,
            include_schedule=False,
        )
        model, _config, _checkpoint = loaders.load_model_from_checkpoint(path)
        assert isinstance(model, EpiForecaster)
        for head in (
            model.ww_head,
            model.hosp_head,
            model.cases_head,
            model.deaths_head,
        ):
            assert float(head._residual_max_log_location_fraction.item()) < 0.0

    def test_enabled_checkpoint_with_mismatched_provenance_fails_closed(self, tmp_path):
        from evaluation import loaders

        path = self._checkpoint(
            tmp_path,
            enabled=True,
            state={
                "enabled": True,
                "phase_index": 1,
                "phase_start_epoch": 1,
                "phase_end_epoch": 3,
                "effective_max_log_location_fraction": 0.2,
            },
        )
        with pytest.raises(ValueError, match="inconsistent"):
            loaders.load_model_from_checkpoint(path)

    def test_legacy_checkpoint_with_schedule_provenance_fails_closed(self, tmp_path):
        from evaluation import loaders

        path = self._checkpoint(
            tmp_path,
            enabled=False,
            include_schedule=False,
            state={
                "enabled": False,
                "phase_index": None,
                "phase_start_epoch": None,
                "phase_end_epoch": None,
                "effective_max_log_location_fraction": None,
            },
        )
        with pytest.raises(ValueError, match="without a residual schedule"):
            loaders.load_model_from_checkpoint(path)
