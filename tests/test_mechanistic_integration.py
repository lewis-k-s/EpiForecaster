"""Mechanistic-integration variants: history roll-forward, window-constant
rates, renewal force of infection, hospitalization week totals, incidence-sourced
cases, and the loss regularizers that go with them."""

from types import SimpleNamespace

import pytest
import torch

from evaluation.losses import JointInferenceLoss
from models.configs import (
    EpiForecasterConfig,
    ModelVariant,
    ObservationHeadConfig,
    SIRPhysicsConfig,
)
from models.epiforecaster import EpiForecaster
from models.observation_heads import ClinicalObservationHead
from models.sir_rollforward import SIRRollForward, generation_interval_weights

pytestmark = pytest.mark.epiforecaster

BASE_LOCAL_CONFIG = "configs/train_epifor_smoke.yaml"
BATCH, WINDOW, HORIZON, HISTORY = 3, 40, 7, 28


def _model(
    sir_physics: SIRPhysicsConfig | None = None,
    observation_heads: ObservationHeadConfig | None = None,
) -> EpiForecaster:
    torch.manual_seed(11)
    return EpiForecaster(
        variant_type=ModelVariant(
            cases=True, mobility=False, regions=False, biomarkers=False
        ),
        sir_physics=sir_physics or SIRPhysicsConfig(),
        observation_heads=observation_heads or ObservationHeadConfig(),
        temporal_input_dim=9,
        forecast_horizon=HORIZON,
        sequence_length=WINDOW,
        head_d_model=16,
    )


def _forward(model: EpiForecaster) -> dict[str, torch.Tensor]:
    torch.manual_seed(5)
    return model(
        hosp_hist=torch.randn(BATCH, WINDOW, 3),
        deaths_hist=torch.randn(BATCH, WINDOW, 3),
        cases_hist=torch.randn(BATCH, WINDOW, 3),
        biomarkers_hist=torch.randn(BATCH, WINDOW, 5),
        mob_graphs=None,
        target_nodes=torch.zeros(BATCH, dtype=torch.long),
        population=torch.full((BATCH,), 1000.0),
    )


class TestHistoryRollForward:
    def test_forecast_outputs_keep_their_shapes(self):
        out = _forward(_model(SIRPhysicsConfig(rollforward_history_steps=HISTORY)))

        for key in ("S", "I", "H", "R", "D"):
            assert out[f"{key}_trajectory"].shape == (BATCH, HORIZON + 1)
        for key in ("infection_flow", "hospitalization_flow", "death_flow", "beta_t"):
            assert out[key].shape == (BATCH, HORIZON)
        for head in ("ww", "hosp", "cases"):
            assert out[f"pred_{head}"].shape == (BATCH, HORIZON + 1)
        assert out["pred_deaths"].shape == (BATCH, HORIZON)
        assert out["beta_rollforward"].shape == (BATCH, HISTORY + HORIZON)

    def test_history_locations_end_where_the_forecast_begins(self):
        heads = ObservationHeadConfig()
        model = _model(SIRPhysicsConfig(rollforward_history_steps=HISTORY), heads)
        out = _forward(model)

        # Nowcast heads: the last history day is the forecast origin.
        for head, kernel_length in (
            ("ww", heads.kernel_length_ww),
            ("hosp", heads.kernel_length_hosp),
            ("cases", heads.kernel_length_cases),
        ):
            history = out[f"mechanistic_history_{head}"]
            assert history.shape == (BATCH, HISTORY + 1 - (kernel_length - 1))
            assert torch.allclose(history[:, -1], out[f"mechanistic_{head}"][:, 0])
        # Deaths have no nowcast: history and forecast tile the roll-forward.
        deaths_rollforward = model.deaths_head.forward_components(
            I_trajectory=out["observation_source_deaths"]
        )["mechanistic_rollforward"]
        assert torch.allclose(
            torch.cat(
                [out["mechanistic_history_deaths"], out["mechanistic_deaths"]], dim=1
            ),
            deaths_rollforward[:, heads.kernel_length_deaths - 1 :],
        )

    def test_origin_admission_flow_comes_from_the_last_history_step(self):
        out = _forward(_model(SIRPhysicsConfig(rollforward_history_steps=HISTORY)))

        # Without history the origin has no admission flow and reads zero.
        assert torch.all(out["observation_source_hosp"][:, HISTORY] > 0)
        assert torch.equal(
            out["observation_source_hosp"][:, HISTORY + 1 :],
            out["hospitalization_flow"],
        )

    def test_history_locations_train_history_rates_and_initial_state(self):
        model = _model(SIRPhysicsConfig(rollforward_history_steps=HISTORY))
        out = _forward(model)

        out["mechanistic_history_cases"].sum().backward()

        history_head = model.backbone.history_rate_projection[2].weight.grad
        initial_state = model.backbone.initial_states_projection[2].weight.grad
        assert torch.count_nonzero(history_head) > 0
        assert torch.count_nonzero(initial_state) > 0

    def test_no_history_outputs_without_history_steps(self):
        out = _forward(_model())

        assert not any(key.startswith("mechanistic_history_") for key in out)
        assert out["beta_rollforward"].shape == (BATCH, HORIZON)

    def test_disabled_mechanistic_head_has_no_history_location(self):
        out = _forward(
            _model(
                SIRPhysicsConfig(rollforward_history_steps=HISTORY),
                ObservationHeadConfig(mechanistic_disabled_heads=["cases"]),
            )
        )

        assert "mechanistic_history_cases" not in out
        assert "mechanistic_history_deaths" in out


class TestWindowConstantRates:
    def test_only_beta_varies_over_the_horizon(self):
        model = _model(SIRPhysicsConfig(time_varying_rates="beta"))
        # Give every rate head a horizon-dependent output.
        with torch.no_grad():
            for projection in (
                model.backbone.beta_projection,
                model.backbone.gamma_projection,
                model.backbone.hospitalization_projection,
            ):
                projection[2].bias.add_(torch.linspace(-0.5, 0.5, HORIZON))
        out = _forward(model)

        assert out["beta_t"].std(dim=1).min() > 0
        for key in (
            "gamma_t",
            "mortality_t",
            "hospitalization_rate_t",
            "hospital_recovery_t",
            "hospital_mortality_t",
        ):
            assert torch.equal(out[key], out[key][:, :1].expand_as(out[key]))

    def test_checkpoints_are_interchangeable_with_per_step_rates(self):
        per_step = _model(SIRPhysicsConfig())
        constant = _model(SIRPhysicsConfig(time_varying_rates="beta"))

        constant.load_state_dict(per_step.state_dict())

    def test_history_rate_head_emits_only_time_varying_rates(self):
        model = _model(
            SIRPhysicsConfig(
                rollforward_history_steps=HISTORY, time_varying_rates="beta"
            )
        )

        assert model.backbone.history_rate_projection[2].out_features == 1
        out = _forward(model)
        assert out["gamma_t"].shape == (BATCH, HORIZON)


class TestRenewalForceOfInfection:
    def _rates(self, steps: int, beta: float) -> dict[str, torch.Tensor]:
        return {
            "beta_t": torch.full((1, steps), beta),
            "gamma_t": torch.full((1, steps), 0.2),
            "mortality_t": torch.full((1, steps), 0.001),
            "hospitalization_rate_t": torch.full((1, steps), 0.01),
            "hospital_recovery_t": torch.full((1, steps), 0.1),
            "hospital_mortality_t": torch.full((1, steps), 0.005),
        }

    def _roll(self, module: SIRRollForward, steps: int, beta: float):
        return module(
            S0=torch.tensor([0.98]),
            I0=torch.tensor([0.01]),
            R0=torch.tensor([0.009]),
            H0=torch.tensor([0.001]),
            population=torch.ones(1),
            **self._rates(steps, beta),
        )

    def test_generation_interval_is_a_distribution_with_the_configured_mean(self):
        weights = generation_interval_weights(mean=5.0, sd=2.0, length=21)

        assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)
        lags = torch.arange(1, 22, dtype=torch.float32)
        assert (weights * lags).sum().item() == pytest.approx(5.0, abs=0.2)

    def test_first_step_matches_mass_action_then_diverges(self):
        renewal = SIRRollForward(
            generation_interval=generation_interval_weights(5.0, 2.0, 14)
        )
        mass_action = SIRRollForward()

        renewal_flow = self._roll(renewal, 12, beta=0.6)["infection_flow"]
        mass_action_flow = self._roll(mass_action, 12, beta=0.6)["infection_flow"]

        # The seeded history sustains I0, so the first flux is beta*S0*I0.
        assert renewal_flow[0, 0].item() == pytest.approx(
            mass_action_flow[0, 0].item(), rel=1e-5
        )
        assert not torch.allclose(renewal_flow[0, 5:], mass_action_flow[0, 5:])

    def test_unit_reproduction_number_keeps_incidence_stationary(self):
        removal = 0.2 + 0.01 + 0.001
        module = SIRRollForward(
            generation_interval=generation_interval_weights(5.0, 2.0, 14)
        )

        out = self._roll(module, 20, beta=removal)

        # R_t = 1: incidence changes only through susceptible depletion.
        flow, susceptible = out["infection_flow"][0], out["S_trajectory"][0]
        assert torch.all(flow[1:] <= flow[:-1])
        assert flow[-1] / flow[0] > (susceptible[-1] / susceptible[0]) ** 20
        total = sum(out[f"{name}_trajectory"] for name in ("S", "I", "H", "R", "D"))
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_model_uses_the_renewal_core_without_new_checkpoint_keys(self):
        mass_action = _model()
        renewal = _model(SIRPhysicsConfig(force_of_infection="renewal"))

        renewal.load_state_dict(mass_action.state_dict())
        out_renewal, out_mass_action = _forward(renewal), _forward(mass_action)
        assert not torch.allclose(
            out_renewal["infection_flow"][:, -1],
            out_mass_action["infection_flow"][:, -1],
        )


class TestHospitalizationWeekTotal:
    def test_week_total_sums_seven_days_of_the_daily_head(self):
        torch.manual_seed(3)
        daily = ClinicalObservationHead(kernel_length=5, learnable_kernel=False)
        weekly = ClinicalObservationHead(
            kernel_length=5, learnable_kernel=False, aggregation_window=7
        )
        weekly.load_state_dict(daily.state_dict())
        source = torch.rand(2, 20) * 1e-4

        daily_count = torch.expm1(daily.forward_components(source)["mechanistic_loc"])
        weekly_count = torch.expm1(weekly.forward_components(source)["mechanistic_loc"])

        assert weekly_count.shape == (2, 14)
        assert torch.allclose(
            weekly_count, daily_count.unfold(1, 7, 1).sum(dim=-1), rtol=1e-4
        )

    def test_model_rolls_six_days_past_the_horizon(self):
        out = _forward(
            _model(observation_heads=ObservationHeadConfig(hosp_week_total=True))
        )

        assert out["pred_hosp"].shape == (BATCH, HORIZON + 1)
        assert out["observation_source_hosp"].shape == (BATCH, HORIZON + 1 + 6)
        assert out["hospitalization_flow"].shape == (BATCH, HORIZON)
        assert out["H_trajectory"].shape == (BATCH, HORIZON + 1)

    def test_week_total_composes_with_history(self):
        heads = ObservationHeadConfig(hosp_week_total=True)
        out = _forward(
            _model(SIRPhysicsConfig(rollforward_history_steps=HISTORY), heads)
        )

        history = out["mechanistic_history_hosp"]
        assert history.shape == (BATCH, HISTORY + 1 - (heads.kernel_length_hosp - 1))
        assert torch.allclose(history[:, -1], out["mechanistic_hosp"][:, 0])


def test_cases_head_can_consume_incidence():
    out = _forward(
        _model(
            observation_heads=ObservationHeadConfig(
                cases_source="infection_flow", relevant_source_cases="infection_flow"
            )
        )
    )

    flow = out["infection_flow"]
    assert torch.equal(
        out["observation_source_cases"], torch.cat([flow[:, :1], flow], dim=1)
    )


class TestConfigValidation:
    def test_relevant_cases_source_must_follow_the_consumed_source(self):
        with pytest.raises(ValueError, match="relevant_source_cases must equal"):
            ObservationHeadConfig(cases_source="infection_flow")

    @pytest.mark.parametrize("steps", [20, 60])
    def test_history_steps_must_cover_kernels_and_fit_the_window(self, steps):
        with pytest.raises(ValueError, match="rollforward_history_steps"):
            EpiForecasterConfig.load(
                BASE_LOCAL_CONFIG,
                overrides=[f"model.sir_physics.rollforward_history_steps={steps}"],
            )

    def test_history_reconstruction_needs_history_steps(self):
        with pytest.raises(ValueError, match="history_reconstruction_weight requires"):
            EpiForecasterConfig.load(
                BASE_LOCAL_CONFIG,
                overrides=["training.loss.joint.history_reconstruction_weight=0.5"],
            )

    def test_residual_penalty_rejects_heads_without_mechanistic_path(self):
        with pytest.raises(ValueError, match="residual_penalty_weight"):
            EpiForecasterConfig.load(
                BASE_LOCAL_CONFIG,
                overrides=[
                    "training.loss.joint.residual_penalty_weight=0.1",
                    "model.observation_heads.mechanistic_disabled_heads=[cases]",
                ],
            )

    def test_overrides_reach_model_and_loss(self):
        cfg = EpiForecasterConfig.load(
            BASE_LOCAL_CONFIG,
            overrides=[
                "model.sir_physics.rollforward_history_steps=56",
                "model.sir_physics.time_varying_rates=beta",
                "model.sir_physics.force_of_infection=renewal",
                "model.observation_heads.hosp_week_total=true",
                "training.loss.joint.history_reconstruction_weight=0.5",
                "training.loss.joint.beta_smoothness_weight=10.0",
            ],
        )

        assert cfg.model.sir_physics.rollforward_history_steps == 56
        assert cfg.model.sir_physics.time_varying_rates == "beta"
        assert cfg.model.sir_physics.force_of_infection == "renewal"
        assert cfg.model.observation_heads.hosp_week_total is True
        assert cfg.training.loss.joint.history_reconstruction_weight == 0.5
        assert cfg.training.loss.joint.beta_smoothness_weight == 10.0


def _targets(horizon: int = 4) -> dict[str, torch.Tensor | None]:
    targets: dict[str, torch.Tensor | None] = {
        f"{name}_target": None for name in ("S", "I", "H", "R", "D", "death_flow")
    }
    for head in ("ww", "hosp", "cases", "deaths"):
        targets[head] = torch.zeros(2, horizon)
        targets[f"{head}_mask"] = torch.ones(2, horizon)
    return targets


def _outputs(horizon: int = 4) -> dict[str, torch.Tensor]:
    outputs = {}
    for head, steps in (("ww", 1), ("hosp", 1), ("cases", 1), ("deaths", 0)):
        outputs[f"pred_{head}"] = torch.zeros(2, horizon + steps)
        outputs[f"residual_{head}"] = torch.zeros(2, horizon + steps)
    return outputs


def _history_batch(length: int = 10) -> SimpleNamespace:
    torch.manual_seed(2)
    clinical = torch.randn(2, length, 3)
    clinical[..., 1] = 1.0
    return SimpleNamespace(
        hosp_hist=clinical.clone(),
        cases_hist=clinical.clone(),
        deaths_hist=clinical.clone(),
        ww_hist=torch.randn(2, length),
        ww_hist_mask=torch.ones(2, length),
    )


class TestRegularizers:
    def test_history_reconstruction_compares_trailing_history_days(self):
        loss_fn = JointInferenceLoss(history_reconstruction_weight=1.0)
        batch = _history_batch()
        outputs = _outputs()
        outputs["mechanistic_history_cases"] = batch.cases_hist[:, -6:, 0].clone()

        aligned = loss_fn.compute_components(outputs, _targets(), batch)
        outputs["mechanistic_history_cases"] = batch.cases_hist[:, -7:-1, 0].clone()
        shifted = loss_fn.compute_components(outputs, _targets(), batch)

        assert aligned["history_reconstruction"].item() == pytest.approx(0.0, abs=1e-7)
        assert shifted["history_reconstruction"].item() > 0.1

    def test_history_reconstruction_skips_unobserved_days_and_heads(self):
        loss_fn = JointInferenceLoss(history_reconstruction_weight=1.0)
        batch = _history_batch()
        batch.cases_hist[:, -3:, 1] = 0.0
        batch.cases_hist[:, -3:, 0] = float("nan")
        batch.ww_hist_mask[:] = 0.0
        outputs = _outputs()
        mechanistic = torch.zeros(2, 6)
        outputs["mechanistic_history_cases"] = mechanistic
        outputs["mechanistic_history_ww"] = torch.full((2, 6), 100.0)

        components = loss_fn.compute_components(outputs, _targets(), batch)

        expected = batch.cases_hist[:, -6:-3, 0].square().mean()
        assert components["history_reconstruction"].item() == pytest.approx(
            expected.item(), rel=1e-5
        )

    def test_residual_penalty_is_the_supervised_mean_square_residual(self):
        loss_fn = JointInferenceLoss(residual_penalty_weight=1.0)
        outputs = _outputs()
        # The nowcast step is not supervised and must not be penalized.
        outputs["residual_cases"] = torch.tensor(
            [[9.0, 1.0, 1.0, 1.0, 1.0], [9.0, 2.0, 2.0, 2.0, 2.0]]
        )
        targets = _targets()
        targets["cases_mask"] = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
        )

        components = loss_fn.compute_components(outputs, targets)

        cases_penalty = (4 * 1.0 + 2 * 4.0) / 6
        assert components["residual_penalty"].item() == pytest.approx(cases_penalty / 4)

    def test_beta_smoothness_is_the_mean_square_log_step(self):
        loss_fn = JointInferenceLoss(beta_smoothness_weight=1.0)
        outputs = _outputs()
        outputs["beta_rollforward"] = torch.tensor([[0.1, 0.2, 0.4], [0.3, 0.3, 0.3]])

        components = loss_fn.compute_components(outputs, _targets())

        expected = 2 * torch.log(torch.tensor(2.0)).square() / 4
        assert components["beta_smoothness"].item() == pytest.approx(expected.item())

    def test_total_adds_only_positively_weighted_regularizers(self):
        outputs = _outputs()
        outputs["beta_rollforward"] = torch.tensor([[0.1, 0.2], [0.1, 0.4]])
        outputs["residual_cases"] = torch.ones(2, 5)
        baseline = JointInferenceLoss().compute_components(outputs, _targets())
        weighted = JointInferenceLoss(
            beta_smoothness_weight=2.0, residual_penalty_weight=3.0
        ).compute_components(outputs, _targets())

        added = 2.0 * weighted["beta_smoothness"] + 3.0 * weighted["residual_penalty"]
        assert baseline["regularization_weighted"].item() == 0.0
        assert weighted["regularization_weighted"].item() == pytest.approx(added.item())
        assert (weighted["total"] - baseline["total"]).item() == pytest.approx(
            added.item(), rel=1e-5
        )
