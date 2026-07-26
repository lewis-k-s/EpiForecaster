from __future__ import annotations

import math

import pytest
import torch

from evaluation.pathway_metrics import (
    TorchLatentCoherenceAccumulator,
    TorchPathwayMetricAccumulator,
)
from training.pathway_gradients import (
    compute_pathway_gradient_metrics,
    should_probe_pathway_gradients,
)


def test_pathway_metrics_quantify_residual_routing_and_mechanistic_ablation() -> None:
    accumulator = TorchPathwayMetricAccumulator(device=torch.device("cpu"))
    mechanistic = torch.tensor([[1.0, 2.0, 3.0]])
    residual = torch.tensor([[1.0, -1.0, 1.0]])
    full = mechanistic + residual
    targets = torch.tensor([[2.0, 2.0, 5.0]])
    weights = torch.tensor([[1.0, 0.0, 1.0]])

    accumulator.update(
        full=full,
        mechanistic=mechanistic,
        residual=residual,
        targets=targets,
        weights=weights,
    )
    result = accumulator.finalize()

    assert result.observed_count == 2
    assert result.effective_count == pytest.approx(2.0)
    assert result.mean_abs_mechanistic == pytest.approx(2.0)
    assert result.mean_abs_residual == pytest.approx(1.0)
    assert result.residual_abs_fraction == pytest.approx(1.0 / 3.0)
    assert result.residual_to_mechanistic_abs_ratio == pytest.approx(0.5)
    assert result.mechanistic_only_mae == pytest.approx(1.5)
    assert result.mechanistic_only_rmse == pytest.approx(math.sqrt(2.5))
    assert result.reconstruction_max_abs_error == pytest.approx(0.0)


def test_latent_metrics_capture_truth_agreement_and_structural_coherence() -> None:
    trajectories = {
        "S_trajectory": torch.tensor([[0.7, 0.6, 0.5]]),
        "I_trajectory": torch.tensor([[0.1, 0.15, 0.2]]),
        "H_trajectory": torch.tensor([[0.05, 0.05, 0.05]]),
        "R_trajectory": torch.tensor([[0.1, 0.13, 0.15]]),
        "D_trajectory": torch.tensor([[0.05, 0.07, 0.1]]),
        "hospitalization_flow": torch.tensor([[0.01, 0.02]]),
        "death_flow": torch.tensor([[0.02, 0.03]]),
        "physics_residual": torch.tensor([[0.0, -0.02]]),
    }
    targets = {
        f"{name}_target": trajectories[f"{name}_trajectory"].clone()
        for name in ("S", "I", "H", "R", "D")
    }
    accumulator = TorchLatentCoherenceAccumulator(device=torch.device("cpu"))
    accumulator.update(model_outputs=trajectories, targets=targets)
    metrics = accumulator.finalize()

    assert metrics["latent_mae_s_fraction"] == pytest.approx(0.0)
    assert metrics["latent_rmse_i_fraction"] == pytest.approx(0.0)
    assert metrics["latent_pearson_d"] == pytest.approx(1.0)
    assert metrics["latent_model_sirhd_mass_error_from_one_mae"] == pytest.approx(
        0.0
    )
    assert metrics["latent_target_exported_sirhd_mass_mean"] == pytest.approx(1.0)
    assert metrics[
        "latent_model_vs_target_exported_sirhd_mass_mae"
    ] == pytest.approx(0.0)
    assert metrics["latent_death_transition_mae"] == pytest.approx(0.0, abs=1.0e-8)
    assert metrics["latent_death_monotonic_violation_fraction"] == pytest.approx(0.0)
    assert metrics["latent_hospitalization_flow_negative_fraction"] == pytest.approx(
        0.0
    )
    assert metrics["latent_death_flow_negative_fraction"] == pytest.approx(0.0)
    assert metrics["latent_physics_residual_mae"] == pytest.approx(0.01)


def test_latent_metrics_do_not_treat_incomplete_export_as_model_mass_failure() -> None:
    trajectories = {
        "S_trajectory": torch.tensor([[0.7, 0.6]]),
        "I_trajectory": torch.tensor([[0.1, 0.15]]),
        "H_trajectory": torch.tensor([[0.05, 0.05]]),
        "R_trajectory": torch.tensor([[0.1, 0.13]]),
        "D_trajectory": torch.tensor([[0.05, 0.07]]),
        "hospitalization_flow": torch.tensor([[0.01]]),
        "death_flow": torch.tensor([[0.02]]),
    }
    targets = {
        f"{name}_target": trajectories[f"{name}_trajectory"] * 0.9
        for name in ("S", "I", "H", "R", "D")
    }
    accumulator = TorchLatentCoherenceAccumulator(device=torch.device("cpu"))
    accumulator.update(model_outputs=trajectories, targets=targets)
    metrics = accumulator.finalize()

    assert metrics["latent_model_sirhd_mass_error_from_one_mae"] == pytest.approx(
        0.0
    )
    assert metrics["latent_target_exported_sirhd_mass_mean"] == pytest.approx(0.9)
    assert metrics[
        "latent_target_exported_sirhd_mass_gap_from_one_mae"
    ] == pytest.approx(0.1)
    assert metrics[
        "latent_model_vs_target_exported_sirhd_mass_mae"
    ] == pytest.approx(0.1)


def test_pathway_gradient_metrics_distinguish_intermediate_routes() -> None:
    source = torch.tensor([[1.0, 2.0]], requires_grad=True)
    residual = torch.tensor([[0.5, -0.5]], requires_grad=True)
    prediction = 2.0 * source + residual
    loss = prediction.square().mean()

    metrics = compute_pathway_gradient_metrics(
        head_name="cases",
        loss=loss,
        mechanistic_source=source,
        residual_input=residual,
        retain_graph=False,
    )

    assert float(metrics["pathgrad_cases_mechanistic_source_grad_norm"]) > 0
    assert float(metrics["pathgrad_cases_residual_input_grad_norm"]) > 0
    fraction = float(metrics["pathgrad_cases_residual_grad_activation_fraction"])
    assert 0.0 < fraction < 1.0
    assert should_probe_pathway_gradients(step=100, frequency=100)
    assert not should_probe_pathway_gradients(step=99, frequency=100)
    assert not should_probe_pathway_gradients(step=100, frequency=0)
