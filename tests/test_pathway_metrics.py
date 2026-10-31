from __future__ import annotations

import math

import pytest
import torch

from evaluation.pathway_metrics import (
    TorchLatentCoherenceAccumulator,
    TorchObservationLatentCoherenceAccumulator,
    TorchPathwayMetricAccumulator,
    _TorchWindowSpearmanAccumulator,
)
from training.pathway_gradients import (
    compute_latent_parameter_gradient_metrics,
    compute_latent_variation_metrics,
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


def test_observation_latent_coherence_tracks_full_and_mechanistic_windows_independently() -> (
    None
):
    accumulator = TorchObservationLatentCoherenceAccumulator(device=torch.device("cpu"))
    accumulator.update(
        full_prediction=torch.tensor(
            [[1.0, 3.0, 2.0, 4.0], [4.0, float("nan"), 2.0, 1.0]]
        ),
        relevant_latent=torch.tensor(
            [[10.0, 30.0, 20.0, 40.0], [40.0, 30.0, 20.0, 10.0]]
        ),
        weights=torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]),
        mechanistic_projection=torch.tensor(
            [[1.0, 3.0, 2.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
        ),
    )
    result = accumulator.finalize()

    assert result.observed_count == 6
    assert result.full_window_spearman.mean == pytest.approx(1.0)
    assert result.full_window_spearman.total_count == 2
    assert result.full_window_spearman.candidate_count == 1
    assert result.full_window_spearman.eligible_count == 1
    assert result.full_window_spearman.candidate_fraction == pytest.approx(0.5)
    assert result.full_window_spearman.eligible_fraction == pytest.approx(0.5)
    assert (
        result.full_window_spearman.eligible_given_candidate_fraction
        == pytest.approx(1.0)
    )
    assert result.mechanistic_only_window_spearman.mean == pytest.approx(1.0)
    assert result.mechanistic_only_window_spearman.total_count == 2
    assert result.mechanistic_only_window_spearman.candidate_count == 2
    assert result.mechanistic_only_window_spearman.eligible_count == 2
    assert result.mechanistic_only_window_spearman.candidate_fraction == pytest.approx(
        1.0
    )
    assert result.mechanistic_only_window_spearman.eligible_fraction == pytest.approx(
        1.0
    )
    assert (
        result.mechanistic_only_window_spearman.eligible_given_candidate_fraction
        == pytest.approx(1.0)
    )


def test_observation_latent_coherence_handles_ties_and_separate_eligibility() -> None:
    accumulator = TorchObservationLatentCoherenceAccumulator(device=torch.device("cpu"))
    accumulator.update(
        full_prediction=torch.tensor([[1.0, 1.0, 2.0, 3.0]]),
        relevant_latent=torch.tensor([[10.0, 10.0, 20.0, 30.0]]),
        weights=torch.ones((1, 4)),
        mechanistic_projection=torch.ones((1, 4)),
    )
    result = accumulator.finalize()

    assert result.full_window_spearman.mean == pytest.approx(1.0)
    assert result.full_window_spearman.candidate_count == 1
    assert result.full_window_spearman.eligible_count == 1
    assert math.isnan(result.mechanistic_only_window_spearman.mean)
    assert result.mechanistic_only_window_spearman.total_count == 1
    assert result.mechanistic_only_window_spearman.candidate_count == 1
    assert result.mechanistic_only_window_spearman.eligible_count == 0
    assert result.mechanistic_only_window_spearman.candidate_fraction == pytest.approx(
        1.0
    )
    assert result.mechanistic_only_window_spearman.eligible_fraction == pytest.approx(
        0.0
    )
    assert (
        result.mechanistic_only_window_spearman.eligible_given_candidate_fraction
        == pytest.approx(0.0)
    )


def test_observation_latent_coherence_serializes_no_candidate_windows() -> None:
    accumulator = TorchObservationLatentCoherenceAccumulator(device=torch.device("cpu"))
    accumulator.update(
        full_prediction=torch.tensor([[1.0, 2.0, 3.0]]),
        relevant_latent=torch.tensor([[10.0, 20.0, 30.0]]),
        weights=torch.tensor([[1.0, 0.0, 0.0]]),
        mechanistic_projection=torch.tensor([[1.0, 2.0, 3.0]]),
    )
    result = accumulator.finalize()

    for summary in (
        result.full_window_spearman,
        result.mechanistic_only_window_spearman,
    ):
        assert math.isnan(summary.mean)
        assert summary.total_count == 1
        assert summary.candidate_count == 0
        assert summary.eligible_count == 0
        assert summary.candidate_fraction == pytest.approx(0.0)
        assert summary.eligible_fraction == pytest.approx(0.0)
        assert summary.eligible_given_candidate_fraction == pytest.approx(0.0)


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
    targets["death_flow_target"] = trajectories["death_flow"].clone()
    accumulator = TorchLatentCoherenceAccumulator(device=torch.device("cpu"))
    accumulator.update(model_outputs=trajectories, targets=targets)
    metrics = accumulator.finalize()

    assert metrics["latent_mae_s_fraction"] == pytest.approx(0.0)
    assert metrics["latent_rmse_i_fraction"] == pytest.approx(0.0)
    assert metrics["latent_pearson_d"] == pytest.approx(1.0)
    assert metrics["latent_prediction_target_std_ratio_i"] == pytest.approx(1.0)
    assert metrics["latent_mean_window_spearman_i"] == pytest.approx(1.0)
    assert metrics["latent_window_spearman_eligible_count_i"] == 1
    assert metrics["latent_mae_i_horizon_02_fraction"] == pytest.approx(0.0)
    assert metrics["latent_rmse_death_flow_fraction"] == pytest.approx(0.0)
    assert metrics["latent_rmse_death_flow_horizon_01_fraction"] == pytest.approx(0.0)
    assert metrics["latent_model_sirhd_mass_error_from_one_mae"] == pytest.approx(0.0)
    assert metrics["latent_target_exported_sirhd_mass_mean"] == pytest.approx(1.0)
    assert metrics["latent_model_vs_target_exported_sirhd_mass_mae"] == pytest.approx(
        0.0
    )
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

    assert metrics["latent_model_sirhd_mass_error_from_one_mae"] == pytest.approx(0.0)
    assert metrics["latent_target_exported_sirhd_mass_mean"] == pytest.approx(0.9)
    assert metrics[
        "latent_target_exported_sirhd_mass_gap_from_one_mae"
    ] == pytest.approx(0.1)
    assert metrics["latent_model_vs_target_exported_sirhd_mass_mae"] == pytest.approx(
        0.1
    )


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


def test_latent_parameter_gradient_metrics_report_conflict_and_clipping() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    obs_loss = (parameter - 1.0).square()
    latent_loss = (parameter + 1.0).square()
    components = {
        "obs_total": obs_loss,
        "total": obs_loss + 0.5 * latent_loss,
        "sird_supervision_weighted": 0.5 * latent_loss,
    }
    for task in ("s", "i", "h", "r", "death_flow"):
        components[f"latent_{task}_raw"] = latent_loss
        components[f"latent_{task}"] = latent_loss
        components[f"latent_{task}_weighted"] = latent_loss / 5.0

    metrics = compute_latent_parameter_gradient_metrics(
        components=components,
        selected_parameters=[parameter],
        all_parameters=[parameter],
        latent_total_weight=0.5,
        clip_value=0.5,
        actual_update_global_norm_preclip=torch.tensor(2.0),
    )

    assert metrics["latentgrad_sidecar_total_global_norm_preclip"].item() == (
        pytest.approx(1.0)
    )
    assert metrics["latentgrad_actual_update_clip_scale"].item() == pytest.approx(0.25)
    assert metrics["latentgrad_actual_update_global_norm_postclip"].item() == (
        pytest.approx(0.5)
    )
    assert metrics["latentgrad_i_shared_grad_norm_preclip"].item() == pytest.approx(0.2)
    assert metrics[
        "latentgrad_i_shared_grad_norm_counterfactual_postclip"
    ].item() == pytest.approx(0.05)
    assert metrics["latentgrad_i_shared_grad_cosine_obs"].item() == pytest.approx(-1.0)
    assert metrics[
        "latentgrad_combined_to_obs_grad_norm_ratio"
    ].item() == pytest.approx(0.5)
    assert metrics["latentgrad_cosine_i_h"].item() == pytest.approx(1.0)
    assert parameter.grad is None


def test_latent_variation_metrics_report_prediction_target_std_ratio() -> None:
    outputs = {
        "I_trajectory": torch.tensor([[0.0, 2.0]]),
    }
    targets = {
        "I_target": torch.tensor([[0.0, 1.0]]),
    }

    metrics = compute_latent_variation_metrics(
        model_outputs=outputs,
        targets=targets,
    )

    assert metrics["latentgrad_i_prediction_std"].item() == pytest.approx(1.0)
    assert metrics["latentgrad_i_target_std"].item() == pytest.approx(0.5)
    assert metrics["latentgrad_i_prediction_target_std_ratio"].item() == pytest.approx(
        2.0
    )


def test_latent_parameter_gradient_cosine_is_nan_for_starved_task() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    obs_loss = (parameter - 1.0).square()
    zero_latent = parameter.square()
    components = {
        "obs_total": obs_loss,
        "total": obs_loss,
        "sird_supervision_weighted": zero_latent,
    }
    for task in ("s", "i", "h", "r", "death_flow"):
        components[f"latent_{task}_raw"] = zero_latent
        components[f"latent_{task}"] = zero_latent
        components[f"latent_{task}_weighted"] = zero_latent

    metrics = compute_latent_parameter_gradient_metrics(
        components=components,
        selected_parameters=[parameter],
        all_parameters=[parameter],
        latent_total_weight=1.0,
        clip_value=1.0,
        actual_update_global_norm_preclip=torch.tensor(2.0),
    )

    assert torch.isnan(metrics["latentgrad_i_shared_grad_cosine_obs"])


def _legacy_window_spearman_update(
    *,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    running_sum: torch.Tensor,
) -> tuple[int, int]:
    """Run the pre-vectorization row loop as an exact test oracle."""
    candidate_count = 0
    eligible_count = 0
    prediction_rows = prediction.float().reshape(-1, prediction.shape[-1])
    reference_rows = reference.float().reshape(-1, reference.shape[-1])
    valid_rows = valid.reshape(-1, valid.shape[-1])
    for prediction_row, reference_row, valid_row in zip(
        prediction_rows,
        reference_rows,
        valid_rows,
        strict=True,
    ):
        if int(valid_row.sum().item()) < 2:
            continue
        candidate_count += 1
        prediction_valid = prediction_row[valid_row]
        reference_valid = reference_row[valid_row]

        def average_tie_ranks(values: torch.Tensor) -> torch.Tensor:
            sorted_values, order = torch.sort(values.float())
            _unique, inverse, counts = torch.unique_consecutive(
                sorted_values,
                return_inverse=True,
                return_counts=True,
            )
            starts = torch.cumsum(counts, dim=0) - counts
            average = starts.float() + (counts.float() - 1.0) / 2.0
            sorted_ranks = average[inverse]
            ranks = torch.empty_like(sorted_ranks)
            ranks[order] = sorted_ranks
            return ranks

        prediction_ranks = average_tie_ranks(prediction_valid)
        reference_ranks = average_tie_ranks(reference_valid)
        prediction_centered = prediction_ranks - prediction_ranks.mean()
        reference_centered = reference_ranks - reference_ranks.mean()
        denominator = torch.sqrt(
            prediction_centered.square().sum() * reference_centered.square().sum()
        )
        if float(denominator.item()) <= 1.0e-12:
            continue
        running_sum += (prediction_centered * reference_centered).sum() / denominator
        eligible_count += 1
    return candidate_count, eligible_count


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_window_spearman_matches_row_loop_bitwise(device: torch.device) -> None:
    generator = torch.Generator().manual_seed(20260824)
    accumulator = _TorchWindowSpearmanAccumulator(device=device)
    legacy_sum = torch.tensor(0.0, device=device)
    total_count = 0
    candidate_count = 0
    eligible_count = 0
    for batch_index in range(8):
        prediction = torch.randint(
            -2,
            5,
            (32, 28),
            generator=generator,
        ).float()
        reference = torch.randint(
            -2,
            5,
            (32, 28),
            generator=generator,
        ).float()
        valid = torch.rand((32, 28), generator=generator) > 0.35
        if batch_index == 0:
            valid[0] = False
            valid[1, 0] = True
            prediction[2] = 1.0
        prediction = prediction.to(device)
        reference = reference.to(device)
        valid = valid.to(device)
        batch_candidates, batch_eligible = _legacy_window_spearman_update(
            prediction=prediction,
            reference=reference,
            valid=valid,
            running_sum=legacy_sum,
        )
        accumulator.update(
            prediction=prediction,
            reference=reference,
            valid=valid,
        )
        total_count += prediction.shape[0]
        candidate_count += batch_candidates
        eligible_count += batch_eligible

    assert torch.equal(accumulator.correlation_sum(), legacy_sum)
    result = accumulator.finalize()
    expected_mean = (
        float((legacy_sum / eligible_count).item())
        if eligible_count > 0
        else float("nan")
    )
    assert result.mean == expected_mean
    assert result.total_count == total_count
    assert result.candidate_count == candidate_count
    assert result.eligible_count == eligible_count
