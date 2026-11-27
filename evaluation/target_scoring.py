"""Per-target scoring of the evaluation pass.

``build_target_forecast`` collects one target's forecast and observations for a
batch, aligned to the forecast horizon. A ``TargetScorer`` accumulates every
per-target indicator over the batches and returns the target's canonical metric
keys. The latent-information accumulator and the granular writer are shared by
all targets; a scorer feeds them but does not own them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from evaluation.granular_export import GranularEvalWriter
from evaluation.latent_information import TorchLatentInformationAccumulator
from evaluation.losses import JointInferenceLoss
from evaluation.metrics import (
    ProbabilisticMetricResult,
    TorchDistributionMetricAccumulator,
    TorchMaskedMetricAccumulator,
)
from evaluation.pathway_metrics import (
    ObservationLatentMetricResult,
    PathwayMetricResult,
    TorchObservationLatentCoherenceAccumulator,
    TorchPathwayMetricAccumulator,
    WindowSpearmanMetricResult,
)
from utils.training_utils import drop_nowcast

# Observation head name to the canonical target name of node metrics.
CANONICAL_TARGET_NAMES = {
    "hosp": "hospitalizations",
    "ww": "wastewater",
    "cases": "cases",
    "deaths": "deaths",
}
_RELEVANT_SOURCE_OUTPUT_KEYS = {
    "hospitalization_flow": "hospitalization_flow",
    "H": "H_trajectory",
    "I": "I_trajectory",
    "death_flow": "death_flow",
    "infection_flow": "infection_flow",
    "D": "D_trajectory",
}
# Stored true value of a relevant source; the admission and infection flows have
# no stored truth.
_RELEVANT_SOURCE_TARGET_KEYS = {
    "H": "H_target",
    "I": "I_target",
    "death_flow": "death_flow_target",
    "D": "D_target",
}

NodeStats = dict[int, dict[str, float]]


@dataclass(frozen=True)
class TargetForecast:
    """One target's forecast and observations for one batch, on the horizon.

    ``prediction`` is the point forecast and ``loc``/``scale`` the predictive
    distribution the loss scores. ``weights`` are the supervision weights.
    ``latent`` is the relevant latent source and ``true_latent`` its stored
    truth, when the dataset has one.
    """

    head: str
    prediction: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    observed: torch.Tensor
    observed_mask: torch.Tensor | None
    weights: torch.Tensor
    mechanistic: torch.Tensor | None
    residual: torch.Tensor | None
    latent_source: str
    latent: torch.Tensor | None
    true_latent: torch.Tensor | None


@dataclass(frozen=True)
class TargetScore:
    """Finalized scores of one target.

    ``metrics`` holds the canonical flat metric keys. ``region_set_metrics``
    holds, per region set, the probabilistic keys before the
    ``region_set_<name>_`` prefix. ``node_stats`` is None when no batch
    supervised the target.
    """

    metrics: dict[str, Any]
    region_set_metrics: dict[str, dict[str, Any]]
    node_stats: NodeStats | None


def _aligned_output(
    model_outputs: dict[str, torch.Tensor],
    key: str,
    horizon: int,
) -> torch.Tensor | None:
    value = model_outputs.get(key)
    if value is None:
        return None
    return drop_nowcast(value, horizon)


def build_target_forecast(
    head: str,
    *,
    model_outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor | None],
    supervision: dict[str, torch.Tensor | None],
    criterion: JointInferenceLoss,
    latent_source: str,
    horizon: int,
) -> TargetForecast | None:
    """Collect one target's forecast for a batch; None without supervision."""
    prediction = model_outputs.get(f"pred_{head}")
    observed = supervision["target"]
    weights = supervision["weights"]
    if prediction is None or observed is None or weights is None:
        return None
    steps = int(observed.shape[1])
    loc, scale = criterion.head_distribution(
        model_outputs, head_name=head, horizon=steps
    )
    true_latent_key = _RELEVANT_SOURCE_TARGET_KEYS.get(latent_source)
    true_latent = None if true_latent_key is None else targets.get(true_latent_key)
    return TargetForecast(
        head=head,
        prediction=drop_nowcast(prediction, horizon),
        loc=loc,
        scale=scale,
        observed=observed,
        observed_mask=targets.get(f"{head}_mask"),
        weights=weights,
        mechanistic=_aligned_output(model_outputs, f"mechanistic_{head}", steps),
        residual=_aligned_output(model_outputs, f"residual_{head}", steps),
        latent_source=latent_source,
        latent=_aligned_output(
            model_outputs, _RELEVANT_SOURCE_OUTPUT_KEYS[latent_source], steps
        ),
        true_latent=None if true_latent is None else drop_nowcast(true_latent, steps),
    )


def region_set_members(
    region_sets: dict[str, frozenset[str]],
    region_ids: Sequence[Any] | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Boolean node-index membership of each named region set."""
    if region_ids is None:
        raise ValueError("Region-set metrics need the dataset's region IDs")
    ids = [str(region_id) for region_id in region_ids]
    return {
        name: torch.tensor(
            [region_id in members for region_id in ids],
            dtype=torch.bool,
            device=device,
        )
        for name, members in sorted(region_sets.items())
    }


class TargetScorer:
    """Accumulates the evaluation indicators of one target over batches."""

    def __init__(
        self,
        head: str,
        *,
        criterion: JointInferenceLoss,
        device: torch.device,
        horizon: int,
        compile_step: bool = False,
        region_set_members: dict[str, torch.Tensor] | None = None,
        latent_information: TorchLatentInformationAccumulator | None = None,
        granular_writer: GranularEvalWriter | None = None,
        temporal_coords: Sequence[Any] | None = None,
        region_ids: Sequence[Any] | None = None,
        region_labels: Sequence[Any] | None = None,
    ) -> None:
        self.head = head
        self._distribution_family = criterion.distribution_family
        self._distribution_df = criterion.distribution_df

        def distribution_accumulator() -> TorchDistributionMetricAccumulator:
            return TorchDistributionMetricAccumulator(
                device=device,
                horizon=horizon,
                distribution_family=criterion.distribution_family,
                distribution_df=criterion.distribution_df,
                scale_floor=criterion.distribution_scale_floor,
                interval_levels=criterion.interval_levels,
                censored=criterion.censored,
                compile_step=compile_step,
            )

        self._point = TorchMaskedMetricAccumulator(
            device=device, horizon=horizon, compile_step=compile_step
        )
        self._distribution = distribution_accumulator()
        self._mechanistic_distribution = distribution_accumulator()
        self._pathway = TorchPathwayMetricAccumulator(
            device=device, compile_step=compile_step
        )
        self._observation_latent = TorchObservationLatentCoherenceAccumulator(
            device=device, compile_step=compile_step
        )
        self._region_set_members = region_set_members or {}
        self._region_set_distributions = {
            name: distribution_accumulator() for name in self._region_set_members
        }
        self._latent_information = latent_information
        self._granular_writer = granular_writer
        self._temporal_coords = temporal_coords
        self._region_ids = region_ids
        self._region_labels = region_labels
        self._node_stats: NodeStats | None = None

    def update(self, forecast: TargetForecast, *, batch_data: Any) -> None:
        nodes = batch_data.target_node
        diff, abs_diff, effective_weights = self._point.update(
            predictions=forecast.prediction,
            targets=forecast.observed,
            observed_mask=forecast.observed_mask,
            sample_weights=forecast.weights,
        )
        self._accumulate_node_stats(nodes, diff, abs_diff, effective_weights)
        batch_scores = self._distribution.update(
            loc=forecast.loc,
            scale=forecast.scale,
            targets=forecast.observed,
            observed_mask=forecast.observed_mask,
            sample_weights=forecast.weights,
        )
        for name, member in self._region_set_members.items():
            in_set = member[nodes.to(member.device)].to(forecast.weights.dtype)
            in_set = in_set.reshape(-1, *([1] * (forecast.weights.dim() - 1)))
            self._region_set_distributions[name].update(
                loc=forecast.loc,
                scale=forecast.scale,
                targets=forecast.observed,
                observed_mask=forecast.observed_mask,
                sample_weights=forecast.weights * in_set,
            )
        self._observation_latent.update(
            full_prediction=forecast.prediction,
            relevant_latent=forecast.latent,
            weights=forecast.weights,
            mechanistic_projection=forecast.mechanistic,
        )
        if (
            self._latent_information is not None
            and forecast.latent is not None
            and forecast.residual is not None
        ):
            self._latent_information.update(
                head=self.head,
                latent=forecast.latent,
                observed=forecast.observed,
                prediction=forecast.prediction,
                residual=forecast.residual,
                weights=forecast.weights,
                nodes=nodes,
                true_latent=forecast.true_latent,
            )
        if forecast.mechanistic is not None and forecast.residual is not None:
            self._pathway.update(
                full=forecast.loc,
                mechanistic=forecast.mechanistic,
                residual=forecast.residual,
                targets=forecast.observed,
                weights=forecast.weights,
            )
            self._mechanistic_distribution.update(
                loc=forecast.mechanistic,
                scale=forecast.scale,
                targets=forecast.observed,
                observed_mask=forecast.observed_mask,
                sample_weights=forecast.weights,
            )
        if self._granular_writer is not None:
            self._granular_writer.write_rows(
                batch_data=batch_data,
                target_name=self.head,
                predictions=forecast.prediction,
                targets=forecast.observed,
                weights=effective_weights,
                nll=batch_scores.nll,
                crps=batch_scores.crps,
                wis=batch_scores.wis,
                distribution_loc=forecast.loc,
                distribution_scale=forecast.scale,
                distribution_family=self._distribution_family,
                distribution_df=self._distribution_df,
                intervals=batch_scores.intervals,
                mechanistic_predictions=forecast.mechanistic,
                residual_contributions=forecast.residual,
                relevant_latent_name=forecast.latent_source,
                relevant_latent_values=forecast.latent,
                temporal_coords=self._temporal_coords,
                region_ids=self._region_ids,
                region_labels=self._region_labels,
            )

    def _accumulate_node_stats(
        self,
        nodes: torch.Tensor,
        diff: torch.Tensor,
        abs_diff: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        if self._node_stats is None:
            self._node_stats = {}
        per_sample_abs_sum = (abs_diff * weights).sum(dim=1).detach().cpu().tolist()
        per_sample_sq_sum = ((diff**2) * weights).sum(dim=1).detach().cpu().tolist()
        per_sample_count = (weights > 0).sum(dim=1).detach().cpu().tolist()
        for node_id, abs_sum, sq_sum, observed_count in zip(
            nodes.detach().cpu().tolist(),
            per_sample_abs_sum,
            per_sample_sq_sum,
            per_sample_count,
            strict=False,
        ):
            count_value = int(observed_count)
            if count_value <= 0:
                continue
            stats = self._node_stats.setdefault(
                int(node_id),
                {"abs_error_sum": 0.0, "sq_error_sum": 0.0, "observed_count": 0.0},
            )
            stats["abs_error_sum"] += float(abs_sum)
            stats["sq_error_sum"] += float(sq_sum)
            stats["observed_count"] += float(count_value)

    def finalize(self) -> TargetScore:
        head = self.head
        point = self._point.finalize()
        metrics: dict[str, Any] = {
            f"mae_{head}_per_h": point.mae_per_h,
            f"rmse_{head}_per_h": point.rmse_per_h,
            f"mae_{head}_log1p_per_100k": point.mae,
            f"rmse_{head}_log1p_per_100k": point.rmse,
            f"r2_{head}_log1p_per_100k": point.r2,
            f"observed_count_{head}": point.observed_count,
            f"effective_count_{head}": point.effective_count,
        }
        _add_probabilistic_metrics(
            metrics=metrics, head_name=head, summary=self._distribution.finalize()
        )
        _add_pathway_metrics(
            metrics=metrics,
            head_name=head,
            summary=self._pathway.finalize(),
            mechanistic_probabilistic_summary=self._mechanistic_distribution.finalize(),
        )
        _add_observation_latent_metrics(
            metrics=metrics,
            head_name=head,
            summary=self._observation_latent.finalize(),
        )
        region_set_metrics: dict[str, dict[str, Any]] = {}
        for name, accumulator in self._region_set_distributions.items():
            region_set_metrics[name] = {}
            _add_probabilistic_metrics(
                metrics=region_set_metrics[name],
                head_name=head,
                summary=accumulator.finalize(),
            )
        return TargetScore(
            metrics=metrics,
            region_set_metrics=region_set_metrics,
            node_stats=self._node_stats,
        )


def _add_probabilistic_metrics(
    *,
    metrics: dict[str, Any],
    head_name: str,
    summary: ProbabilisticMetricResult,
) -> None:
    metrics[f"nll_{head_name}_log1p_per_100k"] = summary.nll
    metrics[f"crps_{head_name}_log1p_per_100k"] = summary.crps
    metrics[f"wis_{head_name}_log1p_per_100k"] = summary.wis
    metrics[f"wis_{head_name}_per_h"] = summary.wis_per_h
    metrics[f"invalid_prediction_count_{head_name}"] = summary.invalid_prediction_count
    metrics[f"scale_floor_count_{head_name}"] = summary.scale_floor_count
    metrics[f"nll_{head_name}_per_h"] = summary.nll_per_h
    metrics[f"crps_{head_name}_per_h"] = summary.crps_per_h
    metrics[f"probabilistic_effective_count_{head_name}"] = summary.effective_count
    for level_name, coverage in summary.coverage_by_level.items():
        metrics[f"coverage_{level_name}_{head_name}"] = coverage
        metrics[f"interval_width_{level_name}_{head_name}_log1p_per_100k"] = (
            summary.interval_width_by_level[level_name]
        )
        metrics[f"coverage_{level_name}_{head_name}_per_h"] = (
            summary.coverage_per_h_by_level[level_name]
        )
        metrics[f"interval_width_{level_name}_{head_name}_per_h"] = (
            summary.interval_width_per_h_by_level[level_name]
        )


def _add_pathway_metrics(
    *,
    metrics: dict[str, Any],
    head_name: str,
    summary: PathwayMetricResult,
    mechanistic_probabilistic_summary: ProbabilisticMetricResult,
) -> None:
    prefix = f"pathway_{head_name}"
    metrics[f"{prefix}_observed_count"] = summary.observed_count
    metrics[f"{prefix}_effective_count"] = summary.effective_count
    metrics[f"{prefix}_mean_abs_mechanistic_log1p_per_100k"] = (
        summary.mean_abs_mechanistic
    )
    metrics[f"{prefix}_mean_abs_residual_log1p_per_100k"] = summary.mean_abs_residual
    metrics[f"{prefix}_residual_abs_fraction"] = summary.residual_abs_fraction
    metrics[f"{prefix}_residual_to_mechanistic_abs_ratio"] = (
        summary.residual_to_mechanistic_abs_ratio
    )
    metrics[f"{prefix}_mechanistic_variance"] = summary.mechanistic_variance
    metrics[f"{prefix}_residual_variance"] = summary.residual_variance
    metrics[f"{prefix}_full_variance"] = summary.full_variance
    metrics[f"{prefix}_mechanistic_variance_share"] = summary.mechanistic_variance_share
    metrics[f"{prefix}_residual_variance_share"] = summary.residual_variance_share
    metrics[f"{prefix}_mechanistic_only_mae_log1p_per_100k"] = (
        summary.mechanistic_only_mae
    )
    metrics[f"{prefix}_mechanistic_only_rmse_log1p_per_100k"] = (
        summary.mechanistic_only_rmse
    )
    metrics[f"{prefix}_mechanistic_only_nll_log1p_per_100k"] = (
        mechanistic_probabilistic_summary.nll
    )
    metrics[f"{prefix}_mechanistic_only_crps_log1p_per_100k"] = (
        mechanistic_probabilistic_summary.crps
    )
    metrics[f"{prefix}_reconstruction_max_abs_error"] = (
        summary.reconstruction_max_abs_error
    )


def _add_window_spearman_metrics(
    *,
    metrics: dict[str, Any],
    prefix: str,
    summary: WindowSpearmanMetricResult,
) -> None:
    metrics[f"{prefix}_mean_window_spearman"] = summary.mean
    metrics[f"{prefix}_window_spearman_total_count"] = summary.total_count
    metrics[f"{prefix}_window_spearman_candidate_count"] = summary.candidate_count
    metrics[f"{prefix}_window_spearman_eligible_count"] = summary.eligible_count
    metrics[f"{prefix}_window_spearman_candidate_fraction"] = summary.candidate_fraction
    metrics[f"{prefix}_window_spearman_eligible_fraction"] = summary.eligible_fraction
    metrics[f"{prefix}_window_spearman_eligible_given_candidate_fraction"] = (
        summary.eligible_given_candidate_fraction
    )


def _add_observation_latent_metrics(
    *,
    metrics: dict[str, Any],
    head_name: str,
    summary: ObservationLatentMetricResult,
) -> None:
    prefix = f"observation_latent_{head_name}"
    metrics[f"{prefix}_observed_count"] = summary.observed_count
    _add_window_spearman_metrics(
        metrics=metrics,
        prefix=f"{prefix}_full",
        summary=summary.full_window_spearman,
    )
    _add_window_spearman_metrics(
        metrics=metrics,
        prefix=f"{prefix}_mechanistic_only",
        summary=summary.mechanistic_only_window_spearman,
    )
