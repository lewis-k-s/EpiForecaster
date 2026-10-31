"""Each pure metric step must trace as one graph and match the eager path.

The eager backend checks graph capture (no breaks, no side effects) without
the cost of generating kernels.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import pytest
import torch

import evaluation.metrics as metrics_module
import evaluation.pathway_metrics as pathway_module
from evaluation.metrics import (
    TorchDistributionMetricAccumulator,
    TorchMaskedMetricAccumulator,
)
from evaluation.pathway_metrics import (
    TorchLatentCoherenceAccumulator,
    TorchLatentProtocolAccumulator,
    TorchObservationLatentCoherenceAccumulator,
    TorchPathwayMetricAccumulator,
)

CPU = torch.device("cpu")
B, H = 9, 4


@pytest.fixture(autouse=True)
def _eager_backend_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route ``compile_step=True`` through a full-graph eager-backend compile."""
    torch._dynamo.reset()
    production = metrics_module.compiled_metric_step

    def compiled(step: Callable) -> Callable:
        return production(step, backend="eager", fullgraph=True)

    monkeypatch.setattr(metrics_module, "compiled_metric_step", compiled)
    monkeypatch.setattr(pathway_module, "compiled_metric_step", compiled)


def _no_recompile_after_first(batch_index: int, monkeypatch: pytest.MonkeyPatch):
    """After the first batch, later batches must reuse the same graphs."""
    if batch_index == 1:
        monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)


def _rows(seed: int) -> int:
    """Full batches, then a short last batch, as in an evaluation loader."""
    return B if seed < 2 else B - 2


def _batch(step: int, salt: int = 0, *, nan_fraction: float = 0.2) -> torch.Tensor:
    """Values for batch ``step``; ``salt`` gives another tensor of that batch."""
    generator = torch.Generator().manual_seed(1000 * salt + step)
    seed = step
    values = torch.randn(_rows(seed), H + 1, generator=generator)
    values[torch.rand(values.shape, generator=generator) < nan_fraction] = float("nan")
    # Batch 0 is a non-contiguous view, like a prediction after drop_nowcast.
    return values[:, 1:] if seed == 0 else values[:, 1:].contiguous()


def _weights(seed: int, salt: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1000 * salt + seed + 500)
    return (torch.rand(_rows(seed), H, generator=generator) > 0.3).float()


def _assert_same(eager: object, compiled: object) -> None:
    left = dataclasses.asdict(eager) if dataclasses.is_dataclass(eager) else eager
    right = (
        dataclasses.asdict(compiled) if dataclasses.is_dataclass(compiled) else compiled
    )
    assert left.keys() == right.keys()
    for key, value in left.items():
        other = right[key]
        if isinstance(value, dict):
            _assert_same(value, other)
        elif isinstance(value, float) and math.isnan(value):
            assert math.isnan(other), key
        else:
            assert other == pytest.approx(value, nan_ok=True), key


@pytest.mark.parametrize("horizon", [None, H])
@pytest.mark.parametrize("use_observed_mask", [False, True])
def test_masked_metric_step_traces_one_graph(
    horizon: int | None, use_observed_mask: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    eager = TorchMaskedMetricAccumulator(device=CPU, horizon=horizon)
    compiled = TorchMaskedMetricAccumulator(
        device=CPU, horizon=horizon, compile_step=True
    )
    for seed in range(3):
        _no_recompile_after_first(seed, monkeypatch)
        outputs = [
            accumulator.update(
                predictions=_batch(seed),
                targets=_batch(seed, 10),
                observed_mask=_weights(seed, 5) if use_observed_mask else None,
                sample_weights=None if use_observed_mask else _weights(seed),
            )
            for accumulator in (eager, compiled)
        ]
        for left, right in zip(*outputs, strict=True):
            torch.testing.assert_close(right, left)
    _assert_same(eager.finalize(), compiled.finalize())


@pytest.mark.parametrize("family", ["normal", "student_t"])
def test_distribution_metric_step_traces_one_graph(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = {"device": CPU, "horizon": H, "distribution_family": family}
    eager = TorchDistributionMetricAccumulator(**kwargs)
    compiled = TorchDistributionMetricAccumulator(**kwargs, compile_step=True)
    for seed in range(3):
        _no_recompile_after_first(seed, monkeypatch)
        left, right = (
            accumulator.update(
                loc=_batch(seed),
                scale=_batch(seed, 1).abs(),
                targets=_batch(seed, 10),
                observed_mask=None,
                sample_weights=_weights(seed),
            )
            for accumulator in (eager, compiled)
        )
        for name in ("nll", "crps", "wis", "weights"):
            torch.testing.assert_close(getattr(right, name), getattr(left, name))
        assert set(right.intervals) == set(compiled.level_names)
        for level_name, (low, high) in left.intervals.items():
            torch.testing.assert_close(right.intervals[level_name][0], low)
            torch.testing.assert_close(right.intervals[level_name][1], high)
    _assert_same(eager.finalize(), compiled.finalize())


def test_pathway_metric_step_traces_one_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    eager = TorchPathwayMetricAccumulator(device=CPU)
    compiled = TorchPathwayMetricAccumulator(device=CPU, compile_step=True)
    for seed in range(3):
        _no_recompile_after_first(seed, monkeypatch)
        for accumulator in (eager, compiled):
            accumulator.update(
                full=_batch(seed),
                mechanistic=_batch(seed, 1),
                residual=_batch(seed, 2),
                targets=_batch(seed, 10),
                weights=_weights(seed),
            )
    _assert_same(eager.finalize(), compiled.finalize())


@pytest.mark.parametrize("with_mechanistic", [False, True])
def test_observation_latent_step_traces_one_graph(
    with_mechanistic: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    eager = TorchObservationLatentCoherenceAccumulator(device=CPU)
    compiled = TorchObservationLatentCoherenceAccumulator(device=CPU, compile_step=True)
    for seed in range(3):
        _no_recompile_after_first(seed, monkeypatch)
        for accumulator in (eager, compiled):
            accumulator.update(
                full_prediction=_batch(seed),
                relevant_latent=_batch(seed, 10, nan_fraction=0.1),
                weights=_weights(seed),
                mechanistic_projection=_batch(seed, 1) if with_mechanistic else None,
            )
    _assert_same(eager.finalize(), compiled.finalize())


def _latent_batch(
    seed: int, generator: torch.Generator, *, with_targets: bool
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
    states = torch.rand(5, _rows(seed), H + 1, generator=generator)
    states = states / states.sum(dim=0, keepdim=True)
    states[0, 0, seed] = float("nan")
    outputs = {
        key: states[index].clone()
        for index, key in enumerate(
            ("S_trajectory", "I_trajectory", "H_trajectory", "R_trajectory")
        )
    }
    outputs["D_trajectory"] = states[4].cumsum(dim=-1)
    outputs["death_flow"] = torch.diff(outputs["D_trajectory"], dim=-1)
    outputs["hospitalization_flow"] = _batch(seed)
    outputs["physics_residual"] = _batch(seed, 1)
    targets: dict[str, torch.Tensor | None] = {}
    if with_targets:
        for key in ("S", "I", "H", "R", "D"):
            targets[f"{key}_target"] = outputs[f"{key}_trajectory"] + 0.01
        death_flow_target = _batch(seed, 2).abs()
        if seed == 1:
            # No finite pairs: the batch must not count toward window totals.
            death_flow_target[:] = float("nan")
        targets["death_flow_target"] = death_flow_target
    return outputs, targets


@pytest.mark.parametrize("with_targets", [False, True])
def test_latent_coherence_steps_trace_one_graph(
    with_targets: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    eager = TorchLatentCoherenceAccumulator(device=CPU)
    compiled = TorchLatentCoherenceAccumulator(device=CPU, compile_step=True)
    generator = torch.Generator().manual_seed(7)
    for seed in range(3):
        _no_recompile_after_first(seed, monkeypatch)
        outputs, targets = _latent_batch(seed, generator, with_targets=with_targets)
        for accumulator in (eager, compiled):
            accumulator.update(model_outputs=outputs, targets=targets)
    _assert_same(eager.finalize(), compiled.finalize())


def test_latent_protocol_steps_trace_one_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eager = TorchLatentProtocolAccumulator(protocol="oracle_x0", device=CPU)
    compiled = TorchLatentProtocolAccumulator(
        protocol="oracle_x0", device=CPU, compile_step=True
    )
    generator = torch.Generator().manual_seed(11)
    for seed in range(3):
        _no_recompile_after_first(seed, monkeypatch)
        outputs, targets = _latent_batch(seed, generator, with_targets=True)
        for accumulator in (eager, compiled):
            accumulator.update(predictions=outputs, targets=targets)
    _assert_same(eager.finalize(), compiled.finalize())
