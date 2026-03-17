from __future__ import annotations

from typing import cast

import pytest
import torch
from torch_geometric.data import Batch

from data.epi_batch import EpiBatch

from utils.device import (
    iter_device_ready_batches,
    prepare_batch_for_device,
    prefetch_enabled,
    setup_device_streams,
)


def _make_batch() -> EpiBatch:
    mob_batch = Batch()
    mob_batch.x_dense = torch.ones(1, 1, 1, dtype=torch.float32)
    mob_batch.global_t = torch.zeros(1, dtype=torch.long)
    mob_batch.target_node = torch.zeros(1, dtype=torch.long)

    return EpiBatch(
        hosp_hist=torch.ones(1, 2, 3, dtype=torch.float32),
        deaths_hist=torch.ones(1, 2, 3, dtype=torch.float32),
        cases_hist=torch.ones(1, 2, 3, dtype=torch.float32),
        bio_node=torch.ones(1, 2, 1, dtype=torch.float32),
        mob_batch=mob_batch,
        population=torch.ones(1, dtype=torch.float32),
        b=1,
        t=2,
        target_node=torch.zeros(1, dtype=torch.long),
        target_region_index=torch.zeros(1, dtype=torch.long),
        window_start=torch.zeros(1, dtype=torch.long),
        node_labels=["node-0"],
        temporal_covariates=torch.ones(1, 2, 1, dtype=torch.float32),
        ww_hist=torch.ones(1, 2, dtype=torch.float32),
        ww_hist_mask=torch.ones(1, 2, dtype=torch.float32),
        hosp_target=torch.ones(1, 1, dtype=torch.float32),
        ww_target=torch.ones(1, 1, dtype=torch.float32),
        cases_target=torch.ones(1, 1, dtype=torch.float32),
        deaths_target=torch.ones(1, 1, dtype=torch.float32),
        hosp_target_mask=torch.ones(1, 1, dtype=torch.float32),
        ww_target_mask=torch.ones(1, 1, dtype=torch.float32),
        cases_target_mask=torch.ones(1, 1, dtype=torch.float32),
        deaths_target_mask=torch.ones(1, 1, dtype=torch.float32),
    )


class _DummyLoader:
    def __init__(self, batches: list[EpiBatch], dataset: object | None) -> None:
        self._batches = batches
        self.dataset = dataset

    def __iter__(self):
        return iter(self._batches)


def test_prefetch_enabled_treats_zero_and_none_as_disabled() -> None:
    assert not prefetch_enabled(None)
    assert not prefetch_enabled(0)
    assert prefetch_enabled(1)


def test_setup_device_streams_noop_on_cpu() -> None:
    streams = setup_device_streams(torch.device("cpu"))
    assert streams.compute is None
    assert streams.transfer is None


def test_prepare_batch_for_device_injects_mobility_and_moves(monkeypatch) -> None:
    calls: list[tuple[object, object, torch.device]] = []
    to_calls: list[tuple[EpiBatch, torch.device, bool]] = []

    def _fake_inject(batch: object, dataset: object, device: torch.device) -> None:
        calls.append((batch, dataset, device))

    original_to = EpiBatch.to

    def _spy_to(
        self: EpiBatch,
        device: torch.device | str,
        non_blocking: bool = True,
    ) -> EpiBatch:
        device_obj = device if isinstance(device, torch.device) else torch.device(device)
        to_calls.append((self, device_obj, non_blocking))
        return original_to(self, device=device, non_blocking=non_blocking)

    monkeypatch.setattr("utils.training_utils.inject_gpu_mobility", _fake_inject)
    monkeypatch.setattr(EpiBatch, "to", _spy_to)

    batch = _make_batch()
    dataset = object()
    device = torch.device("cpu")

    result = prepare_batch_for_device(
        batch,
        dataset=dataset,
        device=device,
        non_blocking=False,
    )

    assert result is batch
    assert calls == [(batch, dataset, device)]
    assert to_calls == [(batch, device, False)]


def test_iter_device_ready_batches_prepares_cpu_batches(monkeypatch) -> None:
    calls: list[tuple[object, object, torch.device]] = []
    to_calls: list[tuple[EpiBatch, torch.device, bool]] = []

    def _fake_inject(batch: object, dataset: object, device: torch.device) -> None:
        calls.append((batch, dataset, device))

    original_to = EpiBatch.to

    def _spy_to(
        self: EpiBatch,
        device: torch.device | str,
        non_blocking: bool = True,
    ) -> EpiBatch:
        device_obj = device if isinstance(device, torch.device) else torch.device(device)
        to_calls.append((self, device_obj, non_blocking))
        return original_to(self, device=device, non_blocking=non_blocking)

    monkeypatch.setattr("utils.training_utils.inject_gpu_mobility", _fake_inject)
    monkeypatch.setattr(EpiBatch, "to", _spy_to)

    dataset = object()
    batches = [_make_batch(), _make_batch()]
    loader = _DummyLoader(batches, dataset=dataset)
    device = torch.device("cpu")

    out = list(
        iter_device_ready_batches(
            loader,
            device=device,
            prefetch_factor=4,
        )
    )

    assert out == batches
    assert calls == [
        (batches[0], dataset, device),
        (batches[1], dataset, device),
    ]
    assert to_calls == [
        (batches[0], device, True),
        (batches[1], device, True),
    ]


def test_prepare_batch_for_device_requires_epibatch_contract() -> None:
    with pytest.raises(AttributeError):
        prepare_batch_for_device(
            cast(EpiBatch, object()),
            dataset=None,
            device=torch.device("cpu"),
        )
