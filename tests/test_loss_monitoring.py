from dataclasses import dataclass

import torch

from utils.loss_monitoring import EmaLossStats, stringify_epi_batch_indices


@dataclass
class DummyBatch:
    target_node: torch.Tensor
    window_start: torch.Tensor
    node_labels: list[str]
    region_ids: list[str]


def test_ema_loss_stats_flags_high_loss_before_updating_baseline() -> None:
    stats = EmaLossStats(decay=0.5, stddev_threshold=2.0)

    for value in [10.0, 12.0, 11.0]:
        assert stats.update(value) is None

    event = stats.update(25.0)

    assert event is not None
    assert event.value == 25.0
    assert event.mean < 25.0
    assert event.threshold < 25.0
    assert event.z_score > 2.0
    assert stats.count == 4


def test_ema_loss_stats_ignores_non_finite_values() -> None:
    stats = EmaLossStats(decay=0.9, stddev_threshold=3.0)
    assert stats.update(1.0) is None
    assert stats.update(float("nan")) is None

    assert stats.count == 1
    assert stats.mean == 1.0


def test_stringify_epi_batch_indices_uses_human_readable_metadata() -> None:
    batch = DummyBatch(
        target_node=torch.tensor([4, 7, 9]),
        window_start=torch.tensor([12, 24, 36]),
        node_labels=["Barcelona", "Girona", "Lleida"],
        region_ids=["08019", "17079", "25120"],
    )

    message = stringify_epi_batch_indices(batch, max_items=2)

    assert "Barcelona (region_id=08019, target_node=4, window_start=12)" in message
    assert "Girona (region_id=17079, target_node=7, window_start=24)" in message
    assert "... 1 more" in message
    assert "Lleida" not in message
