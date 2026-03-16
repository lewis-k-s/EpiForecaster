import pytest
import torch

from models.anchor_utils import reduce_variant_mask, resolve_last_valid_anchor


@pytest.mark.epiforecaster
def test_reduce_variant_mask_uses_or_reduction() -> None:
    mask = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    reduced = reduce_variant_mask(mask)
    expected = torch.tensor(
        [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    assert torch.equal(reduced, expected)


@pytest.mark.epiforecaster
def test_resolve_last_valid_anchor_scans_backwards() -> None:
    values = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    anchor_value, anchor_mask = resolve_last_valid_anchor(values, mask)
    assert torch.equal(anchor_value, torch.tensor([2.0, 12.0], dtype=torch.float32))
    assert torch.equal(anchor_mask, torch.tensor([1.0, 1.0], dtype=torch.float32))


@pytest.mark.epiforecaster
def test_resolve_last_valid_anchor_all_missing_returns_zero_and_disabled() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    mask = torch.zeros_like(values)
    anchor_value, anchor_mask = resolve_last_valid_anchor(values, mask)
    assert torch.equal(anchor_value, torch.tensor([0.0], dtype=torch.float32))
    assert torch.equal(anchor_mask, torch.tensor([0.0], dtype=torch.float32))
