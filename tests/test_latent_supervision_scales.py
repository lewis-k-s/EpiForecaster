import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.latent_supervision import (
    LATENT_SCALE_TASKS,
    fit_latent_scale_state,
    latent_scales_from_checkpoint,
)


FLOORS = {name: 1e-6 for name in LATENT_SCALE_TASKS}


class _ConcatLike:
    def __init__(self, *datasets):
        self.datasets = datasets


def _dataset(
    *,
    index_map=None,
    values=None,
    target_nodes=None,
    run_id="synthetic-a",
    aligned_data_path=Path("/tmp/latent-scales.zarr"),
):
    values = (
        np.asarray(values, dtype=np.float32)
        if values is not None
        else np.array(
            [
                [100.0, 900.0],
                [0.0, 800.0],
                [1.0, 700.0],
                [3.0, 600.0],
                [5.0, 500.0],
            ],
            dtype=np.float32,
        )
    )
    target = torch.from_numpy(values)
    return SimpleNamespace(
        aligned_data_path=aligned_data_path,
        run_id=run_id,
        target_nodes=[0] if target_nodes is None else target_nodes,
        _index_map=[(0, 0), (0, 1)] if index_map is None else index_map,
        _temporal_coords=np.datetime64("2020-01-01")
        + np.arange(values.shape[0], dtype="timedelta64[D]"),
        config=SimpleNamespace(
            model=SimpleNamespace(input_window_length=2, forecast_horizon=2)
        ),
        precomputed_S_target=target,
        precomputed_I_target=target,
        precomputed_H_target=target,
        precomputed_R_target=target,
        precomputed_death_flow_target=target,
    )


@pytest.mark.epiforecaster
def test_scale_fit_uses_unique_train_node_dates_and_target_domains():
    state = fit_latent_scale_state(
        _dataset(index_map=[(0, 0), (0, 1), (1, 0)]), floors=FLOORS
    )

    # States cover [1, 2, 3] and [2, 3, 4], but each target node/date enters once.
    expected_state_values = np.array([0.0, 1.0, 3.0, 5.0])
    expected_flow_values = np.array([1.0, 3.0, 5.0])
    for task in ("s", "i", "h", "r"):
        assert state.count[task] == 4
        assert state.zero_count[task] == 1
        assert state.std[task] == pytest.approx(np.std(expected_state_values))
    assert state.count["death_flow"] == 3
    assert state.zero_count["death_flow"] == 0
    assert state.std["death_flow"] == pytest.approx(np.std(expected_flow_values))


@pytest.mark.epiforecaster
def test_scale_fit_excludes_nonfinite_targets_and_applies_positive_floors():
    values = np.array(
        [
            [10.0, 900.0],
            [0.0, 800.0],
            [np.nan, 700.0],
            [0.0, 600.0],
            [0.0, 500.0],
        ],
        dtype=np.float32,
    )
    floors = {name: 10.0 for name in LATENT_SCALE_TASKS}

    state = fit_latent_scale_state(_dataset(values=values), floors=floors)

    for task in LATENT_SCALE_TASKS:
        assert state.count[task] in {2, 3}
        assert state.scales[task].dtype == np.dtype(np.float32)
        assert state.scales[task] == np.float32(10.0)
    assert state.count["s"] == 3
    assert state.zero_count["s"] == 3
    assert state.count["death_flow"] == 2
    assert state.zero_count["death_flow"] == 2


@pytest.mark.epiforecaster
def test_scale_fit_deduplicates_or_rejects_duplicate_dataset_runs():
    dataset = _dataset()
    duplicated = _ConcatLike(dataset, dataset)

    deduplicated = fit_latent_scale_state(duplicated, floors=FLOORS)
    single = fit_latent_scale_state(dataset, floors=FLOORS)
    assert deduplicated.to_dict() == single.to_dict()

    with pytest.raises(ValueError, match="Duplicate EpiDataset run"):
        fit_latent_scale_state(duplicated, floors=FLOORS, duplicate_policy="error")


@pytest.mark.epiforecaster
def test_scale_state_serialization_and_hash_are_deterministic():
    first = fit_latent_scale_state(_dataset(index_map=[(0, 1), (0, 0)]), floors=FLOORS)
    second = fit_latent_scale_state(_dataset(index_map=[(0, 0), (0, 1)]), floors=FLOORS)

    assert first.to_dict() == second.to_dict()
    assert first.state_hash == second.state_hash
    serialized = json.dumps(first.to_dict(), sort_keys=True, allow_nan=False)
    assert json.loads(serialized)["membership_hash"] == first.membership_hash


@pytest.mark.epiforecaster
def test_checkpoint_scale_state_is_hash_checked():
    state = fit_latent_scale_state(_dataset(), floors=FLOORS)
    checkpoint = {
        "latent_scale_state": state.to_dict(),
        "latent_scale_state_hash": state.state_hash,
    }

    loaded = latent_scales_from_checkpoint(checkpoint)

    assert loaded == {name: float(state.scales[name]) for name in LATENT_SCALE_TASKS}
    checkpoint["latent_scale_state"]["scales"]["i"] = 99.0
    with pytest.raises(ValueError, match="hash does not match"):
        latent_scales_from_checkpoint(checkpoint)


@pytest.mark.epiforecaster
def test_scale_state_uses_canonical_identity_instead_of_staging_path():
    first = fit_latent_scale_state(
        _dataset(aligned_data_path=Path("/nvme/job-1/data.zarr")),
        floors=FLOORS,
        dataset_identity="data/processed/canonical.zarr",
    )
    second = fit_latent_scale_state(
        _dataset(aligned_data_path=Path("/nvme/job-2/data.zarr")),
        floors=FLOORS,
        dataset_identity="data/processed/canonical.zarr",
    )

    assert first.membership_hash == second.membership_hash
    assert first.state_hash == second.state_hash
