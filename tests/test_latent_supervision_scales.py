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
    validate_complete_latent_targets,
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
def test_complete_target_check_ignores_values_outside_supervised_windows():
    values = np.array(
        [
            [np.nan, -1.0],
            [0.0, np.nan],
            [1.0, -1.0],
            [3.0, np.nan],
            [5.0, -1.0],
        ],
        dtype=np.float32,
    )

    validate_complete_latent_targets(_ConcatLike(_dataset(values=values)))


@pytest.mark.epiforecaster
@pytest.mark.parametrize("invalid_value", [np.nan, -1.0])
def test_complete_target_check_rejects_invalid_supervised_value(invalid_value):
    dataset = _dataset()
    death_flow = dataset.precomputed_death_flow_target.clone()
    death_flow[4, 0] = invalid_value
    dataset.precomputed_death_flow_target = death_flow

    with pytest.raises(
        ValueError, match="finite and nonnegative.*precomputed_death_flow_target"
    ):
        validate_complete_latent_targets(dataset)


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


def _per_window_dates(dataset):
    """Reference: the original per-sample accumulation of supervised dates."""
    history = int(dataset.config.model.input_window_length)
    horizon = int(dataset.config.model.forecast_horizon)
    nodes = {int(node) for node in dataset.target_nodes}
    state = {node: set() for node in nodes}
    flow = {node: set() for node in nodes}
    for node, window_start in dataset._index_map:
        if int(node) not in nodes:
            continue
        end = int(window_start) + history
        state[int(node)].update(range(end - 1, end + horizon))
        flow[int(node)].update(range(end, end + horizon))
    return (
        {node: sorted(dates) for node, dates in state.items()},
        {node: sorted(dates) for node, dates in flow.items()},
    )


def _random_dataset(seed: int):
    rng = np.random.default_rng(seed)
    time_count, node_count = 40, 5
    values = rng.random((time_count, node_count)).astype(np.float32)
    # Invalid values anywhere: only those inside supervised windows may count.
    values[rng.random(values.shape) < 0.04] = np.nan
    values[rng.random(values.shape) < 0.04] = -1.0
    # 2 + 3 timesteps per window; stride and node subset vary by seed.
    starts = range(int(rng.integers(0, 3)), time_count - 5, int(rng.integers(1, 6)))
    index_map = [
        (int(node), start)
        for start in starts
        for node in range(node_count)
        if rng.random() < 0.7
    ]
    target_nodes = [int(n) for n in rng.permutation(node_count)[: rng.integers(1, 5)]]
    dataset = _dataset(values=values, index_map=index_map, target_nodes=target_nodes)
    dataset.config = SimpleNamespace(
        model=SimpleNamespace(input_window_length=2, forecast_horizon=3)
    )
    return dataset


@pytest.mark.epiforecaster
@pytest.mark.parametrize("seed", range(12))
def test_vectorized_supervised_dates_match_the_per_window_reference(seed):
    from data.latent_supervision import _supervised_dates_by_node

    dataset = _random_dataset(seed)
    expected_state, expected_flow = _per_window_dates(dataset)
    state, flow = _supervised_dates_by_node(dataset)
    assert state == expected_state
    assert flow == expected_flow
    # Dict order drives the order-dependent float accumulation of the fit.
    assert list(state) == list({int(n) for n in dataset.target_nodes})


@pytest.mark.epiforecaster
@pytest.mark.parametrize("seed", range(12))
def test_vectorized_completeness_check_matches_the_per_window_reference(seed):
    dataset = _random_dataset(seed)
    state_dates, flow_dates = _per_window_dates(dataset)
    expected_invalid = []
    for attribute, dates_by_node in (
        ("precomputed_S_target", state_dates),
        ("precomputed_I_target", state_dates),
        ("precomputed_H_target", state_dates),
        ("precomputed_R_target", state_dates),
        ("precomputed_death_flow_target", flow_dates),
    ):
        values = getattr(dataset, attribute).numpy()
        for node, dates in dates_by_node.items():
            window = values[np.asarray(dates, dtype=np.int64), node]
            if not (np.isfinite(window).all() and (window >= 0).all()):
                expected_invalid.append(attribute)
                break

    if expected_invalid:
        with pytest.raises(ValueError) as error:
            validate_complete_latent_targets(dataset)
        assert str(expected_invalid) in str(error.value)
    else:
        validate_complete_latent_targets(dataset)


@pytest.mark.epiforecaster
def test_a_node_outside_the_target_dimensions_is_rejected():
    dataset = _dataset(target_nodes=[0, 7], index_map=[(0, 0), (7, 0)])
    with pytest.raises(ValueError, match="outside target dimensions"):
        validate_complete_latent_targets(dataset)


@pytest.mark.epiforecaster
def test_a_window_beyond_the_time_range_is_rejected():
    dataset = _dataset(index_map=[(0, 4)])  # 4 + 2 + 2 > 5 dates
    with pytest.raises(ValueError, match="exceeds the temporal coordinate range"):
        validate_complete_latent_targets(dataset)


@pytest.mark.epiforecaster
def test_scale_state_hashes_match_the_pre_vectorization_fit():
    """Golden hashes computed with the per-window implementation (HEAD af1b793+1)."""
    rng = np.random.default_rng(3)
    values = rng.random((12, 4)).astype(np.float32)
    values[3, 1] = np.nan
    dataset = _dataset(
        values=values,
        target_nodes=[0, 1, 3],
        index_map=[(0, 0), (0, 3), (1, 2), (1, 5), (3, 1), (3, 6), (2, 2)],
    )

    state = fit_latent_scale_state(dataset, floors=FLOORS)

    assert state.supervised_date_hash == (
        "d5b71485ea9f0a8ac27983f852d79e46856bf386cf38379c8271b0ddd642e53f"
    )
    assert state.membership_hash == (
        "14a6343a99a127ac230821f2bd3f7f4686b0624db8fb9d3dee93e68314509931"
    )
    assert state.state_hash == (
        "e92c221ecb9388cc102ce020f715ed979414f6a116b8589c44db5534a66f12b0"
    )
