"""Fixed training-split scale fitting for synthetic latent supervision."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, Protocol
from typing import Any

import numpy as np
import torch


LATENT_SCALE_TASKS = ("s", "i", "h", "r", "death_flow")
LATENT_SCALE_STATE_SCHEMA_VERSION = 1


class _LatentScaleDataset(Protocol):
    """The EpiDataset state required to fit latent supervision scales."""

    aligned_data_path: Path
    run_id: str
    target_nodes: list[int]
    _index_map: list[tuple[int, int]]
    _temporal_coords: list[object]
    precomputed_S_target: torch.Tensor | None
    precomputed_I_target: torch.Tensor | None
    precomputed_H_target: torch.Tensor | None
    precomputed_R_target: torch.Tensor | None
    precomputed_death_flow_target: torch.Tensor | None


@dataclass(frozen=True)
class LatentScaleState:
    """Deterministic fixed scales and membership identifiers for one train split."""

    schema_version: int
    scales: dict[str, np.float32]
    std: dict[str, float]
    count: dict[str, int]
    zero_count: dict[str, int]
    floors: dict[str, float]
    run_ids: tuple[str, ...]
    target_node_hash: str
    supervised_date_hash: str
    membership_hash: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-stable representation suitable for a run manifest."""
        return {
            "schema_version": self.schema_version,
            "scales": {name: float(self.scales[name]) for name in LATENT_SCALE_TASKS},
            "std": {name: self.std[name] for name in LATENT_SCALE_TASKS},
            "count": {name: self.count[name] for name in LATENT_SCALE_TASKS},
            "zero_count": {name: self.zero_count[name] for name in LATENT_SCALE_TASKS},
            "floors": {name: self.floors[name] for name in LATENT_SCALE_TASKS},
            "run_ids": list(self.run_ids),
            "target_node_hash": self.target_node_hash,
            "supervised_date_hash": self.supervised_date_hash,
            "membership_hash": self.membership_hash,
        }

    @property
    def state_hash(self) -> str:
        """Hash the complete serialized state with canonical JSON encoding."""
        return _hash_payload(self.to_dict())


@dataclass
class _StreamingPopulationStats:
    """Float64 Welford accumulator for a population standard deviation."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    zero_count: int = 0

    def update(self, values: np.ndarray) -> None:
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return

        batch_count = int(finite_values.size)
        batch_mean = float(finite_values.mean(dtype=np.float64))
        batch_m2 = float(np.square(finite_values - batch_mean, dtype=np.float64).sum())
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total_count
        self.mean += delta * batch_count / total_count
        self.count = total_count
        self.zero_count += int(np.count_nonzero(finite_values == 0.0))

    @property
    def population_std(self) -> float:
        if self.count == 0:
            raise ValueError("Cannot compute a latent scale from zero finite targets.")
        return float(np.sqrt(self.m2 / self.count))


def fit_latent_scale_state(
    training_dataset: _LatentScaleDataset | object,
    *,
    floors: Mapping[str, float],
    dataset_identity: str | None = None,
    duplicate_policy: Literal["deduplicate", "error"] = "deduplicate",
) -> LatentScaleState:
    """Fit fixed scales from unique valid train-window targets.

    Each EpiDataset run is visited once. Within a run, a target node/date pair
    contributes once even when overlapping valid windows cover it repeatedly.
    States use their ``[H+1]`` target domain and death flow uses its destination
    dated ``[H]`` domain. Non-finite values are excluded; finite zeros remain.
    """
    normalized_floors = _validate_floors(floors)
    datasets = _unique_datasets(training_dataset, duplicate_policy=duplicate_policy)
    if not datasets:
        raise ValueError("At least one EpiDataset is required to fit latent scales.")

    stats = {name: _StreamingPopulationStats() for name in LATENT_SCALE_TASKS}
    membership: list[dict[str, object]] = []
    date_membership: list[dict[str, object]] = []

    for dataset in datasets:
        state_dates_by_node, flow_dates_by_node = _supervised_dates_by_node(dataset)
        target_nodes = sorted(int(node) for node in dataset.target_nodes)
        run_id = str(dataset.run_id)
        membership.append(
            {
                "path": (
                    str(dataset_identity)
                    if dataset_identity is not None
                    else str(Path(dataset.aligned_data_path).resolve())
                ),
                "run_id": run_id,
                "target_nodes": target_nodes,
            }
        )

        _update_state_targets(stats, dataset, state_dates_by_node)
        _update_target(
            stats["death_flow"],
            _required_target(dataset, "precomputed_death_flow_target"),
            flow_dates_by_node,
        )

        for domain, dates_by_node in (
            ("state", state_dates_by_node),
            ("death_flow", flow_dates_by_node),
        ):
            for node, dates in sorted(dates_by_node.items()):
                date_membership.append(
                    {
                        "domain": domain,
                        "run_id": run_id,
                        "target_node": node,
                        "dates": [
                            _date_string(dataset._temporal_coords[i]) for i in dates
                        ],
                    }
                )

    std = {name: stats[name].population_std for name in LATENT_SCALE_TASKS}
    scales = {
        name: np.float32(max(std[name], normalized_floors[name]))
        for name in LATENT_SCALE_TASKS
    }
    count = {name: stats[name].count for name in LATENT_SCALE_TASKS}
    zero_count = {name: stats[name].zero_count for name in LATENT_SCALE_TASKS}
    membership.sort(key=lambda entry: (str(entry["path"]), str(entry["run_id"])))
    date_membership.sort(
        key=lambda entry: (
            str(entry["run_id"]),
            str(entry["domain"]),
            int(entry["target_node"]),
        )
    )
    run_ids = tuple(sorted({str(dataset.run_id) for dataset in datasets}))
    target_node_hash = _hash_payload(
        [
            {"run_id": entry["run_id"], "target_nodes": entry["target_nodes"]}
            for entry in membership
        ]
    )
    supervised_date_hash = _hash_payload(date_membership)
    return LatentScaleState(
        schema_version=LATENT_SCALE_STATE_SCHEMA_VERSION,
        scales=scales,
        std=std,
        count=count,
        zero_count=zero_count,
        floors=normalized_floors,
        run_ids=run_ids,
        target_node_hash=target_node_hash,
        supervised_date_hash=supervised_date_hash,
        membership_hash=_hash_payload(membership),
    )


def latent_scales_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> dict[str, float] | None:
    """Load and verify fitted latent scales stored in a trainer checkpoint."""
    payload = checkpoint.get("latent_scale_state")
    stored_hash = checkpoint.get("latent_scale_state_hash")
    if payload is None:
        if stored_hash is not None:
            raise ValueError("Checkpoint has a latent scale hash without scale state.")
        return None
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint latent scale state must be a mapping.")
    if stored_hash != _hash_payload(payload):
        raise ValueError(
            "Checkpoint latent scale state hash does not match its payload."
        )
    if payload.get("schema_version") != LATENT_SCALE_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint latent scale schema: "
            f"{payload.get('schema_version')!r}"
        )
    scales = payload.get("scales")
    if not isinstance(scales, dict) or set(scales) != set(LATENT_SCALE_TASKS):
        raise ValueError(
            f"Checkpoint latent scales must contain exactly {LATENT_SCALE_TASKS}."
        )
    normalized = {name: float(scales[name]) for name in LATENT_SCALE_TASKS}
    if any(not np.isfinite(value) or value <= 0 for value in normalized.values()):
        raise ValueError("Checkpoint latent scales must be finite and positive.")
    return normalized


def _flatten_datasets(dataset: object) -> Iterable[_LatentScaleDataset]:
    children = getattr(dataset, "datasets", None)
    if children is not None:
        for child in children:
            yield from _flatten_datasets(child)
        return

    required_attributes = (
        "aligned_data_path",
        "run_id",
        "target_nodes",
        "_index_map",
        "_temporal_coords",
    )
    missing = [name for name in required_attributes if not hasattr(dataset, name)]
    if missing:
        raise TypeError(
            "Latent scale fitting requires an EpiDataset or ConcatDataset-like "
            f"object; missing attributes: {missing}"
        )
    yield dataset  # type: ignore[misc]


def _unique_datasets(
    training_dataset: object,
    *,
    duplicate_policy: Literal["deduplicate", "error"],
) -> list[_LatentScaleDataset]:
    if duplicate_policy not in {"deduplicate", "error"}:
        raise ValueError(f"Unknown duplicate policy: {duplicate_policy!r}")

    unique: dict[tuple[str, str], _LatentScaleDataset] = {}
    for dataset in _flatten_datasets(training_dataset):
        key = (str(Path(dataset.aligned_data_path).resolve()), str(dataset.run_id))
        if key in unique:
            if duplicate_policy == "error":
                raise ValueError(f"Duplicate EpiDataset run in scale fit: {key}")
            continue
        unique[key] = dataset
    return [unique[key] for key in sorted(unique)]


def _supervised_dates_by_node(
    dataset: _LatentScaleDataset,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    model_config = dataset.config.model
    history_length = int(model_config.input_window_length)
    horizon = int(model_config.forecast_horizon)
    time_count = len(dataset._temporal_coords)
    target_nodes = {int(node) for node in dataset.target_nodes}
    state_dates = {node: set() for node in target_nodes}
    flow_dates = {node: set() for node in target_nodes}

    for node, window_start in dataset._index_map:
        node = int(node)
        if node not in target_nodes:
            continue
        range_end = int(window_start) + history_length
        state_range = range(range_end - 1, range_end + horizon)
        flow_range = range(range_end, range_end + horizon)
        if state_range.start < 0 or state_range.stop > time_count:
            raise ValueError(
                f"Latent supervision window for run {dataset.run_id!r} exceeds "
                "the temporal coordinate range."
            )
        state_dates[node].update(state_range)
        flow_dates[node].update(flow_range)

    return (
        {node: sorted(dates) for node, dates in state_dates.items()},
        {node: sorted(dates) for node, dates in flow_dates.items()},
    )


def _update_state_targets(
    stats: dict[str, _StreamingPopulationStats],
    dataset: _LatentScaleDataset,
    dates_by_node: dict[int, list[int]],
) -> None:
    for task, attribute in (
        ("s", "precomputed_S_target"),
        ("i", "precomputed_I_target"),
        ("h", "precomputed_H_target"),
        ("r", "precomputed_R_target"),
    ):
        _update_target(stats[task], _required_target(dataset, attribute), dates_by_node)


def _required_target(dataset: _LatentScaleDataset, attribute: str) -> torch.Tensor:
    target = getattr(dataset, attribute, None)
    if target is None:
        raise ValueError(
            f"Latent scale fitting requires '{attribute}' for run {dataset.run_id!r}."
        )
    if target.ndim != 2:
        raise ValueError(
            f"Latent target '{attribute}' for run {dataset.run_id!r} must have "
            f"shape [date, node], got {tuple(target.shape)}."
        )
    return target


def _update_target(
    stats: _StreamingPopulationStats,
    target: torch.Tensor,
    dates_by_node: dict[int, list[int]],
) -> None:
    values = target.detach().cpu().numpy()
    for node, dates in dates_by_node.items():
        if node < 0 or node >= values.shape[1]:
            raise ValueError(f"Target node index {node} is outside target dimensions.")
        stats.update(values[np.asarray(dates, dtype=np.int64), node])


def _validate_floors(floors: Mapping[str, float]) -> dict[str, float]:
    expected = set(LATENT_SCALE_TASKS)
    missing = expected - set(floors)
    extra = set(floors) - expected
    if missing or extra:
        raise ValueError(
            f"Latent scale floors must contain exactly {LATENT_SCALE_TASKS}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    normalized = {name: float(floors[name]) for name in LATENT_SCALE_TASKS}
    invalid = [
        name
        for name, floor in normalized.items()
        if not np.isfinite(floor) or floor <= 0
    ]
    if invalid:
        raise ValueError(f"Latent scale floors must be finite and positive: {invalid}")
    return normalized


def _date_string(value: object) -> str:
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="ns")
    return str(value)


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(encoded.encode("utf-8")).hexdigest()
