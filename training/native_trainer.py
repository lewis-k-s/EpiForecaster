"""Source-native heterogeneous training and evaluation.

The municipal trainer belongs to the separate legacy worktree.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from data.native_batch import ROUTED_HEADS, collate_native_batch

from data.native_inputs import native_dataset, split_identity
from data.native_source_sets import (
    CalendarSplit,
    QueryEligibilityRecord,
    SourceSetManifest,
    load_source_set_manifest,
    split_queries_for_run,
    freeze_calendar_split,
)
from data.query_contract import ForecastQuery
from evaluation.eval_loop import evaluate_native_loader
from evaluation.losses import get_native_loss_from_config
from models.configs import (
    EpiForecasterConfig,
    InitWeightsConfig,
    ObservationHeadConfig,
    SIRPhysicsConfig,
    TrainingParams,
)
from models.native_forecaster import NativeEpiForecaster, NativeForecasterConfig
from training.optim_factory import create_optimizer, create_scheduler
from training.training_dynamics import (
    TrainingDynamicsWriter,
    load_training_dynamics_identity,
)
from utils.device import prefetch_enabled, resolve_device
from utils.platform import select_multiprocessing_context
from utils.precision_policy import (
    create_precision_signature,
    resolve_precision_policy,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HierarchicalBalancedSampler",
    "NativeNonFiniteHalt",
    "NativeTrainer",
    "compute_native_selection_score",
    "forecaster_config_from_dict",
    "forecaster_config_to_dict",
    "load_native_checkpoint",
    "native_loader_kwargs",
    "source_set_manifest_split",
    "temporal_tail_split",
]

#: Legacy projected municipal inputs that native mode must never load
#: (plan section 10.2: superseded for native configs).
_LEGACY_PROJECTED_MARKERS = (
    "hospitalizations_municipality",
    "deaths_municipality",
)

#: Routed native heads supervised in real-data training (plan section 11.1).
_NATIVE_HEADS = ("ww", "cases", "hosp", "deaths")

_SELECTION_METRIC_NAME = "mean_target_wis"

_BEST_CHECKPOINT_NAME = "best_model.pt"
_FINAL_CHECKPOINT_NAME = "final_model.pt"
_CHECKPOINT_SCHEMA_VERSION = 1


def _validation_epoch_record(
    *,
    epoch: int,
    step: int,
    joint_loss: float,
    selection_metric: str,
    selection_score: float,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    """Keep the already-computed validation scores for each completed epoch."""
    record: dict[str, object] = {
        "epoch": epoch,
        "step": step,
        "joint_loss": _optional_float(joint_loss),
        "selection_metric": selection_metric,
        "selection_score": _optional_float(selection_score),
    }
    for head in _NATIVE_HEADS:
        for metric in (
            "nll",
            "crps",
            "wis",
            "wis_dispersion",
            "wis_underprediction",
            "wis_overprediction",
            "wis_point",
        ):
            record[f"{metric}_{head}"] = _optional_float(
                metrics.get(f"{metric}_{head}")
            )
        count = metrics.get(f"observed_count_{head}")
        record[f"observed_count_{head}"] = (
            int(count)
            if isinstance(count, int | float) and not isinstance(count, bool)
            else None
        )
    record["censored_nll_ww"] = _optional_float(metrics.get("censored_nll_ww"))
    censored_count = metrics.get("censored_count_ww")
    record["censored_count_ww"] = (
        int(censored_count)
        if isinstance(censored_count, int | float)
        and not isinstance(censored_count, bool)
        else None
    )
    return record


class NativeNonFiniteHalt(RuntimeError):
    """Raised when a non-finite training loss halts a native run."""


def compute_native_selection_score(
    metrics: Mapping[str, object],
) -> tuple[float, str]:
    """Equal-head WIS over observed routed targets, independent of train loss.

    Heads with no observed targets are excluded. An observed head must have
    a finite WIS; silently dropping it would change the selection objective.
    """
    values: list[float] = []
    for head in _NATIVE_HEADS:
        count = metrics.get(f"observed_count_{head}")
        if count == 0:
            continue
        value = _optional_float(metrics.get(f"wis_{head}"))
        if value is None:
            if isinstance(count, int | float) and count > 0:
                raise ValueError(f"Observed head {head!r} has no finite validation WIS")
            continue
        values.append(value)
    if not values:
        raise ValueError("Native validation requires at least one finite observed WIS")
    return float(np.mean(values)), _SELECTION_METRIC_NAME


def _resolve_worker_count(cfg_workers: int) -> int:
    """Mirror ``dataloader_factory._compute_num_workers`` (-1 means auto)."""
    avail_cores = max(0, (os.cpu_count() or 1) - 1)
    if cfg_workers == -1:
        return avail_cores
    return min(avail_cores, cfg_workers)


def native_loader_kwargs(
    training_config: TrainingParams,
    *,
    workers: int,
    device_type: str,
) -> dict[str, Any]:
    """DataLoader keyword arguments honoring the shared worker policy.

    ``pin_memory`` applies only on CUDA. ``persistent_workers`` and
    ``prefetch_factor`` apply only with a positive worker count. CUDA workers
    use spawn because this trainer starts them after model CUDA initialization.
    """
    num_workers = _resolve_worker_count(workers)
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(training_config.pin_memory and device_type == "cuda"),
    }
    if num_workers > 0:
        kwargs["multiprocessing_context"] = (
            "spawn"
            if device_type == "cuda"
            else select_multiprocessing_context(device_type, all_num_workers_zero=False)
        )
        if training_config.persistent_workers:
            kwargs["persistent_workers"] = True
        if prefetch_enabled(training_config.prefetch_factor):
            kwargs["prefetch_factor"] = training_config.prefetch_factor
    return kwargs


def forecaster_config_to_dict(config: NativeForecasterConfig) -> dict[str, Any]:
    """Serialize a :class:`NativeForecasterConfig` for checkpoint storage."""
    return asdict(config)


def forecaster_config_from_dict(
    payload: Mapping[str, Any],
) -> NativeForecasterConfig:
    """Rebuild a :class:`NativeForecasterConfig` from its stored form."""
    data = dict(payload)
    data["sir_physics"] = SIRPhysicsConfig(**dict(data["sir_physics"]))
    data["observation_heads"] = ObservationHeadConfig(**dict(data["observation_heads"]))
    data["init_weights"] = InitWeightsConfig(**dict(data["init_weights"]))
    return NativeForecasterConfig(**data)


def load_native_checkpoint(
    path: Path,
) -> tuple[NativeEpiForecaster, dict[str, Any]]:
    """Rebuild a native forecaster from a canonical checkpoint.

    The checkpoint is a trusted canonical run artifact written by
    :class:`NativeTrainer`; ``weights_only=False`` is required because the
    record carries nested configuration dictionaries.
    """
    record = torch.load(path, map_location="cpu", weights_only=False)
    model = NativeEpiForecaster(
        forecaster_config_from_dict(record["forecaster_config"])
    )
    model.load_state_dict(record["model_state_dict"])
    model.eval()
    return model, record


class HierarchicalBalancedSampler(Sampler[int]):
    """Choose a head, then a source, then an origin with equal exposure.

    Balanced mode oversamples smaller heads to the largest head's item count.
    Inside each head it round-robins sources, and cycles each source's shuffled
    origins. Thus head counts are equal and source counts within a head differ
    by at most one. Plain mode shuffles every unique item once per epoch.
    """

    def __init__(
        self,
        head_source_items: Mapping[str, Mapping[tuple[str, str], Sequence[int]]],
        *,
        seed: int,
        balanced: bool = True,
        head_order: Sequence[str] = ROUTED_HEADS,
    ) -> None:
        self._indices: list[int] = []
        self._seed = seed
        self._plain = not balanced
        self._epoch = 0
        active_heads = [
            head for head in head_order if any(head_source_items.get(head, {}).values())
        ]
        all_items = [
            index
            for head in active_heads
            for items in head_source_items[head].values()
            for index in items
        ]
        if len(all_items) != len(set(all_items)):
            raise ValueError(
                "hierarchical sampler input contains duplicate item indices"
            )
        if not balanced:
            self._indices = sorted(all_items)
            return
        if not active_heads:
            return

        target_per_head = max(
            sum(len(items) for items in head_source_items[head].values())
            for head in active_heads
        )
        streams: dict[str, list[int]] = {}
        for head in active_heads:
            rng = random.Random(f"{seed}|{head}")
            source_order = sorted(
                source for source, items in head_source_items[head].items() if items
            )
            rng.shuffle(source_order)
            source_items: dict[tuple[str, str], list[int]] = {}
            for source in source_order:
                items = sorted(head_source_items[head][source])
                rng.shuffle(items)
                source_items[source] = items
            stream: list[int] = []
            for step in range(target_per_head):
                source = source_order[step % len(source_order)]
                source_step = step // len(source_order)
                items = source_items[source]
                stream.append(items[source_step % len(items)])
            streams[head] = stream

        for step in range(target_per_head):
            for head in active_heads:
                self._indices.append(streams[head][step])

    def __iter__(self) -> Iterator[int]:
        if self._plain:
            indices = self._indices.copy()
            random.Random(f"{self._seed}|{self._epoch}").shuffle(indices)
            self._epoch += 1
            return iter(indices)
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def temporal_tail_split(
    origins: Sequence[int], val_fraction: float, forecast_horizon: int = 1
) -> tuple[list[int], list[int]]:
    """Split item positions into (train, val) by the last ``val_fraction`` of
    distinct forecast origins.

    At least one distinct origin is reserved for validation when
    ``val_fraction > 0``; validation origins are exactly the largest ones,
    so validation never precedes training in time. Positions with origins
    below the threshold train; the rest validate.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not origins:
        raise ValueError("temporal_tail_split requires a non-empty origin list")
    unique_sorted = sorted(set(origins))
    if val_fraction <= 0.0:
        return list(range(len(origins))), []
    n_val_origins = min(
        len(unique_sorted), max(1, math.ceil(val_fraction * len(unique_sorted)))
    )
    threshold = unique_sorted[-n_val_origins]
    train = [
        i for i, origin in enumerate(origins) if origin + forecast_horizon < threshold
    ]
    val = [i for i, origin in enumerate(origins) if origin >= threshold]
    return train, val


def source_set_manifest_split(
    queries: Sequence[ForecastQuery],
    heads: Sequence[str],
    manifest: SourceSetManifest,
    *,
    run_index: int,
) -> tuple[list[int], list[int], list[int]]:
    """Apply a v2 source-set manifest and its frozen calendar intervals."""
    if len(queries) != len(heads):
        raise ValueError("queries and heads must have the same length")
    if not 0 <= run_index < manifest.num_folds:
        raise ValueError(
            f"native fold index {run_index} outside 0..{manifest.num_folds - 1}"
        )
    inventory = [
        QueryEligibilityRecord(
            source_type=str(query.source_type),
            source_id=str(query.source_id),
            head=head,
            origin=int(query.forecast_origin),
        )
        for query, head in zip(queries, heads, strict=True)
    ]
    validation_start = manifest.validation_start_origin
    test_start = manifest.test_start_origin
    calendar = CalendarSplit(
        train=tuple(
            index
            for index, query in enumerate(inventory)
            if query.origin + manifest.forecast_horizon < validation_start
        ),
        validation=tuple(
            index
            for index, query in enumerate(inventory)
            if validation_start <= query.origin
            and query.origin + manifest.forecast_horizon < test_start
        ),
        test=tuple(
            index for index, query in enumerate(inventory) if query.origin >= test_start
        ),
        validation_start_origin=validation_start,
        test_start_origin=test_start,
        forecast_horizon=manifest.forecast_horizon,
    )
    splits = split_queries_for_run(
        inventory,
        manifest.assignments,
        calendar,
        manifest.run_roles[run_index],
    )
    expected_heads = set(heads)
    for name, positions in zip(("train", "validation", "test"), splits, strict=True):
        present = {heads[position] for position in positions}
        missing = sorted(expected_heads - present)
        if missing:
            raise ValueError(
                f"native source-set run {run_index} {name} split lacks heads: {missing}"
            )
    return splits


class NativeRuntime:
    """Shared native data, model, and evaluation state without training artifacts."""

    def __init__(
        self,
        config: EpiForecasterConfig,
        *,
        model_id: str | None = None,
    ) -> None:
        self.config = config
        self._assert_native_contract(config)

        if config.training.seed is not None:
            random.seed(config.training.seed)
            np.random.seed(config.training.seed)
            torch.manual_seed(config.training.seed)

        self.model_id = model_id or "evaluation"

        self.device = resolve_device(config.training.device)

        self.dataset = native_dataset(config)
        queries = self.dataset.queries
        test_positions: list[int] = []
        self.source_sets: SourceSetManifest | None = None
        if config.native.fold_index is None:
            origins = [query.forecast_origin for query in queries]
            train_positions, val_positions = temporal_tail_split(
                origins, config.training.val_split, config.model.forecast_horizon
            )
        else:
            source_sets = load_source_set_manifest(Path(config.native.fold_manifest))
            if source_sets.provenance != split_identity(config, self.dataset):
                raise ValueError(
                    "Native split asset is stale: processed inputs or eligibility/policy changed; rebuild source sets"
                )
            inventory = [
                QueryEligibilityRecord(
                    str(q.source_type),
                    str(q.source_id),
                    self.dataset.head_at(i),
                    int(q.forecast_origin),
                )
                for i, q in enumerate(queries)
            ]
            if set(source_sets.assignments) != {
                (q.source_type, q.source_id) for q in inventory
            }:
                raise ValueError(
                    "Native split asset does not match the exact eligible source inventory"
                )
            expected_calendar = freeze_calendar_split(
                inventory,
                validation_fraction=config.native.validation_fraction,
                test_fraction=config.native.test_fraction,
                forecast_horizon=config.model.forecast_horizon,
            )
            if (source_sets.validation_start_origin, source_sets.test_start_origin) != (
                expected_calendar.validation_start_origin,
                expected_calendar.test_start_origin,
            ):
                raise ValueError(
                    "Native split calendar differs from its input/policy identity"
                )
            self.source_sets = source_sets
            train_positions, val_positions, test_positions = source_set_manifest_split(
                queries,
                [self.dataset.head_at(i) for i in range(len(queries))],
                source_sets,
                run_index=config.native.fold_index,
            )
        self.split_provenance = (
            self.source_sets.provenance
            if self.source_sets is not None
            else split_identity(config, self.dataset)
        )
        logger.info(
            "Native index: %d items (%s), train %d / val %d / test %d",
            len(self.dataset),
            self.dataset.head_counts,
            len(train_positions),
            len(val_positions),
            len(test_positions),
        )

        sampler = HierarchicalBalancedSampler(
            self._head_source_items(train_positions),
            seed=config.training.seed if config.training.seed is not None else 0,
            balanced=config.native.head_balanced_sampling,
        )
        batch_size = config.training.batch_size
        collate = collate_native_batch
        # Loaders are created before any CUDA tensor operation so worker
        # processes fork from a CUDA-clean parent (plan WP-B item 7).
        train_kwargs = native_loader_kwargs(
            config.training,
            workers=config.training.num_workers,
            device_type=self.device.type,
        )
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate,
            **train_kwargs,
        )
        self.val_loader: DataLoader | None = None
        if val_positions:
            val_kwargs = native_loader_kwargs(
                config.training,
                workers=config.training.val_workers,
                device_type=self.device.type,
            )
            self.val_loader = DataLoader(
                Subset(self.dataset, val_positions),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate,
                **val_kwargs,
            )
        self.test_loader: DataLoader | None = None
        if test_positions:
            test_kwargs = native_loader_kwargs(
                config.training,
                workers=config.training.test_workers,
                device_type=self.device.type,
            )
            self.test_loader = DataLoader(
                Subset(self.dataset, test_positions),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate,
                **test_kwargs,
            )
        if config.training.diagnostic_test_trace_epochs and self.test_loader is None:
            raise ValueError("Diagnostic test trace requires a test split")

        # Adapter widths from a COLLATED batch (review R12 collation): the
        # one-hot source-type block is prepended at collation, so batch hist
        # is [B, L, 5 + F_native] while dataset items stay [L, F_native].
        collated_probe = collate_native_batch([self.dataset[0]])
        self.forecaster_config = NativeForecasterConfig(
            f_native=int(collated_probe.hist.shape[2]),
            d_ctx=int(collated_probe.ctx_feat.shape[2]),
            forecast_horizon=config.model.forecast_horizon,
            sequence_length=config.model.input_window_length,
            head_d_model=config.model.head_d_model,
            head_n_heads=config.model.head_n_heads,
            head_num_layers=config.model.head_num_layers,
            head_dropout=config.model.head_dropout,
            sir_physics=config.model.sir_physics,
            observation_heads=config.model.observation_heads,
            init_weights=config.model.init_weights,
            strict=config.model.strict,
            spatial_encoder=config.native.spatial_encoder,
            mobility_enabled=config.native.mobility_enabled,
            wastewater_enabled=config.native.wastewater_enabled,
            include_vaccination=config.native.include_vaccination,
            hetero_hidden_dim=config.native.hetero_hidden_dim,
            hetero_parameter_budget=config.native.hetero_parameter_budget,
        )
        self.model = NativeEpiForecaster(self.forecaster_config).to(self.device)
        self.criterion = get_native_loss_from_config(
            config.training.loss,
            observation_heads_config=config.model.observation_heads,
            horizon_days=config.model.forecast_horizon,
        )
        self.precision_policy = resolve_precision_policy(config.training, self.device)
        n_train_batches = math.ceil(len(sampler) / batch_size)
        self.steps_per_epoch = (
            min(n_train_batches, config.training.max_batches)
            if config.training.max_batches is not None
            else n_train_batches
        )

    # ------------------------------------------------------------------
    # Guards and identity
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_native_contract(config: EpiForecasterConfig) -> None:
        """Construction-time contract checks (plan sections 12.2, 11.6)."""
        if not config.native.enabled:
            raise ValueError(
                "This worktree requires native.enabled=true and the heterogeneous trainer."
            )
        if config.training.curriculum.enabled:
            raise ValueError(
                "Curriculum mixing is municipal-only; native mode trains "
                "real-data-only (plan section 11.7). Disable "
                "training.curriculum.enabled."
            )
        if (config.training.loss.joint.adaptive_scheme or "").lower() == "gradnorm":
            raise ValueError(
                "GradNorm is not wired for native mode in TRN-1 (documented "
                "follow-up); use training.loss.joint.adaptive_scheme=none."
            )
        if config.training.compile:
            raise ValueError(
                "torch.compile is not wired for native mode; disable training.compile."
            )
        if not config.output.save_best_only:
            raise ValueError(
                "Native checkpointing writes best/final checkpoints only; "
                "output.save_best_only must be true."
            )
        test_epochs = config.training.diagnostic_test_trace_epochs
        if any(
            type(epoch) is not int or not 1 <= epoch <= config.training.epochs
            for epoch in test_epochs
        ) or test_epochs != sorted(set(test_epochs)):
            raise ValueError(
                "training.diagnostic_test_trace_epochs must contain unique, "
                "ascending, 1-based epochs within training.epochs"
            )
        for field_name in ("dataset_path", "real_dataset_path"):
            value = getattr(config.data, field_name)
            for marker in _LEGACY_PROJECTED_MARKERS:
                if marker in value:
                    raise ValueError(
                        f"data.{field_name} references the legacy projected "
                        f"municipal input {marker!r}; native mode never "
                        "loads pre-projected hosp/deaths series."
                    )
        NativeRuntime._assert_no_wandb_config(config)

    @staticmethod
    def _assert_no_wandb_config(config: EpiForecasterConfig) -> None:
        """Reject every meaningful W&B setting instead of ignoring it."""
        output = config.output
        explicit = [
            name
            for name, value in (
                ("wandb_group", output.wandb_group),
                ("wandb_entity", output.wandb_entity),
                ("wandb_tags", output.wandb_tags),
            )
            if value
        ]
        if explicit:
            raise ValueError(
                "NativeTrainer does not implement W&B tracking; remove the "
                f"explicit output field(s) {explicit} from native configs."
            )
        if output.wandb_mode != "disabled":
            raise ValueError(
                "NativeTrainer does not implement W&B tracking; set "
                "output.wandb_mode=disabled in native configs (wandb_project "
                "is inert under disabled mode)."
            )

    @staticmethod
    def _resolve_model_id(override: str | None) -> str:
        """Explicit constructor value, then env, then Slurm, then unique."""
        if override:
            return override
        env_id = os.getenv("EPIFORECASTER_MODEL_ID", "").strip()
        if env_id:
            return env_id
        slurm_job_id = os.getenv("SLURM_JOB_ID", "")
        if slurm_job_id:
            job_name = os.getenv("SLURM_JOB_NAME", "")
            job_qos = os.getenv("SLURM_JOB_QOS", "")
            if job_name == "interactive" or "_interactive" in job_qos:
                return f"run_{time.time_ns()}"
            return slurm_job_id
        return f"run_{time.time_ns()}"

    def _head_source_items(
        self, positions: list[int]
    ) -> dict[str, dict[tuple[str, str], list[int]]]:
        """Training indices grouped by routed head and typed source."""
        items: dict[str, dict[tuple[str, str], list[int]]] = {
            head: {} for head in ROUTED_HEADS
        }
        queries = self.dataset.queries
        for position in positions:
            head = self.dataset.head_at(position)
            query = queries[position]
            source = str(query.source_type), str(query.source_id)
            items[head].setdefault(source, []).append(position)
        return items

    def _evaluate(
        self,
        *,
        loader: DataLoader | None = None,
        split_name: str = "Val",
        granular_output_path: Path | None = None,
        aggregate_output_dir: Path | None = None,
        granular_metadata: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Routing-aware metrics over a validation or test loader."""
        selected_loader = self.val_loader if loader is None else loader
        if selected_loader is None:
            raise ValueError(f"native {split_name} evaluation has no loader")
        loss, metrics, _aggregate = evaluate_native_loader(
            model=self.model,
            loader=selected_loader,
            criterion=self.criterion,
            device=self.device,
            split_name=split_name,
            max_batches=self.config.training.max_batches,
            confirmation_guard=self.config.native.confirmation_guard,
            granular_output_path=granular_output_path,
            granular_eval_format=self.config.output.granular_eval_format,
            aggregate_output_dir=aggregate_output_dir,
            autocast_enabled=self.precision_policy.autocast_enabled,
            autocast_dtype=self.precision_policy.autocast_dtype,
            granular_metadata={
                "experiment_name": self.config.output.experiment_name,
                "model_id": self.model_id,
                "spatial_encoder": self.config.native.spatial_encoder,
                "mobility_enabled": self.config.native.mobility_enabled,
                "od_source": self.config.native.od_source,
                "objective": self.config.training.loss.joint.observation_loss,
                "fold": self.config.native.fold_index,
                "seed": self.config.training.seed,
                "split_seed": self.config.training.split_seed,
                "epochs": self.config.training.epochs,
                "batch_size": self.config.training.batch_size,
                "max_batches": self.config.training.max_batches,
                "optimizer": self.config.training.optimizer,
                "learning_rate": self.config.training.learning_rate,
                "steps_per_epoch": self.steps_per_epoch,
                "fold_manifest": self.config.native.fold_manifest,
                "generalization_claim": (
                    "held-out native sources with frozen future calendar intervals; local pre-origin history remains visible"
                    if self.source_sets is not None
                    else "temporal holdout with visible pre-origin source history"
                ),
                **(granular_metadata or {}),
            },
        )
        if split_name.lower() == "val":
            self.last_val_metrics = metrics
        return loss, metrics


class NativeTrainer(NativeRuntime):
    """Native training lifecycle and checkpoint resume."""

    def __init__(
        self,
        config: EpiForecasterConfig,
        *,
        model_id: str | None = None,
        trial: Any = None,
        pruning_start_epoch: int = 10,
    ) -> None:
        super().__init__(config, model_id=model_id)
        self.trial = trial
        self.pruning_start_epoch = pruning_start_epoch
        self.model_id = self._resolve_model_id(model_id)
        self.experiment_dir = (
            Path(config.output.log_dir) / config.output.experiment_name / self.model_id
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        if config.output.save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=config.training.optimizer,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer_eps=self.precision_policy.optimizer_eps,
            device=self.device,
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=config.training.scheduler_type,
            total_steps=max(1, self.steps_per_epoch * config.training.epochs),
            warmup_steps=config.training.warmup_steps,
            eta_min=config.training.min_learning_rate,
        )
        self.training_dynamics_writer = TrainingDynamicsWriter(
            self.experiment_dir,
            progress_log_frequency=config.training.progress_log_frequency,
            grad_norm_log_frequency=config.training.grad_norm_log_frequency,
            mobility_enabled=config.native.mobility_enabled,
        )
        (self.experiment_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        self.global_step = 0
        self.best_val_selection_score: float | None = None
        self.best_val_selection_metric: str | None = None
        self.best_val_joint_loss: float | None = None
        self.best_val_series_metrics: dict[str, float | None] = {}
        self.best_state: dict[str, torch.Tensor] | None = None
        self.last_val_metrics: dict[str, object] = {}
        self.validation_epoch_trace: list[dict[str, object]] = []
        self.train_epoch_trace: list[dict[str, object]] = []
        self.test_epoch_trace: list[dict[str, object]] = []
        self.start_epoch = 0
        self.resumed_train_trace: list[list[float]] = []
        self.resumed_val_trace: list[float | None] = []
        if config.training.init_checkpoint_path is not None:
            raise ValueError(
                "Native training requires resume_checkpoint_path for continuation"
            )
        if config.training.resume_checkpoint_path is not None:
            self._restore_training_checkpoint(
                Path(config.training.resume_checkpoint_path)
            )

    def _restore_training_checkpoint(self, path: Path) -> None:
        """Resume at an epoch boundary without resetting the optimizer schedule."""
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record["forecaster_config"] != forecaster_config_to_dict(
            self.forecaster_config
        ):
            raise ValueError("Native resume forecaster configuration differs")
        previous = record["config"]
        current = self.config.to_dict()
        if record["split_provenance"] != self.split_provenance:
            raise ValueError("Native resume split input identity differs")
        if record["split_asset"] is not None and self.source_sets is not None:
            if (
                record["split_asset"]["assignments"]
                != (self.source_sets.path.parent / "assignments.parquet").read_bytes()
            ):
                raise ValueError("Native resume source assignments differ")
        for block, locations in (
            ("data", {"dataset_path", "real_dataset_path"}),
            ("native", {"root", "fold_manifest"}),
        ):
            saved = {
                key: value
                for key, value in previous[block].items()
                if key not in locations
            }
            active = {
                key: value
                for key, value in current[block].items()
                if key not in locations
            }
            if saved != active:
                raise ValueError(f"Native resume {block} configuration differs")
        for name in (
            "epochs",
            "max_batches",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "scheduler_type",
            "warmup_steps",
            "min_learning_rate",
            "seed",
            "split_seed",
            "loss",
        ):
            if previous["training"][name] != current["training"][name]:
                raise ValueError(f"Native resume training.{name} differs")
        self.start_epoch = int(record["epoch"]) + 1
        if not 0 < self.start_epoch < self.config.training.epochs:
            raise ValueError("Native resume checkpoint has no remaining epochs")
        if record["best_val_selection_score"] is None or (
            record["validation_epoch_trace"][-1]["selection_score"]
            != record["best_val_selection_score"]
        ):
            raise ValueError(
                "Native continuation requires a best-validation checkpoint"
            )
        completed_batches = record["train_loss_trace"]
        if len(completed_batches) != self.start_epoch or int(
            record["global_step"]
        ) != sum(len(epoch_batches) for epoch_batches in completed_batches):
            raise ValueError("Native resume completed-step history differs")
        if previous["training"]["batch_size"] != self.config.training.batch_size:
            logger.info(
                "Native resume batch size %d -> %d; preserving scheduler state, "
                "new steps per epoch=%d",
                previous["training"]["batch_size"],
                self.config.training.batch_size,
                self.steps_per_epoch,
            )
        if record["precision_signature"] != create_precision_signature(
            self.precision_policy
        ):
            raise ValueError("Native resume precision policy differs")
        self.model.load_state_dict(record["model_state_dict"])
        self.optimizer.load_state_dict(record["optimizer_state_dict"])
        scheduler_state = record["scheduler_state_dict"]
        if (self.scheduler is None) != (scheduler_state is None):
            raise ValueError("Native resume scheduler differs")
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
        self.global_step = int(record["global_step"])
        self.best_val_selection_score = record["best_val_selection_score"]
        self.best_val_selection_metric = record["best_val_selection_metric"]
        self.best_val_joint_loss = record["best_val_joint_loss"]
        self.best_val_series_metrics = dict(record["best_val_series_metrics"])
        self.best_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        self.validation_epoch_trace = list(record["validation_epoch_trace"])
        self.train_epoch_trace = list(record.get("train_epoch_trace", []))
        self.resumed_train_trace = list(record["train_loss_trace"])
        self.resumed_val_trace = list(record["val_joint_loss_trace"])
        self._save_checkpoint(
            self.start_epoch - 1,
            is_best=True,
            train_trace=self.resumed_train_trace,
            val_trace=self.resumed_val_trace,
        )
        logger.info(
            "Native resume: completed epochs=%d global_step=%d; continuing to epoch %d",
            self.start_epoch,
            self.global_step,
            self.config.training.epochs,
        )
        logger.info(
            "Parent checkpoint has no RNG state; dropout and worker RNG restart from config seed"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run the bounded training loop and return its summary record."""
        started = time.perf_counter()
        train_trace: list[list[float]] = list(self.resumed_train_trace)
        val_trace: list[float | None] = list(self.resumed_val_trace)
        try:
            for epoch in range(self.start_epoch, self.config.training.epochs):
                epoch_losses = self._train_epoch(epoch)
                train_trace.append(epoch_losses)
                if self.val_loader is None:
                    val_trace.append(None)
                    continue
                val_loss, metrics = self._evaluate()
                val_trace.append(val_loss)
                selection_score, selection_metric = compute_native_selection_score(
                    metrics
                )
                self.validation_epoch_trace.append(
                    _validation_epoch_record(
                        epoch=epoch + 1,
                        step=self.global_step,
                        joint_loss=val_loss,
                        selection_metric=selection_metric,
                        selection_score=selection_score,
                        metrics=metrics,
                    )
                )
                if self.trial is not None and epoch >= self.pruning_start_epoch:
                    self.trial.report(selection_score, epoch)
                    if self.trial.should_prune():
                        from optuna import TrialPruned

                        raise TrialPruned(f"Native trial pruned at epoch {epoch + 1}")
                if (
                    self.best_val_selection_score is None
                    or selection_score < self.best_val_selection_score
                ):
                    self.best_val_selection_score = selection_score
                    self.best_val_selection_metric = selection_metric
                    self.best_val_joint_loss = val_loss
                    self.best_val_series_metrics = {
                        head: _optional_float(metrics.get(f"wis_{head}"))
                        for head in _NATIVE_HEADS
                    }
                    self.best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    if self.config.output.save_checkpoints:
                        self._save_checkpoint(
                            epoch,
                            is_best=True,
                            train_trace=train_trace,
                            val_trace=val_trace,
                        )
                logger.info(
                    "native epoch %d/%d mean_train_loss=%.6f val_loss=%.6f "
                    "selection=%s=%.6f val_joint_loss=%s",
                    epoch + 1,
                    self.config.training.epochs,
                    sum(epoch_losses) / max(1, len(epoch_losses)),
                    selection_score,
                    selection_metric,
                    selection_score,
                    f"{val_loss:.6f}" if val_loss is not None else "n/a",
                )
                if epoch + 1 in self.config.training.diagnostic_test_trace_epochs:
                    assert self.test_loader is not None
                    diagnostic_test_loss, diagnostic_test_metrics = self._evaluate(
                        loader=self.test_loader,
                        split_name="Test",
                    )
                    diagnostic_score, diagnostic_metric = (
                        compute_native_selection_score(diagnostic_test_metrics)
                    )
                    self.test_epoch_trace.append(
                        _validation_epoch_record(
                            epoch=epoch + 1,
                            step=self.global_step,
                            joint_loss=diagnostic_test_loss,
                            selection_metric=diagnostic_metric,
                            selection_score=diagnostic_score,
                            metrics=diagnostic_test_metrics,
                        )
                    )
            if self.best_state is None:
                # No validation available: the final state is the best state.
                self.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                if self.config.output.save_checkpoints:
                    self._save_checkpoint(
                        self.config.training.epochs - 1,
                        is_best=True,
                        train_trace=train_trace,
                        val_trace=val_trace,
                    )
            if self.config.output.save_checkpoints:
                self._save_checkpoint(
                    self.config.training.epochs - 1,
                    is_best=False,
                    train_trace=train_trace,
                    val_trace=val_trace,
                )
            self.model.load_state_dict(
                {key: value.to(self.device) for key, value in self.best_state.items()}
            )
            parameters = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            artifact_paths: dict[str, str] = {}
            test_loss: float | None = None
            test_metrics: dict[str, Any] = {}
            if (
                self.config.training.run_final_evaluation
                and self.config.output.write_granular_eval
                and self.val_loader is not None
            ):
                granular_path = (
                    self.experiment_dir
                    / self.config.output.resolve_granular_eval_filename(split="val")
                )
                self._evaluate(
                    granular_output_path=granular_path,
                    aggregate_output_dir=self.experiment_dir,
                    granular_metadata={
                        "optimizer_steps": sum(len(epoch) for epoch in train_trace),
                        "model_parameters": parameters,
                        "trainable_parameters": trainable,
                        "wall_time_s_before_final_eval": time.perf_counter() - started,
                        "model_id": self.model_id,
                        "best_val_selection_score": self.best_val_selection_score,
                        "best_val_selection_metric": self.best_val_selection_metric,
                        "precision_signature": create_precision_signature(
                            self.precision_policy
                        ),
                    },
                )
                artifact_paths = {
                    "granular": str(granular_path),
                    "granular_metadata": str(
                        granular_path.with_suffix(f"{granular_path.suffix}.meta.json")
                    ),
                    "aggregate_dir": str(self.experiment_dir),
                }
            if (
                self.config.training.run_final_evaluation
                and self.test_loader is not None
            ):
                test_granular_path = (
                    self.experiment_dir
                    / self.config.output.resolve_granular_eval_filename(split="test")
                    if self.config.output.write_granular_eval
                    else None
                )
                test_loss, test_metrics = self._evaluate(
                    loader=self.test_loader,
                    split_name="Test",
                    granular_output_path=test_granular_path,
                    aggregate_output_dir=(
                        self.experiment_dir
                        if self.config.output.write_granular_eval
                        else None
                    ),
                )
                if test_granular_path is not None:
                    artifact_paths["test_granular"] = str(test_granular_path)
                    artifact_paths["test_granular_metadata"] = str(
                        test_granular_path.with_suffix(
                            f"{test_granular_path.suffix}.meta.json"
                        )
                    )
            if self.config.output.save_checkpoints:
                artifact_paths["checkpoint"] = str(
                    self.checkpoint_dir / _BEST_CHECKPOINT_NAME
                )
                artifact_paths["final_checkpoint"] = str(
                    self.checkpoint_dir / _FINAL_CHECKPOINT_NAME
                )
            self.training_dynamics_writer.finalize(
                "complete",
                summary=self._dynamics_summary(train_trace, val_trace),
            )
        except NativeNonFiniteHalt as exc:
            self.training_dynamics_writer.finalize(
                "nonfinite_halt",
                error_type="NativeNonFiniteHalt",
                error_message=str(exc),
                summary=self._dynamics_summary(train_trace, val_trace),
            )
            raise
        except Exception as exc:
            self.training_dynamics_writer.finalize(
                "pruned" if type(exc).__name__ == "TrialPruned" else "failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                summary=self._dynamics_summary(train_trace, val_trace),
            )
            raise
        return {
            "best_val_loss": self.best_val_selection_score,
            "best_val_selection_score": self.best_val_selection_score,
            "best_val_selection_metric": self.best_val_selection_metric,
            "best_val_joint_loss": self.best_val_joint_loss,
            "best_val_series_metrics": dict(self.best_val_series_metrics),
            "total_epochs": self.config.training.epochs,
            "train_loss_trace": train_trace,
            "train_epoch_trace": list(self.train_epoch_trace),
            **self._validation_history(val_trace),
            "validation_epoch_trace": list(self.validation_epoch_trace),
            "test_epoch_trace": list(self.test_epoch_trace),
            "head_counts": self.dataset.head_counts,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "wall_time_s": time.perf_counter() - started,
            "model_info": {
                "parameters": parameters,
                "trainable_parameters": trainable,
            },
            "model_id": self.model_id,
            "model_root": str(self.experiment_dir),
            "precision_signature": create_precision_signature(self.precision_policy),
            "training_dynamics": load_training_dynamics_identity(self.experiment_dir),
            "evaluation_artifacts": artifact_paths,
        }

    def _train_epoch(self, epoch: int) -> list[float]:
        self.model.train()
        # Keep detached metrics on the device until logging or epoch completion.
        # Double precision preserves the previous host-side weighted arithmetic.
        device_losses: list[torch.Tensor] = []
        metric_totals = torch.zeros(10, dtype=torch.float64, device=self.device)
        max_batches = self.config.training.max_batches
        for step, batch in enumerate(self.train_loader):
            if max_batches is not None and step >= max_batches:
                break
            batch = batch.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.precision_policy.autocast_dtype,
                enabled=self.precision_policy.autocast_enabled,
            ):
                outputs = self.model(batch)
                components = self.criterion(outputs, batch)
            total = components["total"]
            if not torch.isfinite(total):
                raise NativeNonFiniteHalt(
                    f"Non-finite training loss at epoch {epoch} step {step} "
                    f"(total={total.item()}): halting native run. Inspect "
                    "the per-head components and targets."
                )
            scores = torch.stack(
                [
                    components[
                        "ww_uncensored_objective" if head == "ww" else head
                    ].detach()
                    for head in _NATIVE_HEADS
                ]
            ).to(dtype=torch.float64)
            counts = torch.stack(
                [
                    components[
                        "ww_uncensored_count" if head == "ww" else f"{head}_n_eff"
                    ].detach()
                    for head in _NATIVE_HEADS
                ]
            ).to(dtype=torch.float64)
            metric_totals[:4].add_(scores * counts)
            metric_totals[4:8].add_(counts)
            ww_count = components["ww_n_eff"].detach().to(dtype=torch.float64)
            metric_totals[8].add_(
                components["ww"].detach().to(dtype=torch.float64) * ww_count
            )
            metric_totals[9].add_(ww_count)
            self.optimizer.zero_grad()
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip_value
            )
            self.global_step += 1
            if not torch.isfinite(grad_norm):
                # Record the offending observation before halting: the
                # writer stores non-finite values as null plus an explicit
                # nonfinite_fields entry (plan WP-B item 5).
                self.training_dynamics_writer.append(
                    step=self.global_step,
                    epoch=epoch,
                    values={
                        "train_loss": float(total.item()),
                        "grad_norm": float(grad_norm),
                    },
                )
                raise NativeNonFiniteHalt(
                    f"Non-finite gradient norm at epoch {epoch} step {step} "
                    f"(loss={float(total.item()):.6f}, "
                    f"grad_norm={float(grad_norm)}): halting native run "
                    "before the optimizer step."
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            device_losses.append(total.detach())
            is_epoch_last = step + 1 == self.steps_per_epoch
            log_progress = (
                self.global_step % self.config.training.progress_log_frequency == 0
                or is_epoch_last
            )
            if log_progress:
                progress_loss, progress_grad_norm = (
                    torch.stack((total.detach(), grad_norm.detach())).cpu().tolist()
                )
                self.training_dynamics_writer.append(
                    step=self.global_step,
                    epoch=epoch,
                    values={
                        "train_loss": progress_loss,
                        "grad_norm": progress_grad_norm,
                    },
                )
                logger.info(
                    "native epoch %d step %d/%d total_loss=%.6f",
                    epoch + 1,
                    step + 1,
                    self.steps_per_epoch,
                    progress_loss,
                )
        epoch_values = (
            torch.cat(
                [metric_totals]
                + (
                    [torch.stack(device_losses).to(dtype=torch.float64)]
                    if device_losses
                    else []
                )
            )
            .cpu()
            .tolist()
        )
        score_sums = dict(zip(_NATIVE_HEADS, epoch_values[:4], strict=True))
        score_counts = dict(zip(_NATIVE_HEADS, epoch_values[4:8], strict=True))
        ww_mixed_sum, ww_mixed_count = epoch_values[8:10]
        losses = epoch_values[10:]
        self.train_epoch_trace.append(
            {
                "epoch": epoch + 1,
                "step": self.global_step,
                "batch_size": self.config.training.batch_size,
                "objective": self.config.training.loss.joint.observation_loss,
                "joint_loss": float(np.mean(losses)) if losses else None,
                "ww_mixed_objective": (
                    ww_mixed_sum / ww_mixed_count if ww_mixed_count > 0 else None
                ),
                "ww_censored_count": ww_mixed_count - score_counts["ww"],
                "ww_uncensored_count": score_counts["ww"],
                **{
                    f"{self.config.training.loss.joint.observation_loss}_{head}": (
                        score_sums[head] / score_counts[head]
                        if score_counts[head] > 0
                        else None
                    )
                    for head in _NATIVE_HEADS
                },
                **{
                    f"observed_count_{head}": (
                        ww_mixed_count if head == "ww" else score_counts[head]
                    )
                    for head in _NATIVE_HEADS
                },
            }
        )
        return losses

    @torch.no_grad()
    def _save_checkpoint(
        self,
        epoch: int,
        *,
        is_best: bool,
        train_trace: list[list[float]] | None = None,
        val_trace: list[float | None] | None = None,
    ) -> Path:
        """Write the canonical checkpoint dictionary (never a bare state dict)."""
        if not self.config.output.save_checkpoints:
            raise RuntimeError(
                "_save_checkpoint requires output.save_checkpoints=true."
            )
        checkpoint = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "model_id": self.model_id,
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "config": self.config.to_dict(),
            "forecaster_config": forecaster_config_to_dict(self.forecaster_config),
            "precision_signature": create_precision_signature(self.precision_policy),
            "split_provenance": self.split_provenance,
            "split_asset": (
                {
                    "manifest": self.source_sets.path.read_bytes(),
                    "assignments": (
                        self.source_sets.path.parent / "assignments.parquet"
                    ).read_bytes(),
                }
                if self.source_sets is not None
                else None
            ),
            "best_val_selection_score": self.best_val_selection_score,
            "best_val_selection_metric": self.best_val_selection_metric,
            "best_val_joint_loss": self.best_val_joint_loss,
            "best_val_series_metrics": dict(self.best_val_series_metrics),
            "train_loss_trace": train_trace or [],
            "train_epoch_trace": list(self.train_epoch_trace),
            **self._validation_history(val_trace or []),
            "validation_epoch_trace": list(self.validation_epoch_trace),
            "head_counts": self.dataset.head_counts,
        }
        filename = _BEST_CHECKPOINT_NAME if is_best else _FINAL_CHECKPOINT_NAME
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        return path

    def _validation_history(self, joint_trace: list[float | None]) -> dict[str, object]:
        """Label the primary WIS history and the joint objective separately."""
        by_epoch = {
            row["epoch"]: row["selection_score"] for row in self.validation_epoch_trace
        }
        return {
            "val_loss_metric": _SELECTION_METRIC_NAME,
            "val_loss_trace": [
                by_epoch.get(epoch + 1) for epoch in range(len(joint_trace))
            ],
            "val_joint_loss_trace": joint_trace,
        }

    def _dynamics_summary(
        self,
        train_trace: list[list[float]],
        val_trace: list[float | None],
    ) -> dict[str, object]:
        """Terminal lifecycle evidence without checkpoint deserialization."""
        return {
            "total_epochs": self.config.training.epochs,
            "global_step": self.global_step,
            "train_loss_trace": train_trace,
            "train_epoch_trace": list(self.train_epoch_trace),
            **self._validation_history(val_trace),
            "validation_epoch_trace": list(self.validation_epoch_trace),
            "test_epoch_trace": list(self.test_epoch_trace),
            "best_val_selection_score": _optional_float(self.best_val_selection_score),
            "best_val_selection_metric": self.best_val_selection_metric,
            "best_val_joint_loss": _optional_float(self.best_val_joint_loss),
            "best_val_series_metrics": {
                head: _optional_float(value)
                for head, value in self.best_val_series_metrics.items()
            },
        }


def _optional_float(value: object) -> float | None:
    """Coerce to a finite float or ``None`` (JSON-safe for manifests)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None
