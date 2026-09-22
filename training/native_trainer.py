"""Production trainer for native-support training (slice TRN-1, WP-B).

Trainer for the native-support path (plan sections 12.2 TRN-1, 11.6, and
12.2d WP-B): it wires :class:`~data.native_dataset.NativeSupportDataset`
through ``collate_native_batch`` into a
:class:`~models.native_forecaster.NativeEpiForecaster` and the routed
:class:`~evaluation.losses.NativeJointInferenceLoss`.

Every run owns a canonical model root
``log_dir / experiment_name / model_id`` — the same layout the municipal
:class:`~training.epiforecaster_trainer.EpiForecasterTrainer` and the
immutable MN5 workflow's ``run_support.train_model`` contract use. All
run-owned artifacts live there: ``checkpoints/best_model.pt`` (canonical
checkpoint dictionary, never a bare state dict), the final-epoch
checkpoint, ``training_dynamics.jsonl`` plus its manifest, and the
final-best-state granular/aggregate evaluation exports. No fold or seed
may write into the shared ``log_dir / experiment_name`` directory.

Best state is selected by the three-head validation-selection score
(CRPS mean, NLL fallback, transformed-rate MAE final fallback — the
municipal chain over native metric keys), not by joint validation loss
alone; the joint loss is recorded separately.

Precision follows the shared policy: float32 parameters with optional
CUDA BF16 autocast. FP16 weights and ``GradScaler`` are intentionally
not supported. Evaluation runs under the same autocast policy.

Still deliberately not ported from the municipal trainer: W&B
(explicitly rejected, not silently ignored), GradNorm, curriculum
mixing, torch.compile, profiler, and mid-training resume. The municipal
path in ``dataloader_factory``/``epiforecaster_trainer`` is untouched —
the native branch lives here and is selected additively in ``cli.py``
when ``config.native.enabled``.

Sampling. With ``native.head_balanced_sampling`` the training order is a
deterministic round-robin interleave of per-head item indices under the
config seed (one item from each routed head in turn, ``cases``/``hosp``/
``deaths`` rotation); with the flag off, a plain seeded shuffle. Validation
uses the temporal tail split: the last ``training.val_split`` fraction of
distinct forecast origins (at least one), so validation never precedes
training in time (plan section 7). Stage-4 configs also select one audited
source fold: training uses assigned sources outside the fold and validation
uses held-out sources inside the fold on the temporal tail.

Bounds. ``training.max_batches`` caps both the per-epoch training pass and
the validation pass, keeping the bounded smoke run CPU-runnable in minutes.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from data.native_batch import ROUTED_HEADS, collate_native_batch
from data.native_crossval import NativeFold, load_native_fold_manifest
from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
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
    "HeadBalancedSampler",
    "NativeNonFiniteHalt",
    "NativeTrainer",
    "compute_native_selection_score",
    "forecaster_config_from_dict",
    "forecaster_config_to_dict",
    "load_native_checkpoint",
    "native_loader_kwargs",
    "source_fold_temporal_split",
    "temporal_tail_split",
]

#: Legacy projected municipal inputs that native mode must never load
#: (plan section 10.2: superseded for native configs).
_LEGACY_PROJECTED_MARKERS = (
    "hospitalizations_municipality",
    "deaths_municipality",
)

#: Routed native heads supervised in real-data training (plan section 11.1).
_NATIVE_HEADS = ("cases", "hosp", "deaths")

_SELECTION_METRIC_NAME = "mean_target_mae"
_CRPS_SELECTION_METRIC_NAME = "mean_target_crps"
_NLL_SELECTION_METRIC_NAME = "mean_target_nll"

_BEST_CHECKPOINT_NAME = "best_model.pt"
_FINAL_CHECKPOINT_NAME = "final_model.pt"
_CHECKPOINT_SCHEMA_VERSION = 1


class NativeNonFiniteHalt(RuntimeError):
    """Raised when a non-finite training loss halts a native run."""


def _finite_metric_mean(metrics: Mapping[str, Any], keys: Sequence[str]) -> float:
    """Mean over finite values among ``keys``; ``inf`` when none qualify."""
    values: list[float] = []
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        value_float = float(value)
        if math.isfinite(value_float):
            values.append(value_float)
    if not values:
        return float("inf")
    return float(np.mean(values))


def compute_native_selection_score(
    metrics: Mapping[str, object],
    *,
    observation_loss: str,
) -> tuple[float, str]:
    """Three-head validation-selection score (municipal chain, native keys).

    CRPS mean over finite routed heads first, NLL mean when the objective
    is NLL and CRPS is unavailable, transformed-rate MAE mean as the final
    fallback. Returns the score and the selected metric name.
    """
    selection_score = _finite_metric_mean(
        metrics, tuple(f"mae_{head}_log1p_per_100k" for head in _NATIVE_HEADS)
    )
    selection_metric = _SELECTION_METRIC_NAME
    if observation_loss in {"nll", "crps"}:
        crps_score = _finite_metric_mean(
            metrics, tuple(f"crps_{head}" for head in _NATIVE_HEADS)
        )
        if math.isfinite(crps_score):
            selection_score = crps_score
            selection_metric = _CRPS_SELECTION_METRIC_NAME
        elif observation_loss == "nll":
            nll_score = _finite_metric_mean(
                metrics, tuple(f"nll_{head}" for head in _NATIVE_HEADS)
            )
            if math.isfinite(nll_score):
                selection_score = nll_score
                selection_metric = _NLL_SELECTION_METRIC_NAME
    return selection_score, selection_metric


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
    ``prefetch_factor`` apply only with a positive worker count; the
    multiprocessing context is selected only when workers will fork.
    """
    num_workers = _resolve_worker_count(workers)
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(training_config.pin_memory and device_type == "cuda"),
    }
    if num_workers > 0:
        kwargs["multiprocessing_context"] = select_multiprocessing_context(
            device_type, all_num_workers_zero=False
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


class HeadBalancedSampler(Sampler[int]):
    """Deterministic round-robin interleave of per-head dataset indices.

    Each head's index list is shuffled under ``seed``, then items are
    yielded one per head in ``head_order`` rotation until all queues drain.
    Heads keep equal exposure per batch window regardless of their item
    counts (plan section 7: log sampling frequencies per head). With
    ``balanced=False`` the sampler is a plain seeded shuffle of the union.
    """

    def __init__(
        self,
        head_items: Mapping[str, Sequence[int]],
        *,
        seed: int,
        balanced: bool = True,
        head_order: Sequence[str] = ROUTED_HEADS,
    ) -> None:
        self._rng = random.Random(seed)
        self._head_order = [head for head in head_order if head_items.get(head)]
        self._indices: list[int] = []
        if balanced:
            queues: dict[str, deque[int]] = {}
            for head in self._head_order:
                items = sorted(head_items[head])
                self._rng.shuffle(items)
                queues[head] = deque(items)
            while any(queues.values()):
                for head in self._head_order:
                    if queues[head]:
                        self._indices.append(queues[head].popleft())
        else:
            union = [index for head in self._head_order for index in head_items[head]]
            self._indices = self._rng.sample(sorted(union), len(union))

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def temporal_tail_split(
    origins: Sequence[int], val_fraction: float
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
    train = [i for i, origin in enumerate(origins) if origin < threshold]
    val = [i for i, origin in enumerate(origins) if origin >= threshold]
    return train, val


def _fold_source_keys(fold: NativeFold) -> set[tuple[str, str]]:
    """Return typed source identities for one audited native fold."""
    return {
        *(("municipality", source_id) for source_id in fold.municipalities),
        *(("abs", source_id) for source_id in fold.abs),
        *(("comarca", source_id) for source_id in fold.comarcas),
    }


def source_fold_temporal_split(
    queries: Sequence[ForecastQuery],
    *,
    val_fraction: float,
    assigned_sources: set[tuple[str, str]],
    held_out_sources: set[tuple[str, str]],
) -> tuple[list[int], list[int]]:
    """Combine an audited source holdout with a forward temporal tail.

    Unassigned sources are excluded. Training uses assigned, non-held-out
    sources before the tail. Validation uses held-out sources in the tail.
    Local pre-origin history remains visible, matching the frozen v1
    generalization claim.
    """
    origins = [query.forecast_origin for query in queries]
    temporal_train, temporal_val = temporal_tail_split(origins, val_fraction)

    def key(position: int) -> tuple[str, str]:
        query = queries[position]
        return str(query.source_type), str(query.source_id)

    train = [
        position
        for position in temporal_train
        if key(position) in assigned_sources and key(position) not in held_out_sources
    ]
    val = [position for position in temporal_val if key(position) in held_out_sources]
    if not train:
        raise ValueError("native source-fold split produced no training queries")
    if not val:
        raise ValueError("native source-fold split produced no validation queries")
    return train, val


class NativeTrainer:
    """Production trainer over the native-support path."""

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

        self.model_id = self._resolve_model_id(model_id)
        self.experiment_dir = (
            Path(config.output.log_dir) / config.output.experiment_name / self.model_id
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        if config.output.save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(config.training.device)

        self.dataset = NativeSupportDataset(
            native_root=config.native.root,
            real_with_id_path=config.data.dataset_path,
            config=NativeDatasetConfig(
                input_window_length=config.model.input_window_length,
                forecast_horizon=config.model.forecast_horizon,
                window_stride=config.data.window_stride,
                min_history_observed_fraction=(
                    config.data.min_history_observed_fraction
                ),
                min_horizon_observed_fraction=(
                    config.data.min_horizon_observed_fraction
                ),
                confirmation_guard=config.native.confirmation_guard,
                spatial_encoder=config.native.spatial_encoder,
                od_source=config.native.od_source,
                od_min_flow_raw=config.native.od_min_flow_raw,
                od_self_flows=config.native.od_self_flows,
                mobility_enabled=config.native.mobility_enabled,
                include_od_context=True,
            ),
        )
        queries = self.dataset.queries
        if config.native.fold_index is None:
            origins = [query.forecast_origin for query in queries]
            train_positions, val_positions = temporal_tail_split(
                origins, config.training.val_split
            )
        else:
            manifest = load_native_fold_manifest(Path(config.native.fold_manifest))
            if config.native.fold_index >= manifest.num_folds:
                raise ValueError(
                    "native.fold_index outside manifest range: "
                    f"{config.native.fold_index} >= {manifest.num_folds}"
                )
            assigned_sources = set().union(
                *(_fold_source_keys(fold) for fold in manifest.folds)
            )
            held_out_sources = _fold_source_keys(
                manifest.folds[config.native.fold_index]
            )
            train_positions, val_positions = source_fold_temporal_split(
                queries,
                val_fraction=config.training.val_split,
                assigned_sources=assigned_sources,
                held_out_sources=held_out_sources,
            )
        logger.info(
            "Native index: %d items (%s), train %d / val %d",
            len(self.dataset),
            self.dataset.head_counts,
            len(train_positions),
            len(val_positions),
        )

        sampler = HeadBalancedSampler(
            self._head_items(train_positions),
            seed=config.training.seed if config.training.seed is not None else 0,
            balanced=config.native.head_balanced_sampling,
        )
        batch_size = config.training.batch_size
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
            collate_fn=collate_native_batch,
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
                collate_fn=collate_native_batch,
                **val_kwargs,
            )

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
        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=config.training.optimizer,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer_eps=self.precision_policy.optimizer_eps,
            device=self.device,
        )
        n_train_batches = math.ceil(len(sampler) / batch_size)
        self.steps_per_epoch = (
            min(n_train_batches, config.training.max_batches)
            if config.training.max_batches is not None
            else n_train_batches
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
        self.global_step = 0
        self.best_val_selection_score: float | None = None
        self.best_val_selection_metric: str | None = None
        self.best_val_joint_loss: float | None = None
        self.best_val_series_metrics: dict[str, float | None] = {}
        self.best_state: dict[str, torch.Tensor] | None = None
        self.last_val_metrics: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Guards and identity
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_native_contract(config: EpiForecasterConfig) -> None:
        """Construction-time contract checks (plan sections 12.2, 11.6)."""
        if not config.native.enabled:
            raise ValueError(
                "NativeTrainer requires native.enabled=true; the municipal "
                "path is EpiForecasterTrainer."
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
        for field_name in ("dataset_path", "real_dataset_path"):
            value = getattr(config.data, field_name)
            for marker in _LEGACY_PROJECTED_MARKERS:
                if marker in value:
                    raise ValueError(
                        f"data.{field_name} references the legacy projected "
                        f"municipal input {marker!r}; native mode never "
                        "loads pre-projected hosp/deaths series."
                    )
        NativeTrainer._assert_no_wandb_config(config)

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

    def _head_items(self, positions: list[int]) -> dict[str, list[int]]:
        items: dict[str, list[int]] = {head: [] for head in ROUTED_HEADS}
        for position in positions:
            items[self.dataset.head_at(position)].append(position)
        return items

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run the bounded training loop and return its summary record."""
        started = time.perf_counter()
        train_trace: list[list[float]] = []
        val_trace: list[float | None] = []
        try:
            for epoch in range(self.config.training.epochs):
                epoch_losses = self._train_epoch(epoch)
                train_trace.append(epoch_losses)
                if self.val_loader is None:
                    val_trace.append(None)
                    continue
                val_loss, metrics = self._evaluate()
                val_trace.append(val_loss)
                selection_score, selection_metric = compute_native_selection_score(
                    metrics,
                    observation_loss=(self.config.training.loss.joint.observation_loss),
                )
                if (
                    self.best_val_selection_score is None
                    or selection_score < self.best_val_selection_score
                ):
                    self.best_val_selection_score = selection_score
                    self.best_val_selection_metric = selection_metric
                    self.best_val_joint_loss = val_loss
                    self.best_val_series_metrics = {
                        head: _optional_float(metrics.get(f"crps_{head}"))
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
                    "native epoch %d/%d mean_train_loss=%.6f val_loss=%s "
                    "selection=%s=%.6f",
                    epoch + 1,
                    self.config.training.epochs,
                    sum(epoch_losses) / max(1, len(epoch_losses)),
                    f"{val_loss:.6f}" if val_loss is not None else "n/a",
                    selection_metric,
                    selection_score,
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
            if self.config.output.write_granular_eval and self.val_loader is not None:
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
                "failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                summary=self._dynamics_summary(train_trace, val_trace),
            )
            raise
        return {
            "best_val_loss": self.best_val_joint_loss,
            "best_val_selection_score": self.best_val_selection_score,
            "best_val_selection_metric": self.best_val_selection_metric,
            "best_val_joint_loss": self.best_val_joint_loss,
            "best_val_series_metrics": dict(self.best_val_series_metrics),
            "total_epochs": self.config.training.epochs,
            "train_loss_trace": train_trace,
            "val_loss_trace": val_trace,
            "head_counts": self.dataset.head_counts,
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
        losses: list[float] = []
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
            losses.append(float(total.item()))
            is_epoch_last = step + 1 == self.steps_per_epoch
            if (
                self.global_step % self.config.training.progress_log_frequency == 0
                or is_epoch_last
            ):
                self.training_dynamics_writer.append(
                    step=self.global_step,
                    epoch=epoch,
                    values={
                        "train_loss": losses[-1],
                        "grad_norm": float(grad_norm),
                    },
                )
            logger.info(
                "native epoch %d step %d/%d total_loss=%.6f",
                epoch + 1,
                step + 1,
                self.steps_per_epoch,
                losses[-1],
            )
        return losses

    @torch.no_grad()
    def _evaluate(
        self,
        *,
        granular_output_path: Path | None = None,
        aggregate_output_dir: Path | None = None,
        granular_metadata: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Routing-aware EVL-1 metrics over the capped validation tail."""
        assert self.val_loader is not None
        loss, metrics, _aggregate = evaluate_native_loader(
            model=self.model,
            loader=self.val_loader,
            criterion=self.criterion,
            device=self.device,
            split_name="Val",
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
                    "held-out source, visible context history "
                    "(temporal-forecasting mode)"
                ),
                **(granular_metadata or {}),
            },
        )
        self.last_val_metrics = metrics
        return loss, metrics

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
            "best_val_selection_score": self.best_val_selection_score,
            "best_val_selection_metric": self.best_val_selection_metric,
            "best_val_joint_loss": self.best_val_joint_loss,
            "best_val_series_metrics": dict(self.best_val_series_metrics),
            "train_loss_trace": train_trace or [],
            "val_loss_trace": val_trace or [],
            "head_counts": self.dataset.head_counts,
        }
        filename = _BEST_CHECKPOINT_NAME if is_best else _FINAL_CHECKPOINT_NAME
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        return path

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
            "val_loss_trace": val_trace,
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
