"""Slim smoke-grade trainer for native-support training (slice TRN-1).

Bounded local acceptance trainer for the native-support path (plan sections
12.2 TRN-1, 11.6): it wires :class:`~data.native_dataset.NativeSupportDataset`
through ``collate_native_batch`` into a :class:`~models.native_forecaster.NativeEpiForecaster`
and the routed :class:`~evaluation.losses.NativeJointInferenceLoss`.

Deliberately NOT a port of the municipal :class:`~training.epiforecaster_trainer.EpiForecasterTrainer`:
no W&B, no checkpoint files, no GradNorm controller (documented follow-up),
no curriculum, no torch.compile, no profiler. The municipal path in
``dataloader_factory``/``epiforecaster_trainer`` is untouched — the native
branch lives here and is selected additively in ``cli.py`` when
``config.native.enabled``.

Sampling. With ``native.head_balanced_sampling`` the training order is a
deterministic round-robin interleave of per-head item indices under the
config seed (one item from each routed head in turn, ``cases``/``hosp``/
``deaths`` rotation); with the flag off, a plain seeded shuffle. Validation
uses the temporal tail split: the last ``training.val_split`` fraction of
distinct forecast origins (at least one), so validation never precedes
training in time (plan section 7).

Bounds. ``training.max_batches`` caps both the per-epoch training pass and
the validation pass, keeping the smoke run CPU-runnable in minutes.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from data.native_batch import ROUTED_HEADS, collate_native_batch
from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
from evaluation.losses import get_native_loss_from_config
from models.configs import EpiForecasterConfig
from models.native_forecaster import NativeEpiForecaster, NativeForecasterConfig
from training.optim_factory import create_optimizer, create_scheduler
from utils.device import resolve_device

logger = logging.getLogger(__name__)

__all__ = ["HeadBalancedSampler", "NativeTrainer", "temporal_tail_split"]

#: Legacy projected municipal inputs that native mode must never load
#: (plan section 10.2: superseded for native configs).
_LEGACY_PROJECTED_MARKERS = (
    "hospitalizations_municipality",
    "deaths_municipality",
)


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
        head_items: dict[str, Sequence[int]],
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


class NativeTrainer:
    """Bounded smoke trainer over the native-support path."""

    def __init__(self, config: EpiForecasterConfig) -> None:
        self.config = config
        self._assert_native_contract(config)

        if config.training.seed is not None:
            random.seed(config.training.seed)
            np.random.seed(config.training.seed)
            torch.manual_seed(config.training.seed)

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
            ),
        )
        origins = [query.forecast_origin for query in self.dataset.queries]
        train_positions, val_positions = temporal_tail_split(
            origins, config.training.val_split
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
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_native_batch,
            num_workers=0,
        )
        self.val_loader: DataLoader | None = None
        if val_positions:
            self.val_loader = DataLoader(
                Subset(self.dataset, val_positions),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_native_batch,
                num_workers=0,
            )

        first = self.dataset[0]
        forecaster_config = NativeForecasterConfig(
            f_native=int(first["hist"].shape[1]),
            d_ctx=int(first["ctx_feat"].shape[1]),
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
        )
        self.model = NativeEpiForecaster(forecaster_config).to(self.device)
        self.criterion = get_native_loss_from_config(
            config.training.loss,
            observation_heads_config=config.model.observation_heads,
            horizon_days=config.model.forecast_horizon,
        )
        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=config.training.optimizer,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer_eps=(
                config.training.optimizer_eps
                if config.training.optimizer_eps is not None
                else 1.0e-8
            ),
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
        #: Optional in-memory best state (the only "checkpointing"; never
        #: written to disk).
        self.best_val_loss: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Guards
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
        for field_name in ("dataset_path", "real_dataset_path"):
            value = getattr(config.data, field_name)
            for marker in _LEGACY_PROJECTED_MARKERS:
                if marker in value:
                    raise ValueError(
                        f"data.{field_name} references the legacy projected "
                        f"municipal input {marker!r}; native mode never "
                        "loads pre-projected hosp/deaths series."
                    )

    def _head_items(self, positions: list[int]) -> dict[str, list[int]]:
        items: dict[str, list[int]] = {head: [] for head in ROUTED_HEADS}
        for position in positions:
            items[self.dataset[position]["head"]].append(position)
        return items

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run the bounded smoke training loop and return a summary."""
        started = time.perf_counter()
        train_trace: list[list[float]] = []
        val_trace: list[float | None] = []
        for epoch in range(self.config.training.epochs):
            epoch_losses = self._train_epoch(epoch)
            train_trace.append(epoch_losses)
            val_loss = self._evaluate() if self.val_loader is not None else None
            val_trace.append(val_loss)
            if val_loss is not None and (
                self.best_val_loss is None or val_loss < self.best_val_loss
            ):
                self.best_val_loss = val_loss
                self.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            logger.info(
                "native epoch %d/%d mean_train_loss=%.6f val_loss=%s",
                epoch + 1,
                self.config.training.epochs,
                sum(epoch_losses) / max(1, len(epoch_losses)),
                f"{val_loss:.6f}" if val_loss is not None else "n/a",
            )
        if self.best_state is not None:
            self.model.load_state_dict(
                {key: value.to(self.device) for key, value in self.best_state.items()}
            )
        parameters = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.config.training.epochs,
            "train_loss_trace": train_trace,
            "val_loss_trace": val_trace,
            "head_counts": self.dataset.head_counts,
            "wall_time_s": time.perf_counter() - started,
            "model_info": {
                "parameters": parameters,
                "trainable_parameters": trainable,
            },
        }

    def _train_epoch(self, epoch: int) -> list[float]:
        self.model.train()
        losses: list[float] = []
        max_batches = self.config.training.max_batches
        for step, batch in enumerate(self.train_loader):
            if max_batches is not None and step >= max_batches:
                break
            batch = batch.to(self.device)
            outputs = self.model(batch)
            components = self.criterion(outputs, batch)
            total = components["total"]
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch} step {step} "
                    f"(total={total.item()}): halting native smoke run. "
                    "Inspect the per-head components and targets."
                )
            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip_value
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            losses.append(float(total.item()))
            logger.info(
                "native epoch %d step %d/%d total_loss=%.6f",
                epoch + 1,
                step + 1,
                self.steps_per_epoch,
                losses[-1],
            )
        return losses

    @torch.no_grad()
    def _evaluate(self) -> float | None:
        """Mean batch loss over the (``max_batches``-capped) val tail."""
        assert self.val_loader is not None
        self.model.eval()
        totals: list[float] = []
        max_batches = self.config.training.max_batches
        for step, batch in enumerate(self.val_loader):
            if max_batches is not None and step >= max_batches:
                break
            batch = batch.to(self.device)
            outputs = self.model(batch)
            components = self.criterion(outputs, batch)
            total = components["total"]
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite validation loss at val step {step} "
                    f"(total={total.item()})."
                )
            totals.append(float(total.item()))
        return sum(totals) / len(totals) if totals else None
