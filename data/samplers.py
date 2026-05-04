import logging
import math
import random
from dataclasses import dataclass
from typing import Iterator

from torch.utils.data import BatchSampler, ConcatDataset

from models.configs import CurriculumConfig

logger = logging.getLogger(__name__)


@dataclass
class CurriculumState:
    """State of the curriculum for the current epoch."""
    epoch: int = 0
    synth_ratio: float = 0.0
    mode: str = "time_major"  # 'time_major' vs 'node_major'
    active_runs: int = 1
    # Sparsity bounds for filtering synthetic runs
    min_sparsity: float | None = None
    max_sparsity: float | None = None


class ShuffledBatchSampler(BatchSampler):
    """Shuffle batch order each epoch while preserving contiguous in-batch indices.

    This keeps `EpiDataset.sample_ordering` semantics *within* each mini-batch
    and randomizes only the order in which those mini-batches are seen.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        if dataset_size < 0:
            raise ValueError(f"dataset_size must be >= 0, got {dataset_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        full_batches = self.dataset_size // self.batch_size
        has_tail = (self.dataset_size % self.batch_size) > 0 and not self.drop_last

        batch_starts = [i * self.batch_size for i in range(full_batches)]
        if has_tail:
            batch_starts.append(full_batches * self.batch_size)

        if self.seed is not None:
            rng = random.Random(self.seed + self._epoch)
        else:
            rng = random.Random()
        rng.shuffle(batch_starts)

        for start in batch_starts:
            stop = min(start + self.batch_size, self.dataset_size)
            batch = list(range(start, stop))
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

        self._epoch += 1

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)


class EpidemicCurriculumSampler(BatchSampler):
    """
    Curriculum sampler that mixes synthetic and real data.
    
    It manages a mixture of "Real" and "Synthetic" data streams, adjusting the
    ratio and sampling mode (Time-Major vs Node-Major) according to a schedule.
    
    To preserve data locality (especially for Zarr/Mobility access), it samples
    indices in contiguous "chunks" from each run before switching.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int,
        config: CurriculumConfig,
        drop_last: bool = False,
        real_run_id: str = "real",
        seed: int | None = None,
    ):
        """
        Args:
            dataset: ConcatDataset containing [RealDataset, SynthRun1, SynthRun2, ...]
                     It is expected that the first dataset is Real, or we identify them by run_id.
                     Actually, we assume the provided datasets are tagged.
            batch_size: Size of mini-batches.
            config: CurriculumConfig object.
            drop_last: Whether to drop the last incomplete batch.
            real_run_id: The run_id that identifies the "real" dataset.
        """
        # We don't call super().__init__ because we override __iter__ completely
        # and don't need the default sampler behavior.
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = config
        self.drop_last = drop_last
        self.real_run_id = real_run_id.strip()
        self.seed = seed

        self.state = CurriculumState()

        # Analyze the ConcatDataset to identify runs
        self._analyze_datasets()

    def _analyze_datasets(self):
        """Identify which sub-datasets are real and which are synthetic.

        Also builds a mapping from dataset index to sparsity level for
        sparsity-based curriculum filtering.
        """
        self.real_dataset_indices = []
        self.synth_dataset_indices = []
        # Map dataset_idx → sparsity level (from processed dataset)
        self._dataset_sparsity: dict[int, float] = {}

        if not isinstance(self.dataset, ConcatDataset):
            # Fallback for single dataset (treated as real)
            self.real_dataset_indices = [0]
            self.dataset_offsets = [0]
            self.cumulative_sizes = [len(self.dataset)]
            self.sub_datasets = [self.dataset]
            return

        self.sub_datasets = self.dataset.datasets
        self.cumulative_sizes = self.dataset.cumulative_sizes
        self.dataset_offsets = [0] + self.cumulative_sizes[:-1]

        for i, ds in enumerate(self.sub_datasets):
            # Check run_id to distinguish real vs synthetic
            # We assume 'real' run_id is "real" or similar
            # If explicit run_id attribute exists, use it.
            run_id = getattr(ds, "run_id", "real")
            run_id_str = str(run_id).strip()

            # Extract sparsity from dataset's precomputed sparsity_level attribute
            if hasattr(ds, 'sparsity_level') and ds.sparsity_level is not None:
                self._dataset_sparsity[i] = ds.sparsity_level

            logger.info(
                f"Sampler Dataset {i}: run_id='{run_id}', "
                f"sparsity={self._dataset_sparsity.get(i, 'N/A')}"
            )

            if run_id_str == self.real_run_id:
                self.real_dataset_indices.append(i)
            else:
                self.synth_dataset_indices.append(i)

        logger.info(
            f"Curriculum Sampler found {len(self.real_dataset_indices)} real datasets "
            f"and {len(self.synth_dataset_indices)} synthetic datasets."
        )

        # Log sparsity distribution
        if self._dataset_sparsity:
            sparsity_values = list(self._dataset_sparsity.values())
            logger.info(
                f"Sparsity levels: min={min(sparsity_values):.2f}, "
                f"max={max(sparsity_values):.2f}, "
                f"num_runs={len(sparsity_values)}"
            )

    def _filter_runs_by_sparsity(
        self,
        run_indices: list[int],
        min_sparsity: float | None,
        max_sparsity: float | None,
    ) -> list[int]:
        """Filter run indices by sparsity range.

        Args:
            run_indices: List of dataset indices to filter.
            min_sparsity: Minimum sparsity threshold (inclusive), or None for no minimum.
            max_sparsity: Maximum sparsity threshold (inclusive), or None for no maximum.

        Returns:
            Filtered list of dataset indices whose sparsity falls within the range.
            Runs with unknown sparsity are excluded when filtering is requested.
        """
        if min_sparsity is None and max_sparsity is None:
            return run_indices

        filtered = []
        for idx in run_indices:
            sparsity = self._dataset_sparsity.get(idx)
            if sparsity is None:
                # Unknown sparsity - exclude when filtering is requested
                # (only include runs with known sparsity in range)
                continue
            if (min_sparsity is None or sparsity >= min_sparsity) and \
               (max_sparsity is None or sparsity <= max_sparsity):
                filtered.append(idx)
        return filtered

    def set_curriculum(self, epoch: int):
        """Update the curriculum state based on the schedule."""
        self.state.epoch = epoch

        # Find the matching phase
        matched_phase = None
        for phase in self.config.schedule:
            if phase.start_epoch <= epoch < phase.end_epoch:
                matched_phase = phase
                break

        if matched_phase:
            self.state.synth_ratio = matched_phase.synth_ratio
            self.state.mode = matched_phase.mode
            # Store sparsity bounds for filtering in __iter__
            self.state.min_sparsity = matched_phase.min_sparsity
            self.state.max_sparsity = matched_phase.max_sparsity
        else:
            # Default fallback if no phase matches (e.g. past last phase)
            if self.config.schedule:
                # Use the last phase's settings
                last_phase = self.config.schedule[-1]
                if epoch >= last_phase.end_epoch:
                    self.state.synth_ratio = last_phase.synth_ratio
                    self.state.mode = last_phase.mode
                    self.state.min_sparsity = last_phase.min_sparsity
                    self.state.max_sparsity = last_phase.max_sparsity
            else:
                # No schedule defined
                self.state.synth_ratio = 0.0
                self.state.mode = "time_major"
                self.state.min_sparsity = None
                self.state.max_sparsity = None

        self.state.active_runs = self.config.active_runs

        sparsity_info = ""
        if self.state.min_sparsity is not None or self.state.max_sparsity is not None:
            sparsity_info = (
                f", sparsity=[{self.state.min_sparsity or 0:.2f}, "
                f"{self.state.max_sparsity or 1:.2f}]"
            )
        logger.info(
            f"Curriculum Epoch {epoch}: ratio={self.state.synth_ratio:.2f}, "
            f"mode={self.state.mode}{sparsity_info}"
        )
        logger.info(
            "Curriculum mode is informational only; dataset-native "
            "data.sample_ordering controls within-dataset iteration order."
        )

    def _get_indices_for_dataset(self, dataset_idx: int) -> list[int]:
        """Get global indices for a specific sub-dataset."""
        offset = self.dataset_offsets[dataset_idx]
        n_samples = len(self.sub_datasets[dataset_idx])
        return [idx + offset for idx in range(n_samples)]

    def _chunk_indices(self, indices: list[int], chunk_size: int) -> list[list[int]]:
        """Split indices into chunks."""
        return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    def _rng_for_epoch(self, epoch: int | None = None) -> random.Random:
        epoch_value = self.state.epoch if epoch is None else epoch
        if self.seed is None:
            return random.Random(epoch_value)
        return random.Random(self.seed + epoch_value)

    def _chunk_to_batches(self, chunk: list[int]) -> list[list[int]]:
        batches = []
        for i in range(0, len(chunk), self.batch_size):
            batch = chunk[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        return batches

    def _build_chunk_streams(
        self,
        dataset_indices: list[int],
        rng: random.Random,
    ) -> list[list[list[int]]]:
        streams: list[list[list[int]]] = []
        for dataset_idx in dataset_indices:
            indices = self._get_indices_for_dataset(dataset_idx)
            chunks = self._chunk_indices(indices, self.config.chunk_size)
            rng.shuffle(chunks)
            for chunk in chunks:
                chunk_batches = self._chunk_to_batches(chunk)
                if chunk_batches:
                    streams.append(chunk_batches)
        rng.shuffle(streams)
        return streams

    @staticmethod
    def _stream_batch_count(streams: list[list[list[int]]]) -> int:
        return sum(len(chunk_batches) for chunk_batches in streams)

    @staticmethod
    def _synth_ratio(real_batches: int, synth_batches: int) -> float:
        total = real_batches + synth_batches
        if total <= 0:
            return 0.0
        return synth_batches / total

    def _resolve_batch_budget(
        self,
        real_available: int,
        synth_available: int,
    ) -> tuple[int, int]:
        target_ratio = self.state.synth_ratio
        if real_available <= 0:
            return 0, synth_available
        if synth_available <= 0:
            return real_available, 0
        if target_ratio <= 0.0:
            return real_available, 0
        if target_ratio >= 1.0:
            return 0, synth_available

        candidates: set[tuple[int, int]] = set()

        synth_from_real = real_available * target_ratio / (1.0 - target_ratio)
        for synth_count in {
            math.floor(synth_from_real),
            math.ceil(synth_from_real),
            synth_available,
        }:
            if 0 <= synth_count <= synth_available:
                candidates.add((real_available, synth_count))

        real_from_synth = synth_available * (1.0 - target_ratio) / target_ratio
        for real_count in {
            math.floor(real_from_synth),
            math.ceil(real_from_synth),
            real_available,
        }:
            if 0 <= real_count <= real_available:
                candidates.add((real_count, synth_available))

        candidates = {
            (real_count, synth_count)
            for real_count, synth_count in candidates
            if real_count > 0 or synth_count > 0
        }
        if not candidates:
            return real_available, synth_available

        return min(
            candidates,
            key=lambda counts: (
                abs(self._synth_ratio(*counts) - target_ratio),
                -(counts[0] + counts[1]),
                -counts[0],
            ),
        )

    def _interleave_chunk_streams(
        self,
        real_streams: list[list[list[int]]],
        synth_streams: list[list[list[int]]],
        real_budget: int,
        synth_budget: int,
    ) -> list[list[int]]:
        batches: list[list[int]] = []
        real_idx = 0
        synth_idx = 0
        real_used = 0
        synth_used = 0

        while real_used < real_budget or synth_used < synth_budget:
            choose_synth = False
            if synth_used >= synth_budget:
                choose_synth = False
            elif real_used >= real_budget:
                choose_synth = True
            else:
                real_progress = real_used / real_budget if real_budget > 0 else 1.0
                synth_progress = synth_used / synth_budget if synth_budget > 0 else 1.0
                choose_synth = synth_progress < real_progress

            if choose_synth:
                if synth_idx >= len(synth_streams):
                    break
                chunk_batches = synth_streams[synth_idx]
                synth_idx += 1
                take = min(len(chunk_batches), synth_budget - synth_used)
                batches.extend(chunk_batches[:take])
                synth_used += take
            else:
                if real_idx >= len(real_streams):
                    break
                chunk_batches = real_streams[real_idx]
                real_idx += 1
                take = min(len(chunk_batches), real_budget - real_used)
                batches.extend(chunk_batches[:take])
                real_used += take

        return batches

    def _resolve_active_synth_indices(self, epoch: int | None = None) -> list[int]:
        """Return the synthetic datasets active for the given epoch."""
        if epoch is not None:
            self.set_curriculum(epoch)

        available_synth = self.synth_dataset_indices
        if self.state.min_sparsity is not None or self.state.max_sparsity is not None:
            available_synth = self._filter_runs_by_sparsity(
                self.synth_dataset_indices,
                self.state.min_sparsity,
                self.state.max_sparsity,
            )
            logger.info(
                f"Sparsity filtering: {len(self.synth_dataset_indices)} -> "
                f"{len(available_synth)} synthetic runs available"
            )
            if not available_synth:
                logger.warning(
                    "Sparsity filtering excluded all synthetic runs; "
                    "falling back to all synthetic runs for this epoch."
                )
                available_synth = self.synth_dataset_indices

        n_synth = len(available_synth)
        if n_synth <= 0 or self.state.active_runs == 0:
            return []

        # -1 means "use all available synthetic runs"
        n_active = n_synth if self.state.active_runs < 0 else min(n_synth, self.state.active_runs)

        if self.config.run_sampling == "round_robin":
            start_idx = (self.state.epoch * n_active) % n_synth
            return [
                available_synth[(start_idx + i) % n_synth]
                for i in range(n_active)
            ]

        rng = self._rng_for_epoch()
        return rng.sample(available_synth, n_active)

    def _build_epoch_batches(self, epoch: int | None = None) -> list[list[int]]:
        if epoch is not None:
            self.set_curriculum(epoch)

        rng = self._rng_for_epoch()

        if not self.config.enabled:
            all_indices = []
            for idx in self.real_dataset_indices:
                all_indices.extend(self._get_indices_for_dataset(idx))
            rng.shuffle(all_indices)
            return self._chunk_to_batches(all_indices)

        active_synth_indices = self._resolve_active_synth_indices()
        real_streams = self._build_chunk_streams(self.real_dataset_indices, rng)
        synth_streams = self._build_chunk_streams(active_synth_indices, rng)

        real_available = self._stream_batch_count(real_streams)
        synth_available = self._stream_batch_count(synth_streams)
        real_budget, synth_budget = self._resolve_batch_budget(
            real_available, synth_available
        )

        return self._interleave_chunk_streams(
            real_streams=real_streams,
            synth_streams=synth_streams,
            real_budget=real_budget,
            synth_budget=synth_budget,
        )

    def num_batches_for_epoch(self, epoch: int | None = None) -> int:
        """Compute the exact number of batches yielded for a curriculum epoch."""
        return len(self._build_epoch_batches(epoch))

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self._build_epoch_batches():
            yield batch

    def __len__(self) -> int:
        return self.num_batches_for_epoch()
