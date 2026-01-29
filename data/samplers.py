import logging
import math
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import BatchSampler, ConcatDataset, Dataset

from models.configs import CurriculumConfig

logger = logging.getLogger(__name__)


@dataclass
class CurriculumState:
    """State of the curriculum for the current epoch."""
    epoch: int = 0
    synth_ratio: float = 0.0
    mode: str = "time_major"  # 'time_major' vs 'node_major'
    active_runs: int = 1


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
    ):
        """
        Args:
            dataset: ConcatDataset containing [RealDataset, SynthRun1, SynthRun2, ...]
                     It is expected that the first dataset is Real, or we identify them by run_id.
                     Actually, we assume the provided datasets are tagged.
            batch_size: Size of mini-batches.
            config: CurriculumConfig object.
            drop_last: Whether to drop the last incomplete batch.
        """
        # We don't call super().__init__ because we override __iter__ completely
        # and don't need the default sampler behavior.
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = config
        self.drop_last = drop_last
        
        self.state = CurriculumState()
        
        # Analyze the ConcatDataset to identify runs
        self._analyze_datasets()
        
    def _analyze_datasets(self):
        """Identify which sub-datasets are real and which are synthetic."""
        self.real_dataset_indices = []
        self.synth_dataset_indices = []
        
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
            logger.info(f"Sampler Dataset {i}: run_id='{run_id}'")
            if str(run_id) == "real":
                self.real_dataset_indices.append(i)
            else:
                self.synth_dataset_indices.append(i)
        
        logger.info(
            f"Curriculum Sampler found {len(self.real_dataset_indices)} real datasets "
            f"and {len(self.synth_dataset_indices)} synthetic datasets."
        )

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
        else:
            # Default fallback if no phase matches (e.g. past last phase)
            if self.config.schedule:
                # Use the last phase's settings
                last_phase = self.config.schedule[-1]
                if epoch >= last_phase.end_epoch:
                    self.state.synth_ratio = last_phase.synth_ratio
                    self.state.mode = last_phase.mode
            else:
                # No schedule defined
                self.state.synth_ratio = 0.0
                self.state.mode = "time_major"
        
        self.state.active_runs = self.config.active_runs
        
        logger.info(f"Curriculum Epoch {epoch}: ratio={self.state.synth_ratio:.2f}, mode={self.state.mode}")

    def _get_indices_for_dataset(self, dataset_idx: int) -> list[int]:
        """Get global indices for a specific sub-dataset."""
        ds = self.sub_datasets[dataset_idx]
        offset = self.dataset_offsets[dataset_idx]
        
        # We can implement 'mode' logic here (Node-Major vs Time-Major)
        # For now, we just take linear indices (which depends on dataset.sample_ordering)
        # If we want to enforce Time-Major or Node-Major, we might need to sort.
        # But EpiDataset is usually sorted by (Target, Time) or (Time, Target).
        # We assume the dataset's native ordering is acceptable for 'node_major' if it was configured as such.
        # But the curriculum might want to CHANGE the ordering.
        
        # NOTE: Re-sorting indices per epoch is expensive if N is large.
        # For Phase 1 (Data Access), let's stick to the dataset's native order
        # and just iterate chunks.
        
        n_samples = len(ds)
        local_indices = list(range(n_samples))
        
        # Apply mode-based shuffling if needed?
        # If mode == 'time_major', we might want random access across nodes but sequential in time?
        # Actually, "Time-Major" usually means [T0_N0, T0_N1, ... T1_N0, T1_N1 ...]
        # "Node-Major" usually means [N0_T0, N0_T1, ... N1_T0, N1_T1 ...]
        # The dataset creation config `sample_ordering` sets the physical layout.
        # Changing it here would mean hopping around the file, hurting locality.
        # We rely on "Chunking" to give us locality.
        
        # If we want "Snapshot-based" (Time-Major) batches, we should ideally ensure
        # the dataset is Time-Major.
        
        global_indices = [idx + offset for idx in local_indices]
        return global_indices

    def _chunk_indices(self, indices: list[int], chunk_size: int) -> list[list[int]]:
        """Split indices into chunks."""
        return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    def __iter__(self) -> Iterator[list[int]]:
        if not self.config.enabled:
            # Fallback to standard sequential/random sampling of the whole concat dataset
            # But normally if disabled, we might not even use this sampler.
            # If we do, just yield all real data.
            all_indices = []
            for idx in self.real_dataset_indices:
                all_indices.extend(self._get_indices_for_dataset(idx))
            
            # Simple shuffle for standard training
            random.shuffle(all_indices)
            
            # Yield batches
            for i in range(0, len(all_indices), self.batch_size):
                batch = all_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch
            return

        # 1. Select active synthetic runs
        # We rotate active runs each epoch or just pick random ones
        n_synth = len(self.synth_dataset_indices)
        if n_synth > 0 and self.state.active_runs > 0:
            if self.config.run_sampling == "round_robin":
                # Deterministic rotation based on epoch
                start_idx = (self.state.epoch * self.state.active_runs) % n_synth
                active_synth_indices = [
                    self.synth_dataset_indices[(start_idx + i) % n_synth]
                    for i in range(self.state.active_runs)
                ]
            else:
                # Random
                active_synth_indices = random.sample(
                    self.synth_dataset_indices, min(n_synth, self.state.active_runs)
                )
        else:
            active_synth_indices = []

        # 2. Gather chunks from Real and Active Synthetic datasets
        real_chunks = []
        for idx in self.real_dataset_indices:
            indices = self._get_indices_for_dataset(idx)
            # Shuffle indices within the dataset before chunking?
            # No, keep them sequential for locality, shuffle chunks later?
            # Or shuffle WITHIN chunks.
            # "Contiguous chunk of chunk_size windows" -> Linear read.
            real_chunks.extend(self._chunk_indices(indices, self.config.chunk_size))
            
        synth_chunks = []
        for idx in active_synth_indices:
            indices = self._get_indices_for_dataset(idx)
            synth_chunks.extend(self._chunk_indices(indices, self.config.chunk_size))

        # Shuffle the order of chunks (but keep items within chunk contiguous-ish?)
        # Actually, if we shuffle chunks, we jump around the file.
        # But we only have a few active runs.
        random.shuffle(real_chunks)
        random.shuffle(synth_chunks)

        # 3. Interleave batches
        # We turn chunks into batches.
        # A chunk (e.g. 512 items) becomes ~16 batches (size 32).
        
        def batches_from_chunks(chunks):
            for chunk in chunks:
                # Optional: Shuffle within the chunk for local randomness
                # (Preserves block locality but randomizes sample order)
                # chunk_copy = chunk[:]
                # random.shuffle(chunk_copy)
                
                # Create batches from this chunk
                for i in range(0, len(chunk), self.batch_size):
                    yield chunk[i : i + self.batch_size]

        real_batch_iter = batches_from_chunks(real_chunks)
        synth_batch_iter = batches_from_chunks(synth_chunks)
        
        # Create a pool of batches
        # We want to yield batches with probability P(synth) = ratio
        
        # We can just collect all batches and shuffle them?
        # If we collect ALL batches and shuffle, we lose the "Chunked Interleaving" benefit
        # which is about temporal locality of access.
        # "Chunked Interleaving": Read a chunk of Synth, yield its batches. Read a chunk of Real...
        
        # Let's implement the "Chunked Interleaving" as described:
        # "Emit k synthetic batches from the chunk and interleave m real batches"
        
        real_batches = list(real_batch_iter)
        synth_batches = list(synth_batch_iter)
        
        n_real = len(real_batches)
        n_synth = len(synth_batches)
        
        # Calculate target number of batches
        # If ratio = 0.8, we want 4 synth for 1 real.
        # Total batches depends on how much data we use.
        # "One epoch" usually means "One pass over the Real data" or "One pass over Active data"?
        # Usually defined by the sampler length.
        
        # Let's assume we want to iterate over the available Real and Active Synth data.
        # But we might drop some data to satisfy the ratio.
        # Or we oversample?
        
        # Simplest approach: Use all available real and active synth data, 
        # but mixed in order.
        # If the ratio implies we have too much Synth, we drop some?
        # Or we just mix what we have.
        # But "Curriculum" implies controlling the distribution.
        
        # If we just mix `real_batches` and `synth_batches`, the ratio is determined by dataset sizes.
        # We want to enforce `self.state.synth_ratio`.
        
        target_ratio = self.state.synth_ratio
        
        # Generator that yields batches according to ratio
        # We pull from real_batches and synth_batches queues.
        
        # We can assign a "score" to each batch? No.
        # We can just iterate and probabilistically pick?
        
        # To strictly respect "Chunked Interleaving" (Process a whole chunk of Synth, then Real...):
        # We should work with Chunks, not Batches.
        
        # Re-think:
        # We have `real_chunks` and `synth_chunks`.
        # We want to form a sequence of chunks that respects the ratio.
        # e.g. [S, S, S, S, R, S, S, S, S, R...]
        
        # If we have N_R real chunks and N_S synth chunks.
        # And we want Ratio P_S.
        # This implies we might need to oversample or undersample chunks.
        
        # Let's try to consume both pools exhaustively if possible, but mixing them.
        # If we just shuffle `real_chunks + synth_chunks`, the ratio is fixed by data size.
        # If we want to CHANGE the ratio (e.g. 80% synth), we need to sample with replacement?
        
        # Given the complexity of "exact ratio" vs "dataset size", and the user's plan:
        # "Emit k synthetic batches ... and interleave m real batches"
        # This implies we are generating the stream.
        
        # Let's use a probabilistic generator that pulls from the pools.
        
        idx_r = 0
        idx_s = 0
        
        # While we have data in EITHER pool
        while idx_r < len(real_batches) or idx_s < len(synth_batches):
            # Decide whether to pick Synth or Real
            # If we run out of one, force the other?
            # Or stop?
            
            # If we want to strictly follow ratio, we toss a coin.
            if idx_s < len(synth_batches) and idx_r < len(real_batches):
                is_synth = random.random() < target_ratio
            elif idx_s < len(synth_batches):
                is_synth = True
            elif idx_r < len(real_batches):
                is_synth = False
            else:
                break
                
            if is_synth:
                batch = synth_batches[idx_s]
                idx_s += 1
            else:
                batch = real_batches[idx_r]
                idx_r += 1
            
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        # This is approximate because of the probabilistic sampling and active runs
        # It is used by tqdm for progress bar.
        # We can calculate exact number if we knew the counts.
        
        # Estimate:
        # We use all real data + active synth data.
        # So len is sum of batches.
        
        n_real_samples = sum(len(self.sub_datasets[i]) for i in self.real_dataset_indices)
        
        n_synth_samples = 0
        # This depends on active runs
        # We don't know EXACTLY which runs are active in __len__ without recalculating.
        # But assuming round robin or random, average size?
        # Or just sum of ALL synth / num_runs * active_runs?
        
        total_synth = sum(len(self.sub_datasets[i]) for i in self.synth_dataset_indices)
        num_synth_runs = len(self.synth_dataset_indices)
        if num_synth_runs > 0:
            avg_synth = total_synth / num_synth_runs
            n_synth_samples = avg_synth * self.state.active_runs
        
        total_samples = n_real_samples + n_synth_samples
        return math.ceil(total_samples / self.batch_size)
