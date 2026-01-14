import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
from torch_geometric.data import Data

from graph.node_encoder import Region2Vec
from models.configs import EpiForecasterConfig

from utils.logging import suppress_zarr_warnings

suppress_zarr_warnings()

from .biomarker_preprocessor import BiomarkerPreprocessor  # noqa: E402
from .cases_preprocessor import CasesPreprocessor, CasesPreprocessorConfig  # noqa: E402
from .mobility_preprocessor import (  # noqa: E402
    MobilityPreprocessor,
    MobilityPreprocessorConfig,
)
from .preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402

logger = logging.getLogger(__name__)

StaticCovariates = dict[str, torch.Tensor]


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (time, region, feature), adding trailing dim if needed."""
    if arr.ndim == 2:
        return arr[..., None]
    return arr


class EpiDatasetItem(TypedDict):
    node_label: str
    target_node: int
    case_node: torch.Tensor
    case_mean: torch.Tensor
    case_std: torch.Tensor
    bio_node: torch.Tensor
    target: torch.Tensor
    target_scale: torch.Tensor
    target_mean: torch.Tensor
    mob: list[Data]
    population: torch.Tensor


class EpiDataset(Dataset):
    """
    Dataset loader for preprocessed epidemiological datasets.

    This class loads preprocessed epidemiological data stored in Zarr format
    and converts it to GraphEgoDataset for ego-graph processing.
    Args:
        zarr_path: Path to the Zarr dataset
        config: EpiForecasterTrainerConfig configuration
    """

    def __init__(
        self,
        config: EpiForecasterConfig,
        target_nodes: list[int],
        context_nodes: list[int],
        biomarker_preprocessor: BiomarkerPreprocessor | None = None,
        cases_preprocessor: CasesPreprocessor | None = None,
        mobility_preprocessor: MobilityPreprocessor | None = None,
    ):
        self.aligned_data_path = Path(config.data.dataset_path).resolve()
        self.config = config

        # Load dataset
        self._dataset = xr.open_zarr(self.aligned_data_path)
        self.num_nodes = self._dataset[REGION_COORD].size
        self.biomarker_available_mask = self._compute_biomarker_available_mask()

        # Setup cases preprocessor
        if cases_preprocessor is None:
            cp_config = CasesPreprocessorConfig(
                history_length=config.model.history_length,
                log_scale=config.data.log_scale,
                scale_epsilon=1e-6,
                per_100k=True,
            )
            self.cases_preprocessor = CasesPreprocessor(cp_config)
        else:
            self.cases_preprocessor = cases_preprocessor

        # Precompute cases
        (
            self.precomputed_cases,
            self.rolling_mean,
            self.rolling_std,
        ) = self.cases_preprocessor.preprocess_dataset(self._dataset)

        # Setup mobility preprocessor
        if mobility_preprocessor is None:
            mp_config = MobilityPreprocessorConfig(
                log_scale=config.data.mobility_log_scale,
                clip_range=config.data.mobility_clip_range,
                scale_epsilon=config.data.mobility_scale_epsilon,
            )
            self.mobility_preprocessor = MobilityPreprocessor(mp_config)

            # Fit scaler on train nodes only
            all_region_ids = self._dataset[REGION_COORD].values
            train_region_ids = [all_region_ids[n] for n in target_nodes]
            self.mobility_preprocessor.fit_scaler(self._dataset, train_region_ids)
        else:
            self.mobility_preprocessor = mobility_preprocessor

        # Always preload mobility data into memory (RAM) to avoid Zarr chunking I/O overhead.
        mobility_da = self._dataset.mobility
        # Ensure proper dimension ordering: Time, Origin, Destination
        mobility_da = self.mobility_preprocessor._ensure_time_first(mobility_da)

        # Load all values into memory
        mobility_np = mobility_da.values

        # Apply preprocessing/scaling once
        mobility_np = self.mobility_preprocessor.transform_values(mobility_np)

        # Convert to torch tensor
        self.preloaded_mobility = torch.from_numpy(mobility_np).to(torch.float32)

        # Optimization: Precompute mobility mask to avoid repeated comparisons in __getitem__
        mobility_threshold = float(config.data.mobility_threshold)
        self.mobility_mask = self.preloaded_mobility >= mobility_threshold

        logger.info(
            f"Mobility preloaded: {self.preloaded_mobility.shape}, "
            f"{self.preloaded_mobility.element_size() * self.preloaded_mobility.numel() / 1e6:.2f} MB"
        )

        # Setup biomarker preprocessor
        if biomarker_preprocessor is None:
            self.biomarker_preprocessor = BiomarkerPreprocessor()

            # Log biomarker availability in train split
            available_mask_np = self.biomarker_available_mask.cpu().numpy().flatten()
            train_nodes_with_bio = [
                n for n in target_nodes if available_mask_np[n] == 1.0
            ]
            logger.info(
                f"Train split: {len(train_nodes_with_bio)}/{len(target_nodes)} "
                f"nodes have biomarker data"
            )

            if len(train_nodes_with_bio) == 0:
                if self.config.model.type.biomarkers:
                    raise ValueError(
                        "Biomarkers enabled in config but no train nodes have biomarker data. "
                        "Disable biomarkers or use a different train split."
                    )
                logger.warning(
                    "No train nodes have biomarker data, using zero encoding"
                )

            # Convert indices to region IDs for fit_scaler
            all_region_ids = self._dataset[REGION_COORD].values
            train_region_ids = [all_region_ids[n] for n in train_nodes_with_bio]

            # Only fit scaler on train nodes that have biomarkers
            self.biomarker_preprocessor.fit_scaler(self._dataset, train_region_ids)
        else:
            self.biomarker_preprocessor = biomarker_preprocessor

        # Precompute biomarkers for the entire dataset
        # This returns a (TotalTime, NumNodes, 3) tensor
        if "edar_biomarker" in self._dataset:
            self.precomputed_biomarkers = torch.from_numpy(
                self.biomarker_preprocessor.preprocess_dataset(self._dataset)
            ).to(torch.float32)
        else:
            T_total = len(self._dataset[TEMPORAL_COORD])
            # 3 channels: value=0, mask=0, age=1
            dummy = torch.zeros((T_total, self.num_nodes, 3), dtype=torch.float32)
            dummy[:, :, 2] = 1.0
            self.precomputed_biomarkers = dummy

        self.region_embeddings = None
        if config.data.region2vec_path:
            # use pre-trained region2vec embeddings and lookup by labeled regions
            # TODO: actually run forward pass region2vec in EpiForecaster
            _, art = Region2Vec.from_weights(config.data.region2vec_path)

            # Filter region embeddings using numpy instead of xarray to avoid requiring a named dimension
            # for the embedding size (since xarray expects all dims to be named).
            region_ids = list(art["region_ids"])
            selected_ids = list(self._dataset[REGION_COORD].values)
            region_id_index = {rid: i for i, rid in enumerate(region_ids)}
            indices = [
                region_id_index[rid] for rid in selected_ids if rid in region_id_index
            ]
            region_embeddings = art["embeddings"][indices]

            assert region_embeddings.shape == (
                self.num_nodes,
                self.config.model.region_embedding_dim,
            ), "Static embeddings shape mismatch"

            self.region_embeddings = torch.as_tensor(
                region_embeddings, dtype=torch.float32
            )

        self.target_nodes = target_nodes
        self._target_node_to_local_idx = {n: i for i, n in enumerate(target_nodes)}
        self.context_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.context_mask[target_nodes] = True
        self.context_mask[context_nodes] = True

        # Set dimensions
        self.time_dim_size = config.model.history_length + config.model.forecast_horizon
        self.window_stride = int(config.data.window_stride)
        self.missing_permit = int(config.data.missing_permit)
        self.window_starts = self._compute_window_starts()
        self._valid_window_starts_by_node = self._compute_valid_window_starts()
        self.window_starts = self._collect_valid_window_starts()
        self._index_map, self._index_lookup = self._build_index_map()

        self.node_static_covariates = self.static_covariates()
        self.scale_epsilon = 1e-6

        # Close dataset and clear reference to avoid pickling issues
        self._dataset.close()
        self._dataset = None

    @property
    def dataset(self) -> xr.Dataset:
        if self._dataset is not None:
            return self._dataset

        # Always cache the dataset handle
        ds = xr.open_zarr(self.aligned_data_path)
        self._dataset = ds
        return ds

    def __getstate__(self):
        """Allow pickling by clearing the dataset handle."""
        state = self.__dict__.copy()
        state["_dataset"] = None
        # preloaded_mobility is a tensor, so it picks fine.
        return state

    def num_windows(self) -> int:
        """Number of window start positions valid for at least one target node."""
        return len(self.window_starts)

    @property
    def cases_output_dim(self) -> int:
        """Temporal input dimension (single feature, always 1)."""
        return 1

    @property
    def biomarkers_output_dim(self) -> int:
        """Biomarkers dim - 4 channels (value, mask, age, has_data)."""
        return 4

    def index_for_target_node_window(self, *, target_node: int, window_idx: int) -> int:
        """Map a (window_idx, target_node) pair to a dataset index.

        window_idx indexes into the stride-based window starts (see ``num_windows``)
        and is filtered by missingness per target node.
        """
        if window_idx < 0 or window_idx >= len(self.window_starts):
            raise IndexError("Requested window exceeds available time windows")

        window_start = self.window_starts[window_idx]
        idx = self._index_lookup.get((target_node, window_start))
        if idx is None:
            raise KeyError(
                "Requested window is not valid for the specified target node"
            )
        return idx

    def __len__(self) -> int:
        """Number of samples in the dataset.

        One sample corresponds to a (window, node) pair. Windows are generated
        with the configured stride and filtered by missingness permit.
        """
        return len(self._index_map)

    def __getitem__(self, idx: int) -> EpiDatasetItem:
        """Return a single target node over one time window.

        Each item is keyed by (target_node, window_start). The mobility slice is
        converted to a PyG ego-graph per time step containing the target node
        and its incoming neighbors.
        """

        L = self.config.model.history_length
        H = self.config.model.forecast_horizon

        try:
            target_idx, range_start = self._index_map[idx]
        except IndexError as exc:
            raise IndexError("Sample index out of range") from exc

        node_label = self.dataset[REGION_COORD].values[target_idx]

        range_end = range_start + L
        forecast_targets = range_end + H
        T = len(self.dataset[TEMPORAL_COORD].values)
        if forecast_targets > T:
            raise IndexError("Requested window exceeds available time steps")

        # Optimization: Only slice mobility from RAM tensor, never from disk/zarr
        # preloaded_mobility is (TotalTime, Origin, Destination)
        # We want [range_start:range_end, :, target_idx]
        mobility_history = self.preloaded_mobility[range_start:range_end, :, target_idx]
        neigh_mask = self.mobility_mask[range_start:range_end, :, target_idx]

        # Cases Processing (using precomputed tensors)
        # 1. Get stats at the end of history window (t + L - 1)
        stat_idx = range_end - 1
        mean = self.rolling_mean[stat_idx]  # (N, 1)
        std = self.rolling_std[stat_idx]  # (N, 1)

        # 2. Slice precomputed cases (L+H, N, 2) - channel 0: value, channel 1: mask
        cases_window = self.precomputed_cases[
            range_start:forecast_targets
        ]  # (L+H, N, 2)

        # 3. Normalize only the value channel (channel 0)
        # Mask channel (channel 1) is already binary and doesn't need normalization
        value_channel = cases_window[..., 0:1]  # (L+H, N, 1)
        mask_channel = cases_window[..., 1:2]  # (L+H, N, 1)

        norm_value = (value_channel - mean) / std
        norm_value = torch.nan_to_num(norm_value, nan=0.0)

        # Recombine normalized value with original mask
        norm_window = torch.cat([norm_value, mask_channel], dim=-1)  # (L+H, N, 2)

        # Split into history and future
        case_history = norm_window[:L]  # (L, N, 2)
        future_cases = norm_window[L:]  # (H, N, 2)

        if self.context_mask is not None:
            # self.context_mask is a tensor
            neigh_mask = neigh_mask & self.context_mask[None, :]
            # Force target node to be included
            neigh_mask[:, target_idx] = True

        # Apply mask to case_history (both channels)
        neigh_mask_t = neigh_mask.to(torch.float32).unsqueeze(-1)
        case_history = case_history * neigh_mask_t

        # Encode all regions in context using 4-channel biomarker encoding
        # Optimized: Use precomputed biomarkers + broadcasted availability mask

        # 1. Get precomputed bio history (value, mask, age) -> (L, N, 3)
        # Note: self.precomputed_biomarkers is a CPU tensor
        bio_slice = self.precomputed_biomarkers[range_start:range_end]

        # 2. Get availability mask -> (N,)
        if self.biomarker_available_mask is not None:
            # We assume dim 0 matches nodes. If available_mask is (N, 1), take col 0.
            has_data = self.biomarker_available_mask[:, 0]
        else:
            has_data = torch.zeros(self.num_nodes, dtype=torch.float32)

        # 3. Broadcast to (L, N, 1)
        has_data_3d = has_data.view(1, self.num_nodes, 1).expand(L, -1, -1)

        # 4. Concatenate -> (L, N, 4)
        biomarker_history = torch.cat([bio_slice, has_data_3d], dim=-1)

        target_np = future_cases[:, target_idx, 0]  # Only value channel for targets
        targets = target_np.squeeze(-1)

        assert mobility_history.shape == (L, self.num_nodes), (
            f"Mob history shape mismatch: expected ({L}, {self.num_nodes}), got {mobility_history.shape}"
        )
        assert case_history.shape == (L, self.num_nodes, 2), (
            f"Case history shape mismatch: expected ({L}, {self.num_nodes}, 2), got {case_history.shape}"
        )
        assert biomarker_history.shape == (L, self.num_nodes, 4), (
            "Biomarker history shape mismatch - expected (T, N, 4)"
        )
        assert targets.shape == (H,), "Targets shape mismatch"

        mob_graphs: list[Data] = []
        for t in range(L):
            mob_graphs.append(
                self._dense_graph_to_ego_pyg(
                    mobility_history[t],
                    case_history[t],
                    biomarker_history[t],
                    target_idx,
                    time_id=t,
                )
            )

        # Slice history for mean and std
        # mean/std are (TotalTime, N) -> need to slice [range_start:range_end] and select target_idx
        # rolling_mean/std are numpy arrays (T, N, 1) or (T, N)
        # Check dim of rolling_mean

        # Original code used rolling_mean[stat_idx] which is one time step.
        # Now we want the sequence.

        # Ensure rolling stats are dense and correct shape (L, 1)
        mean_seq = self.rolling_mean[
            range_start:range_end, target_idx
        ].float()  # (L, 1)
        std_seq = self.rolling_std[range_start:range_end, target_idx].float()  # (L, 1)

        if mean_seq.ndim == 1:
            mean_seq = mean_seq.unsqueeze(-1)
        if std_seq.ndim == 1:
            std_seq = std_seq.unsqueeze(-1)

        population = self.node_static_covariates["Pop"][target_idx]

        return {
            "node_label": node_label,
            "target_node": target_idx,
            "case_node": case_history[:, target_idx, :],  # Already normalized
            "case_mean": mean_seq,
            "case_std": std_seq,
            "bio_node": biomarker_history[:, target_idx, :],
            "target": targets,
            "target_scale": std[target_idx].squeeze(-1),
            "target_mean": mean[target_idx].squeeze(-1),
            "mob": mob_graphs,
            "population": population,
        }

    def static_covariates(self) -> StaticCovariates:
        "Returns static covariates for the dataset. (num_nodes, num_features)"
        population_cov = self.dataset.population
        population_tensor = torch.from_numpy(population_cov.to_numpy()).to(
            torch.float32
        )
        assert population_tensor.shape == (self.num_nodes,), (
            f"Static covariates shape mismatch: expected ({self.num_nodes},), got {population_tensor.shape}"
        )

        return {
            "Pop": population_tensor,
        }

    def _compute_window_starts(self) -> list[int]:
        """Compute window start indices given history, horizon, and stride."""
        L = self.config.model.history_length
        H = self.config.model.forecast_horizon
        T = len(self.dataset[TEMPORAL_COORD].values)
        seg = L + H
        if T < seg:
            return []
        return list(range(0, T - seg + 1, self.window_stride))

    def _compute_biomarker_available_mask(self) -> torch.Tensor | None:
        """Return a (num_nodes, biomarkers_output_dim) availability mask for biomarkers.

        The mask is tiled to cover both raw and change features.
        """
        if "edar_biomarker" not in self.dataset:
            return None
        biomarker_da = self.dataset["edar_biomarker"]
        values = _ensure_3d(biomarker_da.values)
        if values.ndim != 3:
            raise ValueError(
                f"Expected biomarker array with 2 or 3 dims, got shape {values.shape}"
            )
        available = np.isfinite(values).any(axis=0)
        available = available.astype(np.float32)
        if available.shape[0] != self.num_nodes:
            raise ValueError(
                "Biomarker availability mask does not match number of nodes."
            )
        return torch.from_numpy(available).to(torch.float32)

    def _compute_valid_window_starts(self) -> dict[int, list[int]]:
        """Compute valid window starts per target node using missingness permit.

        History windows may include up to ``missing_permit`` missing values, but
        forecast targets must be fully observed (no NaNs).
        """
        if not self.window_starts:
            return {target_idx: [] for target_idx in self.target_nodes}

        other_dims = [
            d
            for d in self.dataset.cases.dims
            if d not in (TEMPORAL_COORD, REGION_COORD)
        ]
        cases_da = self.dataset.cases.transpose(
            TEMPORAL_COORD, REGION_COORD, *other_dims
        )
        cases_np = _ensure_3d(cases_da.values)
        if cases_np.ndim != 3:
            raise ValueError(
                f"Expected cases array with 2 or 3 dims, got shape {cases_np.shape}"
            )

        valid = np.isfinite(cases_np).all(axis=2)
        valid_int = valid.astype(np.int32)

        L = self.config.model.history_length
        H = self.config.model.forecast_horizon

        cumsum = np.concatenate(
            [
                np.zeros((1, self.num_nodes), dtype=np.int32),
                np.cumsum(valid_int, axis=0),
            ],
            axis=0,
        )

        history_counts = cumsum[L:] - cumsum[:-L]
        target_counts = cumsum[L + H :] - cumsum[L:-H]

        starts = np.asarray(self.window_starts, dtype=np.int64)
        history_counts = history_counts[starts]
        target_counts = target_counts[starts]

        history_threshold = max(0, L - self.missing_permit)
        history_ok = history_counts >= history_threshold
        target_ok = target_counts >= H
        valid_mask = history_ok & target_ok

        starts_by_node: dict[int, list[int]] = {}
        for target_idx in self.target_nodes:
            mask = valid_mask[:, target_idx]
            starts_by_node[target_idx] = [
                int(s) for s, ok in zip(starts, mask, strict=False) if ok
            ]

        return starts_by_node

    def _collect_valid_window_starts(self) -> list[int]:
        """Return sorted window starts that are valid for at least one target node."""
        unique_starts: set[int] = set()
        for starts in self._valid_window_starts_by_node.values():
            unique_starts.update(starts)
        return sorted(unique_starts)

    def _build_index_map(
        self,
    ) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
        """Build index mappings for fast (node, window_start) lookup."""
        index_map: list[tuple[int, int]] = []
        index_lookup: dict[tuple[int, int], int] = {}
        for target_idx in self.target_nodes:
            for start in self._valid_window_starts_by_node.get(target_idx, []):
                idx = len(index_map)
                index_map.append((target_idx, start))
                index_lookup[(target_idx, start)] = idx
        return index_map, index_lookup

    def _dense_graph_to_ego_pyg(
        self,
        inflow_t: torch.Tensor,
        case_t: torch.Tensor,
        bio_t: torch.Tensor,
        dst_idx: int,
        time_id: int | None = None,
    ) -> Data:
        """Convert a dense mobility slice into a PyG ego-graph for one target node.

        Nodes include the target and all origins with non-zero incoming flow. Edges
        are directed origin -> target and store the raw flow weight.

        Vectorized Logic:
            We construct node_ids as [neighbor_1, ..., neighbor_k, target].
            We then blindly create edges 0 -> k, 1 -> k, etc.
            If a self-loop exists (neighbor_i == target), we generate an edge i -> k.
            Since x[i] and x[k] are identical feature vectors (derived from the same region),
            the Message Passing Neural Network (MPNN) receives the exact same information
            as a k -> k self-loop.

        Args:
            inflow_t: Vector (N,) of inflow values to dst_idx
            case_t: Vector (N, C) of case features
            bio_t: Vector (N, B) of biomarker features
            dst_idx: Index of the target node
            time_id: Optional time step index
        """
        if self.context_mask is not None:
            inflow_t = inflow_t.clone()
            inflow_t[~self.context_mask] = 0

        # Now inflow_t is already the inflow vector (N,)
        origins = inflow_t.nonzero(as_tuple=False).flatten()

        # Limit to max_neighbors by selecting top-k by inflow
        max_neighbors = self.config.model.max_neighbors
        if origins.numel() > max_neighbors:
            # Get top-k neighbors by inflow value
            top_values = inflow_t[origins]
            top_k_indices = torch.topk(top_values, k=max_neighbors).indices
            origins = origins[top_k_indices]

        # Construct node_ids: [neighbor_1, ..., neighbor_k, target_node]
        # Note: 'origins' might contain dst_idx if self-loop exists.
        # We append dst_idx at the end regardless, effectively treating it as a
        # distinct node in the graph structure if it also appears in origins.
        # This simplifies edge construction to "all nodes in list -> last node".
        target_node_tensor = torch.tensor([dst_idx], dtype=torch.long)
        node_ids = torch.cat([origins, target_node_tensor], dim=0)

        num_neighbors = origins.numel()
        # Edge index: all neighbor indices (0 to num_neighbors-1) -> target index (num_neighbors)
        # Shape (2, num_edges)
        if num_neighbors > 0:
            sources = torch.arange(num_neighbors, dtype=torch.long)
            targets = torch.full((num_neighbors,), num_neighbors, dtype=torch.long)
            edge_index = torch.stack([sources, targets], dim=0)
            edge_weight = inflow_t[origins]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=inflow_t.dtype)

        # TODO: find a better way to do this. the feature concatenation logic is scattered
        # through the model.
        feat = []
        if self.config.model.type.cases:
            feat.append(case_t[node_ids])
        if self.config.model.type.biomarkers:
            feat.append(bio_t[node_ids])

        x = torch.cat(feat, dim=-1)

        g = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        g.num_nodes = node_ids.numel()
        # The target node is always the last one in our construction
        g.target_node = torch.tensor([num_neighbors], dtype=torch.long)
        if time_id is not None:
            g.time_id = torch.tensor([time_id], dtype=torch.long)
        return g

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"EpiDataset(source={self.aligned_data_path}, "
            f"seq_len={self.time_dim_size}, "
            f"nodes={self.num_nodes})"
        )

    def missingness_features(self, data: xr.DataArray) -> torch.Tensor:
        """Return a missingness indicator tensor for a (time, region[, feature]) array."""
        if not isinstance(data, xr.DataArray):
            raise TypeError("missingness_features expects an xarray DataArray")
        mask = _ensure_3d(data.isnull().values)
        return torch.from_numpy(mask).to(torch.float32)

    def calendar_features(
        self, time_index: pd.DatetimeIndex | None = None
    ) -> torch.Tensor:
        """Return simple calendar covariates for each timestamp.

        Features: day-of-week one-hot (7), month one-hot (12), day-of-year sin/cos (2).
        Shape: (time, 21).
        """
        if time_index is None:
            time_index = pd.DatetimeIndex(self.dataset[TEMPORAL_COORD].values)

        dow = time_index.dayofweek.to_numpy()
        months = time_index.month.to_numpy()
        doy = time_index.dayofyear.to_numpy()

        dow_oh = np.eye(7, dtype=np.float32)[dow]
        month_oh = np.eye(12, dtype=np.float32)[months - 1]
        doy_angle = 2 * np.pi * (doy / 365.25)
        doy_sin = np.sin(doy_angle).astype(np.float32)[:, None]
        doy_cos = np.cos(doy_angle).astype(np.float32)[:, None]

        features = np.concatenate([dow_oh, month_oh, doy_sin, doy_cos], axis=1)
        return torch.from_numpy(features).to(torch.float32)

    @classmethod
    def load_canonical_dataset(cls, aligned_data_path: Path) -> xr.Dataset:
        "Load the canonical dataset from the aligned data path."
        return xr.open_zarr(aligned_data_path)
