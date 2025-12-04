"""
Ego-Graph Dataset for Epidemiological Forecasting.

This module implements an ego-graph approach where each dataset item represents
one target node at one specific time point with its neighborhood subgraphs.
This ensures the model never sees the full graph and maintains true
inductive node-level processing.

Key Design Principles:
1. Ego-graph extraction: Each sample contains subgraphs of incoming neighbors
2. Temporal consistency: Ego-graphs built for each time step in history window
3. Inductive learning: Model never sees full graph, only local neighborhoods
4. Memory efficiency: Only subgraph data loaded per sample

Dataset Structure:
- Target node temporal sequence: [L] history + [H] future
- Ego-graphs per time step: subgraphs with incoming neighbors
- Variable neighborhood sizes with proper normalization
"""

import logging
from typing import Any

import numpy as np
import torch
import zarr
from einops import rearrange
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GraphEgoDataset(Dataset):
    """
    Ego-graph dataset for epidemiological forecasting.

    Each dataset item represents one target node at one specific time point
    with its neighborhood subgraphs across the history window.
    """

    def __init__(
        self,
        cases: torch.Tensor,  # (N, T) - case counts per node per time
        biomarkers: torch.Tensor,  # (N, T, F) - biomarkers per node per time
        mobility: list[torch.Tensor],  # [T] mobility matrices (T, N, N) or sparse
        L: int,  # history length
        H: int,  # forecast horizon
        min_flow_threshold: float = 10.0,  # minimum flow to include neighbor
        max_neighbors: int = 20,  # maximum neighbors per ego-graph
        include_target_in_graph: bool = True,  # include target node in subgraph
        validate_data: bool = True,
    ):
        """
        Initialize ego-graph dataset.

        Args:
            cases: Case counts tensor [num_nodes, num_timesteps]
            biomarkers: Biomarker tensor [num_nodes, num_timesteps, biomarker_dim]
            mobility: List of mobility matrices per timestep
            L: History window length for temporal context
            H: Forecast horizon length
            min_flow_threshold: Minimum flow weight to include neighbor
            max_neighbors: Maximum number of neighbors per ego-graph
            include_target_in_graph: Whether to include target node in subgraph
            validate_data: Whether to validate input data consistency
        """
        self.cases = cases
        self.biomarkers = biomarkers
        self.mobility = mobility
        self.L = L
        self.H = H
        self.min_flow_threshold = min_flow_threshold
        self.max_neighbors = max_neighbors
        self.include_target_in_graph = include_target_in_graph

        # Validate data dimensions
        if validate_data:
            self._validate_input_data()

        # Build ego-graph index list
        self.indices = self._build_index_list()
        logger.info(
            f"Created GraphEgoDataset with {len(self.indices)} ego-graph samples"
        )
        logger.info(f"Data shape: cases {cases.shape}, biomarkers {biomarkers.shape}")

    def _validate_input_data(self):
        """Validate input data dimensions and consistency."""
        num_nodes, num_timesteps = self.cases.shape
        biomarker_nodes, biomarker_timesteps, biomarker_dim = self.biomarkers.shape

        if biomarker_nodes != num_nodes:
            raise ValueError(
                f"Cases nodes {num_nodes} != biomarker nodes {biomarker_nodes}"
            )
        if biomarker_timesteps != num_timesteps:
            raise ValueError(
                f"Cases timesteps {num_timesteps} != biomarker timesteps {biomarker_timesteps}"
            )
        if len(self.mobility) != num_timesteps:
            raise ValueError(
                f"Mobility length {len(self.mobility)} != timesteps {num_timesteps}"
            )

        # Check each mobility matrix
        for t, mobility_t in enumerate(self.mobility):
            if mobility_t.shape != (num_nodes, num_nodes):
                raise ValueError(
                    f"Mobility[{t}] shape {mobility_t.shape} != ({num_nodes}, {num_nodes})"
                )

    def _build_index_list(self) -> list[tuple[int, int]]:
        """
        Build list of (target_region, t0) indices for valid samples.

        Returns:
            List of valid (region_id, t0) tuples where we have enough history
            and future data for ego-graph construction.
        """
        num_nodes, num_timesteps = self.cases.shape
        indices = []

        # Valid t0 range: we need L history and H future
        min_t0 = self.L - 1
        max_t0 = num_timesteps - self.H - 1

        for region_id in range(num_nodes):
            for t0 in range(min_t0, max_t0 + 1):
                # Check if we have sufficient data for this region at this time
                if self._has_sufficient_data(region_id, t0):
                    indices.append((region_id, t0))

        logger.info(
            f"Built {len(indices)} valid ego-graph samples from {num_nodes} regions × {num_timesteps} timesteps"
        )
        return indices

    def _has_sufficient_data(self, region_id: int, t0: int) -> bool:
        """
        Check if we have sufficient data for ego-graph construction.

        Args:
            region_id: Target region ID
            t0: Target time point

        Returns:
            True if we have sufficient history and future data
        """
        # Check history window
        history_start = max(0, t0 - self.L + 1)
        history_end = t0 + 1

        # Check if target has cases in history window
        history_cases = self.cases[region_id, history_start:history_end]
        if torch.all(history_cases == 0):
            return False  # No cases in history window

        # Check future window
        future_start = t0 + 1
        future_end = t0 + self.H + 1

        # Check if we have future data
        if future_end > self.cases.shape[1]:
            return False  # Not enough future data

        return True

    def __len__(self) -> int:
        """Return number of ego-graph samples."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get ego-graph sample for specific target node at specific time.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - target_region_id: Target node ID
            - cases_hist: Historical cases [L]
            - biomarkers_hist: Historical biomarkers [L, F]
            - cases_future: Future cases [H]
            - node_features_t: List of L ego-graph node features
            - edge_index_t: List of L ego-graph edge indices
            - edge_weight_t: List of L ego-graph edge weights
            - target_local_idx: List of L target local indices
        """
        region_id, t0 = self.indices[idx]

        # 1) Temporal slices for target region
        history_start = t0 - self.L + 1
        history_end = t0 + 1

        cases_hist = self.cases[region_id, history_start:history_end]  # (L,)
        biomarkers_hist = self.biomarkers[
            region_id, history_start:history_end, :
        ]  # (L, F)
        cases_future = self.cases[region_id, t0 + 1 : t0 + self.H + 1]  # (H,)

        # 2) Build ego-graphs for each time in history window
        node_features_t = []
        edge_index_t = []
        edge_weight_t = []
        target_local_idx_t = []

        for _k, t in enumerate(range(history_start, history_end)):
            ego_graph = self._build_ego_graph(region_id, t)

            node_features_t.append(ego_graph["node_features"])
            edge_index_t.append(ego_graph["edge_index"])
            edge_weight_t.append(ego_graph["edge_weight"])
            target_local_idx_t.append(ego_graph["target_local_idx"])

        sample = {
            "target_region_id": region_id,
            "cases_hist": cases_hist,
            "biomarkers_hist": biomarkers_hist,
            "cases_future": cases_future,
            "node_features_t": node_features_t,
            "edge_index_t": edge_index_t,
            "edge_weight_t": edge_weight_t,
            "target_local_idx": target_local_idx_t,
        }

        return sample

    def _build_ego_graph(self, region_id: int, t: int) -> dict[str, Any]:
        """
        Build ego-graph for target region at specific time step.

        Args:
            region_id: Target region ID
            t: Time step

        Returns:
            Dictionary containing ego-graph structure:
            - node_features: Node features for subgraph
            - edge_index: Edge indices in PyTorch Geometric format [2, E]
            - edge_weight: Normalized edge weights [E]
            - target_local_idx: Local index of target node in subgraph
        """
        # Get mobility matrix for this time step
        M_t = self.mobility[t]  # (N, N)

        # Find incoming flows to target region
        flows_to_i = M_t[:, region_id]  # (N,)

        # Filter origins with sufficient flow
        significant_mask = flows_to_i > self.min_flow_threshold
        origins = torch.nonzero(significant_mask, as_tuple=False).flatten()

        # Get flow weights for significant origins
        flow_weights = flows_to_i[origins]

        # Limit number of neighbors if necessary
        if len(origins) > self.max_neighbors:
            # Sort by flow weight and keep top neighbors
            sorted_indices = torch.argsort(flow_weights, descending=True)[
                : self.max_neighbors
            ]
            origins = origins[sorted_indices]
            flow_weights = flow_weights[sorted_indices]

        # Include target node in subgraph
        if self.include_target_in_graph:
            nodes = torch.cat(
                [origins, torch.tensor([region_id], device=self.cases.device)]
            )
        else:
            nodes = origins

        nodes = nodes.unique()

        # Map global node IDs to local indices
        global_to_local = {int(n): idx for idx, n in enumerate(nodes)}

        # Build node features from cases at time t
        node_features = rearrange(
            self.cases[nodes, t], "nodes -> nodes 1"
        )  # (N_sub, 1)

        # Build ego-graph edges and weights
        if self.include_target_in_graph:
            src_list, dst_list, weight_list = [], [], []

            for j, origin in enumerate(origins):
                # Only include edge if target is in subgraph
                if region_id in global_to_local:
                    src_list.append(global_to_local[int(origin)])
                    dst_list.append(global_to_local[int(region_id)])
                    weight_list.append(float(flow_weights[j]))

            if src_list:  # Only create edges if we have any
                edge_index = torch.stack(
                    [
                        torch.tensor(src_list, device=self.cases.device),
                        torch.tensor(dst_list, device=self.cases.device),
                    ],
                    dim=0,
                )  # (2, E_sub)
                edge_weight = torch.tensor(
                    weight_list, dtype=torch.float, device=self.cases.device
                )  # (E_sub,)

                # Normalize incoming weights to target
                edge_weight = edge_weight / (edge_weight.sum() + 1e-8)
            else:
                edge_index = torch.empty(
                    (2, 0), dtype=torch.long, device=self.cases.device
                )
                edge_weight = torch.empty(
                    0, dtype=torch.float, device=self.cases.device
                )
        else:
            # No target node in graph - no edges
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.cases.device)
            edge_weight = torch.empty(0, dtype=torch.float, device=self.cases.device)

        target_local_idx = global_to_local.get(int(region_id), -1)

        return {
            "node_features": node_features,  # (N_sub, 1)
            "edge_index": edge_index,  # (2, E_sub) or (2, 0)
            "edge_weight": edge_weight,  # (E_sub,) or (0,)
            "target_local_idx": target_local_idx,
        }

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information for logging and debugging."""
        num_nodes, num_timesteps = self.cases.shape
        biomarker_dim = self.biomarkers.shape[2]

        # Compute statistics
        avg_neighbors = []
        for i in range(min(100, len(self.indices))):  # Sample first 100 for efficiency
            region_id, t0 = self.indices[i]
            neighbors = 0
            for t in range(t0 - self.L + 1, t0 + 1):
                flows_to_i = self.mobility[t][:, region_id]
                significant = torch.sum(flows_to_i > self.min_flow_threshold)
                neighbors += significant.item()
            avg_neighbors.append(neighbors / self.L)

        return {
            "num_samples": len(self),
            "num_nodes": num_nodes,
            "num_timesteps": num_timesteps,
            "biomarker_dim": biomarker_dim,
            "history_length": self.L,
            "forecast_horizon": self.H,
            "min_flow_threshold": self.min_flow_threshold,
            "max_neighbors": self.max_neighbors,
            "avg_neighbors_per_sample": np.mean(avg_neighbors) if avg_neighbors else 0,
        }


def create_ego_graph_dataset(
    zarr_path: str,
    L: int = 14,
    H: int = 7,
    min_flow_threshold: float = 10.0,
    max_neighbors: int = 20,
    device: str | torch.device = "cpu",
) -> GraphEgoDataset:
    """
    Convenience function to create ego-graph dataset from Zarr file.

    Args:
        zarr_path: Path to preprocessed Zarr dataset
        L: History length
        H: Forecast horizon
        min_flow_threshold: Minimum flow threshold
        max_neighbors: Maximum neighbors per ego-graph
        device: Target device for tensor placement

    Returns:
        GraphEgoDataset instance
    """

    # Load data from Zarr
    zarr_data = zarr.open(zarr_path)

    # Extract tensors and transpose using einops
    cases = rearrange(
        torch.from_numpy(zarr_data["node_features"][:, :, 0]),
        "time nodes -> nodes time",
    ).to(device)

    # Extract biomarkers if available
    if "node_features" in zarr_data and zarr_data["node_features"].shape[2] > 1:
        biomarkers = rearrange(
            torch.from_numpy(zarr_data["node_features"][:, :, 1:]),
            "time nodes features -> nodes time features",
        ).to(device)
    else:
        biomarkers = torch.zeros(
            cases.shape[0], cases.shape[1], 0, device=device
        )  # (N, T, 0)

    # Reshape mobility data: [T, E, dim] -> list of [T] matrices [N, N]
    edge_attr = zarr_data["edge_attr"]
    edge_index = torch.from_numpy(zarr_data["edge_index"])

    mobility_list = []
    for t in range(len(edge_attr)):
        mobility_t = torch.zeros(cases.shape[1], cases.shape[1], device=device)
        if edge_attr[t].shape[0] > 0:
            # Build sparse matrix from edge_index and edge_attr using einops
            mobility_t[edge_index[0], edge_index[1]] = rearrange(
                torch.from_numpy(edge_attr[t][:, 0]), "edges -> edges"
            )
        mobility_list.append(mobility_t)

    dataset = GraphEgoDataset(
        cases=cases,
        biomarkers=biomarkers,
        mobility=mobility_list,
        L=L,
        H=H,
        min_flow_threshold=min_flow_threshold,
        max_neighbors=max_neighbors,
    )

    return dataset


# Add missing methods to GraphEgoDataset class
def _get_forecast_horizon(self) -> int:
    """Get the forecast horizon for this dataset."""
    return self.H


# Monkey patch the method onto the class
GraphEgoDataset.get_forecast_horizon = _get_forecast_horizon


def ego_graph_collate_fn(batch, device="cpu"):
    """
    Collate function for ego-graph datasets with device awareness.

    Args:
        batch: List of samples from GraphEgoDataset
        device: Target device for tensor placement (None to keep original device)

    Returns:
        Dict containing batched tensors on specified device
    """
    # Stack basic tensors and transfer to device if specified
    stack_fn = lambda tensors: torch.stack(tensors)
    cases_hist = stack_fn([item["cases_hist"] for item in batch])
    biomarkers_hist = stack_fn([item["biomarkers_hist"] for item in batch])
    cases_future = stack_fn([item["cases_future"] for item in batch])

    if device is not None:
        cases_hist = cases_hist.to(device)
        biomarkers_hist = biomarkers_hist.to(device)
        cases_future = cases_future.to(device)

    # Handle lists of tensors (ego-graph data per time step)
    L = len(batch[0]["node_features_t"])
    node_features_t = []
    edge_index_t = []
    edge_weight_t = []
    target_local_idx = []

    for t in range(L):
        nf = torch.stack([item["node_features_t"][t] for item in batch])
        ei = torch.stack([item["edge_index_t"][t] for item in batch])
        ew = torch.stack([item["edge_weight_t"][t] for item in batch])
        # target_local_idx values are integers, not tensors, so create tensor from them
        tli = torch.tensor([item["target_local_idx"][t] for item in batch])

        if device is not None:
            nf = nf.to(device)
            ei = ei.to(device)
            ew = ew.to(device)
            tli = tli.to(device)

        node_features_t.append(nf)
        edge_index_t.append(ei)
        edge_weight_t.append(ew)
        target_local_idx.append(tli)

    return {
        "cases_hist": cases_hist,
        "biomarkers_hist": biomarkers_hist,
        "cases_future": cases_future,
        "node_features_t": node_features_t,
        "edge_index_t": edge_index_t,
        "edge_weight_t": edge_weight_t,
        "target_local_idx": target_local_idx,
        "target_sequences": cases_future,  # For compatibility with trainer
    }
