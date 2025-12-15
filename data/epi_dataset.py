from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torch_geometric.data import Data

from graph.node_encoder import Region2Vec
from models.configs import EpiForecasterConfig

from .preprocess.config import REGION_COORD, TEMPORAL_COORD

StaticCovariates = dict[str, torch.Tensor]


class EpiDatasetItem(TypedDict):
    node_label: str
    target_node: int
    case_node: torch.Tensor
    bio_node: torch.Tensor
    target: torch.Tensor
    mob: list[Data]
    static_covariates: StaticCovariates


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
        aligned_data_path: Path,
        region2vec_path: Path | None,
        config: EpiForecasterConfig,
        target_nodes: list[int],
        context_nodes: list[int],
    ):
        self.aligned_data_path = Path(aligned_data_path)
        self.config = config

        # Load dataset
        self.dataset: xr.Dataset = xr.open_zarr(self.aligned_data_path)
        self.num_nodes = self.dataset[REGION_COORD].size

        if region2vec_path:
            # use pre-trained region2vec embeddings and lookup by labeled regions
            # TODO: actually run forward pass region2vec in EpiForecaster
            _, art = Region2Vec.from_weights(region2vec_path)

            # Filter region embeddings using numpy instead of xarray to avoid requiring a named dimension
            # for the embedding size (since xarray expects all dims to be named).
            region_ids = list(art["region_ids"])
            selected_ids = list(self.dataset[REGION_COORD].values)
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
        self.context_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.context_mask[target_nodes] = True
        self.context_mask[context_nodes] = True

        # Set dimensions
        self.time_dim_size = config.model.history_length + config.model.forecast_horizon

        self.node_static_covariates = self.static_covariates()

    def __len__(self) -> int:
        """Number of samples in the dataset.

        One sample corresponds to a (window, node) pair. We retain the
        non-overlapping stride of `history_length` and multiply by the number
        of nodes to expose a per-node item API.
        """
        L = self.config.model.history_length
        H = self.config.model.forecast_horizon
        T = len(self.dataset[TEMPORAL_COORD].values)
        if T < L + H:
            return 0
        total_windows = ((T - (L + H)) // L) + 1
        return total_windows * len(self.target_nodes)

    def __getitem__(self, idx: int) -> EpiDatasetItem:
        """Return a single target node over one time window.

        Each item is keyed by (window_idx, target_node). The mobility slice is
        converted to a PyG ego-graph per time step containing the target node
        and its incoming neighbors.
        """

        N = len(self.target_nodes)
        L = self.config.model.history_length
        H = self.config.model.forecast_horizon
        BDim = self.config.model.biomarkers_dim
        CDim = self.config.model.cases_dim

        window_idx, local_idx = divmod(idx, N)
        target_idx = self.target_nodes[local_idx]

        node_label = self.dataset[REGION_COORD].values[target_idx]

        range_start = window_idx * L
        range_end = range_start + L
        forecast_targets = range_end + H
        T = len(self.dataset[TEMPORAL_COORD].values)
        if forecast_targets > T:
            raise IndexError("Requested window exceeds available time steps")

        history = self.dataset.isel({TEMPORAL_COORD: slice(range_start, range_end)})
        future = self.dataset.isel({TEMPORAL_COORD: slice(range_end, forecast_targets)})

        mobility_history = torch.from_numpy(history.mobility.values).to(torch.float32)

        cases_np = history.cases.values
        if cases_np.ndim == 2:
            cases_np = cases_np[..., None]
        cases_np = np.nan_to_num(cases_np, nan=0.0)
        case_history = torch.from_numpy(cases_np).to(torch.float32)

        bio_np = history.edar_biomarker.values
        if bio_np.ndim == 2:
            bio_np = bio_np[..., None]
        bio_np = np.nan_to_num(bio_np, nan=0.0)
        biomarker_history = torch.from_numpy(bio_np).to(torch.float32)

        target_np = future.cases.isel({REGION_COORD: target_idx}).values
        targets = torch.from_numpy(target_np).to(torch.float32).squeeze(-1)

        assert mobility_history.shape == (L, self.num_nodes, self.num_nodes), (
            "Mob history shape mismatch"
        )
        assert case_history.shape == (L, self.num_nodes, CDim), (
            "Case history shape mismatch"
        )
        assert biomarker_history.shape == (L, self.num_nodes, BDim), (
            "Biomarker history shape mismatch"
        )
        assert targets.shape == (H,), "Targets shape mismatch"

        mob_graphs: list[Data] = []
        for t in range(L):
            mob_graphs.append(
                self._mobility_ego_to_pyg(
                    mobility_history[t],
                    case_history[t],
                    biomarker_history[t],
                    target_idx,
                    time_id=t,
                )
            )

        return {
            "node_label": node_label,
            "target_node": target_idx,
            "case_node": case_history[:, target_idx, :],
            "bio_node": biomarker_history[:, target_idx, :],
            "target": targets,
            "mob": mob_graphs,
            "static_covariates": self.node_static_covariates,
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

    def _mobility_ego_to_pyg(
        self,
        mob_t: torch.Tensor,
        case_t: torch.Tensor,
        bio_t: torch.Tensor,
        dst_idx: int,
        time_id: int | None = None,
    ) -> Data:
        """Convert a dense mobility slice into a PyG ego-graph for one target node.

        Nodes include the target and all origins with non-zero incoming flow. Edges
        are directed origin -> target and store the raw flow weight.
        """
        if self.context_mask is not None:
            mob_t = mob_t.clone()
            mob_t[~self.context_mask, dst_idx] = 0

        inflow = mob_t[:, dst_idx]
        origins = inflow.nonzero(as_tuple=False).flatten()

        node_ids = torch.cat(
            [origins, torch.tensor([dst_idx], dtype=torch.long)], dim=0
        )
        id_map = {int(n): i for i, n in enumerate(node_ids)}

        if origins.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=mob_t.dtype)
        else:
            edge_index = torch.tensor(
                [[id_map[int(o)], id_map[dst_idx]] for o in origins],
                dtype=torch.long,
            ).t()
            edge_weight = inflow[origins]

        x = torch.cat([case_t[node_ids], bio_t[node_ids]], dim=-1)

        g = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        g.num_nodes = node_ids.numel()
        g.target_node = torch.tensor([id_map[dst_idx]], dtype=torch.long)
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

    @classmethod
    def load_canonical_dataset(cls, aligned_data_path: Path) -> xr.Dataset:
        "Load the canonical dataset from the aligned data path."
        return xr.open_zarr(aligned_data_path)
