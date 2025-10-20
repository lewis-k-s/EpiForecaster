"""
Subgraph loader for temporal mobility sequences using k=1 neighbor sampling.

Implements efficient minibatch training with PyTorch Geometric's NeighborLoader
for temporal epidemic forecasting on mobility graphs.
"""

import logging
from collections.abc import Iterator
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree

from training.windowing import get_window_stats, make_windows_generator

logger = logging.getLogger(__name__)


class TemporalSubgraphLoader:
    """
    Loads k=1 subgraphs for temporal mobility forecasting.

    This loader creates minibatches by sampling target nodes and their 1-hop
    neighborhoods across temporal sequences, enabling efficient training on
    large mobility graphs.
    """

    def __init__(
        self,
        num_neighbors: Union[int, list[int]] = 25,
        batch_size: int = 14,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        degree_balanced_sampling: bool = True,
        min_degree: int = 1,
    ):
        """
        Initialize temporal subgraph loader.

        Args:
            num_neighbors: Number of neighbors to sample per hop (k=1 only)
            batch_size: Number of target nodes per batch
            shuffle: Whether to shuffle target nodes
            drop_last: Whether to drop incomplete batches
            num_workers: Number of worker processes
            degree_balanced_sampling: Whether to balance sampling by node degree
            min_degree: Minimum degree for nodes to be selected as targets
        """
        self.num_neighbors = (
            [num_neighbors] if isinstance(num_neighbors, int) else num_neighbors
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.degree_balanced_sampling = degree_balanced_sampling
        self.min_degree = min_degree

        # Ensure we only use k=1
        if len(self.num_neighbors) > 1:
            logger.warning(
                f"Multiple hop sizes provided: {self.num_neighbors}. Using k=1 only: {self.num_neighbors[0]}"
            )
            self.num_neighbors = [self.num_neighbors[0]]

        logger.info(
            f"Initialized TemporalSubgraphLoader: k=1, neighbors={self.num_neighbors[0]}, batch_size={batch_size}"
        )

    def create_target_node_sampler(self, graph: Data) -> torch.Tensor:
        """
        Create balanced target node indices for sampling.

        Args:
            graph: PyG Data object

        Returns:
            Indices of nodes suitable for target sampling
        """
        if self.degree_balanced_sampling:
            # Calculate node degrees
            node_degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)

            # Filter nodes by minimum degree
            valid_mask = node_degrees >= self.min_degree
            valid_nodes = torch.nonzero(valid_mask).flatten()

            if len(valid_nodes) == 0:
                logger.warning(
                    "No nodes meet minimum degree requirement, using all nodes"
                )
                valid_nodes = torch.arange(graph.num_nodes)

            # Create degree-balanced weights (inverse of degree for balance)
            degrees = node_degrees[valid_nodes].float()
            # Avoid division by zero and extreme weights
            weights = 1.0 / torch.clamp(degrees, min=1.0, max=degrees.quantile(0.9))
            weights = weights / weights.sum()

            logger.info(
                f"Target sampling: {len(valid_nodes)} valid nodes, "
                f"degree range: [{degrees.min():.1f}, {degrees.max():.1f}]"
            )

            return valid_nodes, weights
        else:
            # Simple uniform sampling
            all_nodes = torch.arange(graph.num_nodes)
            weights = torch.ones(graph.num_nodes) / graph.num_nodes
            return all_nodes, weights

    def create_temporal_batches(
        self,
        temporal_graphs: list[Data],
        sequence_length: int = 1,
        forecast_horizon: int = 7,
        stride: int = 1,
    ) -> Iterator[tuple[list[Data], list[Data], torch.Tensor]]:
        """
        Create subgraph batches for temporal sequences using well-tested windowing.

        Args:
            temporal_graphs: List of temporal graph snapshots
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            stride: Step size between consecutive windows (default: 1)

        Yields:
            Tuple of (input_subgraphs, target_subgraphs, target_nodes)
        """
        # Validate input using windowing statistics
        window_stats = get_window_stats(
            temporal_graphs, sequence_length, forecast_horizon, stride
        )

        if not window_stats["valid"]:
            suggested_horizon = max(1, (len(temporal_graphs) - sequence_length) // 3)
            raise ValueError(
                f"Insufficient temporal data for subgraph training:\n"
                f"  - Available temporal graphs: {len(temporal_graphs)}\n"
                f"  - Required: {window_stats['min_required_graphs']} (seq_len={sequence_length} + horizon={forecast_horizon})\n"
                f"  - Valid windows: {window_stats['num_windows']}\n"
                f"  - Suggestion: Use --forecast_horizon {suggested_horizon} or provide more temporal data"
            )

        logger.info("Temporal windowing analysis for subgraph batches:")
        logger.info(f"  Total graphs: {window_stats['total_graphs']}")
        logger.info(f"  Valid windows: {window_stats['num_windows']}")
        logger.info(f"  Window coverage: {window_stats['window_coverage']:.1%}")
        logger.info(f"  Memory estimate: {window_stats['memory_estimate_mb']:.1f} MB")

        # Use the most recent graph for target node sampling (structure should be consistent)
        reference_graph = temporal_graphs[-1]
        target_candidates, sampling_weights = self.create_target_node_sampler(
            reference_graph
        )

        # Use well-tested windowing generator for creating temporal windows
        windows = list(
            make_windows_generator(
                temporal_graphs,
                seq_len=sequence_length,
                horizon=forecast_horizon,
                stride=stride,
                validate_consistency=True,  # Use windowing module's validation
            )
        )

        total_samples = len(windows) * len(target_candidates)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size

        if self.drop_last:
            num_batches = total_samples // self.batch_size

        logger.info(
            f"Creating temporal batches: {len(windows)} windows, "
            f"{len(target_candidates)} target candidates, {num_batches} batches"
        )

        # Create all (window_idx, node_idx) pairs
        all_samples = []
        for window_idx, (_input_seq, _target_seq) in enumerate(windows):
            for node_idx in target_candidates:
                all_samples.append((window_idx, node_idx.item()))

        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(all_samples)

        # Create batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_samples))

            batch_samples = all_samples[start_idx:end_idx]

            if len(batch_samples) == 0:
                continue

            # Group by window index for efficient subgraph extraction
            window_groups = {}
            for window_idx, node_idx in batch_samples:
                if window_idx not in window_groups:
                    window_groups[window_idx] = []
                window_groups[window_idx].append(node_idx)

            # Process each window group separately to maintain sequence integrity
            for window_idx, node_indices in window_groups.items():
                input_sequence, target_sequence = windows[window_idx]

                # Create subgraphs for this window
                target_nodes_tensor = torch.tensor(node_indices, dtype=torch.long)

                # Extract subgraphs using NeighborLoader (only for input, not target)
                input_subgraphs = self._extract_subgraphs(
                    input_sequence, target_nodes_tensor
                )

                # Yield individual window data to maintain sequence length integrity
                yield input_subgraphs, target_sequence, target_nodes_tensor

    def _extract_subgraphs(
        self, graph_sequence: list[Data], target_nodes: torch.Tensor
    ) -> list[Data]:
        """
        Extract k=1 subgraphs around target nodes for a sequence of graphs.

        Args:
            graph_sequence: Temporal sequence of graphs
            target_nodes: Target node indices

        Returns:
            List of subgraph Data objects with n_id mapping preserved
        """
        subgraphs = []

        for graph in graph_sequence:
            # Create NeighborLoader for this graph
            loader = NeighborLoader(
                data=graph,
                num_neighbors=self.num_neighbors,  # k=1 only
                input_nodes=target_nodes,
                batch_size=len(target_nodes),  # Process all targets together
                shuffle=False,
                num_workers=0,  # Disable multiprocessing for temporal consistency
            )

            # Extract the single batch (all target nodes)
            for subgraph_batch in loader:
                # Verify n_id is preserved (should be automatic with NeighborLoader)
                if not hasattr(subgraph_batch, "n_id") or subgraph_batch.n_id is None:
                    logger.error(
                        f"NeighborLoader failed to preserve n_id mapping. "
                        f"Subgraph has {subgraph_batch.num_nodes} nodes but no n_id."
                    )
                    raise RuntimeError(
                        "Critical: Node identity mapping lost during subgraph extraction"
                    )

                # Validate that all target nodes are included in the subgraph
                missing_targets = []
                for target_idx in target_nodes:
                    if target_idx not in subgraph_batch.n_id:
                        missing_targets.append(target_idx.item())

                if missing_targets:
                    logger.warning(
                        f"Target nodes {missing_targets} missing from subgraph. "
                        f"This may cause target extraction errors."
                    )

                subgraphs.append(subgraph_batch)
                break  # Only need the first (and only) batch

        return subgraphs

    def get_batch_stats(self, temporal_graphs: list[Data]) -> dict:
        """
        Get statistics about batch sizes and coverage.

        Args:
            temporal_graphs: List of temporal graphs

        Returns:
            Dictionary with batch statistics
        """
        if not temporal_graphs:
            return {}

        reference_graph = temporal_graphs[0]
        target_candidates, _ = self.create_target_node_sampler(reference_graph)

        # Estimate subgraph sizes using a sample
        sample_targets = target_candidates[: min(10, len(target_candidates))]
        sample_subgraphs = self._extract_subgraphs([reference_graph], sample_targets)

        if sample_subgraphs:
            sample_subgraph = sample_subgraphs[0]
            avg_subgraph_nodes = sample_subgraph.num_nodes / len(sample_targets)
            avg_subgraph_edges = sample_subgraph.num_edges / len(sample_targets)
        else:
            avg_subgraph_nodes = 0
            avg_subgraph_edges = 0

        coverage_per_target = avg_subgraph_nodes / reference_graph.num_nodes
        coverage_per_batch = min(1.0, self.batch_size * coverage_per_target)

        return {
            "total_nodes": reference_graph.num_nodes,
            "target_candidates": len(target_candidates),
            "batch_size": self.batch_size,
            "num_neighbors": self.num_neighbors[0],
            "avg_subgraph_nodes": avg_subgraph_nodes,
            "avg_subgraph_edges": avg_subgraph_edges,
            "coverage_per_target": coverage_per_target * 100,  # Percentage
            "coverage_per_batch": coverage_per_batch * 100,  # Percentage
            "estimated_batches_per_epoch": len(target_candidates) // self.batch_size,
        }


if __name__ == "__main__":
    # Test the temporal subgraph loader

    # Create dummy temporal graphs
    num_nodes = 100
    num_edges = 200
    num_timesteps = 20

    temporal_graphs = []
    for _t in range(num_timesteps):
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, 10)  # 10-dim node features
        edge_attr = torch.randn(num_edges, 1)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        temporal_graphs.append(graph)

    # Create loader
    loader = TemporalSubgraphLoader(
        num_neighbors=15, batch_size=8, degree_balanced_sampling=True
    )

    # Get statistics
    stats = loader.get_batch_stats(temporal_graphs)
    print("Batch Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test batch creation
    print("\nTesting batch creation...")
    batch_count = 0
    for input_graphs, target_graphs, target_nodes in loader.create_temporal_batches(
        temporal_graphs, sequence_length=1, forecast_horizon=7
    ):
        if batch_count == 0:
            print("First batch:")
            print(f"  Input graphs: {len(input_graphs)}")
            print(f"  Target graphs: {len(target_graphs)}")
            print(f"  Target nodes: {len(target_nodes)}")
            if input_graphs:
                print(f"  Subgraph nodes: {input_graphs[0].num_nodes}")
                print(f"  Subgraph edges: {input_graphs[0].num_edges}")

        batch_count += 1
        if batch_count >= 3:  # Test first few batches
            break

    print(f"Successfully created {batch_count} test batches")
