"""
Temporal windowing module for creating sliding window samples from time series graph data.

This module provides utilities for creating temporal windows after train/test/val splits,
enabling flexible sequence modeling for graph neural network time series forecasting.
"""

import logging
from collections.abc import Iterator
from typing import Optional

from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def make_windows(
    graphs: list[Data],
    seq_len: int,
    horizon: int,
    stride: int = 1,
    validate_consistency: bool = True,
) -> list[tuple[list[Data], list[Data]]]:
    """
    Create sliding windows from temporal graph data for time series forecasting.

    Creates sliding windows with the specified input sequence length and forecast horizon.
    Each window contains an input sequence of `seq_len` graphs followed by a target
    sequence of `horizon` graphs.

    Args:
        graphs: List of temporal graphs in chronological order
        seq_len: Length of input sequence (e.g., 14 for 14-day history)
        horizon: Length of forecast sequence (e.g., 7 for 7-day forecast)
        stride: Step size between consecutive windows (default: 1)
        validate_consistency: Whether to validate graph structure consistency (default: True)

    Returns:
        List of (input_sequence, target_sequence) tuples where:
        - input_sequence: List of `seq_len` consecutive graphs
        - target_sequence: List of `horizon` consecutive graphs

    Raises:
        ValueError: If insufficient data or invalid parameters

    Example:
        >>> temporal_graphs = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        >>> windows = make_windows(temporal_graphs, seq_len=3, horizon=2)
        >>> len(windows)  # (10 - 3 - 2 + 1) = 6 windows
        6
        >>> input_seq, target_seq = windows[0]
        >>> len(input_seq), len(target_seq)
        (3, 2)  # [g1,g2,g3] -> [g4,g5]
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    total_graphs = len(graphs)
    min_required = seq_len + horizon

    if total_graphs < min_required:
        raise ValueError(
            f"Insufficient temporal data for windowing:\n"
            f"  Available graphs: {total_graphs}\n"
            f"  Required: {min_required} (seq_len={seq_len} + horizon={horizon})\n"
            f"  Need at least {min_required - total_graphs} more timesteps"
        )

    # Calculate number of valid windows
    num_windows = (total_graphs - min_required) // stride + 1

    if num_windows <= 0:
        logger.warning(
            f"No valid windows can be created with current parameters:\n"
            f"  Graphs: {total_graphs}, seq_len: {seq_len}, horizon: {horizon}, stride: {stride}"
        )
        return []

    # Validate graph structure consistency if requested
    if validate_consistency and total_graphs > 0:
        _validate_graph_consistency(graphs)

    logger.info(
        f"Creating {num_windows} temporal windows "
        f"(seq_len={seq_len}, horizon={horizon}, stride={stride})"
    )

    windows = []
    for i in range(0, total_graphs - min_required + 1, stride):
        input_sequence = graphs[i : i + seq_len]
        target_sequence = graphs[i + seq_len : i + seq_len + horizon]

        # Sanity check
        assert len(input_sequence) == seq_len, (
            f"Input sequence length mismatch: {len(input_sequence)} != {seq_len}"
        )
        assert len(target_sequence) == horizon, (
            f"Target sequence length mismatch: {len(target_sequence)} != {horizon}"
        )

        windows.append((input_sequence, target_sequence))

    logger.debug(f"Created {len(windows)} temporal windows from {total_graphs} graphs")
    return windows


def make_windows_generator(
    graphs: list[Data],
    seq_len: int,
    horizon: int,
    stride: int = 1,
    validate_consistency: bool = True,
) -> Iterator[tuple[list[Data], list[Data]]]:
    """
    Memory-efficient generator version of make_windows for large datasets.

    Creates sliding windows lazily without storing all windows in memory.
    Useful for very large temporal datasets where memory usage is a concern.

    Args:
        graphs: List of temporal graphs in chronological order
        seq_len: Length of input sequence
        horizon: Length of forecast sequence
        stride: Step size between consecutive windows (default: 1)
        validate_consistency: Whether to validate graph structure consistency (default: True)

    Yields:
        Tuple of (input_sequence, target_sequence)

    Example:
        >>> for input_seq, target_seq in make_windows_generator(graphs, 14, 7):
        ...     # Process window without storing all in memory
        ...     process_window(input_seq, target_seq)
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    total_graphs = len(graphs)
    min_required = seq_len + horizon

    if total_graphs < min_required:
        raise ValueError(
            f"Insufficient temporal data for windowing:\n"
            f"  Available graphs: {total_graphs}\n"
            f"  Required: {min_required} (seq_len={seq_len} + horizon={horizon})"
        )

    # Validate graph structure consistency if requested
    if validate_consistency and total_graphs > 0:
        _validate_graph_consistency(graphs)

    num_windows = (total_graphs - min_required) // stride + 1
    logger.info(
        f"Creating generator for {num_windows} temporal windows "
        f"(seq_len={seq_len}, horizon={horizon}, stride={stride})"
    )

    for i in range(0, total_graphs - min_required + 1, stride):
        input_sequence = graphs[i : i + seq_len]
        target_sequence = graphs[i + seq_len : i + seq_len + horizon]

        yield (input_sequence, target_sequence)


def get_window_stats(
    graphs: list[Data], seq_len: int, horizon: int, stride: int = 1
) -> dict:
    """
    Get statistics about temporal windows without creating them.

    Useful for planning memory usage and understanding data requirements.

    Args:
        graphs: List of temporal graphs
        seq_len: Input sequence length
        horizon: Forecast horizon
        stride: Step size between windows

    Returns:
        Dictionary with window statistics:
        - num_windows: Number of valid windows
        - total_graphs: Total number of input graphs
        - window_coverage: Fraction of data covered by windows
        - memory_estimate_mb: Rough memory estimate for all windows
    """
    total_graphs = len(graphs)
    min_required = seq_len + horizon

    if total_graphs < min_required:
        num_windows = 0
        coverage = 0.0
    else:
        num_windows = (total_graphs - min_required) // stride + 1
        last_window_end = (num_windows - 1) * stride + seq_len + horizon
        coverage = last_window_end / total_graphs

    # Rough memory estimate (assuming PyG Data objects ~1KB each on average)
    memory_per_window_kb = (seq_len + horizon) * 1  # 1KB per graph estimate
    total_memory_mb = (num_windows * memory_per_window_kb) / 1024

    return {
        "num_windows": num_windows,
        "total_graphs": total_graphs,
        "seq_len": seq_len,
        "horizon": horizon,
        "stride": stride,
        "min_required_graphs": min_required,
        "window_coverage": coverage,
        "memory_estimate_mb": total_memory_mb,
        "valid": num_windows > 0,
    }


def _validate_graph_consistency(graphs: list[Data]) -> None:
    """
    Validate that graphs have consistent structure for windowing.

    Checks that all graphs have the same number of nodes and consistent
    feature dimensions, which is required for proper batching.

    Args:
        graphs: List of temporal graphs to validate

    Raises:
        ValueError: If graphs have inconsistent structure
    """
    if not graphs:
        return

    reference = graphs[0]
    ref_num_nodes = reference.num_nodes
    ref_node_features = reference.x.shape[1] if reference.x is not None else 0
    ref_edge_features = (
        reference.edge_attr.shape[1] if reference.edge_attr is not None else 0
    )

    inconsistencies = []

    for i, graph in enumerate(graphs[1:], 1):
        if graph.num_nodes != ref_num_nodes:
            inconsistencies.append(
                f"Graph {i}: {graph.num_nodes} nodes (expected {ref_num_nodes})"
            )

        if graph.x is not None and graph.x.shape[1] != ref_node_features:
            inconsistencies.append(
                f"Graph {i}: {graph.x.shape[1]} node features (expected {ref_node_features})"
            )

        if (
            graph.edge_attr is not None
            and graph.edge_attr.shape[1] != ref_edge_features
        ):
            inconsistencies.append(
                f"Graph {i}: {graph.edge_attr.shape[1]} edge features (expected {ref_edge_features})"
            )

    if inconsistencies:
        raise ValueError(
            "Graph structure inconsistencies detected:\n"
            + "\n".join(f"  - {issue}" for issue in inconsistencies[:5])
            + (
                f"\n  ... and {len(inconsistencies) - 5} more"
                if len(inconsistencies) > 5
                else ""
            )
        )

    logger.debug(
        f"Graph consistency validation passed: {len(graphs)} graphs with "
        f"{ref_num_nodes} nodes, {ref_node_features} node features, {ref_edge_features} edge features"
    )


def split_temporal_data(
    graphs: list[Data],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: Optional[float] = None,
    min_graphs_per_split: int = 1,
) -> tuple[list[Data], list[Data], list[Data]]:
    """
    Split temporal graphs into train/validation/test sets with proper chronological ordering.

    Maintains temporal order by using early timesteps for training, middle for validation,
    and latest for testing. Ensures each split has minimum required graphs.

    Args:
        graphs: Temporal graphs in chronological order
        train_ratio: Fraction for training (default: 0.6)
        val_ratio: Fraction for validation (default: 0.2)
        test_ratio: Fraction for testing (default: 1 - train_ratio - val_ratio)
        min_graphs_per_split: Minimum graphs required per split (default: 1)

    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)

    Raises:
        ValueError: If ratios are invalid or insufficient data
    """
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio

    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("All ratios must be between 0 and 1")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.6f}"
        )

    total_graphs = len(graphs)
    min_total_required = 3 * min_graphs_per_split

    if total_graphs < min_total_required:
        raise ValueError(
            f"Insufficient data for splitting: {total_graphs} graphs available, "
            f"need at least {min_total_required} ({min_graphs_per_split} per split)"
        )

    # Calculate split points ensuring minimum graphs per split
    train_size = max(min_graphs_per_split, int(total_graphs * train_ratio))
    val_size = max(min_graphs_per_split, int(total_graphs * val_ratio))
    test_size = max(min_graphs_per_split, total_graphs - train_size - val_size)

    # Adjust if sizes exceed total (shouldn't happen with proper ratios)
    if train_size + val_size + test_size > total_graphs:
        excess = (train_size + val_size + test_size) - total_graphs
        # Remove excess from largest split
        if train_size >= val_size and train_size >= test_size:
            train_size -= excess
        elif val_size >= test_size:
            val_size -= excess
        else:
            test_size -= excess

    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size : train_size + val_size]
    test_graphs = graphs[train_size + val_size : train_size + val_size + test_size]

    logger.info(
        f"Split temporal data: Train={len(train_graphs)}, "
        f"Val={len(val_graphs)}, Test={len(test_graphs)}"
    )

    return train_graphs, val_graphs, test_graphs
