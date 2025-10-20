"""
Unit tests for the temporal windowing module.

Tests the make_windows function and related utilities for creating
sliding window samples from time series graph data.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.windowing import (
    _validate_graph_consistency,
    get_window_stats,
    make_windows,
    make_windows_generator,
    split_temporal_data,
)


def create_mock_graph(num_nodes=10, node_features=3, num_edges=20, seed=None):
    """
    Create a mock PyTorch Geometric Data object for testing.

    Args:
        num_nodes: Number of nodes in the graph
        node_features: Number of features per node
        num_edges: Number of edges
        seed: Random seed for reproducibility

    Returns:
        Data object with random features and edges
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create random node features
    x = torch.randn(num_nodes, node_features)

    # Create random edges (ensuring valid indices)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Create random edge features
    edge_attr = torch.randn(num_edges, 2)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def create_temporal_graphs(num_graphs=10, num_nodes=10, node_features=3):
    """Create a list of temporal graphs for testing."""
    return [
        create_mock_graph(num_nodes, node_features, seed=i) for i in range(num_graphs)
    ]


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_make_windows_basic():
    """Test basic windowing functionality with default parameters."""
    graphs = create_temporal_graphs(10)
    windows = make_windows(graphs, seq_len=3, horizon=2)

    # Expected number of windows: 10 - 3 - 2 + 1 = 6
    assert len(windows) == 6, f"Expected 6 windows, got {len(windows)}"

    # Check each window structure
    for input_seq, target_seq in windows:
        assert len(input_seq) == 3, "Input sequence should have 3 graphs"
        assert len(target_seq) == 2, "Target sequence should have 2 graphs"

        # Verify each is a Data object
        for g in input_seq + target_seq:
            assert isinstance(g, Data), "Each element should be a Data object"


def test_make_windows_exact_fit():
    """Test when data exactly fits window requirements."""
    graphs = create_temporal_graphs(5)  # Exactly seq_len + horizon
    windows = make_windows(graphs, seq_len=3, horizon=2)

    assert len(windows) == 1, "Should create exactly 1 window"
    input_seq, target_seq = windows[0]
    assert len(input_seq) == 3
    assert len(target_seq) == 2


def test_make_windows_single_window():
    """Test creation of a single window with minimal data."""
    graphs = create_temporal_graphs(7)
    windows = make_windows(graphs, seq_len=4, horizon=3)

    assert len(windows) == 1, "Should create exactly 1 window"


# ============================================================================
# Parameter Validation Tests
# ============================================================================


def test_make_windows_invalid_seq_len():
    """Test with invalid sequence length."""
    graphs = create_temporal_graphs(10)

    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        make_windows(graphs, seq_len=0, horizon=2)

    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        make_windows(graphs, seq_len=-1, horizon=2)


def test_make_windows_invalid_horizon():
    """Test with invalid forecast horizon."""
    graphs = create_temporal_graphs(10)

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        make_windows(graphs, seq_len=3, horizon=0)

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        make_windows(graphs, seq_len=3, horizon=-1)


def test_make_windows_invalid_stride():
    """Test with invalid stride parameter."""
    graphs = create_temporal_graphs(10)

    with pytest.raises(ValueError, match="stride must be >= 1"):
        make_windows(graphs, seq_len=3, horizon=2, stride=0)

    with pytest.raises(ValueError, match="stride must be >= 1"):
        make_windows(graphs, seq_len=3, horizon=2, stride=-1)


def test_make_windows_insufficient_data():
    """Test with insufficient temporal data."""
    graphs = create_temporal_graphs(4)  # Less than seq_len + horizon

    with pytest.raises(ValueError, match="Insufficient temporal data"):
        make_windows(graphs, seq_len=3, horizon=2)  # Needs 5, only have 4


# ============================================================================
# Stride Tests
# ============================================================================


def test_make_windows_stride_2():
    """Test windowing with stride=2."""
    graphs = create_temporal_graphs(10)
    windows = make_windows(graphs, seq_len=3, horizon=2, stride=2)

    # With stride=2: (10 - 5) // 2 + 1 = 3 windows
    assert len(windows) == 3, f"Expected 3 windows with stride=2, got {len(windows)}"

    for input_seq, target_seq in windows:
        assert len(input_seq) == 3
        assert len(target_seq) == 2


def test_make_windows_stride_larger():
    """Test with larger stride values."""
    graphs = create_temporal_graphs(20)

    # Test stride=3
    windows = make_windows(graphs, seq_len=4, horizon=3, stride=3)
    expected = (20 - 7) // 3 + 1  # 5 windows
    assert len(windows) == expected

    # Test stride=5
    windows = make_windows(graphs, seq_len=4, horizon=3, stride=5)
    expected = (20 - 7) // 5 + 1  # 3 windows
    assert len(windows) == expected


# ============================================================================
# Edge Cases
# ============================================================================


def test_make_windows_minimal_data():
    """Test with exactly the minimal required data."""
    graphs = create_temporal_graphs(5)
    windows = make_windows(graphs, seq_len=2, horizon=3)

    assert len(windows) == 1
    input_seq, target_seq = windows[0]
    assert len(input_seq) == 2
    assert len(target_seq) == 3


def test_make_windows_empty_list():
    """Test with empty graph list."""
    graphs = []

    with pytest.raises(ValueError, match="Insufficient temporal data"):
        make_windows(graphs, seq_len=3, horizon=2)


def test_make_windows_no_valid_windows():
    """Test when stride is too large to create any windows."""
    graphs = create_temporal_graphs(10)
    windows = make_windows(graphs, seq_len=3, horizon=2, stride=10)

    assert len(windows) == 1, "Should create at least one window at position 0"


# ============================================================================
# Graph Consistency Tests
# ============================================================================


def test_graph_consistency_validation():
    """Test that graph consistency validation works correctly."""
    # Create consistent graphs
    graphs = create_temporal_graphs(5, num_nodes=10, node_features=3)

    # Should not raise any errors
    _validate_graph_consistency(graphs)

    # This should work without errors
    windows = make_windows(graphs, seq_len=2, horizon=2, validate_consistency=True)
    assert len(windows) == 2


def test_graph_inconsistent_nodes():
    """Test validation with inconsistent node counts."""
    graphs = [
        create_mock_graph(num_nodes=10, node_features=3),
        create_mock_graph(num_nodes=15, node_features=3),  # Different node count
        create_mock_graph(num_nodes=10, node_features=3),
    ]

    with pytest.raises(ValueError, match="Graph structure inconsistencies"):
        _validate_graph_consistency(graphs)


def test_graph_inconsistent_features():
    """Test validation with inconsistent feature dimensions."""
    graphs = [
        create_mock_graph(num_nodes=10, node_features=3),
        create_mock_graph(num_nodes=10, node_features=5),  # Different features
        create_mock_graph(num_nodes=10, node_features=3),
    ]

    with pytest.raises(ValueError, match="Graph structure inconsistencies"):
        _validate_graph_consistency(graphs)


def test_graph_consistency_disabled():
    """Test that validation can be disabled."""
    # Create inconsistent graphs
    graphs = [
        create_mock_graph(num_nodes=10, node_features=3),
        create_mock_graph(num_nodes=15, node_features=3),
        create_mock_graph(num_nodes=10, node_features=3),
        create_mock_graph(num_nodes=10, node_features=3),
        create_mock_graph(num_nodes=10, node_features=3),
    ]

    # Should work with validation disabled
    windows = make_windows(graphs, seq_len=2, horizon=2, validate_consistency=False)
    assert len(windows) == 2  # Should create windows despite inconsistency


# ============================================================================
# Generator Function Tests
# ============================================================================


def test_make_windows_generator():
    """Test the generator version of make_windows."""
    graphs = create_temporal_graphs(10)

    # Create generator
    gen = make_windows_generator(graphs, seq_len=3, horizon=2)

    # Collect all windows from generator
    windows_from_gen = list(gen)

    assert len(windows_from_gen) == 6

    for input_seq, target_seq in windows_from_gen:
        assert len(input_seq) == 3
        assert len(target_seq) == 2


def test_generator_vs_list_equivalence():
    """Verify generator produces same results as list version."""
    graphs = create_temporal_graphs(15)

    # Get windows from both methods
    windows_list = make_windows(graphs, seq_len=4, horizon=3, stride=2)
    windows_gen = list(make_windows_generator(graphs, seq_len=4, horizon=3, stride=2))

    assert len(windows_list) == len(windows_gen), (
        "Should produce same number of windows"
    )

    # Check that corresponding windows are identical
    for (list_input, list_target), (gen_input, gen_target) in zip(
        windows_list, windows_gen
    ):
        assert len(list_input) == len(gen_input)
        assert len(list_target) == len(gen_target)

        # Check they reference the same graph objects
        for l_graph, g_graph in zip(list_input + list_target, gen_input + gen_target):
            assert l_graph is g_graph, "Should reference the same graph objects"


# ============================================================================
# Statistics Function Tests
# ============================================================================


def test_get_window_stats():
    """Test window statistics calculation."""
    graphs = create_temporal_graphs(20)
    stats = get_window_stats(graphs, seq_len=5, horizon=3, stride=2)

    assert stats["num_windows"] == 7  # (20 - 8) // 2 + 1
    assert stats["total_graphs"] == 20
    assert stats["seq_len"] == 5
    assert stats["horizon"] == 3
    assert stats["stride"] == 2
    assert stats["min_required_graphs"] == 8
    assert stats["valid"]
    assert 0 < stats["window_coverage"] <= 1.0
    assert stats["memory_estimate_mb"] > 0


def test_get_window_stats_insufficient_data():
    """Test statistics with insufficient data."""
    graphs = create_temporal_graphs(5)
    stats = get_window_stats(graphs, seq_len=4, horizon=3, stride=1)

    assert stats["num_windows"] == 0
    assert stats["total_graphs"] == 5
    assert stats["min_required_graphs"] == 7
    assert not stats["valid"]
    assert stats["window_coverage"] == 0.0


def test_get_window_stats_various_strides():
    """Test statistics calculation with various stride values."""
    graphs = create_temporal_graphs(30)

    # Test different strides
    for stride in [1, 2, 3, 5, 10]:
        stats = get_window_stats(graphs, seq_len=5, horizon=5, stride=stride)
        expected_windows = (30 - 10) // stride + 1
        assert stats["num_windows"] == expected_windows
        assert stats["stride"] == stride


# ============================================================================
# Split Function Tests
# ============================================================================


def test_split_temporal_data():
    """Test temporal data splitting with default ratios."""
    graphs = create_temporal_graphs(100)
    train, val, test = split_temporal_data(graphs)

    # Check sizes (default: 60% train, 20% val, 20% test)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    assert len(train) + len(val) + len(test) == 100

    # Verify chronological order is maintained
    # The graphs should be split sequentially
    assert train[0] is graphs[0]
    assert train[-1] is graphs[59]
    assert val[0] is graphs[60]
    assert test[-1] is graphs[-1]


def test_split_temporal_data_custom_ratios():
    """Test splitting with custom ratios."""
    graphs = create_temporal_graphs(100)
    train, val, test = split_temporal_data(
        graphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_split_temporal_data_minimum_constraint():
    """Test that minimum graphs per split is enforced."""
    graphs = create_temporal_graphs(10)
    train, val, test = split_temporal_data(
        graphs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, min_graphs_per_split=3
    )

    # Each split should have at least 3 graphs
    assert len(train) >= 3
    assert len(val) >= 3
    assert len(test) >= 3
    assert len(train) + len(val) + len(test) == 10


def test_split_temporal_data_insufficient():
    """Test splitting with insufficient data."""
    graphs = create_temporal_graphs(2)  # Only 2 graphs

    with pytest.raises(ValueError, match="Insufficient data for splitting"):
        split_temporal_data(graphs, min_graphs_per_split=1)


def test_split_temporal_data_invalid_ratios():
    """Test with invalid split ratios."""
    graphs = create_temporal_graphs(100)

    # Ratios don't sum to 1
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        split_temporal_data(graphs, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    # Negative ratio
    with pytest.raises(ValueError, match="All ratios must be between 0 and 1"):
        split_temporal_data(graphs, train_ratio=-0.6, val_ratio=0.8, test_ratio=0.8)

    # Ratio > 1
    with pytest.raises(ValueError, match="All ratios must be between 0 and 1"):
        split_temporal_data(graphs, train_ratio=1.5, val_ratio=0.2, test_ratio=0.3)


# ============================================================================
# Integration Tests
# ============================================================================


def test_windowing_pipeline_integration():
    """Test complete windowing pipeline: split -> window."""
    # Create temporal data
    graphs = create_temporal_graphs(100)

    # Split the data
    train, val, test = split_temporal_data(graphs, train_ratio=0.7, val_ratio=0.15)

    # Create windows for each split
    train_windows = make_windows(train, seq_len=10, horizon=5)
    val_windows = make_windows(val, seq_len=10, horizon=5)
    test_windows = make_windows(test, seq_len=10, horizon=5)

    # Verify we get expected number of windows
    assert len(train_windows) == 70 - 10 - 5 + 1  # 56
    assert len(val_windows) == 15 - 10 - 5 + 1  # 1
    assert len(test_windows) == 15 - 10 - 5 + 1  # 1

    # Verify window structure
    for windows in [train_windows, val_windows, test_windows]:
        for input_seq, target_seq in windows:
            assert len(input_seq) == 10
            assert len(target_seq) == 5


def test_windowing_with_different_parameters():
    """Test windowing with various parameter combinations."""
    graphs = create_temporal_graphs(50)

    # Test different parameter combinations
    test_cases = [
        (14, 7, 1),  # Two weeks history, one week forecast
        (7, 1, 1),  # One week history, one day forecast
        (30, 14, 7),  # Monthly history, two week forecast, weekly stride
        (10, 10, 1),  # Equal history and forecast
        (5, 15, 2),  # Shorter history, longer forecast
    ]

    for seq_len, horizon, stride in test_cases:
        # Only test if we have enough data
        if len(graphs) >= seq_len + horizon:
            windows = make_windows(graphs, seq_len, horizon, stride)
            expected = (len(graphs) - seq_len - horizon) // stride + 1
            assert len(windows) == expected

            # Verify structure
            if len(windows) > 0:
                input_seq, target_seq = windows[0]
                assert len(input_seq) == seq_len
                assert len(target_seq) == horizon
