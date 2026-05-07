import numpy as np

from dataviz.adjacency_graph_comparison import (
    _build_spatial_knn,
    _ego_receptive_field,
)


def test_spatial_knn_is_symmetric_and_excludes_diagonal() -> None:
    centroids = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=np.float32,
    )

    adjacency, scores = _build_spatial_knn(centroids, k=1)

    assert adjacency.dtype == bool
    assert np.array_equal(adjacency, adjacency.T)
    assert not adjacency.diagonal().any()
    assert adjacency[0, 1]
    assert adjacency[1, 2]
    assert adjacency[2, 3]
    assert scores[adjacency].min() > 0.0


def test_ego_receptive_field_expands_through_incoming_edges() -> None:
    adjacency = np.zeros((4, 4), dtype=bool)
    adjacency[1, 0] = True
    adjacency[2, 1] = True
    adjacency[3, 2] = True
    scores = adjacency.astype(np.float32)

    assert _ego_receptive_field(
        adjacency,
        scores,
        center=0,
        max_hops=2,
        neighbor_ceiling=None,
    ) == {0, 1, 2}


def test_ego_receptive_field_applies_neighbor_ceiling_per_hop() -> None:
    adjacency = np.zeros((4, 4), dtype=bool)
    adjacency[1, 0] = True
    adjacency[2, 0] = True
    adjacency[3, 0] = True
    scores = np.zeros((4, 4), dtype=np.float32)
    scores[1, 0] = 0.5
    scores[2, 0] = 2.0
    scores[3, 0] = 1.0

    assert _ego_receptive_field(
        adjacency,
        scores,
        center=0,
        max_hops=1,
        neighbor_ceiling=2,
    ) == {0, 2, 3}
