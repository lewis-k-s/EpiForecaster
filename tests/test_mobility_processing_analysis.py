import numpy as np
import pytest

from scripts.analysis.analyze_mobility_processing import (
    finalize_scope,
    new_scope_accumulator,
    update_scope,
)


@pytest.mark.epiforecaster
def test_scope_metrics_use_one_raw_threshold_mask_before_clipping() -> None:
    raw_flow = np.array(
        [
            [
                [0.0, 9.0, 10.0],
                [11.0, 0.0, 100.0],
                [0.0, 1.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    accumulator = new_scope_accumulator("known graph", np.arange(3))

    update_scope(
        accumulator,
        np.log1p(raw_flow),
        min_flow_raw=10.0,
        weight_clip_max=3.0,
    )
    metrics = finalize_scope(accumulator)

    assert metrics["positive_edge_count"] == 5
    assert metrics["selected_edge_count"] == 3
    assert metrics["positive_observation_retained_fraction"] == pytest.approx(3 / 5)
    assert metrics["mean_incoming_degree_unthresholded"] == pytest.approx(5 / 3)
    assert metrics["mean_incoming_degree_effective"] == pytest.approx(1.0)
    assert metrics["clipped_edge_fraction"] == pytest.approx(1 / 3)
    assert metrics["non_finite_weight_count"] == 0
    assert metrics["negative_weight_count"] == 0
