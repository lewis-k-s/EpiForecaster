import numpy as np
import pytest

from data.mobility_preprocessor import (
    MobilityPreprocessor,
    MobilityPreprocessorConfig,
)


@pytest.mark.epiforecaster
def test_raw_flow_threshold_selects_equal_and_above_before_clipping() -> None:
    raw_flow = np.array(
        [
            [100.0, 9.0, 10.0],
            [11.0, 100.0, 100.0],
            [0.0, 1.0, 100.0],
        ],
        dtype=np.float32,
    )
    stored_log_flow = np.log1p(raw_flow)
    preprocessor = MobilityPreprocessor(
        MobilityPreprocessorConfig(
            min_flow_raw=10.0,
            weight_clip_max=3.0,
        )
    )

    edge_mask = preprocessor.build_edge_mask(stored_log_flow)

    assert not edge_mask[0, 1]
    assert edge_mask[0, 2]
    assert edge_mask[1, 0]
    assert edge_mask[1, 2]
    assert not edge_mask[2, 1]
    assert not edge_mask.diagonal().any()
    assert preprocessor.stored_log_threshold == pytest.approx(np.log1p(10.0))


@pytest.mark.epiforecaster
def test_weight_transform_is_masked_finite_nonnegative_and_upper_clipped() -> None:
    stored_log_flow = np.array(
        [
            [np.log1p(100.0), np.log1p(9.0), np.log1p(10.0)],
            [np.log1p(11.0), np.log1p(100.0), np.log1p(1000.0)],
            [-1.0, np.inf, np.nan],
        ],
        dtype=np.float32,
    )
    preprocessor = MobilityPreprocessor(
        MobilityPreprocessorConfig(
            min_flow_raw=10.0,
            weight_clip_max=3.0,
        )
    )
    edge_mask = preprocessor.build_edge_mask(stored_log_flow)

    weights = preprocessor.transform_edge_weights(stored_log_flow, edge_mask)

    assert np.all(weights[~edge_mask] == 0.0)
    assert weights[0, 2] == pytest.approx(np.log1p(10.0))
    assert weights[1, 2] == pytest.approx(3.0)
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)


@pytest.mark.epiforecaster
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("min_flow_raw", -1.0),
        ("weight_clip_max", 0.0),
    ],
)
def test_preprocessor_config_rejects_invalid_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        MobilityPreprocessorConfig(**{field_name: value})
