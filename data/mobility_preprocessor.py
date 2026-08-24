from dataclasses import dataclass

import numpy as np
import xarray as xr

from .preprocess.config import TEMPORAL_COORD


@dataclass(frozen=True)
class MobilityPreprocessorConfig:
    """Configuration for selecting and weighting mobility edges."""

    min_flow_raw: float = 10.0
    weight_clip_max: float = 8.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_flow_raw) or self.min_flow_raw < 0:
            raise ValueError("min_flow_raw must be finite and non-negative")
        if (
            not np.isfinite(self.weight_clip_max)
            or self.weight_clip_max <= 0
            or self.weight_clip_max > np.finfo(np.float16).max
        ):
            raise ValueError(
                "weight_clip_max must be finite, positive, and representable in float16"
            )


class MobilityPreprocessor:
    """Select mobility edges in raw-flow units and build nonnegative weights.

    The canonical Zarr stores ``log1p(raw_flow)``. Edge selection therefore uses
    ``log1p(min_flow_raw)`` directly in stored space, before upper clipping.
    Retained weights remain in stored log space; there is no centering.
    """

    def __init__(self, config: MobilityPreprocessorConfig | None = None) -> None:
        self.config = config or MobilityPreprocessorConfig()

    @property
    def stored_log_threshold(self) -> float:
        """Return the exact stored-space equivalent of the raw-flow threshold."""
        return float(np.log1p(self.config.min_flow_raw))

    def build_edge_mask(
        self,
        stored_log_flow: np.ndarray,
        min_flow_raw: float | None = None,
    ) -> np.ndarray:
        """Select positive, finite, non-self edges before weight transformation."""
        threshold_raw = (
            self.config.min_flow_raw if min_flow_raw is None else float(min_flow_raw)
        )
        if not np.isfinite(threshold_raw) or threshold_raw < 0:
            raise ValueError("min_flow_raw must be finite and non-negative")

        values = np.asarray(stored_log_flow)
        threshold_log = np.log1p(threshold_raw)
        edge_mask = np.isfinite(values) & (values > 0.0) & (values >= threshold_log)
        if values.ndim >= 2 and values.shape[-2] == values.shape[-1]:
            diagonal = np.arange(values.shape[-1])
            edge_mask[..., diagonal, diagonal] = False
        return edge_mask

    def transform_edge_weights(
        self,
        stored_log_flow: np.ndarray,
        edge_mask: np.ndarray,
    ) -> np.ndarray:
        """Upper-clip retained log-space weights and keep all other edges zero."""
        values = np.asarray(stored_log_flow)
        mask = np.asarray(edge_mask, dtype=bool)
        if values.shape != mask.shape:
            raise ValueError(
                "stored_log_flow and edge_mask must have identical shapes, got "
                f"{values.shape} and {mask.shape}"
            )

        finite_nonnegative = np.nan_to_num(
            values.astype(np.float32, copy=False),
            nan=0.0,
            posinf=self.config.weight_clip_max,
            neginf=0.0,
        )
        finite_nonnegative = np.clip(
            finite_nonnegative,
            0.0,
            self.config.weight_clip_max,
        )
        weights = np.where(mask, finite_nonnegative, 0.0).astype(np.float32)
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(
                "Mobility weight transform must produce finite nonnegative values"
            )
        return weights

    def preprocess_dataset(self, dataset: xr.Dataset) -> np.ndarray:
        """Select and transform a full canonical mobility tensor."""
        mobility_da = dataset.mobility
        mobility_da = self._ensure_time_first(mobility_da)
        values = mobility_da.values
        edge_mask = self.build_edge_mask(values)
        return self.transform_edge_weights(values, edge_mask)

    def _ensure_time_first(self, mobility_da: xr.DataArray) -> xr.DataArray:
        if TEMPORAL_COORD in mobility_da.dims:
            spatial_dims = [d for d in mobility_da.dims if d != TEMPORAL_COORD]
            if len(spatial_dims) != 2:
                raise ValueError(
                    f"Expected mobility array with 2 spatial dims, got {mobility_da.dims}"
                )
            return mobility_da.transpose(
                TEMPORAL_COORD, spatial_dims[0], spatial_dims[1]
            )
        return mobility_da

    @staticmethod
    def compute_imported_risk(
        cases: np.ndarray,
        mobility: np.ndarray,
        lags: list[int],
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Compute mobility-weighted lagged case features (imported risk).

        Computes Risk[t, i] = sum_j (Mobility[t, j, i] * Cases[t-lag, j])
        using normalized incoming flow weights.

        Args:
            cases: Normalized cases array (T, N, 1) or (T, N)
            mobility: Mobility matrix (T, N, N) or (N, N)
                      mobility[..., j, i] is flow from j to i
            lags: List of lag days to compute (e.g. [1, 7, 14])
            epsilon: Small constant for normalization stability

        Returns:
            Array of shape (T, N, len(lags)) containing imported risk features.
        """
        # Ensure cases is (T, N, 1); reject extra trailing channels up front so
        # every return path (including the empty-lag one) enforces the contract.
        if cases.ndim == 2:
            cases = cases[..., None]

        T, N, channels = cases.shape
        if channels != 1:
            raise ValueError(
                "cases must be (T, N) or (T, N, 1) for imported risk; "
                f"got trailing dimension {channels}"
            )

        if not lags:
            return np.zeros((T, N, 0), dtype=np.float32)

        # Normalize mobility (incoming flow normalization)
        # Sum over origins (axis -2) for each destination
        # Mobility shape: (..., origin, dest)
        incoming_sums = np.sum(mobility, axis=-2, keepdims=True)
        mobility_norm = mobility / (incoming_sums + epsilon)

        # Transpose to (..., dest, origin) for matmul: Dest <- Origin
        if mobility_norm.ndim == 3:
            # (T, N, N) -> (T, N, N)
            mob_t = mobility_norm.transpose(0, 2, 1)
        else:
            # (N, N) -> (N, N)
            mob_t = mobility_norm.transpose(1, 0)

        features = []
        for lag in lags:
            # Shift cases by lag (pad with 0 at start)
            # cases[t] should be cases[t-lag]
            shifted = np.zeros_like(cases)
            if lag < T:
                shifted[lag:] = cases[:-lag]

            # Compute Risk = M.T @ C_shifted
            # (..., N, N) @ (..., N, 1) -> (..., N, 1)
            risk = np.matmul(mob_t, shifted)
            features.append(risk)

        return np.concatenate(features, axis=-1).astype(np.float32)
