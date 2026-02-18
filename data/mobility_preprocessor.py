from dataclasses import dataclass
import logging

import numpy as np
import xarray as xr

from .preprocess.config import REGION_COORD, TEMPORAL_COORD

logger = logging.getLogger(__name__)


def _compute_quantiles_via_sort(
    values: np.ndarray, quantiles: list[float]
) -> list[float]:
    """Compute quantiles via sorting to avoid float16 overflow in np.percentile.

    np.percentile internally computes (n-1) * q which overflows float16
    for large arrays (256M+ values). Sorting stays in the input dtype,
    and index calculation uses Python int (unlimited precision).

    Args:
        values: 1D array of finite values (any dtype including float16)
        quantiles: List of quantile values in [0, 1] (e.g., [0.25, 0.5, 0.75])

    Returns:
        List of quantile values as Python floats
    """
    if len(values) == 0:
        raise ValueError("Cannot compute quantiles of empty array")

    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    results = []
    for q in quantiles:
        idx = int((n - 1) * q)
        idx = max(0, min(idx, n - 1))  # Clamp to valid range
        results.append(float(sorted_vals[idx]))

    return results


@dataclass
class MobilityScalerParams:
    """Fitted scaler parameters for train-only mobility scaling.

    Stores robust scaling parameters (median/IQR) computed from training data.
    """

    center: float
    scale: float
    is_fitted: bool = False


@dataclass
class MobilityPreprocessorConfig:
    """Configuration for mobility preprocessing.

    Note: log1p transform is applied in the preprocessing pipeline, not here.
    This preprocessor only applies robust scaling to already-log-transformed values.
    """

    clip_range: tuple[float, float] = (-8.0, 8.0)
    scale_epsilon: float = 1e-6


class MobilityPreprocessor:
    """Handles mobility normalization with train-only robust scaling.

    Values are already log1p-transformed from the preprocessing pipeline.
    This preprocessor applies robust scaling (median/IQR normalization) only.
    """

    def __init__(self, config: MobilityPreprocessorConfig | None = None) -> None:
        self.config = config or MobilityPreprocessorConfig()
        self.scaler_params: MobilityScalerParams | None = None

    def fit_scaler(self, dataset: xr.Dataset, train_nodes: list[int]) -> None:
        """Fit robust scaler on train nodes only (all timesteps).

        Args:
            dataset: xarray Dataset containing mobility variable (already log1p-transformed)
            train_nodes: List of region indices or region IDs for training split
        """
        mobility_da = dataset.mobility
        mobility_train = self._select_train_mobility(mobility_da, train_nodes, dataset)

        values = mobility_train.values
        finite_values = values[np.isfinite(values)].astype(np.float32)
        if len(finite_values) == 0:
            logger.warning(
                "No finite mobility values in train nodes. Using default scaler (center=0, scale=1). "
                "This is expected if using broken synthetic data for testing."
            )
            self.scaler_params = MobilityScalerParams(
                center=0.0, scale=1.0, is_fitted=True
            )
            return

        # Values are already log1p-transformed from preprocessing pipeline
        # Fit scaler directly on log-transformed values
        center = np.median(finite_values)
        q75 = np.percentile(finite_values, 75)
        q25 = np.percentile(finite_values, 25)
        scale = np.maximum(q75 - q25, 1.0)

        self.scaler_params = MobilityScalerParams(
            center=float(center), scale=float(scale), is_fitted=True
        )

    def set_scaler_params(self, params: MobilityScalerParams) -> None:
        """Set pre-fitted scaler params (for val/test datasets)."""
        self.scaler_params = params

    def preprocess_dataset(self, dataset: xr.Dataset) -> np.ndarray:
        """Vectorized preprocessing of the entire dataset.

        Returns:
            np.ndarray: Shape (time, nodes, nodes) containing normalized mobility.
        """
        mobility_da = dataset.mobility
        mobility_da = self._ensure_time_first(mobility_da)
        return self.transform_values(mobility_da.values)

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        """Apply robust scaling to a mobility array (already log1p-transformed).

        Args:
            values: Mobility array with log1p-transformed values

        Returns:
            Robust-scaled mobility array
        """
        out = values.astype(np.float32, copy=True)

        # Values are already log1p-transformed from preprocessing pipeline
        # Apply robust scaling only
        if self.scaler_params and self.scaler_params.is_fitted:
            scale = max(self.scaler_params.scale, self.config.scale_epsilon)
            out = (out - self.scaler_params.center) / scale
            out = np.clip(out, *self.config.clip_range)

        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    def _select_train_mobility(
        self, mobility_da: xr.DataArray, train_nodes: list[int], dataset: xr.Dataset
    ) -> xr.DataArray:
        spatial_dims = [d for d in mobility_da.dims if d != TEMPORAL_COORD]
        if len(spatial_dims) != 2:
            raise ValueError(
                f"Expected mobility array with 2 spatial dims, got {mobility_da.dims}"
            )
        origin_dim, dest_dim = spatial_dims

        region_ids = dataset[REGION_COORD].values
        region_id_index = {rid: i for i, rid in enumerate(region_ids)}
        region_id_set = set(region_id_index.keys())

        if all(node in region_id_set for node in train_nodes):
            train_ids = list(train_nodes)
            train_indices = [region_id_index[rid] for rid in train_ids]
        else:
            train_indices = list(train_nodes)
            train_ids = [region_ids[i] for i in train_indices]

        try:
            return mobility_da.sel({origin_dim: train_ids, dest_dim: train_ids})
        except (KeyError, ValueError):
            return mobility_da.isel(
                {origin_dim: train_indices, dest_dim: train_indices}
            )

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
        if not lags:
            if cases.ndim == 2:
                T, N = cases.shape
            else:
                T, N, _ = cases.shape
            return np.zeros((T, N, 0), dtype=np.float32)

        # Ensure cases is (T, N, 1)
        if cases.ndim == 2:
            cases = cases[..., None]

        T, N, _ = cases.shape

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
