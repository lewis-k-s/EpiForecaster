from dataclasses import dataclass

import numpy as np
import xarray as xr

from .preprocess.config import REGION_COORD, TEMPORAL_COORD


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
    """Configuration for mobility preprocessing."""

    log_scale: bool = True
    clip_range: tuple[float, float] = (-8.0, 8.0)
    scale_epsilon: float = 1e-6


class MobilityPreprocessor:
    """Handles mobility normalization with train-only robust scaling."""

    def __init__(self, config: MobilityPreprocessorConfig | None = None) -> None:
        self.config = config or MobilityPreprocessorConfig()
        self.scaler_params: MobilityScalerParams | None = None

    def fit_scaler(self, dataset: xr.Dataset, train_nodes: list[int]) -> None:
        """Fit robust scaler on train nodes only (all timesteps).

        Args:
            dataset: xarray Dataset containing mobility variable
            train_nodes: List of region indices or region IDs for training split
        """
        mobility_da = dataset.mobility
        mobility_train = self._select_train_mobility(mobility_da, train_nodes, dataset)

        values = mobility_train.values
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            raise ValueError("No finite mobility values in train nodes")

        if self.config.log_scale:
            finite_values = np.log1p(finite_values)

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
        """Apply log + robust scaling to a mobility array."""
        out = values.astype(np.float32, copy=True)

        if self.config.log_scale:
            out = np.log1p(out)

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
            return mobility_da.isel({origin_dim: train_indices, dest_dim: train_indices})

    def _ensure_time_first(self, mobility_da: xr.DataArray) -> xr.DataArray:
        if TEMPORAL_COORD in mobility_da.dims:
            spatial_dims = [d for d in mobility_da.dims if d != TEMPORAL_COORD]
            if len(spatial_dims) != 2:
                raise ValueError(
                    f"Expected mobility array with 2 spatial dims, got {mobility_da.dims}"
                )
            return mobility_da.transpose(TEMPORAL_COORD, spatial_dims[0], spatial_dims[1])
        return mobility_da
