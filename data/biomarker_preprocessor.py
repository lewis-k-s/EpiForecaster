from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class BiomarkerScalerParams:
    """Fitted scaler parameters for train-only scaling.

    Stores robust scaling parameters (median/IQR) computed from training data.
    """

    center: float
    scale: float
    is_fitted: bool = False


class BiomarkerPreprocessor:
    """Handles 3-channel biomarker encoding with train-only robust scaling.

    Features:
    - Value channel: LOCF with log1p + robust scaling + clipping [-8, 8]
    - Mask channel: 1.0 if measured today, else 0.0
    - Age channel: Days since last measurement, normalized to [0, 1] (max 14 days)
    """

    def __init__(
        self, age_max: int = 14, clip_range: tuple[float, float] = (-8.0, 8.0)
    ) -> None:
        self.age_max = age_max
        self.clip_range = clip_range
        self.scaler_params: BiomarkerScalerParams | None = None

    def fit_scaler(self, dataset: xr.Dataset, train_nodes: list[int]) -> None:
        """Fit robust scalers on train nodes only (all timesteps).

        Args:
            dataset: xarray Dataset containing edar_biomarker variable
            train_nodes: List of region indices for training split
        """
        biomarker_da = dataset.edar_biomarker

        train_mask = np.isin(biomarker_da.region_id.values, train_nodes)

        all_values = biomarker_da.isel(region_id=train_mask).values

        # Exclude zeros (below detection limit) from scaler fitting
        finite_values = all_values[np.isfinite(all_values) & (all_values > 0)]

        if len(finite_values) == 0:
            raise ValueError("No finite biomarker values in train nodes")

        log_values = np.log1p(finite_values)

        center = np.median(log_values)
        q75 = np.percentile(log_values, 75)
        q25 = np.percentile(log_values, 25)
        scale = np.maximum(q75 - q25, 1.0)

        self.scaler_params = BiomarkerScalerParams(
            center=float(center), scale=float(scale), is_fitted=True
        )

    def set_scaler_params(self, params: BiomarkerScalerParams) -> None:
        """Set pre-fitted scaler params (for val/test datasets).

        Args:
            params: Scaler parameters fitted on training data
        """
        self.scaler_params = params

    def preprocess_dataset(self, dataset: xr.Dataset) -> np.ndarray:
        """Vectorized preprocessing of the entire dataset.

        Returns:
            np.ndarray: Shape (time, nodes, 3) containing [value, mask, age] channels.
        """
        # (time, nodes)
        biomarker_da = dataset.edar_biomarker

        # expect 1 feature currently
        if not biomarker_da.ndim == 2:
            raise ValueError("Biomarker data must be 2-dimensional")

        # ensure expected coord ordering
        biomarker_da = biomarker_da.transpose("date", "region_id")

        values = biomarker_da.values  # (T, N)
        T, N = values.shape

        # --- Mask Channel ---
        # 1.0 if measured (finite and positive), 0.0 otherwise
        # Zeros are below detection limit and treated as non-measurements
        mask_channel = (np.isfinite(values) & (values > 0)).astype(np.float32)

        # --- Value Channel (LOCF + Log + Scale) ---
        # Convert zeros/negatives to NaN for LOCF (they mean "not measured")
        values_for_locf = np.where(values > 0, values, np.nan)
        df = pd.DataFrame(values_for_locf)
        # ffill propagates last valid observation forward
        # fillna(0) handles leading NaNs (before first measurement)
        filled_values = df.ffill().fillna(0.0).values

        # Log1p
        value_channel = np.log1p(filled_values)

        # Robust Scaling
        if self.scaler_params and self.scaler_params.is_fitted:
            value_channel = (
                value_channel - self.scaler_params.center
            ) / self.scaler_params.scale
            value_channel = np.clip(value_channel, *self.clip_range)

        value_channel = value_channel.astype(np.float32)

        # --- Age Channel ---
        # Calculate days since last measurement
        # We can use the mask to find indices of measurements
        age_channel = np.full_like(values, self.age_max, dtype=np.float32)

        # Vectorized age calculation
        # We want: for each t, how many steps back was the last non-nan value?
        # Approach:
        # 1. Create an array of time indices [0, 1, ..., T-1] broadcasted to (T, N)
        # 2. Where mask is 1, keep the index. Where mask is 0, set to NaN or -1.
        # 3. Forward fill these indices to propagate "last seen time".
        # 4. Age = current_time - last_seen_time.

        time_indices = np.arange(T)[:, None]  # (T, 1)
        # broadcast to (T, N)
        last_seen_indices = np.where(mask_channel > 0, time_indices, np.nan)

        last_seen_df = pd.DataFrame(last_seen_indices)
        last_seen_filled = last_seen_df.ffill().values  # Propagate last seen index

        # Calculate diff. For leading NaNs (no previous measurement), age remains max_age
        # We only update where we have a valid last_seen
        valid_history_mask = ~np.isnan(last_seen_filled)

        current_age = np.zeros_like(age_channel)
        # age = t - last_seen_t
        # We need to be careful with broadcasting. time_indices is (T, 1).
        current_age[valid_history_mask] = (
            time_indices * np.ones((1, N)) - last_seen_filled
        )[valid_history_mask]

        # Where we haven't seen any data yet, set to max_age (already initialized or set here)
        # leading NaNs are age_max.

        # clip to age_max
        final_age = np.where(
            valid_history_mask, np.minimum(current_age, self.age_max), self.age_max
        )

        # Normalize
        age_channel = final_age / self.age_max
        age_channel = age_channel.astype(np.float32)

        # Stack channels: (T, N, 3)
        return np.stack([value_channel, mask_channel, age_channel], axis=-1)
