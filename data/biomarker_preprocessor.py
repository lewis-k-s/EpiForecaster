import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from constants import (
    EDAR_BIOMARKER_PREFIX,
    EDAR_BIOMARKER_VARIANTS,
)

logger = logging.getLogger(__name__)


@dataclass
class BiomarkerScalerParams:
    """Fitted scaler parameters for train-only scaling.

    Stores robust scaling parameters (median/IQR) computed from training data.
    """

    center: dict[str, float]
    scale: dict[str, float]
    is_fitted: bool = False


class BiomarkerPreprocessor:
    """Handles per-variant biomarker encoding with train-only robust scaling.

    Features:
    - Value channel: already log1p-transformed from pipeline + robust scaling + clipping [-8, 8]
    - Mask channel: 1.0 if measured today (finite and positive), else 0.0 (required, pre-computed)
    - Censor channel: 1.0 if censored at LD floor (from Kalman), else 0.0 (required, pre-computed)
    - Age channel: Days since last measurement, stored as uint8 (0-14), normalized to [0, 1] at load time

    Channel layout: [value, mask, censor, age] - 4 channels per variant

    NOTE: Mask, censor, and age channels must be pre-computed and present in the dataset.
    Kalman filter already handles interpolation, so no LOCF is applied.
    Values are already log1p-transformed from the preprocessing pipeline.

    IMPORTANT: fit_scaler() requires {variant}_mask channels and only uses values where
    mask > 0 to compute statistics. This ensures interpolated/filled values don't corrupt
    the robust scaling parameters.
    """

    def __init__(
        self, age_max: int = 14, clip_range: tuple[float, float] = (-8.0, 8.0)
    ) -> None:
        self.age_max = age_max
        self.clip_range = clip_range
        self.scaler_params: BiomarkerScalerParams | None = None

    def _get_variant_names(self, dataset: xr.Dataset) -> list[str]:
        """Get biomarker variant names present in the dataset.

        Looks for edar_biomarker_{variant} variables where variant is one of
        EDAR_BIOMARKER_VARIANTS (N1, N2, IP4), excluding channel suffixes.
        """
        expected_names = [
            f"{EDAR_BIOMARKER_PREFIX}{v}" for v in EDAR_BIOMARKER_VARIANTS
        ]
        variant_names = [
            str(name) for name in dataset.data_vars if str(name) in expected_names
        ]
        return sorted(
            variant_names,
            key=lambda x: EDAR_BIOMARKER_VARIANTS.index(
                x.replace(EDAR_BIOMARKER_PREFIX, "")
            ),
        )

    def fit_scaler(
        self, dataset: xr.Dataset, train_nodes: list[str] | list[int]
    ) -> None:
        """Fit robust scalers on train nodes using OBSERVED values only (mask > 0).

        This ensures interpolated/filled values don't corrupt the robust scaling
        parameters (median/IQR). Only values where mask > 0 are used.

        Args:
            dataset: xarray Dataset containing EDAR biomarker variables (already log1p-transformed)
                     and {variant}_mask channels
            train_nodes: List of region IDs for training split

        Raises:
            ValueError: If no observed values found or non-finite scaler params result
        """
        variant_names = self._get_variant_names(dataset)
        if not variant_names:
            raise ValueError("Dataset missing EDAR biomarker variables")

        centers: dict[str, float] = {}
        scales: dict[str, float] = {}

        for variant_name in variant_names:
            if variant_name not in dataset:
                continue

            biomarker_da = dataset[variant_name]
            mask_var = f"{variant_name}_mask"

            if mask_var not in dataset:
                raise ValueError(f"Missing mask channel for scaler fitting: {mask_var}")

            mask_da = dataset[mask_var]

            train_mask = np.isin(biomarker_da.region_id.values, train_nodes)
            values = biomarker_da.isel(region_id=train_mask).values
            masks = mask_da.isel(region_id=train_mask).values

            # KEY: Use mask channel to select only OBSERVED values
            # - mask > 0: measurement was actually observed (not interpolated/filled)
            # - finite: no NaN/inf (isfinite handles both)
            # - positive: exclude below-detection-limit zeros
            observed = (masks > 0) & np.isfinite(values) & (values > 0)
            finite_values = values[observed].astype(np.float32)

            if len(finite_values) == 0:
                # Skip variants with no observed values (e.g., not collected at site)
                # This variant will use raw values (no scaling) during preprocessing
                logger.warning(
                    f"No observed biomarker values in train nodes for {variant_name}, "
                    f"skipping scaler (variant will use unscaled values)"
                )
                continue

            # Compute robust statistics
            center = np.median(finite_values)
            q75 = np.percentile(finite_values, 75)
            q25 = np.percentile(finite_values, 25)
            scale = np.maximum(q75 - q25, 1.0)

            # Validate scaler parameters are finite
            if not (np.isfinite(center) and np.isfinite(scale)):
                raise ValueError(
                    f"Non-finite scaler params for {variant_name}: "
                    f"center={center}, scale={scale}"
                )

            centers[variant_name] = float(center)
            scales[variant_name] = float(scale)
            logger.debug(
                f"Scaler params for {variant_name}: center={center:.4f}, scale={scale:.4f} "
                f"(n={len(finite_values)} observed values)"
            )

        if not centers:
            # No variants had observed values - log warning and set default scaler
            # This can happen when training on regions without biomarker data
            logger.warning(
                "No biomarker variants had observed values in train nodes. "
                "Using default scaler (center=0, scale=1) for all variants."
            )
            # Fit default scaler for all variants (no-op scaling)
            for variant_name in variant_names:
                centers[variant_name] = 0.0
                scales[variant_name] = 1.0

        self.scaler_params = BiomarkerScalerParams(
            center=centers, scale=scales, is_fitted=True
        )
        logger.info(
            f"Biomarker scaler fitted on {len(centers)} variants using mask-filtered values"
        )

    def set_scaler_params(self, params: BiomarkerScalerParams) -> None:
        """Set pre-fitted scaler params (for val/test datasets).

        Args:
            params: Scaler parameters fitted on training data
        """
        self.scaler_params = params

    def _preprocess_values(
        self,
        values: np.ndarray,
        center: float | None,
        scale: float | None,
        mask: np.ndarray,
        censor: np.ndarray,
        age: np.ndarray,
    ) -> np.ndarray:
        """Preprocess biomarker values with pre-computed mask/censor/age.

        Values are already log1p-transformed from preprocessing pipeline.
        Kalman filter handles interpolation, so no LOCF needed.

        Args:
            values: Biomarker values (T, N) - already log1p-transformed and Kalman-filtered
            mask: Pre-computed mask channel (T, N) - MUST be provided
            censor: Pre-computed censor channel (T, N) - MUST be provided
            age: Pre-computed age channel (T, N) as uint8 (0-14) - MUST be provided
            center: Robust scaler center (median)
            scale: Robust scaler scale (IQR)

        Returns:
            Array with shape (T, N, 4) containing [value, mask, censor, age]
        """
        # Value channel: values are already log1p-transformed from pipeline
        # Handle NaN by filling with 0 (no measurement)
        value_channel = np.where(
            np.isfinite(values) & (values > 0), values, 0.0
        ).astype(np.float32)

        # Robust Scaling
        if center is not None and scale is not None:
            value_channel = (value_channel - center) / scale
            value_channel = np.clip(value_channel, *self.clip_range)

        # Normalize age from uint8 (0-14) to float [0, 1]
        age_normalized = (age.astype(np.float32) / self.age_max).astype(np.float32)

        # Convert mask and censor to float32
        mask_float = mask.astype(np.float32)
        censor_float = censor.astype(np.float32)

        return np.stack(
            [value_channel, mask_float, censor_float, age_normalized], axis=-1
        )

    def preprocess_dataset(self, dataset: xr.Dataset) -> np.ndarray:
        """Vectorized preprocessing of the entire dataset.

        Reads pre-computed mask, censor, and age channels from Zarr. These channels
        are required and must be present in the dataset.

        Returns:
            np.ndarray: Shape (time, nodes, 4 * variants) containing
            [value, mask, censor, age] channels per variant.
        """
        variant_names = self._get_variant_names(dataset)
        if not variant_names:
            raise ValueError("Dataset missing EDAR biomarker variables")

        outputs: list[np.ndarray] = []

        for variant_name in variant_names:
            biomarker_da = dataset[variant_name]

            # expect 1 feature currently
            if not biomarker_da.ndim == 2:
                raise ValueError("Biomarker data must be 2-dimensional")

            # ensure expected coord ordering
            biomarker_da = biomarker_da.transpose("date", "region_id")

            values = biomarker_da.values  # (T, N)
            center = None
            scale = None
            if self.scaler_params and self.scaler_params.is_fitted:
                center = self.scaler_params.center.get(variant_name)
                scale = self.scaler_params.scale.get(variant_name)

            # Require pre-computed mask, censor, and age channels
            # variant_name is like "edar_biomarker_N1"
            mask_var = f"{variant_name}_mask"
            censor_var = f"{variant_name}_censor"
            age_var = f"{variant_name}_age"
            if mask_var not in dataset:
                raise ValueError(f"Missing required channel: {mask_var}")
            if censor_var not in dataset:
                raise ValueError(f"Missing required channel: {censor_var}")
            if age_var not in dataset:
                raise ValueError(f"Missing required channel: {age_var}")

            mask = dataset[mask_var].transpose("date", "region_id").values  # type: ignore[arg-type]
            censor = dataset[censor_var].transpose("date", "region_id").values  # type: ignore[arg-type]
            age = dataset[age_var].transpose("date", "region_id").values  # type: ignore[arg-type]

            outputs.append(
                self._preprocess_values(values, center, scale, mask, censor, age)
            )

        return np.concatenate(outputs, axis=-1)
