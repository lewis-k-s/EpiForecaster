from dataclasses import dataclass

import numpy as np
import xarray as xr


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
    - Value channel: log1p + robust scaling + clipping [-8, 8] (Kalman filter handles interpolation)
    - Mask channel: 1.0 if measured today (finite and positive), else 0.0 (required, pre-computed)
    - Censor channel: 1.0 if censored at LD floor (from Kalman), else 0.0 (required, pre-computed)
    - Age channel: Days since last measurement, normalized to [0, 1] (max 14 days, required, pre-computed)

    Channel layout: [value, mask, censor, age] - 4 channels per variant

    NOTE: Mask, censor, and age channels must be pre-computed and present in the dataset.
    Kalman filter already handles interpolation, so no LOCF is applied.
    """

    def __init__(
        self, age_max: int = 14, clip_range: tuple[float, float] = (-8.0, 8.0)
    ) -> None:
        self.age_max = age_max
        self.clip_range = clip_range
        self.scaler_params: BiomarkerScalerParams | None = None

    def _get_variant_names(self, dataset: xr.Dataset) -> list[str]:
        variant_names = [
            str(name)
            for name in dataset.data_vars
            if str(name).startswith("edar_biomarker_")
            and not str(name).endswith(("_mask", "_age", "_censor"))
        ]
        return sorted(variant_names) if variant_names else []

    def fit_scaler(
        self, dataset: xr.Dataset, train_nodes: list[str] | list[int]
    ) -> None:
        """Fit robust scalers on train nodes only (all timesteps).

        Args:
            dataset: xarray Dataset containing EDAR biomarker variables
            train_nodes: List of region IDs for training split
        """
        variant_names = self._get_variant_names(dataset)
        if not variant_names:
            raise ValueError("Dataset missing EDAR biomarker variables")

        centers: dict[str, float] = {}
        scales: dict[str, float] = {}

        for variant_name in variant_names:
            # Check if variant exists in dataset
            if variant_name not in dataset:
                continue

            biomarker_da = dataset[variant_name]
            train_mask = np.isin(biomarker_da.region_id.values, train_nodes)
            all_values = biomarker_da.isel(region_id=train_mask).values

            # Exclude zeros (below detection limit) from scaler fitting
            finite_values = all_values[np.isfinite(all_values) & (all_values > 0)]

            if len(finite_values) == 0:
                raise ValueError(
                    f"No finite biomarker values in train nodes for {variant_name}"
                )

            log_values = np.log1p(finite_values)

            center = np.median(log_values)
            q75 = np.percentile(log_values, 75)
            q25 = np.percentile(log_values, 25)
            scale = np.maximum(q75 - q25, 1.0)

            centers[variant_name] = float(center)
            scales[variant_name] = float(scale)

        if not centers:
            raise ValueError("No valid biomarker variants found to fit scaler")

        self.scaler_params = BiomarkerScalerParams(
            center=centers, scale=scales, is_fitted=True
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

        Kalman filter handles interpolation, so no LOCF needed.

        Args:
            values: Raw biomarker values (T, N) - already Kalman-filtered
            mask: Pre-computed mask channel (T, N) - MUST be provided
            censor: Pre-computed censor channel (T, N) - MUST be provided
            age: Pre-computed age channel (T, N) - MUST be provided
            center: Robust scaler center (median)
            scale: Robust scaler scale (IQR)

        Returns:
            Array with shape (T, N, 4) containing [value, mask, censor, age]
        """
        # Value channel: log transform + scaling (no LOCF needed)
        # Handle NaN by filling with 0 (no measurement)
        value_channel = np.where(
            np.isfinite(values) & (values > 0), np.log1p(values), 0.0
        ).astype(np.float32)

        # Robust Scaling
        if center is not None and scale is not None:
            value_channel = (value_channel - center) / scale
            value_channel = np.clip(value_channel, *self.clip_range)

        return np.stack([value_channel, mask, censor, age], axis=-1)

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
