from dataclasses import dataclass

import bottleneck as bn
import numpy as np
import torch
import xarray as xr


@dataclass
class CasesPreprocessorConfig:
    history_length: int
    log_scale: bool = False
    scale_epsilon: float = 1e-6
    per_100k: bool = True


class CasesPreprocessor:
    """Preprocesses case data with scaling, log-transform, and rolling statistics."""

    def __init__(self, config: CasesPreprocessorConfig):
        self.config = config

    def preprocess_dataset(
        self, dataset: xr.Dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Precomputes cases, means, and stds for the entire dataset.

        Args:
            dataset: xarray Dataset containing 'cases' and optionally 'population'.

        Returns:
            processed_cases: (T, N, 1) Tensor of scaled, log-transformed (if set) cases.
            rolling_mean: (T, N, 1) Tensor of rolling means (right-aligned).
            rolling_std: (T, N, 1) Tensor of rolling stds (right-aligned).
        """
        # 1. Get raw cases (T, N)
        cases_da = dataset.cases
        # Handle potential feature dimension (e.g. from tests)
        if cases_da.ndim == 3:
            cases_da = cases_da.squeeze(drop=True)

        # Ensure (time, region) order
        cases_da = cases_da.transpose("date", "region_id")
        values = cases_da.values.astype(np.float32)  # (T, N)

        # 2. Scale per 100k
        if self.config.per_100k and "population" in dataset:
            population = dataset.population
            # Handle if population is DataArray or just values.
            # Assuming population is aligned with region_id.
            # If population has shape (N,), broadcasting works.
            # Check for invalid population
            pop_values = population.values
            pop_values = np.where(
                (pop_values > 0) & np.isfinite(pop_values), pop_values, np.nan
            )
            per_100k = 100000.0 / pop_values
            # Broadcast: (T, N) * (N,) or (1, N) -> (T, N)
            values = values * per_100k

        # 3. Log transform
        if self.config.log_scale:
            values = np.log1p(values)

        # 4. Compute Rolling Stats
        # bottleneck.move_mean/std are fast moving window functions.
        # axis=0 is time.
        window = self.config.history_length

        # move_mean/std result includes the current value in the window.
        # So at index t, it is mean(values[t-L+1 : t+1]).
        # We need the stats for the history window [t : t+L].
        # In __getitem__, we are at window_start `t`. The history is `values[t : t+L]`.
        # The rolling stats we want are those computed over `values[t : t+L]`.
        # This corresponds to the rolling stat at index `t + L - 1` (since move_mean is right-aligned).

        # bottleneck handles NaNs by default?
        # bn.move_mean(a, window, min_count=None, axis=-1)
        # min_count=None defaults to window.
        # We probably want min_count=1 to match "ignore NaNs" behavior (like np.nanmean)

        rolling_mean = bn.move_mean(values, window=window, min_count=1, axis=0)
        rolling_std = bn.move_std(values, window=window, min_count=1, axis=0)

        # Fix NaNs in stats (e.g. at the beginning before window fills, or if all are NaN)
        # In our case, we want 0.0 for mean and 1.0 (or epsilon) for std if undefined?
        # Or keep them as NaN and handle later?
        # Let's clean them up to be safe for torch conversion.

        # For std, ensure min value
        rolling_std = np.nan_to_num(rolling_std, nan=0.0)
        rolling_std = np.maximum(rolling_std, self.config.scale_epsilon)

        # For mean, nan -> 0
        rolling_mean = np.nan_to_num(rolling_mean, nan=0.0)

        # Also clean values if there are NaNs (though we might want to preserve them for masking?)
        # EpiDataset uses NaNs to detect missingness.
        # But for the output tensor, usually we want 0s where data is missing.
        # However, _normalize_cases did imputation.
        # "history_imputed = np.where(np.isfinite(history_values), history_values, series_mean)"
        # So we should probably return values with NaNs still in them,
        # and let the final lookup handle imputation?
        # Or better: pre-impute?
        # If we pre-impute with rolling mean, we need to do it carefully.
        # Let's keep NaNs in `values` for now, so we can see where data is missing.
        # But we must ensure `rolling_mean` and `rolling_std` are dense (no NaNs).

        # 5. Convert to Tensor and add feature dim (T, N, 1)
        values_t = torch.from_numpy(values).float().unsqueeze(-1)
        mean_t = torch.from_numpy(rolling_mean).float().unsqueeze(-1)
        std_t = torch.from_numpy(rolling_std).float().unsqueeze(-1)

        return values_t, mean_t, std_t
