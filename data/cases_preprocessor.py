from dataclasses import dataclass

import bottleneck as bn
import numpy as np
import pandas as pd
import torch
import xarray as xr


@dataclass
class CasesPreprocessorConfig:
    history_length: int
    log_scale: bool = False
    scale_epsilon: float = 1e-6
    per_100k: bool = True
    age_max: int = 14  # Max days for LOCF age channel


class CasesPreprocessor:
    """Preprocesses case data with scaling, log-transform, and rolling statistics.

    Contract:
    - `preprocess_dataset()` computes transformed case values + rolling mean/std.
    - `make_normalized_window()` produces a reversible per-window normalization:
        norm = (x - mean_anchor) / std_anchor
      where anchor stats are taken at the *last* history step.
    """

    def __init__(self, config: CasesPreprocessorConfig):
        self.config = config
        self.processed_cases: torch.Tensor | None = None  # (T, N, 3) [value, mask, age]
        self.rolling_mean: torch.Tensor | None = None  # (T, N, 1)
        self.rolling_std: torch.Tensor | None = None  # (T, N, 1)

    def preprocess_dataset(
        self, dataset: xr.Dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute cases + rolling stats over full dataset.

        Returns:
            processed_cases: (T, N, 3) tensor with [value, mask, age] channels.
                Channel 0: scaled/log-transformed (if configured) case values (NaN preserved).
                Channel 1: mask (1.0 if finite, 0.0 if NaN).
                Channel 2: LOCF age normalized to [0, 1] (days since last observation / age_max).
            rolling_mean: (T, N, 1) rolling mean tensor (right-aligned, NaN-aware).
            rolling_std: (T, N, 1) rolling std tensor (right-aligned, NaN-aware, >= scale_epsilon).
        """
        cases_da = dataset.cases
        if cases_da.ndim == 3:
            cases_da = cases_da.squeeze(drop=True)

        cases_da = cases_da.transpose("date", "region_id")
        values = cases_da.values.astype(np.float32)  # (T, N)

        if self.config.per_100k and "population" in dataset:
            pop_values = dataset.population.values
            pop_values = np.where(
                (pop_values > 0) & np.isfinite(pop_values), pop_values, np.nan
            )
            per_100k = 100000.0 / pop_values
            values = values * per_100k

        if self.config.log_scale:
            values = np.log1p(values)

        window = int(self.config.history_length)
        rolling_mean = bn.move_mean(values, window=window, min_count=1, axis=0)
        rolling_std = bn.move_std(values, window=window, min_count=1, axis=0)

        rolling_mean = np.nan_to_num(rolling_mean, nan=0.0)
        rolling_std = np.nan_to_num(rolling_std, nan=0.0)
        rolling_std = np.maximum(rolling_std, float(self.config.scale_epsilon))

        # Mask channel: 1.0 if finite, 0.0 otherwise
        mask = np.isfinite(values).astype(np.float32)

        # --- Age Channel (LOCF) ---
        # Calculate days since last observation, normalized to [0, 1]
        T, N = values.shape
        age_channel = np.full_like(values, self.config.age_max, dtype=np.float32)

        # Vectorized age calculation
        # 1. Create time indices [0, 1, ..., T-1] broadcasted to (T, N)
        # 2. Where mask is 1, keep the index. Where mask is 0, set to NaN.
        # 3. Forward fill to propagate "last seen time".
        # 4. Age = current_time - last_seen_time.
        time_indices = np.arange(T)[:, None]  # (T, 1)
        last_seen_indices = np.where(mask > 0, time_indices, np.nan)

        last_seen_df = pd.DataFrame(last_seen_indices)
        last_seen_filled = last_seen_df.ffill().values  # Propagate last seen index

        # Calculate diff. For leading NaNs (no previous measurement), age remains max_age
        valid_history_mask = ~np.isnan(last_seen_filled)

        current_age = np.zeros_like(age_channel)
        current_age[valid_history_mask] = (
            time_indices * np.ones((1, N)) - last_seen_filled
        )[valid_history_mask]

        # Clip to age_max and normalize to [0, 1]
        final_age = np.where(
            valid_history_mask,
            np.minimum(current_age, self.config.age_max),
            self.config.age_max,
        )
        age_channel = (final_age / self.config.age_max).astype(np.float32)

        # Stack channels: (T, N, 3) -> [value, mask, age]
        values_t = torch.from_numpy(values).to(torch.float32)
        mask_t = torch.from_numpy(mask).to(torch.float32)
        age_t = torch.from_numpy(age_channel).to(torch.float32)
        cases_3ch = torch.stack([values_t, mask_t, age_t], dim=-1)

        mean_t = torch.from_numpy(rolling_mean).to(torch.float32).unsqueeze(-1)
        std_t = torch.from_numpy(rolling_std).to(torch.float32).unsqueeze(-1)

        self.processed_cases = cases_3ch
        self.rolling_mean = mean_t
        self.rolling_std = std_t

        return cases_3ch, mean_t, std_t

    def _require_fitted(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.processed_cases is None
            or self.rolling_mean is None
            or self.rolling_std is None
        ):
            raise RuntimeError(
                "CasesPreprocessor not initialized. Call preprocess_dataset() first."
            )
        return self.processed_cases, self.rolling_mean, self.rolling_std

    @staticmethod
    def _normalize_window(
        *, cases_window: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Normalize value channel of a (L+H, N, 3) window; preserve mask and age channels."""
        value_channel = cases_window[..., 0:1]
        mask_channel = cases_window[..., 1:2]
        age_channel = cases_window[..., 2:3]
        norm_value = (value_channel - mean) / std
        norm_value = torch.nan_to_num(norm_value, nan=0.0)
        return torch.cat([norm_value, mask_channel, age_channel], dim=-1)

    def make_normalized_window(
        self, *, range_start: int, history_length: int, forecast_horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized (L+H) window and anchor mean/std.

        Anchor mean/std are taken at `stat_idx = range_start + L - 1` and applied
        across the full (L+H) window, matching the reversible scheme used by
        `unscale_forecasts`.

        Returns:
            norm_window: (L+H, N, 3) tensor with [norm_value, mask, age]
            mean_anchor: (N, 1) rolling mean at stat_idx
            std_anchor: (N, 1) rolling std at stat_idx
        """
        processed_cases, rolling_mean, rolling_std = self._require_fitted()

        L = int(history_length)
        H = int(forecast_horizon)
        range_end = int(range_start) + L
        forecast_end = range_end + H
        stat_idx = range_end - 1

        mean_anchor = rolling_mean[stat_idx]  # (N, 1)
        std_anchor = rolling_std[stat_idx]  # (N, 1)

        cases_window = processed_cases[range_start:forecast_end]  # (L+H, N, 2)
        norm_window = self._normalize_window(
            cases_window=cases_window, mean=mean_anchor, std=std_anchor
        )
        return norm_window, mean_anchor, std_anchor

    def get_stats_sequence_for_target(
        self, *, range_start: int, history_length: int, target_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rolling mean/std sequences for the target node over history.

        These are used as additional model inputs (not for unscaling).

        Returns:
            mean_seq: (L, 1)
            std_seq: (L, 1)
        """
        _processed_cases, rolling_mean, rolling_std = self._require_fitted()

        L = int(history_length)
        range_end = int(range_start) + L

        mean_seq = rolling_mean[range_start:range_end, target_idx].float()
        std_seq = rolling_std[range_start:range_end, target_idx].float()

        if mean_seq.ndim == 1:
            mean_seq = mean_seq.unsqueeze(-1)
        if std_seq.ndim == 1:
            std_seq = std_seq.unsqueeze(-1)

        return mean_seq, std_seq
