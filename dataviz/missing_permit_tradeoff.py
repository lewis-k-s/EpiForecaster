"""Analyze missing-permit tradeoffs against pass rate and flat-window incidence.

For each series, this script computes:
1) Window pass rate over a (history_permit, target_permit) grid.
2) Incidence of zero-variation windows among the passed windows.

Zero-variation is computed on value channels using std <= epsilon, with windows
that have fewer than 2 finite values treated as zero-variation (degenerate).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from models.configs import EpiForecasterConfig
from utils.logging import setup_logging, suppress_zarr_warnings

suppress_zarr_warnings()
logger = logging.getLogger(__name__)


def _as_2d(da: xr.DataArray) -> xr.DataArray:
    """Select run_id/feature dims if present and return (date, region_id)."""
    if "run_id" in da.dims:
        da = da.isel(run_id=0)
    if "feature" in da.dims:
        da = da.isel(feature=0)
    return da.transpose(TEMPORAL_COORD, REGION_COORD)


def _load_series(
    ds: xr.Dataset,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return mapping: series_name -> (values[T,N], obs_mask[T,N], valid_range[T,N])."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for name in ("cases", "hospitalizations", "deaths"):
        value_da = _as_2d(ds[name])
        mask_da = _as_2d(ds[f"{name}_mask"])
        values = value_da.values.astype(np.float32)
        mask = (mask_da.values > 0).astype(np.int32)
        valid_range = np.ones_like(mask, dtype=bool)
        out[name] = (values, mask, valid_range)

    variants = [
        ("edar_biomarker_N1", "edar_biomarker_N1_mask"),
        ("edar_biomarker_N2", "edar_biomarker_N2_mask"),
        ("edar_biomarker_IP4", "edar_biomarker_IP4_mask"),
    ]
    present = [(v, m) for v, m in variants if v in ds and m in ds]
    if present:
        values_stack: list[np.ndarray] = []
        masks_stack: list[np.ndarray] = []
        for value_name, mask_name in present:
            v = _as_2d(ds[value_name]).values.astype(np.float32)
            m = (_as_2d(ds[mask_name]).values > 0).astype(np.int32)
            values_stack.append(v)
            masks_stack.append(m)

        stacked_v = np.stack(values_stack, axis=0)  # (K,T,N)
        stacked_m = np.stack(masks_stack, axis=0)  # (K,T,N)
        observed_count = stacked_m.sum(axis=0)  # (T,N)
        weighted_sum = (stacked_v * stacked_m).sum(axis=0)
        combined_values = np.full_like(weighted_sum, np.nan, dtype=np.float32)
        np.divide(
            weighted_sum,
            observed_count,
            out=combined_values,
            where=observed_count > 0,
        )
        combined_mask = (observed_count > 0).astype(np.int32)

        # Restrict biomarkers to source regions when available.
        if "edar_has_source" in ds:
            source_mask = ds["edar_has_source"].values.astype(bool)
            combined_values = combined_values[:, source_mask]
            combined_mask = combined_mask[:, source_mask]

        valid_range = np.ones_like(combined_mask, dtype=bool)
        if "biomarker_data_start" in ds:
            starts = ds["biomarker_data_start"].values
            if starts.ndim == 2:
                starts = starts[0]
            starts = starts[source_mask].astype(np.int32)
            t_index = np.arange(combined_mask.shape[0], dtype=np.int32)[:, None]
            valid_range = t_index >= starts[None, :]
            # Ensure values are undefined outside valid range.
            combined_values = np.where(valid_range, combined_values, np.nan)

        out["biomarkers_combined"] = (combined_values, combined_mask, valid_range)

    return out


def _window_counts(mask: np.ndarray, length: int, starts: np.ndarray) -> np.ndarray:
    """Count observed points in each sliding window over time."""
    csum = np.concatenate(
        [
            np.zeros((1, mask.shape[1]), dtype=np.int32),
            np.cumsum(mask, axis=0),
        ],
        axis=0,
    )
    counts = csum[length:] - csum[:-length]
    return counts[starts]


def _window_zero_variation_flags(
    values: np.ndarray,
    starts: np.ndarray,
    length: int,
    epsilon: float,
) -> np.ndarray:
    """Return (num_windows, num_regions) bool for zero-variation windows."""
    num_starts = len(starts)
    num_regions = values.shape[1]
    out = np.zeros((num_starts, num_regions), dtype=bool)

    for i, s in enumerate(starts):
        w = values[s : s + length, :]
        finite = np.isfinite(w)
        finite_count = finite.sum(axis=0)
        masked = np.where(finite, w, 0.0)
        sum_x = masked.sum(axis=0)
        sum_x2 = (masked * masked).sum(axis=0)
        mean_x = np.divide(
            sum_x, finite_count, out=np.zeros_like(sum_x), where=finite_count > 0
        )
        mean_x2 = np.divide(
            sum_x2, finite_count, out=np.zeros_like(sum_x2), where=finite_count > 0
        )
        var_x = np.maximum(mean_x2 - (mean_x * mean_x), 0.0)
        std = np.sqrt(var_x)
        out[i] = (finite_count < 2) | (std <= epsilon)

    return out


def analyze_series(
    values: np.ndarray,
    mask: np.ndarray,
    valid_range: np.ndarray,
    history_length: int,
    forecast_horizon: int,
    hist_permits: list[int],
    target_permits: list[int],
    epsilon: float,
) -> pd.DataFrame:
    """Build permit grid with pass rate and zero-variation incidence."""
    t, _ = mask.shape
    segment = history_length + forecast_horizon
    if t < segment:
        return pd.DataFrame()

    starts = np.arange(0, t - segment + 1, dtype=np.int32)
    # Eligible window-node pairs: full window lies in the valid range.
    vr = valid_range.astype(np.int32)
    vr_csum = np.concatenate(
        [
            np.zeros((1, vr.shape[1]), dtype=np.int32),
            np.cumsum(vr, axis=0),
        ],
        axis=0,
    )
    valid_counts = (vr_csum[segment:] - vr_csum[:-segment])[starts]
    eligible = valid_counts >= segment
    hist_obs = _window_counts(mask, history_length, starts)
    target_obs_counts = _window_counts(mask[history_length:], forecast_horizon, starts)

    hist_missing = history_length - hist_obs
    target_missing = forecast_horizon - target_obs_counts

    z_hist = _window_zero_variation_flags(values, starts, history_length, epsilon)
    z_target = _window_zero_variation_flags(
        values[history_length:], starts, forecast_horizon, epsilon
    )
    z_full = _window_zero_variation_flags(values, starts, segment, epsilon)

    n_total = len(starts) * mask.shape[1]
    n_eligible = int(eligible.sum())
    rows: list[dict[str, float | int]] = []
    for hp in hist_permits:
        if hp > history_length:
            continue
        hist_ok = hist_missing <= hp
        for tp in target_permits:
            if tp > forecast_horizon:
                continue
            passed = hist_ok & (target_missing <= tp) & eligible
            n_pass = int(passed.sum())

            if n_pass == 0:
                z_hist_rate = np.nan
                z_target_rate = np.nan
                z_full_rate = np.nan
                target_obs_median = np.nan
                target_obs_p25 = np.nan
                target_obs_p75 = np.nan
                target_obs_min = np.nan
                target_obs_max = np.nan
                target_obs_le1_pct = np.nan
            else:
                z_hist_rate = float(z_hist[passed].mean() * 100.0)
                z_target_rate = float(z_target[passed].mean() * 100.0)
                z_full_rate = float(z_full[passed].mean() * 100.0)
                target_obs = target_obs_counts[passed]
                target_obs_median = float(np.median(target_obs))
                target_obs_p25 = float(np.percentile(target_obs, 25))
                target_obs_p75 = float(np.percentile(target_obs, 75))
                target_obs_min = float(np.min(target_obs))
                target_obs_max = float(np.max(target_obs))
                target_obs_le1_pct = float(np.mean(target_obs <= 1) * 100.0)

            rows.append(
                {
                    "history_permit": hp,
                    "target_permit": tp,
                    "n_total_window_node_pairs": n_total,
                    "n_eligible_window_node_pairs": n_eligible,
                    "n_pass": n_pass,
                    "pass_rate_pct_all_pairs": float(n_pass / n_total * 100.0),
                    "pass_rate_pct_eligible_pairs": (
                        float(n_pass / n_eligible * 100.0) if n_eligible > 0 else np.nan
                    ),
                    "target_obs_count_median": target_obs_median,
                    "target_obs_count_p25": target_obs_p25,
                    "target_obs_count_p75": target_obs_p75,
                    "target_obs_count_min": target_obs_min,
                    "target_obs_count_max": target_obs_max,
                    "target_obs_count_le1_pct": target_obs_le1_pct,
                    "zero_var_history_pct": z_hist_rate,
                    "zero_var_target_pct": z_target_rate,
                    "zero_var_full_window_pct": z_full_rate,
                }
            )

    return pd.DataFrame(rows)


def summarize_baseline(
    values: np.ndarray, mask: np.ndarray, valid_range: np.ndarray, epsilon: float
) -> dict[str, float]:
    """Return overall sparsity/variation summary independent of permits."""
    observed = (mask > 0) & valid_range
    valid_count = valid_range.sum(axis=0)
    density = np.divide(
        observed.sum(axis=0),
        valid_count,
        out=np.zeros_like(valid_count, dtype=np.float32),
        where=valid_count > 0,
    )
    sparsity = 1.0 - density

    finite = np.isfinite(values) & valid_range
    count = finite.sum(axis=0)
    masked = np.where(finite, values, 0.0)
    sum_x = masked.sum(axis=0)
    sum_x2 = (masked * masked).sum(axis=0)
    mean_x = np.divide(sum_x, count, out=np.zeros_like(sum_x), where=count > 0)
    mean_x2 = np.divide(sum_x2, count, out=np.zeros_like(sum_x2), where=count > 0)
    var_x = np.maximum(mean_x2 - (mean_x * mean_x), 0.0)
    std = np.sqrt(var_x)
    zero_var_region = (count < 2) | (std <= epsilon)

    return {
        "n_regions": int(values.shape[1]),
        "valid_range_days_median": float(np.median(valid_count)),
        "valid_range_days_mean": float(np.mean(valid_count)),
        "obs_density_median_pct": float(np.median(density) * 100.0),
        "obs_density_mean_pct": float(np.mean(density) * 100.0),
        "sparsity_median_pct": float(np.median(sparsity) * 100.0),
        "sparsity_mean_pct": float(np.mean(sparsity) * 100.0),
        "zero_variation_regions_pct": float(np.mean(zero_var_region) * 100.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze missing permit tradeoff by series."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_epifor_full.yaml"),
        help="Training config path (default: configs/train_epifor_full.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/missing_permit_tradeoff"),
        help="Directory for CSV outputs",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Std threshold for zero-variation windows",
    )
    parser.add_argument(
        "--history-permits",
        type=str,
        default="0,2,4,6,8,10,12,14,16,20,24,28",
        help="Comma-separated history permits",
    )
    parser.add_argument(
        "--target-permits",
        type=str,
        default="0,1,2,3,4,6,8,10,12,14",
        help="Comma-separated target permits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    cfg = EpiForecasterConfig.from_file(str(args.config))
    dataset_path = Path(cfg.data.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ds = xr.open_zarr(dataset_path)
    series = _load_series(ds)

    history_length = int(cfg.model.history_length)
    forecast_horizon = int(cfg.model.forecast_horizon)
    hist_permits = [int(x) for x in args.history_permits.split(",") if x.strip()]
    target_permits = [int(x) for x in args.target_permits.split(",") if x.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows: list[dict[str, float | int | str]] = []
    grid_rows: list[pd.DataFrame] = []
    for series_name, (values, mask, valid_range) in series.items():
        logger.info("Analyzing %s (shape=%s)", series_name, values.shape)

        baseline = summarize_baseline(
            values, mask, valid_range, epsilon=float(args.epsilon)
        )
        baseline["series"] = series_name
        baseline_rows.append(baseline)

        grid = analyze_series(
            values=values,
            mask=mask,
            valid_range=valid_range,
            history_length=history_length,
            forecast_horizon=forecast_horizon,
            hist_permits=hist_permits,
            target_permits=target_permits,
            epsilon=float(args.epsilon),
        )
        if not grid.empty:
            grid.insert(0, "series", series_name)
            grid_rows.append(grid)
            per_series_path = args.output_dir / f"{series_name}_permit_grid.csv"
            grid.to_csv(per_series_path, index=False)
            logger.info("Saved %s", per_series_path)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_path = args.output_dir / "series_baseline_summary.csv"
    baseline_df.to_csv(baseline_path, index=False)
    logger.info("Saved %s", baseline_path)

    if grid_rows:
        all_grid_df = pd.concat(grid_rows, ignore_index=True)
        all_grid_path = args.output_dir / "permit_grid_all_series.csv"
        all_grid_df.to_csv(all_grid_path, index=False)
        logger.info("Saved %s", all_grid_path)

    print("\n=== Baseline series summary ===")
    if not baseline_df.empty:
        print(
            baseline_df.sort_values("sparsity_median_pct").to_string(
                index=False, float_format=lambda x: f"{x:6.2f}"
            )
        )
    else:
        print("No series available.")


if __name__ == "__main__":
    main()
