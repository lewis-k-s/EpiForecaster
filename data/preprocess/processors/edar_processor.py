"""
Processor for EDAR wastewater biomarker data.

This module handles the conversion of wastewater biomarker data from EDAR
(Environmental DNA Analysis and Recovery) systems into canonical tensor formats.
It processes variant selection, flow calculations, temporal alignment, and
creates temporal tensors for downstream aggregation to target regions.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.stats import norm
from statsmodels.tsa.statespace.structural import UnobservedComponents

from ..config import REGION_COORD, PreprocessingConfig


class _TobitKalman:
    def __init__(
        self,
        *,
        process_var: float,
        measurement_var: float,
        censor_inflation: float = 4.0,
    ) -> None:
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self.censor_inflation = float(censor_inflation)
        self.state = 0.0
        self.state_var = 1.0
        self.initialized = False

    def _initialize(self, first_log_value: float) -> None:
        self.state = float(first_log_value)
        self.state_var = float(self.measurement_var)
        self.initialized = True

    def filter_series(
        self, values: np.ndarray, limits: np.ndarray
    ) -> tuple[list[float], list[int]]:
        filtered: list[float] = []
        flags: list[int] = []

        finite_mask = np.isfinite(values) & (values > 0)
        if finite_mask.any():
            first_value = float(np.log(values[finite_mask][0]))
            self._initialize(first_value)

        for value, limit in zip(values, limits, strict=False):
            # Predict
            pred_state = self.state
            pred_var = self.state_var + self.process_var
            pred_sigma = float(np.sqrt(pred_var + self.measurement_var))

            limit_valid = np.isfinite(limit) and limit > 0

            if not np.isfinite(value):
                # Missing observation
                z_eff = pred_state
                r_eff = 1e9
                flag = 2
            elif limit_valid and value <= limit:
                if not self.initialized:
                    self._initialize(float(np.log(limit) - 0.5))
                    pred_state = self.state
                    pred_var = self.state_var + self.process_var
                    pred_sigma = float(np.sqrt(pred_var + self.measurement_var))

                log_limit = float(np.log(limit + 1e-9))
                alpha = (log_limit - pred_state) / pred_sigma
                pdf = float(norm.pdf(alpha))
                cdf = float(norm.cdf(alpha))
                cdf = max(cdf, 1e-9)
                z_eff = pred_state - pred_sigma * (pdf / cdf)
                r_eff = self.measurement_var * self.censor_inflation
                flag = 1
            else:
                log_value = float(np.log(value + 1e-9))
                if not self.initialized:
                    self._initialize(log_value)
                    pred_state = self.state
                z_eff = log_value
                r_eff = self.measurement_var
                flag = 0

            s = pred_var + r_eff
            k_gain = pred_var / s
            self.state = pred_state + k_gain * (z_eff - pred_state)
            self.state_var = (1 - k_gain) * pred_var

            filtered.append(float(self.state))
            flags.append(flag)

        return filtered, flags


class _KalmanFilter:
    """
    Standard Kalman filter for time series smoothing.

    Simpler version of _TobitKalman without censoring support.
    Used for hospitalization data where there's no detection limit.

    State space model:
    - State transition: x_t = x_{t-1} + w_t,  w_t ~ N(0, process_var)
    - Measurement: z_t = x_t + v_t,  v_t ~ N(0, measurement_var)
    """

    def __init__(
        self,
        *,
        process_var: float,
        measurement_var: float,
    ) -> None:
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self.state = 0.0
        self.state_var = 1.0
        self.initialized = False

    def _initialize(self, first_log_value: float) -> None:
        self.state = float(first_log_value)
        self.state_var = float(self.measurement_var)
        self.initialized = True

    def filter_series(self, values: np.ndarray) -> tuple[list[float], list[int]]:
        """
        Apply Kalman filter to a time series.

        Args:
            values: Array of measurements (can contain NaN for missing)

        Returns:
            Tuple of (filtered_values, flags)
            flags: 0=normal, 2=missing
        """
        filtered: list[float] = []
        flags: list[int] = []

        finite_mask = np.isfinite(values) & (values > 0)
        if finite_mask.any():
            first_value = float(np.log(values[finite_mask][0]))
            self._initialize(first_value)

        for value in values:
            # Predict
            pred_state = self.state
            pred_var = self.state_var + self.process_var

            if not np.isfinite(value) or value <= 0:
                # Missing observation - use prediction only
                z_eff = pred_state
                r_eff = 1e9
                flag = 2
            else:
                # Normal observation
                log_value = float(np.log(value + 1e-9))
                if not self.initialized:
                    self._initialize(log_value)
                    pred_state = self.state
                z_eff = log_value
                r_eff = self.measurement_var
                flag = 0

            # Update
            s = pred_var + r_eff
            k_gain = pred_var / s if s > 0 else 0.0
            self.state = pred_state + k_gain * (z_eff - pred_state)
            self.state_var = (1 - k_gain) * pred_var

            filtered.append(float(self.state))
            flags.append(flag)

        return filtered, flags


class EDARProcessor:
    """
    Converts EDAR wastewater biomarker data to canonical tensors.

    This processor handles:
    - Loading and parsing wastewater biomarker data
    - Variant selection (N2, IP4, etc.) based on data quality
    - Duplicate removal and temporal aggregation
    - Flow calculation and normalization
    - Resampling to daily frequency
    - Creation of temporal tensors for EDAR site features

    The output includes EDAR features and metadata for integration with other
    data sources. Biomarker data is aggregated to target regions in _process_edar_mapping.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the EDAR processor.

        Args:
            config: Preprocessing configuration with EDAR processing options
        """
        self.config = config
        self.validation_options = config.validation_options

    def _fit_kalman_params(self, series: pd.Series) -> tuple[float, float]:
        series = series.where(series > 0)
        series_log = pd.Series(np.log(series), index=series.index)
        if series_log.dropna().empty:
            raise ValueError("No finite observations to fit Kalman params")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(series_log, level="local level")
            result = model.fit(disp=False)

        params = dict(zip(result.param_names, result.params, strict=False))
        process_var = float(params.get("sigma2.level", 0.0))
        measurement_var = float(params.get("sigma2.irregular", 0.0))

        process_var = max(process_var, 1e-6)
        measurement_var = max(measurement_var, 1e-6)
        return process_var, measurement_var

    def _apply_tobit_kalman(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        if daily_data.empty:
            return daily_data

        if "limit_flow" not in daily_data.columns:
            return daily_data

        if not np.isfinite(daily_data["limit_flow"]).any():
            return daily_data

        censor_inflation = float(self.config.censor_inflation)
        fallback_process = float(self.validation_options.get("process_var", 0.05))
        fallback_measure = float(self.validation_options.get("measurement_var", 0.5))

        filtered_frames: list[pd.DataFrame] = []
        for (edar_id, variant), group in daily_data.groupby(["edar_id", "variant"]):
            series = group.set_index("date").sort_index().asfreq("D")
            values = series["total_covid_flow"].to_numpy()
            limits = (
                series["limit_flow"].ffill().bfill().to_numpy()
                if "limit_flow" in series
                else np.full_like(values, np.nan)
            )

            fit_series = pd.Series(values, index=series.index)
            censored_mask = np.isfinite(limits) & (values <= limits)
            fit_series = fit_series.mask(censored_mask)

            try:
                process_var, measurement_var = self._fit_kalman_params(fit_series)
            except ValueError:
                process_var = fallback_process
                measurement_var = fallback_measure

            filter_model = _TobitKalman(
                process_var=process_var,
                measurement_var=measurement_var,
                censor_inflation=censor_inflation,
            )
            filtered_log, flags = filter_model.filter_series(values, limits)

            series["total_covid_flow"] = np.exp(filtered_log)
            series["censor_flag"] = flags
            series["process_var"] = process_var
            series["measurement_var"] = measurement_var
            series["edar_id"] = edar_id
            series["variant"] = variant
            filtered_frames.append(series.reset_index())

        return pd.concat(filtered_frames, ignore_index=True)

    def _aggregate_censor_flags(
        self,
        censor_xr: xr.DataArray,
        emap: xr.DataArray,
    ) -> xr.DataArray:
        """Aggregate censor flags from EDAR sites to regions.

        Uses max-severity aggregation: missing (2) > censored (1) > uncensored (0).
        Where multiple EDAR sites contribute to a region, takes the maximum flag value.

        Preserves run_id dimension if present in input.

        Args:
            censor_xr: Censor flags with dimensions (run_id?, date, edar_id, variant)
            emap: EDAR contribution matrix (edar_id, region_id)

        Returns:
            Censor flags aggregated to regions with dimensions (run_id?, date, region_id, variant)
        """
        # Align and take max over EDAR sites contributing to each region
        censor_xr_aligned, emap_aligned = xr.align(censor_xr, emap, join="inner")

        # run_id dimension is always present
        assert "run_id" in censor_xr_aligned.dims, "run_id dimension is required"

        coords = {
            "run_id": censor_xr_aligned["run_id"].values,
            "date": censor_xr_aligned["date"].values,
            REGION_COORD: emap_aligned[REGION_COORD].values,
            "variant": censor_xr_aligned["variant"].values,
        }
        dims = ["run_id", "date", REGION_COORD, "variant"]

        result = xr.DataArray(coords=coords, dims=dims)

        for variant in censor_xr_aligned["variant"].values:
            variant_censor = censor_xr_aligned.sel(variant=variant)
            # Fill NaN with 0 (uncensored) for sites without data
            variant_censor_filled = variant_censor.fillna(0)

            # Max-severity aggregation: for each region, take max of contributing sites
            # We compute this by iterating over regions and taking max where emap > 0
            for region in emap_aligned[REGION_COORD].values:
                # Find EDAR sites that contribute to this region
                contributing_sites = emap_aligned.sel({REGION_COORD: region}) > 0

                # Take max censor flag among contributing sites
                region_censor = variant_censor_filled.where(contributing_sites).max(
                    dim="edar_id"
                )
                result.loc[{"variant": variant, REGION_COORD: region}] = (
                    region_censor.values
                )

        return result

    def _compute_age_channel(
        self,
        values: xr.DataArray,
        max_age: int = 14,
    ) -> xr.DataArray:
        """Compute age channel (days since last measurement).

        Age is normalized to [0, 1] with max_age days as the maximum.
        Leading NaNs (before first measurement) get age = 1.0.

        run_id dimension is ALWAYS present (synthetic: multiple runs, real: run_id="real").

        Args:
            values: DataArray with shape (run_id, date, region_id) or (run_id, date, region_id, variant)
            max_age: Maximum age in days for normalization

        Returns:
            DataArray with normalized age values [0, 1]
        """
        assert "run_id" in values.dims, "run_id dimension is required"
        return self._compute_age_core(values, max_age)

    def _compute_age_core(
        self,
        values: xr.DataArray,
        max_age: int,
    ) -> xr.DataArray:
        """Core age computation that handles all dimensions via broadcasting.

        run_id is always present and handled implicitly via xarray operations.
        Uses groupby for better chunking support instead of explicit loops.

        Args:
            values: DataArray with shape (run_id, date, region_id) or (run_id, date, region_id, variant)
            max_age: Maximum age in days for normalization

        Returns:
            DataArray with normalized age values [0, 1]
        """
        # Handle variant dimension using groupby for better chunking support
        has_variant = "variant" in values.dims
        if has_variant:
            # Use groupby for better chunking support vs explicit loop
            return values.groupby("variant").map(
                lambda g: self._compute_age_channel_2d(g, max_age)
            )
        else:
            return self._compute_age_channel_2d(values, max_age)

    def _compute_age_channel_2d(
        self,
        values: xr.DataArray,
        max_age: int,
    ) -> xr.DataArray:
        """Compute age channel for DataArray with run_id support.

        Handles both 2D (date, region_id) and 3D (run_id, date, region_id) inputs.
        Uses xarray operations for proper chunking support.

        Args:
            values: DataArray with shape (run_id?, date, region_id)
            max_age: Maximum age in days for normalization

        Returns:
            DataArray with normalized age values [0, 1]
        """
        # Get dimension order - run_id may or may not be present
        dims = values.dims
        date_dim = "date"
        region_dim = REGION_COORD
        has_run_id = "run_id" in dims

        # Create time indices along date dimension
        time_indices = xr.DataArray(
            np.arange(len(values[date_dim])),
            dims=[date_dim],
            coords={date_dim: values[date_dim].values},
        )

        # Mask: 1.0 if measured (finite and positive), 0.0 otherwise
        mask = xr.where(values.notnull() & (values > 0), 1.0, 0.0)

        # Find last measurement time for each (run_id?, region)
        # We use cumsum to find the last time index where mask=1
        # For each position, we want the most recent time where data was observed

        # For each date, find if there's data at this or any previous date
        # We use a reverse cumulative approach to track last seen time

        # Get time indices where mask=1, else NaN
        last_seen_indices = xr.where(mask > 0, time_indices, np.nan)

        # Forward fill through time (carries last seen index forward)
        # This needs to be done along the date dimension
        last_seen_filled = last_seen_indices.ffill(dim=date_dim)

        # Current time index for each position
        current_time_indices = time_indices

        # Expand current_time_indices to match the shape of last_seen_filled
        # If run_id exists, we need to broadcast to (run_id, date, region_id)
        if has_run_id:
            # Broadcast: (date,) -> (run_id, date, region_id)
            current_time_indices = current_time_indices.expand_dims(
                {region_dim: len(values[region_dim]), "run_id": len(values["run_id"])}
            )
        else:
            # Broadcast: (date,) -> (date, region_id)
            current_time_indices = current_time_indices.expand_dims(
                {region_dim: len(values[region_dim])}
            )

        # Calculate age = current_time - last_seen_time
        current_age = current_time_indices - last_seen_filled

        # For positions with no history (NaN after ffill), set age to max_age
        valid_history = last_seen_filled.notnull()
        final_age = xr.where(valid_history, np.minimum(current_age, max_age), max_age)

        # Normalize to [0, 1]
        age_normalized = final_age / max_age

        # Preserve the same dimensions as input
        return age_normalized.transpose(*dims)

    def process(self, wastewater_file: str, region_metadata_file: str) -> xr.Dataset:
        """
        Process real EDAR wastewater data from CSV file.

        This is the entry point for real EDAR data processing. It loads and tidies
        the raw CSV data, then processes it using the shared aggregation logic
        that is also used for synthetic data.

        Args:
            wastewater_file: Path to CSV/Excel file with wastewater data
            region_metadata_file: Path to EDAR-to-region contribution matrix

        Returns:
            Biomarker time series aggregated to regions, as an xarray Dataset
            with per-variant variables shaped `[date, region_id]`.
        """
        print(f"Processing EDAR wastewater data from {wastewater_file}")

        # Load and tidy raw data (CSV → xarray)
        flow_xr, censor_xr = self._load_and_tidy_raw_data(wastewater_file)

        # Process from xarray (shared with synthetic)
        return self.process_from_xarray(flow_xr, censor_xr, region_metadata_file)

    def _load_and_tidy_raw_data(
        self,
        wastewater_file: str,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Load and tidy raw CSV wastewater data into xarray intermediate format.

        This is the only step that differs between real and synthetic data.
        Synthetic data is already in xarray format.

        The intermediate format has dimensions (run_id, date, edar_id, variant).
        Real data gets run_id="real" to match synthetic format.

        Returns:
            Tuple of (flow_xr, censor_xr) with dimensions (run_id, date, edar_id, variant)
        """
        # Load wastewater data
        wastewater_df = self._load_wastewater_data(wastewater_file)

        # Select best variant for each site
        selected_data = self._select_variants(wastewater_df)

        # Remove duplicates and aggregate
        aggregated_data = self._remove_duplicates_and_aggregate(selected_data)

        # Calculate flow rates
        flow_data = self._calculate_flow_rates(aggregated_data)

        # Resample to daily frequency
        daily_data = self._resample_to_daily(flow_data)
        daily_data = self._apply_tobit_kalman(daily_data)

        # Convert to xarray - include both flow values and censor flags
        daily_data_xr_flow = daily_data.set_index(["date", "edar_id", "variant"])[
            "total_covid_flow"
        ].to_xarray()
        # Handle missing censor_flag column (when no LD data available)
        if "censor_flag" in daily_data.columns:
            daily_data_xr_censor = daily_data.set_index(["date", "edar_id", "variant"])[
                "censor_flag"
            ].to_xarray()
        else:
            # Create default censor flags (all uncensored = 0)
            daily_data_xr_censor = daily_data_xr_flow * 0

        # Add run_id dimension to match synthetic format
        # Real data gets run_id="real" to distinguish from synthetic runs
        flow_xr = daily_data_xr_flow.expand_dims(run_id=["real"])
        censor_xr = daily_data_xr_censor.expand_dims(run_id=["real"])

        return flow_xr, censor_xr

    def process_from_xarray(
        self,
        flow_xr: xr.DataArray,
        censor_xr: xr.DataArray,
        region_metadata_file: str,
    ) -> xr.Dataset:
        """
        Process EDAR data from xarray intermediate format to regions.

        This is the shared code path for both real and synthetic data.
        Both real and synthetic data use this method for aggregation.

        Preserves run_id dimension for curriculum training.
        Uses xarray broadcasting to handle single-run and multi-run data uniformly.

        Args:
            flow_xr: Flow/concentration values with dimensions (run_id?, date, edar_id, variant)
            censor_xr: Censor flags with dimensions (run_id?, date, edar_id, variant)
            region_metadata_file: Path to EDAR-to-region contribution matrix

        Returns:
            Dataset with biomarker variables indexed by (run_id?, region_id)
        """
        assert "run_id" in flow_xr.dims, "run_id dimension is required"
        run_count = len(flow_xr.run_id)
        print(f"Processing {run_count} run(s) of EDAR data...")
        result = self._process_broadcast(flow_xr, censor_xr, region_metadata_file)
        print(f"  ✓ Processed EDAR data: {result.dims}")
        return result

    def _process_broadcast(
        self,
        flow_xr: xr.DataArray,
        censor_xr: xr.DataArray,
        region_metadata_file: str,
    ) -> xr.Dataset:
        """
        Process EDAR data using xarray broadcasting.

        Handles both single-run and multi-run data uniformly using xarray's
        automatic broadcasting. Operations like xr.dot, xr.align preserve
        additional dimensions like run_id.

        Args:
            flow_xr: Flow/concentration values with dimensions (run_id?, date, edar_id, variant)
            censor_xr: Censor flags with dimensions (run_id?, date, edar_id, variant)
            region_metadata_file: Path to EDAR-to-region contribution matrix

        Returns:
            Dataset with biomarker variables indexed by (run_id?, region_id)
        """
        # Load contribution matrix
        emap = xr.open_dataarray(region_metadata_file)
        # EDAR contribution matrices are typically stored sparsely with NaNs where
        # there is no contribution. `xr.dot` does not skip NaNs, so we must treat
        # missing contributions as zeros.
        emap = emap.fillna(0)
        emap = emap.rename({"home": REGION_COORD})

        print(
            f"Transforming EDAR data to regions using contribution matrix from {region_metadata_file}"
        )
        if "edar_id" not in flow_xr.dims:
            raise ValueError(
                "Expected 'edar_id' dimension in processed wastewater data, "
                f"got dims={tuple(flow_xr.dims)!r}"
            )
        if "edar_id" not in emap.dims:
            raise ValueError(
                "Expected 'edar_id' dimension in EDAR contribution matrix, "
                f"got dims={tuple(emap.dims)!r}"
            )

        # Align IDs before dot product. A common silent failure mode is that
        # `xr.dot` aligns on labels and produces all-NaNs when there is no overlap.
        flow_xr = flow_xr.assign_coords(edar_id=flow_xr["edar_id"].astype(str))
        censor_xr = censor_xr.assign_coords(edar_id=censor_xr["edar_id"].astype(str))
        emap = emap.assign_coords(edar_id=emap["edar_id"].astype(str))

        wastewater_ids = set(flow_xr["edar_id"].values.tolist())
        mapping_ids = set(emap["edar_id"].values.tolist())
        overlap = sorted(wastewater_ids.intersection(mapping_ids))
        if not overlap:
            wastewater_sample = sorted(wastewater_ids)[:10]
            mapping_sample = sorted(mapping_ids)[:10]
            raise ValueError(
                "No overlapping 'edar_id' labels between wastewater data and "
                "EDAR contribution matrix. This would produce an all-NaN biomarker.\n"
                f"- wastewater edar_id sample: {wastewater_sample}\n"
                f"- mapping edar_id sample: {mapping_sample}\n"
                "Fix by normalizing IDs (leading zeros, prefixes) or updating the "
                "contribution matrix to match the raw wastewater data."
            )

        # Align both flow and censor data with emap
        flow_xr_aligned, emap_aligned = xr.align(flow_xr, emap, join="inner")
        censor_xr_aligned, _ = xr.align(censor_xr, emap, join="inner")

        # xr.dot propagates NaN values, causing all-NaN output even when only
        # some EDAR sites have missing data. We need a masked dot product that
        # only sums valid contributions per region, then normalizes by the
        # number of contributing sites.
        # Note: This is before Kalman imputation; NaN values represent truly
        # missing measurements that shouldn't contribute zero flow.
        mask = flow_xr_aligned.notnull()
        masked_data = flow_xr_aligned.fillna(0)
        weighted_sum = xr.dot(masked_data, emap_aligned, dim="edar_id")
        contribution_count = xr.dot(
            mask.astype(float),
            emap_aligned.astype(bool).astype(float),
            dim="edar_id",
        )
        # Normalize by contribution count (avoid division by zero)
        result = weighted_sum / contribution_count.where(contribution_count > 0, 1)

        # Aggregate censor flags to regions using max-severity
        censor_aggregated = self._aggregate_censor_flags(
            censor_xr_aligned, emap_aligned
        )

        # Skip early data quality validation - we'll assess quality at aligned stage

        biomarkers: dict[str, xr.DataArray] = {}
        for variant in result["variant"].values.tolist():
            variant_da = result.sel(variant=variant).drop_vars("variant")
            variant_name = f"edar_biomarker_{variant}"
            variant_da.name = variant_name
            biomarkers[variant_name] = variant_da

            # Mask channel: 1.0 if measured, 0.0 otherwise
            # Fill NaN with 0.0 (no measurement) to prevent NaN propagation
            mask = xr.where(variant_da.notnull() & (variant_da > 0), 1.0, 0.0).fillna(
                0.0
            )
            biomarkers[f"{variant_name}_mask"] = mask

            # Censor flag channel: 0=uncensored, 1=censored, 2=missing
            # Fill NaN with 0.0 (uncensored) for regions without EDAR data
            censor_variant = (
                censor_aggregated.sel(variant=variant).drop_vars("variant").fillna(0.0)
            )
            biomarkers[f"{variant_name}_censor"] = censor_variant

            # Age channel: normalized days since last measurement
            # Fill NaN with 1.0 (max age) for regions without data
            age = self._compute_age_channel(variant_da).fillna(1.0)
            biomarkers[f"{variant_name}_age"] = age

        return xr.Dataset(biomarkers)

    def _load_wastewater_data(self, wastewater_file: str) -> pd.DataFrame:
        header = pd.read_csv(wastewater_file, nrows=0)
        available_columns = set(header.columns)
        usecols = [
            "id mostra",
            "Cabal últimes 24h(m3)",
            "IP4(CG/L)",
            "N1(CG/L)",
            "N2(CG/L)",
        ]
        if "LD(CG/L)" in available_columns:
            usecols.append("LD(CG/L)")
        if "depuradora" in available_columns:
            usecols.append("depuradora")

        df = pd.read_csv(
            wastewater_file,
            usecols=usecols,  # type: ignore[arg-type]
            dtype={
                "id mostra": str,
                "depuradora": str,
                "Cabal últimes 24h(m3)": float,
                "IP4(CG/L)": float,
                "N1(CG/L)": float,
                "N2(CG/L)": float,
                "LD(CG/L)": float,
            },
        )

        df = df.rename(
            columns={
                "id mostra": "date",
                "Cabal últimes 24h(m3)": "flow_rate",
                "IP4(CG/L)": "IP4",
                "N1(CG/L)": "N1",
                "N2(CG/L)": "N2",
                "LD(CG/L)": "detection_limit",
            }
        )

        # Parse date from 'id mostra' (format: XXXX-YYYY-MM-DD)
        dates = df["date"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
        edar_codes = df["date"].astype(str).str.extract(r"^(\w+)-\d{4}-\d{2}-\d{2}$")[0]
        df["date"] = pd.to_datetime(dates)
        df["edar_id"] = edar_codes

        time_mask = df["date"].isin(
            pd.date_range(
                start=self.config.start_date, end=self.config.end_date, freq="D"
            )
        )
        df = df[time_mask]

        id_vars = ["date", "edar_id", "flow_rate"]
        if "detection_limit" in df.columns:
            id_vars.append("detection_limit")

        df = df.melt(
            id_vars=id_vars,
            value_vars=["N2", "IP4", "N1"],
            var_name="variant",
            value_name="viral_load",
        )

        df = df.dropna(subset=["date", "edar_id", "variant", "viral_load", "flow_rate"])

        # Remove negative values
        df = df[(df["viral_load"] >= 0) & (df["flow_rate"] >= 0)]

        return df

    def _select_variants(self, wastewater_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select top 2 most prevalent variants for each EDAR site.

        Args:
            wastewater_df: DataFrame with all variants

        Returns:
            DataFrame with top 2 variants per site
        """
        assert not wastewater_df.empty, "No wastewater data to select variants from"

        # Count non-null viral_load entries per (edar_id, variant)
        # Filter valid entries first
        valid_entries = wastewater_df[wastewater_df["viral_load"].notna()]

        if valid_entries.empty:
            raise ValueError("No valid entries to select variants from")

        # Group and count
        variant_counts = (
            valid_entries.groupby(["edar_id", "variant"])
            .size()
            .reset_index(name="count")  # type: ignore[call-arg]
        )

        # Rank variants by count within each edar_id
        variant_counts["rank"] = variant_counts.groupby("edar_id")["count"].rank(
            method="first", ascending=False
        )

        # Select top 2 variants
        top_variants = variant_counts[variant_counts["rank"] <= 2][
            ["edar_id", "variant"]
        ]

        # Filter original dataframe to keep only selected variants
        selected_df = wastewater_df.merge(  # type: ignore[arg-type]
            top_variants, on=["edar_id", "variant"], how="inner"
        )

        return selected_df

    def _remove_duplicates_and_aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate measurements and aggregate temporally.

        Args:
            df: DataFrame with potential duplicates

        Returns:
            Aggregated DataFrame
        """
        # Sort by date and site
        df = df.sort_values(["edar_id", "date", "variant"])

        # Remove exact duplicates
        df = df.drop_duplicates(subset=["edar_id", "date", "variant"], keep="last")

        # Aggregate multiple measurements on same day for same site
        agg_functions = {
            "viral_load": "mean",
            "flow_rate": "mean",
        }
        if "detection_limit" in df.columns:
            agg_functions["detection_limit"] = "max"

        aggregated = (
            df.groupby(["edar_id", "date", "variant"]).agg(agg_functions).reset_index()
        )

        return aggregated

    def _calculate_flow_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate viral load flow rates.

        Args:
            df: DataFrame with viral loads and flow rates

        Returns:
            DataFrame with calculated flow metrics
        """
        # Calculate viral concentration (viral load per unit flow)
        df["viral_concentration"] = df["viral_load"] / (df["flow_rate"] + 1e-8)

        flow_mode = self.config.wastewater_flow_mode
        if flow_mode not in {"total_flow", "concentration"}:
            raise ValueError(
                "Unsupported wastewater_flow_mode. Expected 'total_flow' or "
                f"'concentration', got {flow_mode!r}."
            )

        if flow_mode == "total_flow":
            # Calculate total COVID flow (viral_load * flow_rate)
            df["total_covid_flow"] = df["viral_load"] * df["flow_rate"]
            if "detection_limit" in df.columns:
                df["limit_flow"] = df["detection_limit"] * df["flow_rate"]
        else:
            # Use concentration directly without flow weighting
            df["total_covid_flow"] = df["viral_load"]
            if "detection_limit" in df.columns:
                df["limit_flow"] = df["detection_limit"]

        agg_columns = {"total_covid_flow": "sum"}
        if "limit_flow" in df.columns:
            agg_columns["limit_flow"] = "sum"

        df = df.groupby(["edar_id", "date", "variant"]).agg(agg_columns).reset_index()

        return df

    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("date")
        df = (
            df.groupby(["edar_id", "variant"])
            .resample("D")
            .sum(numeric_only=True)
            .reset_index()
        )
        return df

    def _compute_edar_statistics(self, edar_features: torch.Tensor) -> dict[str, float]:
        """Compute statistics for EDAR data."""
        return {
            "total_measurements": float(torch.sum(edar_features > 0)),
            "mean_viral_load": float(edar_features[:, :, 0].mean()),
            "mean_flow_rate": float(edar_features[:, :, 1].mean()),
            "mean_viral_concentration": float(edar_features[:, :, 2].mean()),
            "mean_total_covid_flow": float(edar_features[:, :, 3].mean()),
            "sites_with_data": int(torch.sum(edar_features.sum(dim=0).sum(dim=1) > 0)),
            "temporal_coverage": float(
                torch.mean(edar_features.sum(dim=1).sum(dim=1) > 0)
            ),
        }

    def _compute_quality_metrics(self, edar_features: torch.Tensor) -> dict[str, Any]:
        """Compute data quality metrics for EDAR data."""
        # Site-level coverage
        site_coverage = (edar_features.sum(dim=0).sum(dim=1) > 0).float().tolist()

        # Temporal coverage per site
        temporal_coverage = []
        for site_idx in range(edar_features.shape[1]):
            site_data = edar_features[:, site_idx, :] > 0
            coverage = site_data.any(dim=1).float().mean().item()
            temporal_coverage.append(coverage)

        return {
            "site_coverage": site_coverage,
            "mean_temporal_coverage": float(np.mean(temporal_coverage)),
            "median_temporal_coverage": float(np.median(temporal_coverage)),
            "data_completeness_score": float(np.mean(temporal_coverage)),
            "sites_with_continuous_data": int(
                np.sum(np.array(temporal_coverage) > 0.9)
            ),
        }

    def transform_to_regions(
        self,
        single_covid: pd.DataFrame,
        edar_muni_mapping: xr.DataArray,
    ) -> dict[str, Any]:
        """
        Transform wastewater data to region features using EDAR to municipality mapping.
        """
        print(single_covid.info())
        print(edar_muni_mapping)

        assert set(single_covid.edar_id) == set(edar_muni_mapping.edar_id.values), (
            "EDAR IDs in single_covid and edar_muni_mapping do not match"
        )

        edar_features = (
            single_covid.groupby(["date", "edar_id"])["total_covid_flow"]
            .sum()
            .to_xarray()
        )
        print(edar_features)

        result = xr.dot(edar_features, edar_muni_mapping, dims="edar_id")
        print(result)

        return result
