"""
Processor for Catalonia official COVID-19 case data.

This module handles the conversion of official Catalonia open data CSV files
into canonical xarray formats. It processes daily case counts at municipality
level and produces output compatible with the existing pipeline.

Data source: Registre de casos de COVID-19 a Catalunya per municipi i sexe
https://datos.gob.es/ca/catalogo/a09002970-registro-de-test-de-covid-19-realizados-en-catalunya-segregacion-por-sexo-y-municipio
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.statespace.structural import UnobservedComponents

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from .edar_processor import _KalmanFilter


class CataloniaCasesProcessor:
    """
    Converts official Catalonia COVID case data to xarray Dataset with mask and age channels.

    Input format: Registre de casos de COVID-19 a Catalunya per municipi i sexe
    https://datos.gob.es/ca/catalogo/a09002970-registro-de-test-de-covid-19-realizados-en-catalunya-segregacion-por-sexo-y-municipio

    Expected CSV columns (Catalan):
    - TipusCasData: Date in DD/MM/YYYY format
    - MunicipiCodi: Municipality code (with leading zeros)
    - TipusCasDescripcio: Test type description
    - NumCasos: Number of cases

    Features:
    - Kalman smoothing for noise reduction
    - Mask channel (1.0 for observed, 0.0 for missing/interpolated)
    - Age channel (days since last observation, normalized)
    - Optional PCR/TAR breakdown by test type
    """

    # Column mapping from Catalan to English
    COLUMN_MAPPING = {
        "TipusCasData": "date",
        "MunicipiCodi": "municipality_code",
        "TipusCasDescripcio": "test_type",
        "NumCasos": "n_cases",
    }

    # Data types to preserve leading zeros
    DTYPES = {
        "TipusCasData": str,
        "MunicipiCodi": str,  # Critical: preserve leading zeros
        "ComarcaCodi": str,
        "SexeCodi": str,
        "NumCasos": int,
    }

    # Test type categorization for inferred testing rate
    TEST_TYPE_CATEGORIES = {
        "Positiu TAR": "TAR",
        "Positiu per Test Ràpid": "TAR",
        "Positiu PCR": "PCR",
        "PCR probable": "PCR",
        "Positiu per ELISA": "ELISA",
        "Epidemiològic": "EPI",
    }

    def __init__(self, config: PreprocessingConfig):
        """Initialize the cases processor."""
        self.config = config
        self.validation_options = config.validation_options

    def _load_raw_data(self, cases_file: Path) -> pd.DataFrame:
        """Load raw cases data from Catalonia official CSV format."""
        print(f"  Loading cases from {cases_file}")

        if not cases_file.exists():
            raise FileNotFoundError(f"Cases file not found: {cases_file}")

        # Read with explicit dtypes to preserve leading zeros
        df = pd.read_csv(
            cases_file,
            dtype=self.DTYPES,
            usecols=list(self.DTYPES.keys()) + ["TipusCasDescripcio"],
        )

        # Rename columns to English
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Drop rows where municipality_code is NaN or empty string
        df = df[df["municipality_code"].notna() & (df["municipality_code"] != "")]

        # Municipality codes are already strings from dasymetric_mob output
        # Ensure they are strings
        df["municipality_code"] = df["municipality_code"].astype(str)

        # Parse dates from DD/MM/YYYY format
        df["date"] = (
            pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
            .dt.tz_localize(None)
            .dt.floor("D")
        )

        # Remove rows with invalid data
        df = df.dropna(subset=["date", "municipality_code", "n_cases"])

        # Remove negative values
        df = df[df["n_cases"] >= 0]

        return df

        # Read with explicit dtypes to preserve leading zeros
        df = pd.read_csv(  # type: ignore[call-arg]
            cases_file,
            dtype=self.DTYPES,
            usecols=list(self.DTYPES.keys()) + ["TipusCasDescripcio"],
        )

        # Rename columns to English
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Parse dates from DD/MM/YYYY format
        df["date"] = (
            pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
            .dt.tz_localize(None)
            .dt.floor("D")
        )

        # Remove invalid rows
        df = df.dropna(subset=["date", "municipality_code", "n_cases"])

        # Filter to valid cases only (remove empty municipality codes)
        df = df[df["municipality_code"] != ""]

        # Remove negative cases if any
        df = df[df["n_cases"] >= 0]

        print(f"  Loaded {len(df):,} case records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique municipalities: {df['municipality_code'].nunique()}")

        return df

    def _categorize_test_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standardized test type category column."""
        df["test_category"] = df["test_type"].map(self.TEST_TYPE_CATEGORIES)  # type: ignore[arg-type]
        # Keep original test type as is for unmapped categories
        df["test_category"] = df["test_category"].fillna(df["test_type"])
        return df

    def _aggregate_by_municipality(
        self, df: pd.DataFrame, by_test_type: bool = False
    ) -> pd.DataFrame:
        """Aggregate cases to municipality-date level."""
        group_cols = ["date", "municipality_code"]
        if by_test_type:
            group_cols.append("test_category")

        aggregated = df.groupby(group_cols, dropna=False)["n_cases"].sum().reset_index()
        # Rename to 'cases' for downstream compatibility
        aggregated = aggregated.rename(columns={"n_cases": "cases"})

        return aggregated

    def _fit_kalman_params(self, series: pd.Series) -> tuple[float, float]:
        """Fit Kalman filter parameters from time series data.

        Uses statsmodels UnobservedComponents to fit a local level model
        and extract the process and measurement variances.

        Args:
            series: Time series of case values

        Returns:
            Tuple of (process_var, measurement_var)
        """
        # Filter to positive values for log transform
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

        # Ensure positive variances
        process_var = max(process_var, 1e-6)
        measurement_var = max(measurement_var, 1e-6)

        return process_var, measurement_var

    def _apply_kalman_smoothing(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard Kalman filtering for case data smoothing.

        Helps reduce reporting noise (weekend effects, holiday delays).
        """
        print("  Applying Kalman smoothing to cases...")

        fallback_process = float(
            self.config.validation_options.get("process_var", 0.05)
        )
        fallback_measure = float(
            self.config.validation_options.get("measurement_var", 0.5)
        )

        smoothed_records = []

        for muni_code in daily_df["municipality_code"].unique():
            muni_data = daily_df[daily_df["municipality_code"] == muni_code].copy()
            muni_data = muni_data.set_index("date").sort_index()

            # Get values as numpy array
            values = muni_data["cases"].values

            # Fit Kalman parameters from data (mask zero/negative for fitting)
            fit_series = pd.Series(values, index=muni_data.index)
            fit_series = fit_series.where(fit_series > 0)

            process_var, measurement_var = self._fit_kalman_params(fit_series)

            # Initialize Kalman filter with fitted params
            kf = _KalmanFilter(
                process_var=process_var,
                measurement_var=measurement_var,
            )

            # Apply filter
            filtered_values, flags = kf.filter_series(values)

            # Create smoothed records
            for i, date in enumerate(muni_data.index):
                smoothed_records.append(
                    {
                        "date": date,
                        "municipality_code": muni_code,
                        "cases": np.exp(
                            filtered_values[i]
                        ),  # Back-transform from log space
                        "cases_log": filtered_values[i],
                        "missing_flag": flags[i],  # 0=normal, 2=missing
                    }
                )

        smoothed_df = pd.DataFrame(smoothed_records)

        # Check for non-finite values
        non_finite = (~np.isfinite(smoothed_df["cases"])).sum()
        if non_finite > 0:
            print(f"  Warning: {non_finite} non-finite values after smoothing")
            smoothed_df["cases"] = smoothed_df["cases"].replace(
                [np.inf, -np.inf], np.nan
            )

        print(
            f"  Smoothing complete for {smoothed_df['municipality_code'].nunique()} municipalities"
        )

        return smoothed_df

    def _create_mask_and_age_channels(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Create mask and age channels for cases.

        Mask: 1.0 if case count is observed (not missing), 0.0 if missing.
        Age: Days since last observation, normalized to [0, 1] (max 14 days).
        """
        daily_df = daily_df.copy()

        # Mask: 1.0 for actual observations (not marked as missing by Kalman)
        if "missing_flag" in daily_df.columns:
            is_not_missing = daily_df["missing_flag"] < 1.5
        else:
            is_not_missing = daily_df["cases"].notna()

        daily_df["cases_mask"] = is_not_missing.astype(float)

        # Age: Days since last actual observation
        age_series_list = []
        for muni_code in daily_df["municipality_code"].unique():
            muni_data = daily_df[
                daily_df["municipality_code"] == muni_code
            ].sort_values("date")
            muni_mask = muni_data["cases_mask"] > 0.5

            # Create a group ID that increments each time a measurement is seen
            groups = muni_mask.cumsum()
            # Count days within each group
            age = muni_data.groupby(groups).cumcount() + 1

            # Handle the leading zeros (before first measurement) - age should be max
            first_true_idx = muni_mask.idxmax() if muni_mask.any() else None
            if first_true_idx is not None:
                age.loc[muni_data["date"] < muni_data.loc[first_true_idx, "date"]] = 14
            else:
                age[:] = 14

            muni_data["cases_age"] = age.clip(upper=14).astype(float)
            age_series_list.append(muni_data)

        return pd.concat(age_series_list, ignore_index=True)

    def _build_cases_dataset(
        self,
        aggregated: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        apply_smoothing: bool = True,
    ) -> xr.Dataset:
        """Build cases dataset with Kalman smoothing and mask/age channels.

        Args:
            aggregated: DataFrame with columns [date, municipality_code, cases]
            date_range: Complete date range for reindexing
            apply_smoothing: Whether to apply Kalman smoothing

        Returns:
            xr.Dataset with cases, cases_mask, cases_age variables
        """
        # Create pivot table: date x municipality_code
        pivot = aggregated.pivot_table(
            index="date",
            columns="municipality_code",
            values="cases",
            aggfunc="sum",
            fill_value=0,
        )

        # Reindex to complete date range
        pivot = pivot.reindex(date_range, fill_value=0)

        # Rename for compatibility
        pivot.columns.name = REGION_COORD
        pivot.index.name = TEMPORAL_COORD

        # Convert to DataFrame for processing
        daily_df = pivot.reset_index().melt(
            id_vars=[TEMPORAL_COORD],
            var_name="municipality_code",
            value_name="cases",
        )

        # Apply Kalman smoothing
        if apply_smoothing:
            daily_df = self._apply_kalman_smoothing(daily_df)

        # Create mask and age channels
        daily_df = self._create_mask_and_age_channels(daily_df)

        # Pivot back to wide format for xarray
        pivot_cases = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="cases",
            aggfunc="sum",
        ).reindex(date_range, fill_value=0)

        pivot_mask = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="cases_mask",
            aggfunc="max",
        ).reindex(date_range, fill_value=0)

        pivot_age = daily_df.pivot_table(
            index="date",
            columns="municipality_code",
            values="cases_age",
            aggfunc="max",
        ).reindex(date_range, fill_value=1)

        # Rename columns
        pivot_cases.columns.name = REGION_COORD
        pivot_cases.index.name = TEMPORAL_COORD
        pivot_mask.columns.name = REGION_COORD
        pivot_mask.index.name = TEMPORAL_COORD
        pivot_age.columns.name = REGION_COORD
        pivot_age.index.name = TEMPORAL_COORD

        # Convert to xarray DataArrays
        cases_da = xr.DataArray(
            pivot_cases.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot_cases.index,
                REGION_COORD: pivot_cases.columns.astype(str),
            },
            name="cases",
        )

        mask_da = xr.DataArray(
            pivot_mask.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot_mask.index,
                REGION_COORD: pivot_mask.columns.astype(str),
            },
            name="cases_mask",
        )

        age_da = xr.DataArray(
            pivot_age.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot_age.index,
                REGION_COORD: pivot_age.columns.astype(str),
            },
            name="cases_age",
        )

        # Add run_id dimension to match other datasets (real data gets run_id="real")
        cases_da = cases_da.expand_dims(run_id=["real"])
        mask_da = mask_da.expand_dims(run_id=["real"])
        age_da = age_da.expand_dims(run_id=["real"])

        return xr.Dataset(
            {"cases": cases_da, "cases_mask": mask_da, "cases_age": age_da}
        )

    def process(
        self,
        cases_file: str | Path,
        by_test_type: bool = False,
        apply_smoothing: bool = True,
    ) -> xr.Dataset:
        """
        Process COVID case data into canonical xarray Dataset.

        Args:
            cases_file: Path to cases CSV file (Catalonia or Flowmaps format)
            by_test_type: If True, return separate variables for PCR/TAR cases
            apply_smoothing: If True, apply Kalman smoothing to case data

        Returns:
            xarray Dataset with structure:
            - cases: total cases (date, region_id)
            - cases_mask: observation mask (date, region_id)
            - cases_age: days since observation (date, region_id)
            - cases_pcr: PCR cases (optional, if by_test_type=True)
            - cases_tar: TAR cases (optional, if by_test_type=True)
        """
        print("Processing case data")

        cases_file = Path(cases_file)
        if not cases_file.exists():
            raise FileNotFoundError(f"Cases file not found: {cases_file}")

        # Load raw data
        cases_df = self._load_raw_data(cases_file)

        # Add test type categories
        cases_df = self._categorize_test_types(cases_df)

        # Filter to config date range
        cases_df = cases_df[
            (cases_df["date"] >= self.config.start_date)
            & (cases_df["date"] <= self.config.end_date)
        ]

        if cases_df.empty:
            raise ValueError("No case data in specified date range")

        # Create complete date range
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )

        # Aggregate to municipality level for total cases
        aggregated_total = self._aggregate_by_municipality(cases_df, by_test_type=False)

        # Build main cases dataset with smoothing and mask/age
        result_ds = self._build_cases_dataset(
            aggregated_total, date_range, apply_smoothing=apply_smoothing
        )

        if by_test_type:
            # Process PCR cases
            aggregated_pcr = self._aggregate_by_municipality(
                cases_df[cases_df["test_category"] == "PCR"], by_test_type=False
            )
            if not aggregated_pcr.empty:
                pcr_ds = self._build_cases_dataset(
                    aggregated_pcr.rename(columns={"n_cases": "cases"}),
                    date_range,
                    apply_smoothing=apply_smoothing,
                )
                result_ds["cases_pcr"] = pcr_ds["cases"]

            # Process TAR cases
            aggregated_tar = self._aggregate_by_municipality(
                cases_df[cases_df["test_category"] == "TAR"], by_test_type=False
            )
            if not aggregated_tar.empty:
                tar_ds = self._build_cases_dataset(
                    aggregated_tar.rename(columns={"n_cases": "cases"}),
                    date_range,
                    apply_smoothing=apply_smoothing,
                )
                result_ds["cases_tar"] = tar_ds["cases"]

        print(
            f"  Processed cases: {result_ds['cases'].sizes[TEMPORAL_COORD]} dates x "
            f"{result_ds['cases'].sizes[REGION_COORD]} regions"
        )
        print(f"  Total cases: {int(result_ds['cases'].sum())}")

        return result_ds
