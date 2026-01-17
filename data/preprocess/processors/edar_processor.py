"""
Processor for EDAR wastewater biomarker data.

This module handles the conversion of wastewater biomarker data from EDAR
(Environmental DNA Analysis and Recovery) systems into canonical tensor formats.
It processes variant selection, flow calculations, temporal alignment, and
creates temporal tensors for downstream aggregation to target regions.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from ..config import REGION_COORD, PreprocessingConfig
from .quality_checks import DataQualityThresholds, validate_notna_and_std


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

    def process(self, wastewater_file: str, region_metadata_file: str) -> xr.DataArray:
        """
        Process EDAR wastewater biomarker data into a canonical xarray DataArray.

        Args:
            wastewater_file: Path to CSV/Excel file with wastewater data
            edar_mapping: Optional mapping from EDAR plants to municipalities

        Returns:
            Biomarker time series aggregated to regions, as an xarray DataArray
            with dims `[date, region_id]`.
        """
        print(f"Processing EDAR wastewater data from {wastewater_file}")

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

        daily_data_xr = daily_data.set_index(["date", "edar_id"])[
            "total_covid_flow"
        ].to_xarray()
        emap = xr.open_dataarray(region_metadata_file)
        # EDAR contribution matrices are typically stored sparsely with NaNs where
        # there is no contribution. `xr.dot` does not skip NaNs, so we must treat
        # missing contributions as zeros.
        emap = emap.fillna(0)
        emap = emap.rename({"home": REGION_COORD})

        print(
            f"Transforming EDAR data to regions using contribution matrix from {region_metadata_file}"
        )
        if "edar_id" not in daily_data_xr.dims:
            raise ValueError(
                "Expected 'edar_id' dimension in processed wastewater data, "
                f"got dims={tuple(daily_data_xr.dims)!r}"
            )
        if "edar_id" not in emap.dims:
            raise ValueError(
                "Expected 'edar_id' dimension in EDAR contribution matrix, "
                f"got dims={tuple(emap.dims)!r}"
            )

        # Align IDs before dot product. A common silent failure mode is that
        # `xr.dot` aligns on labels and produces all-NaNs when there is no overlap.
        daily_data_xr = daily_data_xr.assign_coords(
            edar_id=daily_data_xr["edar_id"].astype(str)
        )
        emap = emap.assign_coords(edar_id=emap["edar_id"].astype(str))

        wastewater_ids = set(daily_data_xr["edar_id"].values.tolist())
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

        daily_data_xr_aligned, emap_aligned = xr.align(
            daily_data_xr, emap, join="inner"
        )
        result = xr.dot(daily_data_xr_aligned, emap_aligned, dims="edar_id")

        result.name = "edar_biomarker"

        thresholds = DataQualityThresholds(
            min_notna_fraction=float(
                self.validation_options.get("min_notna_fraction", 0.99)
            ),
            min_std_epsilon=float(
                self.validation_options.get("min_std_epsilon", 1e-12)
            ),
        )
        validate_notna_and_std(result, name="edar_biomarker", thresholds=thresholds)

        return result

    def _load_wastewater_data(self, wastewater_file: str) -> pd.DataFrame:
        df = pd.read_csv(
            wastewater_file,
            usecols=[  # type: ignore[arg-type]
                "id mostra",
                "Cabal últimes 24h(m3)",
                "IP4(CG/L)",
                "N1(CG/L)",
                "N2(CG/L)",
            ],
            dtype={
                "id mostra": str,
                "depuradora": str,
                "Cabal últimes 24h(m3)": float,
                "IP4(CG/L)": float,
                "N1(CG/L)": float,
                "N2(CG/L)": float,
            },
        )

        df = df.rename(
            columns={
                "id mostra": "date",
                "Cabal últimes 24h(m3)": "flow_rate",
                "IP4(CG/L)": "IP4",
                "N1(CG/L)": "N1",
                "N2(CG/L)": "N2",
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

        df = df.melt(
            id_vars=["date", "edar_id", "flow_rate"],
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

        # Calculate total COVID flow (viral_load * flow_rate)
        df["total_covid_flow"] = df["viral_load"] * df["flow_rate"]

        df = df.groupby(["edar_id", "date"])["total_covid_flow"].sum().reset_index()

        return df

    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df.pivot(index="date", columns="edar_id", values="total_covid_flow")
            .resample("D")
            .sum()
        )
        # Unpivot the date index to a column and edar_id index to a column
        df = df.reset_index().melt(
            id_vars="date", var_name="edar_id", value_name="total_covid_flow"
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
