"""
Processor for EDAR wastewater biomarker data.

This module handles the conversion of wastewater biomarker data from EDAR
(Environmental DNA Analysis and Recovery) systems into canonical tensor formats.
It processes variant selection, flow calculations, temporal alignment, and
creates attention masks for the bipartite graph between municipalities and
EDAR plants.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import PreprocessingConfig


class EDARProcessor:
    """
    Converts EDAR wastewater biomarker data to canonical tensors.

    This processor handles:
    - Loading and parsing wastewater biomarker data
    - Variant selection (N2, IP4, etc.) based on data quality
    - Duplicate removal and temporal aggregation
    - Flow calculation and normalization
    - Resampling to daily frequency
    - Creation of municipality-EDAR bipartite graph connectivity

    The output includes EDAR features, attention masks for graph connectivity,
    and metadata for integration with other data sources.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the EDAR processor.

        Args:
            config: Preprocessing configuration with EDAR processing options
        """
        self.config = config
        self.validation_options = config.validation_options

    def process(
        self, wastewater_file: str, edar_mapping: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process EDAR wastewater biomarker data into canonical tensors.

        Args:
            wastewater_file: Path to CSV/Excel file with wastewater data
            edar_mapping: Optional mapping from EDAR plants to municipalities

        Returns:
            Dictionary containing processed EDAR data:
            - edar_features: [time, num_edar_sites, feature_dim] tensor
            - edar_attention_mask: [num_municipalities, num_edar_sites] tensor
            - edar_metadata: Dictionary with site information
            - metadata: Processing metadata and statistics
        """
        print(f"Processing EDAR wastewater data from {wastewater_file}")

        # Load wastewater data
        wastewater_df = self._load_wastewater_data(wastewater_file)

        # Process EDAR mapping
        mapping_info = self._process_edar_mapping(wastewater_df, edar_mapping)

        # Select best variant for each site
        selected_data = self._select_variants(wastewater_df)

        # Remove duplicates and aggregate
        aggregated_data = self._remove_duplicates_and_aggregate(selected_data)

        # Calculate flow rates
        flow_data = self._calculate_flow_rates(aggregated_data)

        # Resample to daily frequency
        daily_data = self._resample_to_daily(flow_data)

        # Create temporal tensors
        edar_features = self._create_edar_tensors(daily_data, mapping_info)

        # Create attention mask for bipartite graph
        edar_attention_mask = self._create_attention_mask(mapping_info)

        # Validate data quality
        self._validate_edar_data(edar_features, edar_attention_mask)

        # Create metadata
        metadata = {
            "num_timepoints": edar_features.shape[0],
            "num_edar_sites": edar_features.shape[1],
            "feature_dim": edar_features.shape[2],
            "date_range": {
                "start": self.config.start_date.isoformat(),
                "end": self.config.end_date.isoformat(),
            },
            "variants_used": list(selected_data["variant"].unique()),
            "data_stats": self._compute_edar_statistics(edar_features),
            "quality_metrics": self._compute_quality_metrics(edar_features),
        }

        return {
            "edar_features": edar_features,
            "edar_attention_mask": edar_attention_mask,
            "edar_metadata": mapping_info,
            "metadata": metadata,
        }

    def _load_wastewater_data(self, wastewater_file: str) -> pd.DataFrame:
        """
        Load and validate wastewater data.

        Args:
            wastewater_file: Path to wastewater data file

        Returns:
            DataFrame with wastewater data
        """
        file_path = Path(wastewater_file)

        # Determine file format
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(wastewater_file)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(wastewater_file)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Validate required columns
        required_columns = ["date", "edar_id", "variant", "viral_load", "flow_rate"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert date column
        df["date"] = pd.to_datetime(df["date"])

        # Clean and validate data
        df = self._clean_wastewater_data(df)

        return df

    def _clean_wastewater_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate wastewater data.

        Args:
            df: Raw wastewater DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Convert numeric columns
        numeric_columns = ["viral_load", "flow_rate"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove invalid rows
        df = df.dropna(subset=["date", "edar_id", "variant", "viral_load", "flow_rate"])

        # Remove negative values
        df = df[(df["viral_load"] >= 0) & (df["flow_rate"] >= 0)]

        # Filter by flow rate threshold
        min_flow = self.config.min_flow_threshold
        df = df[df["flow_rate"] >= min_flow]

        return df

    def _process_edar_mapping(
        self, wastewater_df: pd.DataFrame, edar_mapping: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Process EDAR to municipality mapping.

        Args:
            wastewater_df: DataFrame with wastewater data
            edar_mapping: Optional mapping dictionary

        Returns:
            Dictionary with mapping information
        """
        # Get unique EDAR sites from data
        unique_edar_sites = sorted(wastewater_df["edar_id"].unique())
        num_edar_sites = len(unique_edar_sites)

        # Create EDAR site to index mapping
        edar_id_to_index = {
            edar_id: idx for idx, edar_id in enumerate(unique_edar_sites)
        }

        # Create default attention mask (all-to-all connectivity)
        num_municipalities = 100  # Default, should be updated from actual data
        attention_mask = torch.ones(num_municipalities, num_edar_sites)

        # Apply custom mapping if provided
        if edar_mapping is not None:
            attention_mask = self._apply_custom_mapping(
                edar_mapping, unique_edar_sites, edar_id_to_index
            )

        mapping_info = {
            "unique_edar_sites": unique_edar_sites,
            "num_edar_sites": num_edar_sites,
            "edar_id_to_index": edar_id_to_index,
            "attention_mask": attention_mask,
            "custom_mapping": edar_mapping is not None,
        }

        return mapping_info

    def _apply_custom_mapping(
        self,
        edar_mapping: dict[str, Any],
        unique_edar_sites: list[int],
        edar_id_to_index: dict[int, int],
    ) -> torch.Tensor:
        """
        Apply custom EDAR to municipality mapping.

        Args:
            edar_mapping: Mapping from EDAR sites to municipalities
            unique_edar_sites: List of EDAR site IDs
            edar_id_to_index: EDAR ID to index mapping

        Returns:
            Attention mask tensor
        """
        # Determine number of municipalities from mapping
        all_municipalities = set()
        for _site_id, municipalities in edar_mapping.items():
            all_municipalities.update(municipalities)

        num_municipalities = max(all_municipalities) + 1
        num_edar_sites = len(unique_edar_sites)

        # Create attention mask
        attention_mask = torch.zeros(num_municipalities, num_edar_sites)

        for site_id, municipalities in edar_mapping.items():
            if site_id in edar_id_to_index:
                site_idx = edar_id_to_index[site_id]
                for mun_id in municipalities:
                    attention_mask[mun_id, site_idx] = 1.0

        return attention_mask

    def _select_variants(self, wastewater_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select best variant for each EDAR site based on data quality.

        Args:
            wastewater_df: DataFrame with all variants

        Returns:
            DataFrame with selected variants
        """
        # Prioritize variants in order of preference
        variant_preference = ["N2", "IP4", "N1", "E", "S"]

        selected_rows = []

        for edar_id in wastewater_df["edar_id"].unique():
            site_data = wastewater_df[wastewater_df["edar_id"] == edar_id]

            # Try variants in order of preference
            for variant in variant_preference:
                variant_data = site_data[site_data["variant"] == variant]

                if len(variant_data) > 0:
                    # Check data quality
                    data_completeness = 1.0 - variant_data[
                        "viral_load"
                    ].isna().sum() / len(variant_data)

                    if data_completeness >= 0.5:  # At least 50% complete
                        selected_rows.append(variant_data)
                        break
            else:
                # If no preferred variant found, use the one with most data
                best_variant = site_data.groupby("variant").size().idxmax()
                selected_rows.append(site_data[site_data["variant"] == best_variant])

        return pd.concat(selected_rows, ignore_index=True)

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
            "flow_rate": "sum",  # Sum flow rates for multiple measurements
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

        return df

    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to daily frequency.

        Args:
            df: DataFrame with potentially irregular timestamps

        Returns:
            DataFrame resampled to daily frequency
        """
        # Set date as index
        df = df.set_index("date")

        # Group by EDAR site and resample
        daily_data = []

        for edar_id in df["edar_id"].unique():
            site_data = df[df["edar_id"] == edar_id]

            # Resample to daily frequency with linear interpolation
            resampled = site_data.resample("D").mean()

            # Forward fill missing values
            resampled = resampled.fillna(method="ffill").fillna(method="bfill")

            # Keep EDAR ID
            resampled["edar_id"] = edar_id
            daily_data.append(resampled.reset_index())

        if daily_data:
            return pd.concat(daily_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def _create_edar_tensors(
        self, daily_data: pd.DataFrame, mapping_info: dict[str, Any]
    ) -> torch.Tensor:
        """
        Create temporal tensors from daily EDAR data.

        Args:
            daily_data: Daily resampled data
            mapping_info: EDAR mapping information

        Returns:
            [time, num_edar_sites, feature_dim] tensor
        """
        # Create date range
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )

        num_timepoints = len(date_range)
        num_edar_sites = mapping_info["num_edar_sites"]
        edar_id_to_index = mapping_info["edar_id_to_index"]

        # Feature columns to include
        feature_columns = [
            "viral_load",
            "flow_rate",
            "viral_concentration",
            "total_covid_flow",
        ]
        feature_dim = len(feature_columns)

        # Initialize tensor
        edar_features = torch.zeros(num_timepoints, num_edar_sites, feature_dim)

        # Fill tensor with data
        for _, row in daily_data.iterrows():
            date = row["date"]
            edar_id = int(row["edar_id"])

            # Check if date is in our range
            if self.config.start_date <= date <= self.config.end_date:
                date_idx = (date - self.config.start_date).days
                if date_idx < num_timepoints and edar_id in edar_id_to_index:
                    site_idx = edar_id_to_index[edar_id]

                    for feat_idx, col in enumerate(feature_columns):
                        if col in row and not pd.isna(row[col]):
                            edar_features[date_idx, site_idx, feat_idx] = float(
                                row[col]
                            )

        # Apply log1p normalization to viral-related features
        viral_features = [0, 2, 3]  # viral_load, viral_concentration, total_covid_flow
        for feat_idx in viral_features:
            edar_features[:, :, feat_idx] = torch.log1p(
                torch.clamp(edar_features[:, :, feat_idx], min=0)
            )

        return edar_features

    def _create_attention_mask(self, mapping_info: dict[str, Any]) -> torch.Tensor:
        """
        Create attention mask for municipality-EDAR connectivity.

        Args:
            mapping_info: EDAR mapping information

        Returns:
            [num_municipalities, num_edar_sites] attention mask
        """
        return mapping_info["attention_mask"]

    def _validate_edar_data(
        self, edar_features: torch.Tensor, edar_attention_mask: torch.Tensor
    ):
        """
        Validate processed EDAR data quality.

        Args:
            edar_features: Processed EDAR features tensor
            edar_attention_mask: Attention mask tensor
        """
        # Check for NaN values
        if torch.isnan(edar_features).any():
            raise ValueError("NaN values found in processed EDAR features")

        if torch.isnan(edar_attention_mask).any():
            raise ValueError("NaN values found in attention mask")

        # Check attention mask values
        if not torch.all((edar_attention_mask >= 0) & (edar_attention_mask <= 1)):
            raise ValueError("Attention mask contains values outside [0, 1] range")

        # Check for sites with no connectivity
        site_connectivity = edar_attention_mask.sum(dim=0)
        zero_connectivity_sites = (site_connectivity == 0).sum().item()
        if zero_connectivity_sites > 0:
            print(
                f"Warning: {zero_connectivity_sites} EDAR sites have no connectivity to municipalities"
            )

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
