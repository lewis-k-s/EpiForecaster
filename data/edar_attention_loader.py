"""
EDAR-Municipality Attention Mask Loader

Loads and processes the contribution ratios from EDAR wastewater treatment plants
to municipalities, creating an attention mask for the model.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray as xr
from einops import einsum, reduce

from .zone_registry import ZoneRegistry

logger = logging.getLogger(__name__)


class EDARAttentionLoader:
    """
    Loads and manages EDAR-municipality contribution ratios for attention masking.

    The contribution_ratio matrix defines how much each municipality contributes
    to each EDAR's wastewater signal, which we invert to determine how each EDAR's
    signal should be weighted when forecasting for each municipality.
    """

    def __init__(
        self,
        edar_edges_path: str,
        normalize_contributions: bool = True,
        min_contribution_threshold: float = 0.0,
        device: str = "cpu",
    ):
        """
        Initialize EDAR attention loader.

        Args:
            edar_edges_path: Path to edar_muni_edges.nc file
            normalize_contributions: Whether to normalize contribution weights
            min_contribution_threshold: Minimum contribution to include (filters noise)
            device: Device for tensors ('cpu' or 'cuda')
        """
        self.edar_edges_path = Path(edar_edges_path)
        self.normalize_contributions = normalize_contributions
        self.min_contribution_threshold = min_contribution_threshold
        self.device = device

        # Load data
        self.ds = None
        self.contribution_matrix = None
        self.attention_mask = None
        self.edar_ids = None
        self.municipality_ids = None
        self.edar_to_idx = {}
        self.muni_to_idx = {}

        # Extension state
        self._extended_mask = None
        self._extended_zone_registry = None
        self._extension_stats = None

        self._load_data()

    def _load_data(self):
        """Load and process the EDAR-municipality edges data.

        Key tensor shapes:
        - Raw contribution_matrix: [n_edars, n_municipalities]
        - Processed attention_mask: [n_municipalities, n_edars]
        """
        if not self.edar_edges_path.exists():
            raise FileNotFoundError(
                f"EDAR edges file not found: {self.edar_edges_path}"
            )

        logger.info(f"Loading EDAR-municipality edges from {self.edar_edges_path}")

        # Open NetCDF dataset
        self.ds = xr.open_dataset(self.edar_edges_path)

        # Extract dimensions and data
        self.edar_ids = self.ds["edar_id"].values
        self.municipality_ids = self.ds["home"].values
        contribution_data = self.ds["contribution_ratio"].values

        # Validate input data
        assert contribution_data.ndim == 2, (
            f"Expected 2D contribution data, got {contribution_data.ndim}"
        )
        n_edars_raw, n_municipalities_raw = contribution_data.shape
        assert n_edars_raw == len(self.edar_ids), (
            f"EDAR count mismatch: {n_edars_raw} vs {len(self.edar_ids)}"
        )
        assert n_municipalities_raw == len(self.municipality_ids), (
            f"Municipality count mismatch: {n_municipalities_raw} vs {len(self.municipality_ids)}"
        )

        # Create ID to index mappings
        self.edar_to_idx = {str(eid): i for i, eid in enumerate(self.edar_ids)}
        self.muni_to_idx = {str(mid): i for i, mid in enumerate(self.municipality_ids)}

        # Process contribution matrix
        # Original shape: [n_edars, n_municipalities]
        # We need: for each municipality, which EDARs contribute and by how much

        # Replace NaN with 0 and validate
        contribution_data = np.nan_to_num(contribution_data, nan=0.0)
        assert not np.isnan(contribution_data).any(), (
            "Contribution data contains NaN after processing"
        )

        # Apply threshold
        contribution_data[contribution_data < self.min_contribution_threshold] = 0.0

        # Store raw contribution matrix: [n_edars, n_municipalities]
        self.contribution_matrix = contribution_data

        # Create attention mask (transpose to get municipality x EDAR)
        # Shape: [n_municipalities, n_edars]
        # This tells us for each municipality, which EDARs are relevant
        self.attention_mask = contribution_data.T
        n_municipalities, n_edars = self.attention_mask.shape

        # Normalize if requested using einops for clarity
        if self.normalize_contributions:
            # Normalize each municipality's EDAR contributions to sum to 1
            # Row-wise normalization: sum over EDAR dimension for each municipality
            row_sums = reduce(
                self.attention_mask, "municipalities edars -> municipalities 1", "sum"
            )
            # Avoid division by zero for municipalities with no EDAR contributions
            row_sums[row_sums == 0] = 1.0
            self.attention_mask = self.attention_mask / row_sums

            # Validate normalization
            row_sums_after = reduce(
                self.attention_mask, "municipalities edars -> municipalities 1", "sum"
            )
            assert np.allclose(row_sums_after, 1.0), (
                "Attention mask rows should sum to 1 after normalization"
            )

        logger.info(f"Loaded attention mask with shape: {self.attention_mask.shape}")
        logger.info(f"Number of EDARs: {n_edars}")
        logger.info(f"Number of municipalities: {n_municipalities}")

        # Log sparsity statistics
        nonzero_entries = np.count_nonzero(self.attention_mask)
        total_entries = self.attention_mask.size
        sparsity = 1.0 - (nonzero_entries / total_entries)
        logger.info(
            f"Attention mask sparsity: {sparsity:.2%} ({nonzero_entries}/{total_entries} non-zero entries)"
        )

    def get_attention_tensor(self) -> torch.Tensor:
        """
        Get the attention mask as a PyTorch tensor.

        Returns:
            Attention mask tensor of shape [n_municipalities, n_edars]
            Each row sums to 1.0 (if normalized), representing EDAR contribution weights
            for each municipality's wastewater signal.
        """
        # Validate tensor shape and values before conversion
        assert self.attention_mask.ndim == 2, (
            f"Expected 2D attention mask, got {self.attention_mask.shape}"
        )
        assert not np.isnan(self.attention_mask).any(), (
            "Attention mask contains NaN values"
        )
        assert not np.isinf(self.attention_mask).any(), (
            "Attention mask contains infinite values"
        )

        attention_tensor = torch.tensor(
            self.attention_mask, dtype=torch.float32, device=self.device
        )

        # Final validation of tensor properties
        assert attention_tensor.shape[0] == len(self.municipality_ids), (
            "Municipality dimension mismatch"
        )
        assert attention_tensor.shape[1] == len(self.edar_ids), (
            "EDAR dimension mismatch"
        )

        return attention_tensor

    def get_municipality_edars(self, municipality_id: str) -> list[tuple[str, float]]:
        """
        Get EDARs that contribute to a specific municipality.

        Args:
            municipality_id: Municipality identifier

        Returns:
            List of (edar_id, contribution_weight) tuples
        """
        if municipality_id not in self.muni_to_idx:
            logger.warning(f"Municipality {municipality_id} not found")
            return []

        muni_idx = self.muni_to_idx[municipality_id]
        contributions = self.attention_mask[muni_idx]

        # Get non-zero contributions
        contributing_edars = []
        for edar_idx, weight in enumerate(contributions):
            if weight > 0:
                edar_id = str(self.edar_ids[edar_idx])
                contributing_edars.append((edar_id, float(weight)))

        # Sort by contribution weight
        contributing_edars.sort(key=lambda x: x[1], reverse=True)

        return contributing_edars

    def get_edar_municipalities(self, edar_id: str) -> list[tuple[str, float]]:
        """
        Get municipalities that contribute to a specific EDAR.

        Args:
            edar_id: EDAR identifier

        Returns:
            List of (municipality_id, contribution_weight) tuples
        """
        if edar_id not in self.edar_to_idx:
            logger.warning(f"EDAR {edar_id} not found")
            return []

        edar_idx = self.edar_to_idx[edar_id]
        contributions = self.contribution_matrix[edar_idx]

        # Get non-zero contributions
        contributing_munis = []
        for muni_idx, weight in enumerate(contributions):
            if weight > 0:
                muni_id = str(self.municipality_ids[muni_idx])
                contributing_munis.append((muni_id, float(weight)))

        # Sort by contribution weight
        contributing_munis.sort(key=lambda x: x[1], reverse=True)

        return contributing_munis

    def create_sparse_attention(self) -> torch.sparse.FloatTensor:
        """
        Create a sparse tensor representation of the attention mask.
        More memory efficient for large, sparse matrices.

        Returns:
            Sparse attention mask tensor
        """
        # Find non-zero entries
        nonzero_indices = np.nonzero(self.attention_mask)
        indices = torch.LongTensor(np.vstack(nonzero_indices))
        values = torch.FloatTensor(self.attention_mask[nonzero_indices])
        shape = self.attention_mask.shape

        sparse_mask = torch.sparse.FloatTensor(
            indices, values, shape, device=self.device
        )

        return sparse_mask

    def apply_attention_mask(
        self,
        edar_embeddings: torch.Tensor,
        municipality_indices: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Apply attention mask to EDAR embeddings to get municipality-specific signals.

        Args:
            edar_embeddings: EDAR node embeddings of shape [n_edars, embedding_dim]
            municipality_indices: Optional subset of municipalities to compute
                List of municipality indices

        Returns:
            Municipality-specific EDAR signals of shape [n_municipalities, embedding_dim]
            Each municipality's signal is a weighted sum of EDAR embeddings based on
            wastewater contribution ratios.
        """
        # Validate input tensor shape
        assert edar_embeddings.ndim == 2, (
            f"Expected 2D EDAR embeddings, got {edar_embeddings.shape}"
        )
        assert edar_embeddings.shape[0] == len(self.edar_ids), (
            f"EDAR embeddings dimension mismatch: expected {len(self.edar_ids)} EDARs, "
            f"got {edar_embeddings.shape[0]}"
        )

        attention_mask = self.get_attention_tensor()  # [n_municipalities, n_edars]

        # Subset municipalities if specified
        if municipality_indices is not None:
            attention_mask = attention_mask[
                municipality_indices
            ]  # [num_municipalities_subset, n_edars]

        # Apply attention mask using einsum for clarity
        # attention_mask: [municipalities, edars]
        # edar_embeddings: [edars, embedding_dim]
        # masked_signals: [municipalities, embedding_dim]
        masked_signals = einsum(
            attention_mask,
            edar_embeddings,
            "municipalities edars, edars embedding_dim -> municipalities embedding_dim",
        )

        # Validate output shape
        expected_shape = (attention_mask.shape[0], edar_embeddings.shape[1])
        assert masked_signals.shape == expected_shape, (
            f"Output shape mismatch: expected {expected_shape}, got {masked_signals.shape}"
        )

        return masked_signals

    def get_statistics(self) -> dict[str, float]:
        """
        Get statistics about the attention mask.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "n_edars": len(self.edar_ids),
            "n_municipalities": len(self.municipality_ids),
            "sparsity": 1.0
            - (np.count_nonzero(self.attention_mask) / self.attention_mask.size),
            "avg_edars_per_muni": np.mean(np.sum(self.attention_mask > 0, axis=1)),
            "avg_munis_per_edar": np.mean(np.sum(self.contribution_matrix > 0, axis=1)),
            "max_contribution": float(np.max(self.attention_mask)),
            "mean_nonzero_contribution": float(
                np.mean(self.attention_mask[self.attention_mask > 0])
            ),
        }

        return stats

    def extend_to_full_zone_set(
        self,
        zone_registry: ZoneRegistry,
        fill_value: float = 0.0,
        update_normalization: bool = True,
    ) -> "EDARAttentionLoader":
        """
        Extend EDAR attention mask to cover the full zone set from registry.

        This creates a natural control group by assigning zero EDAR attention weights
        to municipalities that don't have wastewater treatment coverage.

        Args:
            zone_registry: ZoneRegistry with full set of zones/municipalities
            fill_value: Value to assign to zones not covered by EDAR (default 0.0)
            update_normalization: Whether to renormalize after extension

        Returns:
            Self for method chaining
        """
        logger.info("Extending EDAR attention mask to full zone set")

        # Convert municipality IDs to strings for consistency
        edar_municipality_ids = [str(mid) for mid in self.municipality_ids]

        # Create extended mask for each municipality dimension
        extended_mask, missing_zones = zone_registry.create_extended_mask(
            source_zones=edar_municipality_ids,
            source_weights=self.attention_mask,
            fill_value=fill_value,
        )

        # Store extension results
        self._extended_mask = extended_mask
        self._extended_zone_registry = zone_registry
        self._extension_stats = {
            "original_municipalities": len(edar_municipality_ids),
            "extended_municipalities": zone_registry.num_zones,
            "missing_municipalities": len(missing_zones),
            "fill_value": fill_value,
            "coverage_ratio": len(edar_municipality_ids) / zone_registry.num_zones,
            "missing_zone_ids": missing_zones[:10]
            if len(missing_zones) > 10
            else missing_zones,  # Sample for logging
        }

        # Optional renormalization
        if update_normalization and self.normalize_contributions:
            # Renormalize each municipality's EDAR contributions to sum to 1
            row_sums = extended_mask.sum(axis=1, keepdims=True)
            # Avoid division by zero - for municipalities with no EDAR coverage, keep zeros
            nonzero_rows = row_sums > 0
            extended_mask[nonzero_rows.flatten()] /= row_sums[nonzero_rows]

        self._extended_mask = extended_mask

        # Log extension statistics
        self._log_extension_stats()

        return self

    def get_extended_attention_tensor(self) -> torch.Tensor:
        """
        Get the extended attention mask as a PyTorch tensor.

        Returns:
            Extended attention mask tensor [n_all_municipalities, n_edars]

        Raises:
            ValueError: If extension hasn't been performed yet
        """
        if self._extended_mask is None:
            raise ValueError(
                "Attention mask not extended yet. Call extend_to_full_zone_set() first."
            )

        return torch.tensor(
            self._extended_mask, dtype=torch.float32, device=self.device
        )

    def get_zone_edar_coverage(self, zone_id: str) -> list[tuple[str, float]]:
        """
        Get EDAR coverage for any zone in the extended set.

        Args:
            zone_id: Zone/municipality identifier

        Returns:
            List of (edar_id, contribution_weight) tuples
        """
        if self._extended_zone_registry is None:
            # Fall back to original implementation if not extended
            return self.get_municipality_edars(zone_id)

        zone_idx = self._extended_zone_registry.get_zone_index(zone_id)
        if zone_idx == -1:
            logger.warning(f"Zone {zone_id} not found in extended registry")
            return []

        contributions = self._extended_mask[zone_idx]

        # Get non-zero contributions
        contributing_edars = []
        for edar_idx, weight in enumerate(contributions):
            if weight > 0:
                edar_id = str(self.edar_ids[edar_idx])
                contributing_edars.append((edar_id, float(weight)))

        # Sort by contribution weight
        contributing_edars.sort(key=lambda x: x[1], reverse=True)

        return contributing_edars

    def is_extended(self) -> bool:
        """Check if the attention mask has been extended."""
        return self._extended_mask is not None

    def get_extension_statistics(self) -> dict:
        """
        Get statistics about the extension operation.

        Returns:
            Dictionary with extension statistics, or empty dict if not extended
        """
        if self._extension_stats is None:
            return {}

        return self._extension_stats.copy()

    def create_sparse_extended_attention(self) -> torch.sparse.FloatTensor:
        """
        Create a sparse tensor representation of the extended attention mask.

        Returns:
            Sparse extended attention mask tensor

        Raises:
            ValueError: If extension hasn't been performed yet
        """
        if self._extended_mask is None:
            raise ValueError(
                "Attention mask not extended yet. Call extend_to_full_zone_set() first."
            )

        # Find non-zero entries
        nonzero_indices = np.nonzero(self._extended_mask)
        indices = torch.LongTensor(np.vstack(nonzero_indices))
        values = torch.FloatTensor(self._extended_mask[nonzero_indices])
        shape = self._extended_mask.shape

        sparse_mask = torch.sparse.FloatTensor(
            indices, values, shape, device=self.device
        )

        return sparse_mask

    def apply_extended_attention_mask(
        self,
        edar_embeddings: torch.Tensor,
        municipality_indices: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Apply extended attention mask to EDAR embeddings.

        Args:
            edar_embeddings: EDAR node embeddings [n_edars, embedding_dim]
            municipality_indices: Optional subset of municipalities to compute

        Returns:
            Municipality-specific EDAR signals [n_all_municipalities, embedding_dim]

        Raises:
            ValueError: If extension hasn't been performed yet
        """
        if self._extended_mask is None:
            raise ValueError(
                "Attention mask not extended yet. Call extend_to_full_zone_set() first."
            )

        attention_mask = self.get_extended_attention_tensor()

        # Subset municipalities if specified
        if municipality_indices is not None:
            attention_mask = attention_mask[municipality_indices]

        # Matrix multiplication: [n_all_munis, n_edars] x [n_edars, embed_dim]
        masked_signals = torch.matmul(attention_mask, edar_embeddings)

        return masked_signals

    def get_control_group_indices(self) -> list[int]:
        """
        Get indices of municipalities that have no EDAR coverage (control group).

        Returns:
            List of municipality indices with zero EDAR attention weights

        Raises:
            ValueError: If extension hasn't been performed yet
        """
        if self._extended_mask is None:
            raise ValueError(
                "Attention mask not extended yet. Call extend_to_full_zone_set() first."
            )

        # Find municipalities with no EDAR coverage (all zeros in their row)
        row_sums = self._extended_mask.sum(axis=1)
        control_indices = np.where(row_sums == 0)[0].tolist()

        return control_indices

    def get_treatment_group_indices(self) -> list[int]:
        """
        Get indices of municipalities that have EDAR coverage (treatment group).

        Returns:
            List of municipality indices with non-zero EDAR attention weights

        Raises:
            ValueError: If extension hasn't been performed yet
        """
        if self._extended_mask is None:
            raise ValueError(
                "Attention mask not extended yet. Call extend_to_full_zone_set() first."
            )

        # Find municipalities with EDAR coverage (non-zero sum in their row)
        row_sums = self._extended_mask.sum(axis=1)
        treatment_indices = np.where(row_sums > 0)[0].tolist()

        return treatment_indices

    def _log_extension_stats(self):
        """Log extension operation statistics."""
        if self._extension_stats is None:
            return

        stats = self._extension_stats
        logger.info("EDAR attention mask extension completed:")
        logger.info(f"  Original municipalities: {stats['original_municipalities']}")
        logger.info(f"  Extended municipalities: {stats['extended_municipalities']}")
        logger.info(f"  Missing municipalities: {stats['missing_municipalities']}")
        logger.info(f"  Coverage ratio: {stats['coverage_ratio']:.2%}")
        logger.info(f"  Fill value for missing: {stats['fill_value']}")

        if len(stats["missing_zone_ids"]) > 0:
            logger.debug(f"  Sample missing zones: {stats['missing_zone_ids']}")

        # Log sparsity of extended mask
        nonzero_entries = np.count_nonzero(self._extended_mask)
        total_entries = self._extended_mask.size
        sparsity = 1.0 - (nonzero_entries / total_entries)
        logger.info(f"  Extended mask sparsity: {sparsity:.2%}")

    def close(self):
        """Close the underlying dataset."""
        if self.ds is not None:
            self.ds.close()


def create_edar_attention_loader(
    data_dir: str,
    normalize: bool = True,
    threshold: float = 0.01,
    attention_path: str | None = None,
) -> EDARAttentionLoader:
    """
    Factory function to create EDAR attention loader.

    Args:
        data_dir: Directory containing data files
        normalize: Whether to normalize contributions
        threshold: Minimum contribution threshold
        attention_path: Optional override for the EDAR edge weights file

    Returns:
        Configured EDARAttentionLoader
    """
    data_base = Path(data_dir)

    def _resolve_path(candidate: str | Path) -> Path:
        candidate_path = Path(candidate)
        if candidate_path.is_absolute():
            return candidate_path
        base_parts = data_base.parts
        candidate_parts = candidate_path.parts
        if candidate_parts[: len(base_parts)] == base_parts:
            return candidate_path
        return data_base / candidate_path

    if attention_path is None:
        edar_edges_path = data_base / "edar_muni_edges.nc"
    else:
        edar_edges_path = _resolve_path(attention_path)

    loader = EDARAttentionLoader(
        edar_edges_path=str(edar_edges_path),
        normalize_contributions=normalize,
        min_contribution_threshold=threshold,
    )

    # Log statistics
    stats = loader.get_statistics()
    logger.info("EDAR Attention Loader Statistics:")
    for key, value in stats.items():
        logger.info(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    return loader


if __name__ == "__main__":
    # Test the loader

    logging.basicConfig(level=logging.INFO)

    # Assuming we're in the sage_experiments directory
    data_dir = "data"

    # Create loader
    loader = create_edar_attention_loader(data_dir)

    # Test municipality lookup
    if len(loader.municipality_ids) > 0:
        test_muni = str(loader.municipality_ids[0])
        edars = loader.get_municipality_edars(test_muni)
        print(f"\nMunicipality {test_muni} receives signals from {len(edars)} EDARs:")
        for edar_id, weight in edars[:5]:  # Show top 5
            print(f"  EDAR {edar_id}: weight={weight:.4f}")

    # Test EDAR lookup
    if len(loader.edar_ids) > 0:
        test_edar = str(loader.edar_ids[0])
        munis = loader.get_edar_municipalities(test_edar)
        print(
            f"\nEDAR {test_edar} receives wastewater from {len(munis)} municipalities:"
        )
        for muni_id, weight in munis[:5]:  # Show top 5
            print(f"  Municipality {muni_id}: weight={weight:.4f}")

    # Test attention application
    dummy_edar_embeddings = torch.randn(len(loader.edar_ids), 32)
    muni_signals = loader.apply_attention_mask(dummy_edar_embeddings)
    print(f"\nMunicipality EDAR signals shape: {muni_signals.shape}")

    loader.close()
    print("\nEDAR Attention Loader test completed successfully!")
