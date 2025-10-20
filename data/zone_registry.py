"""
Centralized Zone Registry

Single source of truth for zone/municipality ID management across the mobility
and EDAR data loading pipeline. Provides consistent zone mapping, validation,
and extension operations to eliminate duplication and ensure data alignment.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ZoneRegistry:
    """
    Centralized registry for zone/municipality ID management.

    This class serves as the single source of truth for zone identification,
    mapping, and validation across all data loaders (mobility, EDAR, etc.).
    """

    def __init__(
        self,
        filter_zones: Optional[set[str]] = None,
        sort_zones: bool = True,
    ):
        """
        Initialize ZoneRegistry.

        Args:
            filter_zones: Set of zone IDs to filter out (e.g., {"out_cat"})
            sort_zones: Whether to sort zone IDs for consistent ordering
        """
        self.filter_zones = filter_zones or {"out_cat"}
        self.sort_zones = sort_zones

        # Core zone data structures
        self._zone_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: list[str] = []

        # Validation and stats
        self._is_built = False
        self._zone_sources: dict[
            str, set[str]
        ] = {}  # Track which component added each zone

    def build_from_zone_lists(
        self, zone_lists: dict[str, list[str]], source_names: Optional[list[str]] = None
    ) -> "ZoneRegistry":
        """
        Build registry from multiple zone lists (e.g., from different data sources).

        Args:
            zone_lists: Dictionary mapping source name to list of zone IDs
            source_names: Optional list of source names for backward compatibility

        Returns:
            Self for method chaining
        """
        if source_names is not None:
            # Backward compatibility: convert lists to dict
            zone_lists = dict(zip(source_names, zone_lists))

        all_zones = set()

        # Collect zones from all sources
        for source, zones in zone_lists.items():
            source_zones = {
                str(zone) for zone in zones if str(zone) not in self.filter_zones
            }
            all_zones.update(source_zones)

            # Track zone sources for validation
            for zone in source_zones:
                if zone not in self._zone_sources:
                    self._zone_sources[zone] = set()
                self._zone_sources[zone].add(source)

        # Create sorted zone list
        self._zone_ids = sorted(all_zones) if self.sort_zones else list(all_zones)

        # Build mappings
        self._id_to_idx = {zone_id: idx for idx, zone_id in enumerate(self._zone_ids)}
        self._idx_to_id = self._zone_ids.copy()

        self._is_built = True

        logger.info(
            f"ZoneRegistry built with {len(self._zone_ids)} zones from {len(zone_lists)} sources"
        )
        self._log_zone_statistics(zone_lists)

        return self

    def build_from_arrays(self, zone_arrays: dict[str, np.ndarray]) -> "ZoneRegistry":
        """
        Build registry from numpy arrays (e.g., from NetCDF coordinates).

        Args:
            zone_arrays: Dictionary mapping source name to zone array

        Returns:
            Self for method chaining
        """
        zone_lists = {
            source: [str(zone) for zone in zones]
            for source, zones in zone_arrays.items()
        }
        return self.build_from_zone_lists(zone_lists)

    def extend_with_zones(
        self, new_zones: list[str], source_name: str
    ) -> "ZoneRegistry":
        """
        Extend registry with additional zones from a new source.

        Args:
            new_zones: List of new zone IDs to add
            source_name: Name of the source providing these zones

        Returns:
            Self for method chaining
        """
        if not self._is_built:
            raise ValueError("Registry must be built before extending")

        # Filter and add new zones
        filtered_zones = [
            str(zone) for zone in new_zones if str(zone) not in self.filter_zones
        ]
        new_zone_set = set(filtered_zones) - set(self._zone_ids)

        if new_zone_set:
            # Add new zones and rebuild mappings
            self._zone_ids.extend(
                sorted(new_zone_set) if self.sort_zones else new_zone_set
            )
            if self.sort_zones:
                self._zone_ids.sort()

            # Rebuild mappings
            self._id_to_idx = {
                zone_id: idx for idx, zone_id in enumerate(self._zone_ids)
            }
            self._idx_to_id = self._zone_ids.copy()

            logger.info(
                f"Extended registry with {len(new_zone_set)} new zones from {source_name}"
            )

        # Track all zones from this source
        for zone in filtered_zones:
            if zone not in self._zone_sources:
                self._zone_sources[zone] = set()
            self._zone_sources[zone].add(source_name)

        return self

    @property
    def zone_ids(self) -> list[str]:
        """Get list of all zone IDs in order."""
        self._check_built()
        return self._idx_to_id.copy()

    @property
    def num_zones(self) -> int:
        """Get total number of zones."""
        self._check_built()
        return len(self._zone_ids)

    @property
    def id_to_idx(self) -> dict[str, int]:
        """Get zone ID to index mapping."""
        self._check_built()
        return self._id_to_idx.copy()

    @property
    def idx_to_id(self) -> list[str]:
        """Get index to zone ID mapping."""
        self._check_built()
        return self._idx_to_id.copy()

    def get_zone_index(self, zone_id: str) -> int:
        """
        Get index for a zone ID.

        Args:
            zone_id: Zone identifier

        Returns:
            Zone index, or -1 if not found
        """
        self._check_built()
        return self._id_to_idx.get(str(zone_id), -1)

    def get_zone_indices(self, zone_ids: list[str]) -> np.ndarray:
        """
        Get indices for multiple zone IDs.

        Args:
            zone_ids: List of zone identifiers

        Returns:
            Array of indices (-1 for not found)
        """
        self._check_built()
        return np.array([self.get_zone_index(zone_id) for zone_id in zone_ids])

    def get_zone_id(self, idx: int) -> Optional[str]:
        """
        Get zone ID for an index.

        Args:
            idx: Zone index

        Returns:
            Zone ID or None if index out of range
        """
        self._check_built()
        if 0 <= idx < len(self._idx_to_id):
            return self._idx_to_id[idx]
        return None

    def contains_zone(self, zone_id: str) -> bool:
        """Check if zone exists in registry."""
        self._check_built()
        return str(zone_id) in self._id_to_idx

    def get_zone_intersection(
        self, zone_sets: dict[str, set[str]]
    ) -> dict[str, set[str]]:
        """
        Analyze zone intersections between different sources.

        Args:
            zone_sets: Dictionary mapping source names to sets of zone IDs

        Returns:
            Dictionary with intersection analysis results
        """
        self._check_built()

        # Convert to registry zone IDs
        registry_zones = set(self._zone_ids)

        intersections = {}
        for source_name, zones in zone_sets.items():
            source_zones = {str(z) for z in zones if str(z) not in self.filter_zones}
            intersections[source_name] = {
                "total_zones": len(source_zones),
                "in_registry": len(source_zones.intersection(registry_zones)),
                "missing_from_registry": source_zones - registry_zones,
                "coverage_ratio": len(source_zones.intersection(registry_zones))
                / max(len(source_zones), 1),
            }

        return intersections

    def create_extended_mask(
        self,
        source_zones: list[str],
        source_weights: np.ndarray,
        fill_value: float = 0.0,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Create extended mask that covers all registry zones.

        Args:
            source_zones: List of zone IDs from source data
            source_weights: Weights for source zones (1D or 2D array)
            fill_value: Value to use for zones not in source

        Returns:
            Tuple of (extended_weights, missing_zones)
        """
        self._check_built()

        if len(source_zones) != source_weights.shape[0]:
            raise ValueError("Source zones and weights must have same length")

        # Determine output shape based on input
        if source_weights.ndim == 1:
            # 1D case: single value per zone
            output_shape = (len(self._zone_ids),)
        else:
            # 2D case: multiple features per zone
            output_shape = (len(self._zone_ids),) + source_weights.shape[1:]

        # Create extended weights array
        extended_weights = np.full(output_shape, fill_value, dtype=source_weights.dtype)
        missing_zones = []

        # Create mapping from source zones to indices in source_weights
        source_zone_to_idx = {str(zone): i for i, zone in enumerate(source_zones)}

        for idx, zone_id in enumerate(self._zone_ids):
            if zone_id in source_zone_to_idx:
                source_idx = source_zone_to_idx[zone_id]
                extended_weights[idx] = source_weights[source_idx]
            else:
                missing_zones.append(zone_id)

        logger.info(
            f"Extended mask: {len(source_zones)} source zones -> {len(self._zone_ids)} registry zones "
            f"({len(missing_zones)} filled with {fill_value})"
        )

        return extended_weights, missing_zones

    def get_statistics(self) -> dict[str, any]:
        """Get registry statistics."""
        self._check_built()

        stats = {
            "total_zones": len(self._zone_ids),
            "source_coverage": {},
            "zone_sources": {},
        }

        # Analyze source coverage
        for source in set().union(*self._zone_sources.values()):
            source_zones = [
                z for z, sources in self._zone_sources.items() if source in sources
            ]
            stats["source_coverage"][source] = len(source_zones)

        # Count zones by number of sources
        for _zone, sources in self._zone_sources.items():
            source_count = len(sources)
            if source_count not in stats["zone_sources"]:
                stats["zone_sources"][source_count] = 0
            stats["zone_sources"][source_count] += 1

        return stats

    def _check_built(self):
        """Ensure registry is built before operations."""
        if not self._is_built:
            raise ValueError("ZoneRegistry must be built before use")

    def _log_zone_statistics(self, zone_lists: dict[str, list[str]]):
        """Log zone coverage statistics."""
        stats = self.get_statistics()

        logger.info("Zone registry statistics:")
        logger.info(f"  Total zones: {stats['total_zones']}")
        logger.info(f"  Sources: {list(stats['source_coverage'].keys())}")

        for source, count in stats["source_coverage"].items():
            logger.info(f"    {source}: {count} zones")

        # Log zones appearing in multiple sources
        multi_source_zones = sum(
            count
            for source_count, count in stats["zone_sources"].items()
            if source_count > 1
        )
        if multi_source_zones > 0:
            logger.info(f"  Zones in multiple sources: {multi_source_zones}")


def create_zone_registry_from_mobility_coords(
    home_coords: np.ndarray,
    dest_coords: np.ndarray,
    filter_zones: Optional[set[str]] = None,
) -> ZoneRegistry:
    """
    Convenience function to create ZoneRegistry from mobility coordinate arrays.

    Args:
        home_coords: Home/origin zone coordinates
        dest_coords: Destination zone coordinates
        filter_zones: Set of zones to filter out

    Returns:
        Built ZoneRegistry instance
    """
    registry = ZoneRegistry(filter_zones=filter_zones)
    zone_arrays = {"home_zones": home_coords, "dest_zones": dest_coords}
    return registry.build_from_arrays(zone_arrays)
