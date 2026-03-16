import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torch_geometric.data import Data

from constants import (
    EDAR_BIOMARKER_PREFIX,
    EDAR_BIOMARKER_VARIANTS,
)
from data import dtypes as dtype_utils
from data.epi_batch import _replace_non_finite
from models.configs import EpiForecasterConfig
from utils.logging import suppress_zarr_warnings

suppress_zarr_warnings()

from .biomarker_preprocessor import BiomarkerPreprocessor  # noqa: E402
from .clinical_series_preprocessor import (  # noqa: E402
    ClinicalSeriesPreprocessor,
    ClinicalSeriesPreprocessorConfig,
)
from .mobility_preprocessor import (  # noqa: E402
    MobilityPreprocessor,
    MobilityPreprocessorConfig,
)
from .preprocess.config import REGION_COORD, TEMPORAL_COORD  # noqa: E402
from .region_embedding_store import RegionEmbeddingStore  # noqa: E402

logger = logging.getLogger(__name__)
_DEFAULT_REGION_NAME_SOURCE = Path("data/files/geo/fl_municipios_catalonia.geojson")

StaticCovariates = dict[str, torch.Tensor]


@dataclass
class SharedSparseTopology:
    """Shared full-network sparse topology for all timesteps."""

    edge_index_by_time: list[torch.Tensor]
    edge_weight_by_time: list[torch.Tensor]
    num_nodes: int
    num_timesteps: int


class EpiDatasetItem(TypedDict):
    node_label: str
    region_id: str
    target_node: int
    window_start: int
    bio_node: torch.Tensor
    # Clinical series inputs (3-channel: value, mask, age)
    hosp_hist: torch.Tensor  # [L, 3] Hospitalization history
    deaths_hist: torch.Tensor  # [L, 3] Deaths history
    cases_hist: torch.Tensor  # [L, 3] Reported cases history
    # mob: list[Data]  <-- REMOVED
    mob_x: torch.Tensor
    mob_edge_index: list[torch.Tensor]
    mob_edge_weight: list[torch.Tensor]
    mob_target_node_idx: torch.Tensor  # Local index of target node as 0-dim tensor
    population: torch.Tensor
    run_id: int | str | None
    target_region_index: int | None
    # Temporal covariates (day-of-week sin/cos + holiday indicator)
    temporal_covariates: torch.Tensor  # [L, 3] or [L, 0] if not available
    # Joint inference targets (log1p per-100k space)
    ww_hist: torch.Tensor  # [L] Wastewater history in log1p target space
    ww_hist_mask: torch.Tensor  # [L] 1.0 if wastewater history is observed
    hosp_target: torch.Tensor  # [H] Hospitalization targets in log1p(per-100k)
    ww_target: torch.Tensor  # [H] Wastewater targets in log1p(per-100k)
    cases_target: torch.Tensor  # [H] Reported cases targets in log1p(per-100k)
    deaths_target: torch.Tensor  # [H] Deaths targets in log1p(per-100k)
    hosp_target_mask: torch.Tensor  # [H] 1.0 if hospitalization target is observed
    ww_target_mask: torch.Tensor  # [H] 1.0 if wastewater target is observed
    cases_target_mask: torch.Tensor  # [H] 1.0 if cases target is observed
    deaths_target_mask: torch.Tensor  # [H] 1.0 if deaths target is observed


class EpiDataset(Dataset):
    """
    Dataset loader for preprocessed epidemiological datasets.

    This class loads preprocessed epidemiological data stored in Zarr format
    and converts it to GraphEgoDataset for ego-graph processing.
    Args:
        zarr_path: Path to the Zarr dataset
        config: EpiForecasterTrainerConfig configuration
    """

    def __init__(
        self,
        config: EpiForecasterConfig,
        target_nodes: list[int],
        context_nodes: list[int],
        biomarker_preprocessor: BiomarkerPreprocessor | None = None,
        mobility_preprocessor: MobilityPreprocessor | None = None,
        preloaded_mobility: torch.Tensor | None = None,
        mobility_mask: torch.Tensor | None = None,
        shared_sparse_topology: SharedSparseTopology | None = None,
        time_range: tuple[int, int] | None = None,
        run_id: str | None = None,
        region_embedding_store: RegionEmbeddingStore | None = None,
    ):
        self.aligned_data_path = Path(config.data.dataset_path).resolve()
        self.config = config
        self.time_range = time_range

        # Determine effective run_id (argument overrides config)
        effective_run_id = run_id if run_id is not None else config.data.run_id

        if not effective_run_id:
            raise ValueError(
                "run_id must be provided either as argument or in config.data.run_id. "
                "This is required to prevent loading all runs into memory."
            )

        # Store run_id for curriculum sampler to identify real vs synthetic datasets
        self.run_id = effective_run_id
        self._region_embedding_store = region_embedding_store
        self._local_to_global_region_index: torch.Tensor | None = None

        # Load dataset with run_id filtering for memory efficiency
        # This ensures only the required run is loaded, not all runs
        self._dataset = self.load_canonical_dataset(
            self.aligned_data_path,
            run_id=effective_run_id,
            run_id_chunk_size=config.data.run_id_chunk_size,
        )

        self.num_nodes = self._dataset[REGION_COORD].size

        # Load biomarker data start offset if available
        if "biomarker_data_start" in self._dataset:
            self.biomarker_data_start = torch.from_numpy(
                self._dataset["biomarker_data_start"].values
            ).to(torch.long)
        else:
            # Fallback: all regions have data starting at index 0
            self.biomarker_data_start = torch.zeros(self.num_nodes, dtype=torch.long)

        self.biomarker_variants = self._get_biomarker_variants()

        # Setup clinical series preprocessors for 3-channel [value, mask, age] format
        # Data is already log1p(per-100k) transformed from preprocessing pipeline
        clinical_config = ClinicalSeriesPreprocessorConfig(
            age_max=14,
        )

        # Precompute hospitalizations (3-channel)
        self.hosp_preprocessor = ClinicalSeriesPreprocessor(
            config=clinical_config,
            var_name="hospitalizations",
        )
        self.precomputed_hosp_hist = self.hosp_preprocessor.preprocess_dataset(
            self._dataset,
            population=self._dataset.get("population"),
        )
        self.precomputed_hosp_hist[..., 0] = _replace_non_finite(
            self.precomputed_hosp_hist[..., 0]
        )

        # Precompute deaths (3-channel)
        self.deaths_preprocessor = ClinicalSeriesPreprocessor(
            config=clinical_config,
            var_name="deaths",
        )
        self.precomputed_deaths_hist = self.deaths_preprocessor.preprocess_dataset(
            self._dataset,
            population=self._dataset.get("population"),
        )
        self.precomputed_deaths_hist[..., 0] = _replace_non_finite(
            self.precomputed_deaths_hist[..., 0]
        )

        # Precompute reported cases (3-channel)
        self.cases_preprocessor = ClinicalSeriesPreprocessor(
            config=clinical_config,
            var_name="cases",
        )
        self.precomputed_cases_hist = self.cases_preprocessor.preprocess_dataset(
            self._dataset,
            population=self._dataset.get("population"),
        )
        self.precomputed_cases_hist[..., 0] = _replace_non_finite(
            self.precomputed_cases_hist[..., 0]
        )

        # Precompute joint inference targets + masks (1D for loss computation).
        # Clinical targets use log1p(per-100k), while wastewater targets stay in
        # measurement space (log1p on raw concentration/proxy values).
        self.precomputed_hosp, self.precomputed_hosp_mask = (
            self._precompute_joint_target("hospitalizations", per_100k=True)
        )
        self.precomputed_ww, self.precomputed_ww_mask = (
            self._precompute_wastewater_target()
        )
        # Reported cases and deaths (optional, default to all-missing if not in dataset)
        self.precomputed_cases_target, self.precomputed_cases_mask = (
            self._precompute_joint_target("cases", per_100k=True)
        )
        self.precomputed_deaths, self.precomputed_deaths_mask = (
            self._precompute_joint_target("deaths", per_100k=True)
        )

        # Setup mobility preprocessor
        # Data is already log1p-transformed from preprocessing pipeline
        if mobility_preprocessor is None:
            mp_config = MobilityPreprocessorConfig(
                clip_range=config.data.mobility_clip_range,
                scale_epsilon=config.data.mobility_scale_epsilon,
            )
            self.mobility_preprocessor = MobilityPreprocessor(mp_config)

            # Fit scaler on train nodes only
            all_region_ids = self._dataset[REGION_COORD].values
            train_region_ids = [all_region_ids[n] for n in target_nodes]
            self.mobility_preprocessor.fit_scaler(self._dataset, train_region_ids)
        else:
            self.mobility_preprocessor = mobility_preprocessor

        if (preloaded_mobility is None) != (mobility_mask is None):
            raise ValueError(
                "preloaded_mobility and mobility_mask must both be provided "
                "together, or both be None."
            )

        if preloaded_mobility is not None and mobility_mask is not None:
            # Reuse shared tensors from another dataset (e.g., train -> val/test)
            self.preloaded_mobility = preloaded_mobility
            self.mobility_mask = mobility_mask
            logger.info(
                f"Reusing shared mobility tensor: {self.preloaded_mobility.shape}, "
                f"{self.preloaded_mobility.element_size() * self.preloaded_mobility.numel() / 1e6:.2f} MB"
            )
        else:
            # Mobility: load full tensor into memory
            mobility_da = self._dataset.mobility
            # Ensure proper dimension ordering: Time, Origin, Destination
            mobility_da = self.mobility_preprocessor._ensure_time_first(mobility_da)

            # Load all values into memory
            mobility_np = mobility_da.values

            # Apply preprocessing/scaling once
            mobility_np = self.mobility_preprocessor.transform_values(mobility_np)

            # Convert to torch tensor (float16 for memory efficiency)
            self.preloaded_mobility = torch.from_numpy(mobility_np).to(
                dtype_utils.STORAGE_DTYPES["continuous"]
            )
            self.preloaded_mobility = _replace_non_finite(self.preloaded_mobility)

            # Optimization: Precompute mobility mask to avoid repeated comparisons in __getitem__
            mobility_threshold = float(config.data.mobility_threshold)
            self.mobility_mask = self.preloaded_mobility >= mobility_threshold

            logger.info(
                f"Mobility preloaded: {self.preloaded_mobility.shape}, "
                f"{self.preloaded_mobility.element_size() * self.preloaded_mobility.numel() / 1e6:.2f} MB"
            )

        # Lagged Mobility Features
        self.mobility_lags = config.data.mobility_lags
        self.use_imported_risk = config.data.use_imported_risk
        self.lagged_risk = None

        if self.mobility_lags and self.use_imported_risk:
            logger.info(f"Pre-computing imported risk for lags: {self.mobility_lags}")
            # Use cases history value channel (index 0) for imported risk
            cases_val = self.precomputed_cases_hist[..., 0]
            if isinstance(cases_val, torch.Tensor):
                cases_val = cases_val.numpy()

            # Compute risk using dense mobility (supports time-varying)
            risk_np = self.mobility_preprocessor.compute_imported_risk(
                cases_val, self.preloaded_mobility.numpy(), self.mobility_lags
            )
            self.lagged_risk = torch.from_numpy(risk_np).to(
                dtype_utils.STORAGE_DTYPES["continuous"]
            )
            self.lagged_risk = _replace_non_finite(self.lagged_risk)

        # Cache for full graphs keyed by time step (CPU tensors only)
        self._full_graph_cache: dict[int, Data] = {}
        self.shared_sparse_topology: SharedSparseTopology | None = (
            shared_sparse_topology
        )

        # Cache for adjacency matrices keyed by time step (CPU tensors only)
        self._adjacency_cache: dict[int, torch.Tensor] = {}

        # Cache for global-to-local node index mapping keyed by time step
        # Each entry is a (N,) tensor with local indices or -1 for non-context nodes
        self._global_to_local_cache: dict[int, torch.Tensor] = {}

        # Precomputed k-hop reachability masks for all timesteps
        # Dict mapping time_step -> (N, N) boolean tensor
        # This replaces on-demand computation to eliminate CPU bottleneck in __getitem__
        self._precomputed_k_hop_masks: dict[int, torch.Tensor] = {}

        # Setup biomarker preprocessor
        if biomarker_preprocessor is None:
            self.biomarker_preprocessor = BiomarkerPreprocessor()

            # Log biomarker availability in train split
            # Regions with biomarker data have biomarker_data_start >= 0
            train_nodes_with_bio = [
                n for n in target_nodes if self.biomarker_data_start[n] >= 0
            ]
            logger.info(
                f"Train split: {len(train_nodes_with_bio)}/{len(target_nodes)} "
                f"nodes have biomarker data"
            )

            if len(train_nodes_with_bio) == 0:
                if self.config.model.type.biomarkers:
                    raise ValueError(
                        "Biomarkers enabled in config but no train nodes have biomarker data. "
                        "Disable biomarkers or use a different train split."
                    )
                logger.warning(
                    "No train nodes have biomarker data, using zero encoding"
                )

            if self.biomarker_variants:
                # Convert indices to region IDs for fit_scaler
                all_region_ids = self._dataset[REGION_COORD].values
                train_region_ids = [all_region_ids[n] for n in train_nodes_with_bio]

                # Only fit scaler on train nodes that have biomarkers
                self.biomarker_preprocessor.fit_scaler(self._dataset, train_region_ids)
        else:
            self.biomarker_preprocessor = biomarker_preprocessor

        # Precompute biomarkers for the entire dataset
        # This returns a (TotalTime, NumNodes, 3 * variants) tensor
        if self.biomarker_variants:
            self.precomputed_biomarkers = torch.from_numpy(
                self.biomarker_preprocessor.preprocess_dataset(self._dataset)
            ).to(dtype_utils.STORAGE_DTYPES["continuous"])
            self.precomputed_biomarkers[..., 0::4] = _replace_non_finite(
                self.precomputed_biomarkers[..., 0::4]
            )
        else:
            T_total = len(self._dataset[TEMPORAL_COORD])
            # 4 channels per variant: value, mask, censor, age
            channel_count = max(1, len(self.biomarker_variants)) * 4
            # channels: value=0, mask=0, censor=0, age=1
            dummy = torch.zeros(
                (T_total, self.num_nodes, channel_count),
                dtype=dtype_utils.STORAGE_DTYPES["continuous"],
            )
            # For 4-channel layout [value, mask, censor, age], age is at indices 3, 7, 11, ...
            for idx in range(3, channel_count, 4):
                dummy[:, :, idx] = 1.0
            self.precomputed_biomarkers = dummy

        # Load temporal covariates if available in dataset
        if "temporal_covariates" in self._dataset:
            self.temporal_covariates = torch.from_numpy(
                self._dataset["temporal_covariates"].values
            ).to(dtype_utils.STORAGE_DTYPES["continuous"])
            self.temporal_covariates = _replace_non_finite(self.temporal_covariates)
            self.temporal_covariates_dim = self.temporal_covariates.shape[1]
            logger.info(
                f"Loaded temporal covariates: shape={self.temporal_covariates.shape}"
            )
        else:
            T_total = len(self._dataset[TEMPORAL_COORD])
            self.temporal_covariates = torch.zeros(
                (T_total, 0), dtype=dtype_utils.STORAGE_DTYPES["continuous"]
            )
            self.temporal_covariates_dim = 0
            logger.warn("No temporal covariates found in dataset")

        self.region_embeddings = (
            region_embedding_store.embeddings
            if region_embedding_store is not None
            else None
        )

        self.target_nodes = target_nodes
        self.context_mask = torch.zeros(
            self.num_nodes, dtype=dtype_utils.STORAGE_DTYPES["mask"]
        )
        self.context_mask[target_nodes] = True
        self.context_mask[context_nodes] = True
        self._target_khop_mask: torch.Tensor | None = None

        # Set dimensions
        self.time_dim_size = (
            config.model.input_window_length + config.model.forecast_horizon
        )
        self.window_stride = int(config.data.window_stride)
        self.missing_permit_map = config.data.resolve_missing_permit_map()
        self.window_starts = self._compute_window_starts()
        self._valid_window_starts_by_node = self._compute_valid_window_starts()
        self.window_starts = self._collect_valid_window_starts()
        self._index_map, self._index_lookup = self._build_index_map()

        self.node_static_covariates = self.static_covariates()
        self.scale_epsilon = 1e-6

        # Validate config dims match dataset expectations
        # Clinical series: 3 channels (value, mask, age) per series, 3 series total
        expected_clinical_dim = 9  # hosp(3) + deaths(3) + cases(3)
        if self.use_imported_risk:
            lag_dim = len(self.mobility_lags) if hasattr(self, "mobility_lags") else 0
            expected_clinical_dim += lag_dim
        expected_bio_dim = self.biomarkers_output_dim

        # Note: cases_dim is deprecated, clinical series use fixed 3-channel format
        if self.config.model.cases_dim != expected_clinical_dim:
            logger.info(
                "Updating cases_dim from %d to %d (3 clinical series x 3 channels)",
                self.config.model.cases_dim,
                expected_clinical_dim,
            )
            self.config.model.cases_dim = expected_clinical_dim
        if self.config.model.biomarkers_dim != expected_bio_dim:
            logger.info(
                "Updating biomarkers_dim from %d to %d based on dataset",
                self.config.model.biomarkers_dim,
                expected_bio_dim,
            )
            self.config.model.biomarkers_dim = expected_bio_dim

        # Extract metadata for workers to avoid zarr access in forked processes
        # This prevents hangs when workers try to reopen zarr files
        self._region_ids = [
            str(region_id) for region_id in self._dataset[REGION_COORD].values
        ]
        self._region_name_source = self._resolve_region_name_source()
        self._region_name_by_id = self._load_region_name_map(self._region_name_source)
        self._region_labels = [
            self._region_name_by_id.get(region_id, region_id)
            for region_id in self._region_ids
        ]
        self._temporal_coords = list(self._dataset[TEMPORAL_COORD].values)
        if self._region_embedding_store is not None:
            self._local_to_global_region_index = (
                self._region_embedding_store.build_local_to_global_index(
                    self._region_ids
                )
            )

        # Precompute k-hop masks for all timesteps to avoid CPU bottleneck in __getitem__
        # This moves expensive matrix operations from per-sample to startup time
        # Must be done after _temporal_coords is set but before dataset is closed
        self._precomputed_k_hop_masks = self._precompute_k_hop_masks()

        # Extract sparsity level once during initialization (before dataset is closed)
        # Dataset is already filtered to a single run, so synthetic_sparsity_level is a scalar
        # Use .values to handle dask arrays (chunked loading)
        if "synthetic_sparsity_level" in self._dataset:
            self.sparsity_level = float(
                self._dataset["synthetic_sparsity_level"].values.item()
            )
        else:
            self.sparsity_level = None

        # Get run_id once and store as scalar
        try:
            run_id_val = self._dataset.mobility.run_id.item()
            try:
                self._run_id_value = int(run_id_val) if run_id_val is not None else None
            except (ValueError, TypeError):
                self._run_id_value = (
                    str(run_id_val).strip() if run_id_val is not None else None
                )
        except Exception:
            self._run_id_value = self.run_id  # Fallback to config value

        self._target_khop_mask = self._compute_target_khop_mask()
        logger.info(
            f"Target k-hop mask: {self._target_khop_mask.sum().item()}/{self.num_nodes} nodes "
            f"(depth={self.config.model.gnn_depth}, targets={len(self.target_nodes)})"
        )

        # Close dataset and clear reference to avoid pickling issues
        self._dataset.close()
        self._dataset = None

    def _get_biomarker_variants(self) -> list[str]:
        """Get biomarker variant names present in the dataset.

        Looks for edar_biomarker_{variant} variables where variant is one of
        EDAR_BIOMARKER_VARIANTS (N1, N2, IP4), excluding channel suffixes.
        """
        expected_names = [
            f"{EDAR_BIOMARKER_PREFIX}{v}" for v in EDAR_BIOMARKER_VARIANTS
        ]
        variant_names = [
            str(name) for name in self.dataset.data_vars if str(name) in expected_names
        ]
        return sorted(
            variant_names,
            key=lambda x: EDAR_BIOMARKER_VARIANTS.index(
                x.replace(EDAR_BIOMARKER_PREFIX, "")
            ),
        )

    @property
    def dataset(self) -> xr.Dataset:
        if self._dataset is not None:
            return self._dataset

        # Reload dataset after unpickling
        ds = xr.open_zarr(self.aligned_data_path)
        # Re-apply run_id filter (critical for DataLoader workers)
        effective_run_id = (
            self.run_id if self.run_id is not None else self.config.data.run_id
        )
        ds = self._filter_dataset_by_runs(ds, effective_run_id)
        self._dataset = ds
        return ds

    def __getstate__(self):
        """Allow pickling by clearing the dataset handle."""
        state = self.__dict__.copy()
        state["_dataset"] = None
        # Full shared topology is only needed during init-time pruning.
        state["shared_sparse_topology"] = None
        return state

    def close(self) -> None:
        """Explicitly close the zarr dataset to release resources.

        This is important for preventing semaphore leaks when used with
        PyTorch DataLoader workers. Each worker opens a zarr dataset,
        and without proper cleanup, file handles and semaphores accumulate.
        """
        if self._dataset is not None:
            try:
                self._dataset.close()
            except Exception:
                pass
            self._dataset = None

    def release_shared_sparse_topology(self) -> None:
        """Drop shared full sparse topology after split-specific pruning."""
        if self.shared_sparse_topology is not None:
            self.shared_sparse_topology = None
            logger.info("Released shared full sparse topology")

    def __del__(self) -> None:
        """Cleanup zarr dataset on garbage collection.

        Provides automatic resource cleanup when the dataset object is
        destroyed, even if close() was not explicitly called.
        """
        self.close()

    def num_windows(self) -> int:
        """Number of window start positions valid for at least one target node."""
        return len(self.window_starts)

    @property
    def cases_output_dim(self) -> int:
        """DEPRECATED: Clinical series now use 3-channel format per series.

        Returns total dimension for all clinical series inputs:
        - hospitalizations: 3 channels (value, mask, age)
        - deaths: 3 channels (value, mask, age)
        - cases: 3 channels (value, mask, age)
        Plus optional imported risk lag features (value-only).
        """
        base_dim = 9  # 3 series x 3 channels
        if self.use_imported_risk:
            lag_dim = len(self.mobility_lags) if hasattr(self, "mobility_lags") else 0
            return base_dim + lag_dim
        return base_dim

    @property
    def biomarkers_output_dim(self) -> int:
        """Biomarkers dim (value/mask/censor/age per variant from preprocessor + has_data).

        Note: The preprocessor outputs 4 channels per variant. The has_data channel
        is added dynamically in __getitem__ based on biomarker_data_start.
        """
        variant_count = max(1, len(self.biomarker_variants))
        return variant_count * 4 + 1

    @property
    def biomarker_available_mask(self) -> torch.Tensor:
        """Compatibility property for plotting functions.

        Returns a (N, B) tensor indicating region-level availability based on
        biomarker_data_start. Regions with data start >= 0 have data (1.0),
        regions with sentinel -1 don't have data (0.0).
        """
        # Create (N, B) tensor with 1.0 for regions with data, 0.0 otherwise
        has_data = (self.biomarker_data_start >= 0).to(torch.float16)
        return has_data.unsqueeze(-1).expand(-1, self.biomarkers_output_dim)

    def index_for_target_node_window(self, *, target_node: int, window_idx: int) -> int:
        """Map a (window_idx, target_node) pair to a dataset index.

        window_idx indexes into the stride-based window starts (see ``num_windows``)
        and is filtered by missingness per target node.
        """
        if window_idx < 0 or window_idx >= len(self.window_starts):
            raise IndexError("Requested window exceeds available time windows")

        window_start = self.window_starts[window_idx]
        idx = self._index_lookup.get((target_node, window_start))
        if idx is None:
            raise KeyError(
                "Requested window is not valid for the specified target node"
            )
        return idx

    def __len__(self) -> int:
        """Number of samples in the dataset.

        One sample corresponds to a (window, node) pair. Windows are generated
        with the configured stride and filtered by missingness permit.
        """
        return len(self._index_map)

    def __getitem__(self, idx: int) -> EpiDatasetItem:
        """Return a single target node over one time window.

        Each item is keyed by (target_node, window_start). The mobility slice is
        converted to a PyG ego-graph per time step containing the target node
        and its incoming neighbors.
        """

        L = self.config.model.input_window_length
        H = self.config.model.forecast_horizon

        try:
            target_idx, range_start = self._index_map[idx]
        except IndexError as exc:
            raise IndexError("Sample index out of range") from exc

        # Use pre-extracted metadata to avoid zarr access in workers
        node_label = self._region_labels[target_idx]
        region_id = self._region_ids[target_idx]
        target_region_index = target_idx
        if self._local_to_global_region_index is not None:
            target_region_index = int(
                self._local_to_global_region_index[target_idx].item()
            )

        range_end = range_start + L
        forecast_targets = range_end + H
        T = len(self._temporal_coords)
        if forecast_targets > T:
            raise IndexError("Requested window exceeds available time steps")

        # Get mobility history from preloaded tensor
        mobility_history = self.preloaded_mobility[range_start:range_end, :, target_idx]
        neigh_mask = self.mobility_mask[range_start:range_end, :, target_idx]

        # Use pre-extracted run_id to avoid zarr access in workers
        run_id = self._run_id_value

        # Clinical Series Processing - extract 3-channel history windows
        # Each is (L, N, 3) with [value, mask, age] channels
        hosp_history = self.precomputed_hosp_hist[range_start:range_end]  # (L, N, 3)
        deaths_history = self.precomputed_deaths_hist[
            range_start:range_end
        ]  # (L, N, 3)
        cases_history = self.precomputed_cases_hist[range_start:range_end]  # (L, N, 3)

        if self.context_mask is not None:
            # self.context_mask is a tensor
            neigh_mask = neigh_mask & self.context_mask[None, :]

        # Force target node to be included (always, regardless of context_mask)
        neigh_mask[:, target_idx] = True

        # Apply neighborhood mask to clinical histories
        neigh_mask_t = neigh_mask.unsqueeze(-1).to(torch.float16)
        hosp_history = hosp_history * neigh_mask_t
        deaths_history = deaths_history * neigh_mask_t
        cases_history = cases_history * neigh_mask_t

        # Concatenate lagged risk features if available
        # Lag features are value-only (no mask/age channels) for efficiency
        if self.use_imported_risk and self.lagged_risk is not None:
            # Slice lagged risk for the current window [range_start:range_end]
            # (L, N, Lags) - value channels only, no mask/age
            risk_slice = self.lagged_risk[range_start:range_end]

            # Apply neighborhood mask (broadcasts to all lag channels)
            risk_slice = risk_slice * neigh_mask_t

            # Concat to cases history: (L, N, 3) + (L, N, Lags) -> (L, N, 3+Lags)
            # Cases keep their 3 channels (value, mask, age); lags are value-only
            cases_history = torch.cat([cases_history, risk_slice], dim=-1)

        # Encode all regions in context using biomarker encoding
        # Optimized: Use precomputed biomarkers + dynamic has_data based on data start offset

        # 1. Get precomputed bio history (value, mask, age per variant)
        # Note: self.precomputed_biomarkers is a CPU tensor
        bio_slice = self.precomputed_biomarkers[range_start:range_end]

        # 2. Get availability mask based on data start offset -> (N,)
        # has_data = 1.0 if current time >= region's data start, else 0.0
        # Use -1 sentinel for regions with no data
        range_end_idx = range_end - 1  # Last index of history window
        has_data = (range_end_idx >= self.biomarker_data_start).to(torch.float16)
        # Regions with sentinel -1 never have data
        has_data[self.biomarker_data_start < 0] = 0.0

        # 3. Broadcast to (L, N, 1)
        has_data_3d = has_data.view(1, self.num_nodes, 1).expand(L, -1, -1)

        # 4. Concatenate -> (L, N, 4)
        biomarker_history = torch.cat([bio_slice, has_data_3d], dim=-1)

        # Extract joint inference targets (hospitalizations and wastewater in log1p per-100k)
        # These are already precomputed in __init__
        ww_hist = self.precomputed_ww[range_start:range_end, target_idx]
        ww_hist_mask = self.precomputed_ww_mask[range_start:range_end, target_idx]
        hosp_target = self.precomputed_hosp[range_end:forecast_targets, target_idx]
        hosp_target_mask = self.precomputed_hosp_mask[
            range_end:forecast_targets, target_idx
        ]
        ww_target = self.precomputed_ww[range_end:forecast_targets, target_idx]
        ww_target_mask = self.precomputed_ww_mask[
            range_end:forecast_targets, target_idx
        ]
        cases_target = self.precomputed_cases_target[
            range_end:forecast_targets, target_idx
        ]
        cases_target_mask = self.precomputed_cases_mask[
            range_end:forecast_targets, target_idx
        ]
        deaths_target = self.precomputed_deaths[range_end:forecast_targets, target_idx]
        deaths_target_mask = self.precomputed_deaths_mask[
            range_end:forecast_targets, target_idx
        ]

        assert mobility_history.shape == (L, self.num_nodes), (
            f"Mob history shape mismatch: expected ({L}, {self.num_nodes}), got {mobility_history.shape}"
        )
        # Clinical series: always 3 channels (value, mask, age)
        # Only cases_hist gets lag features concatenated
        assert hosp_history.shape == (L, self.num_nodes, 3), (
            f"Hosp history shape mismatch: expected ({L}, {self.num_nodes}, 3), "
            f"got {hosp_history.shape}"
        )
        assert deaths_history.shape == (L, self.num_nodes, 3), (
            f"Deaths history shape mismatch: expected ({L}, {self.num_nodes}, 3), "
            f"got {deaths_history.shape}"
        )
        expected_cases_dim = 3 + (
            len(self.mobility_lags) if self.use_imported_risk else 0
        )
        assert cases_history.shape == (L, self.num_nodes, expected_cases_dim), (
            f"Cases history shape mismatch: expected ({L}, {self.num_nodes}, {expected_cases_dim}), "
            f"got {cases_history.shape}"
        )
        expected_bio_dim = self.biomarkers_output_dim
        assert biomarker_history.shape == (L, self.num_nodes, expected_bio_dim), (
            "Biomarker history shape mismatch - expected (T, N, B)"
        )

        mob_x_list: list[torch.Tensor] = []

        # Find local index of target node (constant across window)
        # Use tensor to defer GPU->CPU sync until actually needed
        global_to_local = self._get_global_to_local_at_time(range_start)
        local_target_idx_tensor = global_to_local[target_idx]
        # Extract Python int once for indexing operations inside the loop
        local_target_idx = int(local_target_idx_tensor.item())

        # Determine node IDs in the context mask
        node_mask = self._get_graph_node_mask()
        node_ids = torch.where(node_mask)[0]
        mob_real_node_idx = node_ids
        if self._local_to_global_region_index is not None:
            mob_real_node_idx = self._local_to_global_region_index[node_ids]

        for t in range(L):
            global_t = range_start + t

            # Build node features for this time step
            hosp_t = hosp_history[t]  # (N, 3)
            deaths_t = deaths_history[t]  # (N, 3)
            cases_t = cases_history[t]  # (N, 3)
            bio_t = biomarker_history[t]  # (N, B)

            # Build feature tensor for all nodes in context
            feat = []
            if self.config.model.type.cases:
                # Concatenate all clinical series: hosp + deaths + cases
                feat.append(hosp_t[node_ids])
                feat.append(deaths_t[node_ids])
                feat.append(cases_t[node_ids])
            if self.config.model.type.biomarkers:
                feat.append(bio_t[node_ids])
            x = torch.cat(feat, dim=-1)  # (num_nodes, feat_dim)

            # Apply k-hop feature masking based on target node
            # Use precomputed masks to avoid expensive CPU matrix operations
            if (
                self.config.model.gnn_depth > 0
                and global_t in self._precomputed_k_hop_masks
            ):
                # Get precomputed k-hop mask for this timestep (N, N) boolean
                k_hop_mask_full = self._precomputed_k_hop_masks[global_t]
                # Extract mask for target node: (N,) boolean
                k_hop_mask = k_hop_mask_full[target_idx]

                # Map global mask to local indices and ensure target is included
                # NOTE: k_hop_mask excludes the target node (diagonal is False),
                # so we need to ensure the target node is NOT masked
                local_k_hop_mask = k_hop_mask[node_ids].clone()
                local_k_hop_mask[local_target_idx] = True
                # Zero out features for nodes outside k-hop (including target would be wrong)
                # Using torch.where instead of boolean indexing for better GPU performance
                x_masked = torch.where(
                    local_k_hop_mask.unsqueeze(-1).expand_as(x), x, torch.zeros_like(x)
                )
            else:
                # No masking (all nodes contribute) or no precomputed mask
                x_masked = x

            # --- OPTIMIZED BATCHING CHANGE ---
            mob_x_list.append(x_masked)

        population = self.node_static_covariates["Pop"][target_idx]

        # Extract temporal covariates for the history window
        temporal_covariates_hist = self.temporal_covariates[range_start:range_end]

        return {
            "node_label": node_label,
            "region_id": region_id,
            "target_node": target_idx,
            "target_region_index": target_region_index,
            "window_start": range_start,
            "hosp_hist": hosp_history[:, target_idx, :],  # (L, 3)
            "deaths_hist": deaths_history[:, target_idx, :],  # (L, 3)
            "cases_hist": cases_history[:, target_idx, :],  # (L, 3)
            "bio_node": biomarker_history[:, target_idx, :],
            "mob_x": torch.stack(mob_x_list),  # (L, N_ctx, F)
            "mob_t": torch.arange(range_start, range_start + L, dtype=torch.long),
            "mob_target_node_idx": local_target_idx_tensor,
            "mob_real_node_idx": mob_real_node_idx,
            "population": population,
            "run_id": run_id,
            "temporal_covariates": temporal_covariates_hist,  # (L, cov_dim)
            "ww_hist": ww_hist,
            "ww_hist_mask": ww_hist_mask,
            "hosp_target": hosp_target,
            "ww_target": ww_target,
            "cases_target": cases_target,
            "deaths_target": deaths_target,
            "hosp_target_mask": hosp_target_mask,
            "ww_target_mask": ww_target_mask,
            "cases_target_mask": cases_target_mask,
            "deaths_target_mask": deaths_target_mask,
        }

    def _resolve_region_name_source(self) -> Path | None:
        configured = self.config.data.regions_data_path.strip()
        if configured:
            configured_path = Path(configured)
            if not configured_path.is_absolute():
                configured_path = (Path.cwd() / configured_path).resolve()
            if (
                configured_path.exists()
                and configured_path.suffix.lower() == ".geojson"
            ):
                return configured_path
            logger.warning(
                "Configured regions_data_path is unavailable or unsupported for "
                "region-name lookup: %s",
                configured_path,
            )

        default_path = (Path.cwd() / _DEFAULT_REGION_NAME_SOURCE).resolve()
        if default_path.exists():
            return default_path
        return None

    def _load_region_name_map(self, source_path: Path | None) -> dict[str, str]:
        if source_path is None or source_path.suffix.lower() != ".geojson":
            return {}

        try:
            with source_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            logger.warning(
                "Failed to load region name source: %s", source_path, exc_info=True
            )
            return {}

        mapping: dict[str, str] = {}
        for feature in payload.get("features", []):
            properties = feature.get("properties", {})
            region_id = properties.get("id")
            region_name = properties.get("name")
            if region_id is None or region_name is None:
                continue
            mapping[str(region_id)] = str(region_name)
        return mapping

    def _load_target_values_and_mask(
        self, var_name: str
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load a target variable and its observation mask from the canonical dataset."""
        ds = self._dataset
        if ds is None:
            raise RuntimeError(
                "Dataset not loaded. Call load_canonical_dataset() first."
            )

        if var_name not in ds.data_vars:
            return None, None

        da = ds[var_name]
        if da.ndim == 3:
            da = da.squeeze(drop=True)
        da = da.transpose(TEMPORAL_COORD, REGION_COORD)
        values = da.values.astype(np.float32)

        mask_name = f"{var_name}_mask"
        if mask_name in ds.data_vars:
            mask_da = ds[mask_name]
            if mask_da.ndim == 3:
                mask_da = mask_da.squeeze(drop=True)
            mask_da = mask_da.transpose(TEMPORAL_COORD, REGION_COORD)
            mask = (mask_da.values > 0).astype(np.float32)
        else:
            mask = np.isfinite(values).astype(np.float32)

        mask = (mask > 0) & np.isfinite(values) & (values >= 0.0)
        return values, mask.astype(np.float32)

    def _precompute_wastewater_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build wastewater (biomarker) supervision target from EDAR variant channels.

        Values are already log1p-transformed from preprocessing pipeline.
        Returns masked mean across available biomarker variants.
        """
        ds = self._dataset
        if ds is None:
            raise RuntimeError(
                "Dataset not loaded. Call load_canonical_dataset() first."
            )

        # WW target is defined as masked mean across biomarker variants.
        ww_components = [
            "edar_biomarker_N1",
            "edar_biomarker_N2",
            "edar_biomarker_IP4",
        ]
        component_tensors: list[np.ndarray] = []
        component_masks: list[np.ndarray] = []

        for var_name in ww_components:
            values, mask = self._load_target_values_and_mask(var_name)
            if values is None or mask is None:
                continue
            component_tensors.append(values)
            component_masks.append(mask)

        if not component_tensors:
            logger.info(
                "No wastewater variable or EDAR components found; using all-missing WW target."
            )
            T = len(ds[TEMPORAL_COORD])
            zeros = torch.zeros((T, self.num_nodes), dtype=torch.float16)
            return zeros, zeros

        stacked_values = np.stack(component_tensors, axis=0)  # (C, T, N)
        stacked_masks = np.stack(component_masks, axis=0).astype(
            np.float16
        )  # (C, T, N)

        valid = (stacked_masks > 0.0) & np.isfinite(stacked_values)
        valid_count = valid.sum(axis=0).astype(np.float16)
        weighted_sum = np.where(valid, stacked_values, 0.0).sum(axis=0)
        combined_values = np.divide(
            weighted_sum,
            np.where(valid_count > 0.0, valid_count, 1.0),
        ).astype(np.float16)
        combined_mask = np.any(valid, axis=0)

        # Values are already log1p-transformed from preprocessing pipeline
        # Preserve missingness in WW targets when no observed variant is available.
        combined_values = np.where(combined_mask, combined_values, np.nan).astype(
            np.float16
        )

        logger.info(
            "Using WW target from biomarker components (masked mean): %s",
            ",".join(ww_components),
        )
        return (
            torch.from_numpy(combined_values),
            torch.from_numpy(combined_mask.astype(np.float16)),
        )

    def _precompute_joint_target(
        self,
        var_name: str,
        *,
        per_100k: bool,  # Ignored - kept for API compatibility
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute joint inference target (already log1p(per-100k) transformed).

        Args:
            var_name: Name of the variable in the dataset (e.g., "hospitalizations", "cases")
            per_100k: Ignored - values are already log1p(per-100k) transformed from pipeline

        Returns:
            Tuple of:
                - values: (T, N) tensor with log1p(per-100k) values (NaNs preserved)
                - mask: (T, N) tensor where 1.0 indicates observed/valid target
        """
        ds = self._dataset
        if ds is None:
            raise RuntimeError(
                "Dataset not loaded. Call load_canonical_dataset() first."
            )

        values, mask = self._load_target_values_and_mask(var_name)
        if values is None or mask is None:
            logger.info(
                f"Joint target '{var_name}' not found in dataset, using all-missing mask"
            )
            T = len(ds[TEMPORAL_COORD])
            return (
                torch.zeros((T, self.num_nodes), dtype=torch.float16),
                torch.zeros((T, self.num_nodes), dtype=torch.float16),
            )

        # Values are already log1p(per-100k) transformed from preprocessing pipeline
        # No additional transforms needed
        mask = (mask > 0) & np.isfinite(values)
        values = np.where(np.isfinite(values), values, np.nan)
        mask_t = torch.from_numpy(mask.astype(np.float16)).to(torch.float16)

        return torch.from_numpy(values.astype(np.float16)).to(torch.float16), mask_t

    def static_covariates(self) -> StaticCovariates:
        "Returns static covariates for the dataset. (num_nodes, num_features)"
        population_cov = self.dataset.population
        population_tensor = torch.from_numpy(population_cov.to_numpy()).to(
            torch.float16
        )
        population_tensor = _replace_non_finite(population_tensor)
        assert population_tensor.shape == (self.num_nodes,), (
            f"Static covariates shape mismatch: expected ({self.num_nodes},), got {population_tensor.shape}"
        )

        return {
            "Pop": population_tensor,
        }

    def _compute_window_starts(self) -> list[int]:
        """Compute window start indices given history, horizon, and stride.

        If time_range is set, only windows fully contained within that range
        (i.e., start + L + H <= end) are included.

        For multi-run datasets, also filters out windows that cross run boundaries
        to prevent context leakage between simulation runs.
        """
        L = self.config.model.input_window_length
        H = self.config.model.forecast_horizon
        T = len(self.dataset[TEMPORAL_COORD].values)
        seg = L + H
        if T < seg:
            return []

        max_lag = (
            max(self.mobility_lags, default=0) if hasattr(self, "mobility_lags") else 0
        )
        # Start at max_lag to avoid leakage/padding at start
        all_starts = list(range(max_lag, T - seg + 1, self.window_stride))

        # Filter by time_range if specified
        if self.time_range is not None:
            start_idx, end_idx = self.time_range
            # Only include windows where:
            # - Start is within or after start_idx
            # - Window fits entirely (start + L + H <= end_idx)
            valid_starts = [
                ws for ws in all_starts if ws >= start_idx and (ws + L + H) <= end_idx
            ]
            all_starts = valid_starts

        # Per curriculum architecture, EpiDataset always receives a singleton run_id.
        # The curriculum sampler handles run mixing at a higher level, so no run
        # boundary filtering is needed here.
        return all_starts

    def _compute_valid_window_starts(self) -> dict[int, list[int]]:
        return self.get_valid_window_starts_dict(mode="any")

    def get_valid_window_starts_dict(
        self, mode: str = "any", required_targets: list[str] | None = None
    ) -> dict[int, list[int]]:
        """Compute valid window starts per target node using mask-based permits.

        Args:
            mode: "any" (valid if any target meets threshold) or "all" (valid only if all required targets meet threshold)
            required_targets: list of target names that must pass if mode="all". If None, defaults to all available targets.
        """
        if not self.window_starts:
            return {target_idx: [] for target_idx in self.target_nodes}

        if mode not in ("any", "all"):
            raise ValueError(f"Unknown mode: {mode}")

        L = self.config.model.input_window_length
        H = self.config.model.forecast_horizon
        starts = np.asarray(self.window_starts, dtype=np.int64)

        mask_by_target = {
            "cases": self.precomputed_cases_mask,
            "deaths": self.precomputed_deaths_mask,
            "hospitalizations": self.precomputed_hosp_mask,
            "wastewater": self.precomputed_ww_mask,
        }

        if required_targets is not None:
            mask_by_target = {
                k: v for k, v in mask_by_target.items() if k in required_targets
            }

        # Presentation logic:
        # mode="any": Include a window for a node if ANY target has enough observed values
        # mode="all": Include a window for a node if ALL required targets have enough observed values
        valid_mask = (
            np.zeros((len(starts), self.num_nodes), dtype=bool)
            if mode == "any"
            else np.ones((len(starts), self.num_nodes), dtype=bool)
        )

        # If no targets required, valid_mask will just stay all ones
        if mode == "all" and not mask_by_target:
            valid_mask = np.zeros((len(starts), self.num_nodes), dtype=bool)

        for target_name, target_mask_t in mask_by_target.items():
            target_mask = target_mask_t.detach().cpu().numpy()
            observed = (target_mask > 0).astype(np.int32)

            cumsum = np.concatenate(
                [
                    np.zeros((1, self.num_nodes), dtype=np.int32),
                    np.cumsum(observed, axis=0),
                ],
                axis=0,
            )

            history_counts = cumsum[L:] - cumsum[:-L]
            target_counts = cumsum[L + H :] - cumsum[L:-H]

            history_counts = history_counts[starts]
            target_counts = target_counts[starts]

            input_permit = int(self.missing_permit_map["input"].get(target_name, 0))
            horizon_permit = int(self.missing_permit_map["horizon"].get(target_name, 0))
            history_threshold = max(0, L - input_permit)
            target_threshold = max(0, H - horizon_permit)
            target_valid = (history_counts >= history_threshold) & (
                target_counts >= target_threshold
            )

            if mode == "any":
                valid_mask |= target_valid
            else:
                valid_mask &= target_valid

        starts_by_node: dict[int, list[int]] = {}
        for target_idx in self.target_nodes:
            mask = valid_mask[:, target_idx]
            starts_by_node[target_idx] = [
                int(s) for s, ok in zip(starts, mask, strict=False) if ok
            ]

        return starts_by_node

    def _collect_valid_window_starts(self) -> list[int]:
        """Return sorted window starts that are valid for at least one target node."""
        unique_starts: set[int] = set()
        for starts in self._valid_window_starts_by_node.values():
            unique_starts.update(starts)
        return sorted(unique_starts)

    def _build_index_map(
        self,
    ) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
        """Build index mappings for fast (node, window_start) lookup."""
        index_map: list[tuple[int, int]] = []
        index_lookup: dict[tuple[int, int], int] = {}
        sample_ordering = self.config.data.sample_ordering

        if sample_ordering == "time":
            starts_to_nodes = {start: [] for start in self.window_starts}
            for target_idx in self.target_nodes:
                for start in self._valid_window_starts_by_node.get(target_idx, []):
                    starts_to_nodes[start].append(target_idx)
            for start in self.window_starts:
                for target_idx in starts_to_nodes[start]:
                    idx = len(index_map)
                    index_map.append((target_idx, start))
                    index_lookup[(target_idx, start)] = idx
        else:
            for target_idx in self.target_nodes:
                for start in self._valid_window_starts_by_node.get(target_idx, []):
                    idx = len(index_map)
                    index_map.append((target_idx, start))
                    index_lookup[(target_idx, start)] = idx
        return index_map, index_lookup

    def _get_adjacency_at_time(self, time_step: int) -> torch.Tensor:
        if time_step in self._adjacency_cache:
            return self._adjacency_cache[time_step]

        # Get mobility matrix for this time step from preloaded tensor
        mobility_matrix = self.preloaded_mobility[time_step]

        adjacency = mobility_matrix > 0
        if self.context_mask is not None:
            mask = self.context_mask
            adjacency = adjacency & mask[:, None] & mask[None, :]

        self._adjacency_cache[time_step] = adjacency
        return adjacency

    def _get_global_to_local(self) -> torch.Tensor:
        """Compute global to local node index mapping (static, computed once).

        Since context_mask is immutable after initialization, this mapping
        is the same for all timesteps.
        """
        node_ids = torch.where(self._get_graph_node_mask())[0]

        global_to_local = torch.full(
            (self.num_nodes,), -1, dtype=torch.long, device=node_ids.device
        )
        global_to_local[node_ids] = torch.arange(
            node_ids.numel(), device=node_ids.device
        )
        return global_to_local

    def _get_graph_node_mask(self) -> torch.Tensor:
        """Get the node mask used for sparse graph topology construction."""
        if self._target_khop_mask is not None:
            return self._target_khop_mask
        if self.context_mask is not None:
            return self.context_mask
        return torch.ones(self.num_nodes, dtype=torch.bool)

    def _get_global_to_local_at_time(self, time_step: int) -> torch.Tensor:
        """Get global to local mapping (now static, ignores time_step)."""
        # Mapping is static since context_mask never changes after init
        # We compute it once lazily and cache
        if not hasattr(self, "_global_to_local_static"):
            self._global_to_local_static = self._get_global_to_local()
        return self._global_to_local_static

    def _compute_target_khop_mask(self) -> torch.Tensor:
        """Compute nodes that are within gnn_depth hops of any target node."""
        if self.config.model.gnn_depth <= 0:
            if self.context_mask is not None:
                return self.context_mask.clone()
            return torch.ones(self.num_nodes, dtype=torch.bool)

        target_idx = torch.tensor(self.target_nodes, dtype=torch.long)
        target_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        target_mask[target_idx] = True
        khop_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        khop_mask[target_idx] = True

        # Fast path: reuse precomputed per-target k-hop masks if available.
        if self._precomputed_k_hop_masks:
            for reach in self._precomputed_k_hop_masks.values():
                if target_idx.numel() == 0:
                    break
                khop_mask = khop_mask | reach[target_idx].any(dim=0)
        else:
            # Fallback path for safety if precomputed masks are unavailable.
            for time_step in range(len(self._temporal_coords)):
                adjacency = self._get_adjacency_at_time(time_step).to(torch.float32)
                reach = target_mask.clone()
                frontier = target_mask.to(torch.float32)
                for _ in range(self.config.model.gnn_depth):
                    frontier = (adjacency @ frontier) > 0
                    reach = reach | frontier
                khop_mask = khop_mask | reach

        if self.context_mask is not None:
            khop_mask = khop_mask & self.context_mask
        khop_mask[target_idx] = True
        return khop_mask

    def _precompute_k_hop_masks(self) -> dict[int, torch.Tensor]:
        """Precompute k-hop reachability masks for all timesteps at startup.

        This moves expensive CPU matrix operations from per-sample __getitem__
        to initialization time, eliminating the 67% data loading bottleneck.
        Masks are stored as CPU tensors and shared across forked workers.

        Returns:
            Dict mapping time_step -> (N, N) boolean tensor where
            mask[target_idx, node_idx] = True if node is within k-hop of target
        """
        if self.config.model.gnn_depth <= 0:
            return {}

        gnn_depth = self.config.model.gnn_depth
        logger.info(
            f"Precomputing k-hop masks for {len(self._temporal_coords)} timesteps "
            f"(depth={gnn_depth}, nodes={self.num_nodes})..."
        )

        masks = {}
        for time_step in range(len(self._temporal_coords)):
            # Get adjacency for this timestep
            adjacency = self._get_adjacency_at_time(time_step)

            # Compute k-hop reachability via matrix multiplication
            reach = adjacency.clone()
            if gnn_depth > 1:
                adj_f = adjacency.to(torch.float32)
                reach_f = reach.to(torch.float32)
                for _ in range(1, gnn_depth):
                    new_reach = (reach_f @ adj_f) > 0
                    reach = reach | new_reach
                    reach_f = reach.to(torch.float32)

            # Exclude self (diagonal = False)
            reach.fill_diagonal_(False)
            masks[time_step] = reach

        logger.info(f"K-hop mask precomputation complete: {len(masks)} masks")
        return masks

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"EpiDataset(source={self.aligned_data_path}, "
            f"seq_len={self.time_dim_size}, "
            f"nodes={self.num_nodes})"
        )

    @classmethod
    def load_canonical_dataset(
        cls,
        aligned_data_path: Path,
        run_id: str,
        run_id_chunk_size: int = 1,
    ) -> xr.Dataset:
        """Load the canonical dataset from the aligned data path.

        Args:
            aligned_data_path: Path to the Zarr dataset
            run_id: Specific run_id to filter by (e.g., "real", "0_Baseline").
                This is now mandatory to prevent accidental loading of all runs.
            run_id_chunk_size: Chunk size for run_id dimension. Default is 1 to
                ensure memory-efficient loading. Must be a positive integer.

        Returns:
            xarray Dataset filtered to the specified run_id with chunked loading.

        Raises:
            ValueError: If run_id is empty or not found in the dataset.
        """
        if not run_id:
            raise ValueError(
                "run_id is mandatory and must be a non-empty string. "
                "This prevents accidental loading of all runs which can cause OOM."
            )

        if run_id_chunk_size < 1:
            raise ValueError(
                f"run_id_chunk_size must be >= 1, got {run_id_chunk_size}. "
                "Use 1 for memory-efficient loading of a single run."
            )

        # Always chunk by run_id to prevent loading entire dataset into memory
        chunks = {"run_id": run_id_chunk_size}

        dataset = xr.open_zarr(aligned_data_path, chunks=chunks, zarr_format=2)

        # Validate run_id exists in dataset
        if "run_id" not in dataset.coords and "run_id" not in dataset.dims:
            dataset.close()
            raise ValueError(
                f"Dataset at {aligned_data_path} does not have a run_id dimension or coordinate."
            )

        # Filter by run_id (handle whitespace padding)
        if "run_id" in dataset.coords:
            available_runs = [str(r).strip() for r in dataset.run_id.values]
            if run_id not in available_runs:
                dataset.close()
                raise ValueError(
                    f"run_id '{run_id}' not found in dataset. "
                    f"Available runs: {available_runs[:10]}{'...' if len(available_runs) > 10 else ''}"
                )
            mask = dataset.run_id.str.strip() == run_id
            dataset = dataset.sel(run_id=mask).squeeze(drop=True)

        return dataset

    @classmethod
    def discover_available_runs(
        cls,
        aligned_data_path: Path,
    ) -> list[str]:
        """Discover all available run_ids in the dataset without loading data.

        This is a lightweight method that only reads the run_id coordinate
        to discover what runs are available. It does not load any actual data.

        Args:
            aligned_data_path: Path to the Zarr dataset

        Returns:
            List of available run_id strings

        Raises:
            ValueError: If the dataset does not have a run_id dimension or coordinate.
        """
        # Use chunking to prevent loading all data
        dataset = xr.open_zarr(aligned_data_path, chunks={"run_id": 1}, zarr_format=2)

        try:
            if "run_id" not in dataset.coords and "run_id" not in dataset.dims:
                raise ValueError(
                    f"Dataset at {aligned_data_path} does not have a run_id dimension or coordinate."
                )

            # Get run_id values without loading the full dataset
            if "run_id" in dataset.coords:
                run_vals = dataset.run_id.values
            else:
                run_vals = dataset.coords["run_id"].values

            unique_runs = np.unique(run_vals)
            return sorted([str(r).strip() for r in unique_runs])
        finally:
            dataset.close()

    @classmethod
    def get_valid_nodes(
        cls,
        dataset_path: Path,
        run_id: str,
    ) -> np.ndarray:
        """Get valid node mask for a specific run_id.

        This is a lightweight method for the trainer to determine which nodes
        are valid before creating EpiDataset instances. It uses xarray's lazy
        loading to avoid loading the full dataset into memory.

        Args:
            dataset_path: Path to the Zarr dataset
            run_id: Specific run_id to get valid nodes for (e.g., "real", "0_Baseline").

        Returns:
            Boolean numpy array of shape (num_nodes,) where True indicates valid nodes.

        Raises:
            ValueError: If run_id is not found in the dataset.
        """
        aligned_dataset = cls.load_canonical_dataset(
            dataset_path, run_id_chunk_size=1, run_id=run_id
        )

        if "valid_targets" not in aligned_dataset:
            # No valid_targets filter - all nodes are valid
            num_nodes = aligned_dataset[REGION_COORD].size
            return np.ones(num_nodes, dtype=bool)

        # valid_targets is now 1D since we filtered by run_id at load time
        return aligned_dataset.valid_targets.values.astype(bool)

    @classmethod
    def _filter_dataset_by_runs(
        cls, dataset: xr.Dataset, run_id: str | None
    ) -> xr.Dataset:
        """Filter dataset by run_id (always present, no conditional logic).

        Args:
            dataset: xarray Dataset with run_id dimension/coordinate
            run_id: Single run_id string to filter by (e.g., "real", "synth_run_001")
                   If None, returns all runs (for future curriculum mode)

        Returns:
            Filtered xarray Dataset
        """
        assert "run_id" in dataset.coords or "run_id" in dataset.dims, (
            "run_id dimension or coordinate must be present in dataset"
        )

        if run_id is None:
            return dataset

        # Filter 1: Dimension-based (e.g., valid_targets with run_id dim)
        if "run_id" in dataset.dims:
            # Use flexible matching to handle whitespace padding
            mask = dataset.run_id.str.strip() == run_id
            dataset = dataset.sel(run_id=mask)
            # Load data into memory to avoid zarr indexing issues
            dataset = dataset.load()
            # Squeeze run_id dimension if it has size 1
            if dataset.sizes.get("run_id") == 1:
                dataset = dataset.squeeze("run_id")

        # Filter 2: Coordinate-based (e.g., mobility with run_id as time coord)
        if TEMPORAL_COORD in dataset.dims and "run_id" in dataset.coords:
            if dataset.run_id.dims == (TEMPORAL_COORD,):
                mask = dataset.run_id.str.strip() == run_id
                dataset = dataset.sel({TEMPORAL_COORD: mask})
                # Load data into memory
                dataset = dataset.load()

        return dataset
