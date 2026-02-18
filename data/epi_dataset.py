import logging
from pathlib import Path
from typing import TypedDict, Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from graph.node_encoder import Region2Vec
from models.configs import EpiForecasterConfig

from constants import (
    EDAR_BIOMARKER_PREFIX,
    EDAR_BIOMARKER_VARIANTS,
)
from utils import dtypes as dtype_utils
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

logger = logging.getLogger(__name__)

StaticCovariates = dict[str, torch.Tensor]


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (time, region, feature), adding trailing dim if needed."""
    if arr.ndim == 2:
        return arr[..., None]
    return arr


def _replace_non_finite(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf values in floating tensors with finite zeros."""
    if not torch.is_floating_point(tensor):
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


class EpiDatasetItem(TypedDict):
    node_label: str
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
        time_range: tuple[int, int] | None = None,
        run_id: str | None = None,
        region_id_index: dict[str, int] | None = None,
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
        self._region_id_index = region_id_index

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
            logger.info("No temporal covariates found in dataset")

        self.region_embeddings = None
        if config.data.region2vec_path:
            # use pre-trained region2vec embeddings and lookup by labeled regions
            # TODO: actually run forward pass region2vec in EpiForecaster
            _, art = Region2Vec.from_weights(config.data.region2vec_path)

            # Filter region embeddings using numpy instead of xarray to avoid requiring a named dimension
            # for the embedding size (since xarray expects all dims to be named).
            region_ids = list(art.get("region_ids", []))  # type: ignore[typeddict-item]
            selected_ids = list(self._dataset[REGION_COORD].values)
            region_id_index = {rid: i for i, rid in enumerate(region_ids)}
            indices = [
                region_id_index[rid] for rid in selected_ids if rid in region_id_index
            ]
            embeddings = art.get("embeddings")  # type: ignore[typeddict-item]
            if embeddings is None:
                raise ValueError("Region embeddings not found in artifact")
            region_embeddings = embeddings[indices]

            assert region_embeddings.shape == (
                self.num_nodes,
                self.config.model.region_embedding_dim,
            ), "Static embeddings shape mismatch"

            self.region_embeddings = torch.as_tensor(
                region_embeddings, dtype=dtype_utils.STORAGE_DTYPES["continuous"]
            )
            self.region_embeddings = _replace_non_finite(self.region_embeddings)

        self.target_nodes = target_nodes
        self.context_mask = torch.zeros(
            self.num_nodes, dtype=dtype_utils.STORAGE_DTYPES["mask"]
        )
        self.context_mask[target_nodes] = True
        self.context_mask[context_nodes] = True

        # Set dimensions
        self.time_dim_size = config.model.history_length + config.model.forecast_horizon
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
        self._region_labels = list(self._dataset[REGION_COORD].values)
        self._temporal_coords = list(self._dataset[TEMPORAL_COORD].values)

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

        L = self.config.model.history_length
        H = self.config.model.forecast_horizon

        try:
            target_idx, range_start = self._index_map[idx]
        except IndexError as exc:
            raise IndexError("Sample index out of range") from exc

        # Use pre-extracted metadata to avoid zarr access in workers
        node_label = self._region_labels[target_idx]
        target_region_index = target_idx
        if self._region_id_index is not None:
            key = str(node_label)
            if key not in self._region_id_index:
                raise ValueError(f"Region ID '{key}' missing from embedding index.")
            target_region_index = self._region_id_index[key]

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
        mob_edge_index_list: list[torch.Tensor] = []
        mob_edge_weight_list: list[torch.Tensor] = []

        # Find local index of target node (constant across window)
        # Use tensor to defer GPU->CPU sync until actually needed
        global_to_local = self._get_global_to_local_at_time(range_start)
        local_target_idx_tensor = global_to_local[target_idx]
        # Extract Python int once for indexing operations inside the loop
        local_target_idx = int(local_target_idx_tensor.item())

        for t in range(L):
            global_t = range_start + t

            # Get cached full graph topology for this time step
            base_graph = self._build_full_graph_topology_at_time(global_t)

            # Build node features for this time step
            hosp_t = hosp_history[t]  # (N, 3)
            deaths_t = deaths_history[t]  # (N, 3)
            cases_t = cases_history[t]  # (N, 3)
            bio_t = biomarker_history[t]  # (N, B)

            # Build feature tensor for all nodes in context
            node_ids = base_graph.node_ids
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
                x_masked = x.clone()
                x_masked[~local_k_hop_mask] = 0
            else:
                # No masking (all nodes contribute) or no precomputed mask
                x_masked = x
                local_k_hop_mask = torch.ones(x.size(0), dtype=torch.bool)

            # --- OPTIMIZED BATCHING CHANGE ---
            mob_x_list.append(x_masked)
            # Ensure not None for type safety
            assert base_graph.edge_index is not None
            assert base_graph.edge_weight is not None
            mob_edge_index_list.append(base_graph.edge_index)
            mob_edge_weight_list.append(base_graph.edge_weight)

        population = self.node_static_covariates["Pop"][target_idx]

        # Extract temporal covariates for the history window
        temporal_covariates_hist = self.temporal_covariates[range_start:range_end]

        return {
            "node_label": node_label,
            "target_node": target_idx,
            "target_region_index": target_region_index,
            "window_start": range_start,
            "hosp_hist": hosp_history[:, target_idx, :],  # (L, 3)
            "deaths_hist": deaths_history[:, target_idx, :],  # (L, 3)
            "cases_hist": cases_history[:, target_idx, :],  # (L, 3)
            "bio_node": biomarker_history[:, target_idx, :],
            "mob_x": torch.stack(mob_x_list),  # (L, N_ctx, F)
            "mob_edge_index": mob_edge_index_list,  # List[Tensor(2, E)]
            "mob_edge_weight": mob_edge_weight_list,  # List[Tensor(E)]
            "mob_target_node_idx": local_target_idx_tensor,
            "population": population,
            "run_id": run_id,
            "temporal_covariates": temporal_covariates_hist,  # (L, cov_dim)
            "hosp_target": hosp_target,
            "ww_target": ww_target,
            "cases_target": cases_target,
            "deaths_target": deaths_target,
            "hosp_target_mask": hosp_target_mask,
            "ww_target_mask": ww_target_mask,
            "cases_target_mask": cases_target_mask,
            "deaths_target_mask": deaths_target_mask,
        }

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
        L = self.config.model.history_length
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
        """Compute valid window starts per target node using mask-based permits."""
        if not self.window_starts:
            return {target_idx: [] for target_idx in self.target_nodes}

        L = self.config.model.history_length
        H = self.config.model.forecast_horizon
        starts = np.asarray(self.window_starts, dtype=np.int64)

        mask_by_target = {
            "cases": self.precomputed_cases_mask,
            "deaths": self.precomputed_deaths_mask,
            "hospitalizations": self.precomputed_hosp_mask,
            "wastewater": self.precomputed_ww_mask,
        }

        # Presentation logic:
        # Include a window for a node if ANY target has enough observed values
        # in both history and forecast horizon for that target.
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

            permit = int(self.missing_permit_map.get(target_name, 0))
            history_threshold = max(0, L - permit)
            target_threshold = max(0, H - permit)
            target_valid = (history_counts >= history_threshold) & (
                target_counts >= target_threshold
            )
            valid_mask |= target_valid

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

    def _get_global_to_local_at_time(self, time_step: int) -> torch.Tensor:
        if time_step in self._global_to_local_cache:
            return self._global_to_local_cache[time_step]

        if self.context_mask is not None:
            node_ids = torch.where(self.context_mask)[0]
        else:
            node_ids = torch.arange(self.num_nodes)

        global_to_local = torch.full(
            (self.num_nodes,), -1, dtype=torch.long, device=node_ids.device
        )
        global_to_local[node_ids] = torch.arange(
            node_ids.numel(), device=node_ids.device
        )
        self._global_to_local_cache[time_step] = global_to_local
        return global_to_local

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

    def _build_full_graph_topology_at_time(
        self,
        time_step: int,
    ) -> Data:
        """Build full PyG graph topology for a given time step (cached).

        The graph includes all context nodes with full topology.
        Only edge structure is cached - features are added per-sample in __getitem__
        since they require k-hop masking per target node.

        Args:
            time_step: Global time index

        Returns:
            PyG Data object with full graph topology (edge_index, edge_weight, node_ids)
        """
        # Check cache first
        if time_step in self._full_graph_cache:
            return self._full_graph_cache[time_step]

        # Warm adjacency cache for this time step
        _ = self._get_adjacency_at_time(time_step)

        # Get full mobility matrix at this time step from preloaded tensor
        mobility_matrix = self.preloaded_mobility[time_step]  # (N, N)

        # Apply context mask if set
        if self.context_mask is not None:
            mobility_matrix = mobility_matrix.clone()
            mobility_matrix[~self.context_mask] = 0
            mobility_matrix[:, ~self.context_mask] = 0

        # Find all non-zero edges
        edge_mask = mobility_matrix > 0
        origins, destinations = torch.where(edge_mask)
        edge_weight = mobility_matrix[origins, destinations]

        # Map global to local node indices
        node_mask = (
            self.context_mask
            if self.context_mask is not None
            else torch.ones(self.num_nodes, dtype=torch.bool)
        )
        node_ids = torch.where(node_mask)[0]
        global_to_local = {int(idx): i for i, idx in enumerate(node_ids.tolist())}

        local_origins = torch.tensor(
            [global_to_local[int(o)] for o in origins], dtype=torch.long
        )
        local_destinations = torch.tensor(
            [global_to_local[int(d)] for d in destinations], dtype=torch.long
        )
        edge_index = torch.stack([local_origins, local_destinations], dim=0)

        # Create topology-only graph (no features yet)
        g = Data(edge_index=edge_index, edge_weight=edge_weight)
        g.num_nodes = node_ids.numel()
        g.node_ids = node_ids  # Store global node ids for mapping

        # Cache and return
        self._full_graph_cache[time_step] = g
        return g

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

    @classmethod
    def create_temporal_splits(
        cls,
        config: EpiForecasterConfig,
        train_end_date: str,
        val_end_date: str,
        test_end_date: str | None = None,
    ) -> tuple["EpiDataset", "EpiDataset", "EpiDataset"]:
        """Create train/val/test datasets with the same nodes but different time ranges.

        All splits use all available nodes as targets, but data is divided by date ranges.
        Preprocessors are fitted on the train data only and shared across splits.

        Args:
            config: EpiForecasterConfig with dataset path and model parameters.
            train_end_date: Train split end date (YYYY-MM-DD). Exclusive.
            val_end_date: Validation split end date (YYYY-MM-DD). Exclusive.
            test_end_date: Optional test split end date. If None, uses end of dataset.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).

        Raises:
            ValueError: If temporal boundaries are invalid or out of range.
        """
        from utils.temporal import (
            format_date_range,
            get_temporal_boundaries,
            validate_temporal_range,
        )

        # Load canonical dataset to get node list and temporal boundaries
        if not config.data.run_id:
            raise ValueError("run_id must be specified in config for temporal splits")
        aligned_dataset = cls.load_canonical_dataset(
            Path(config.data.dataset_path),
            run_id=config.data.run_id,
            run_id_chunk_size=config.data.run_id_chunk_size,
        )

        num_nodes = aligned_dataset[REGION_COORD].size
        all_nodes = list(range(num_nodes))

        # Check for valid_targets filter
        if config.data.use_valid_targets and "valid_targets" in aligned_dataset:
            valid_targets = aligned_dataset.valid_targets

            # Aggregate across run_id dimension (always present)
            if "run_id" in valid_targets.dims:
                valid_targets = valid_targets.any(dim="run_id")

            valid_mask = valid_targets.values.astype(bool)
            all_nodes = [i for i in all_nodes if valid_mask[i]]
            logger.info(
                f"Using valid_targets filter: {len(all_nodes)}/{num_nodes} training regions"
            )

        # Get temporal boundaries
        train_start, train_end, val_end, test_end = get_temporal_boundaries(
            aligned_dataset,
            train_end_date=train_end_date,
            val_end_date=val_end_date,
            test_end_date=test_end_date,
        )

        L = config.model.history_length
        H = config.model.forecast_horizon
        total_time_steps = len(aligned_dataset[TEMPORAL_COORD])

        # Validate each temporal range
        for name, time_range in [
            ("train", (train_start, train_end)),
            ("val", (train_end, val_end)),
            ("test", (val_end, test_end)),
        ]:
            try:
                validate_temporal_range(time_range, L, H, total_time_steps)
            except ValueError as e:
                raise ValueError(
                    f"{name.upper()} split temporal range invalid: {e}"
                ) from e

        # Log date ranges
        logger.info("Temporal split boundaries:")
        logger.info(
            f"  TRAIN: {format_date_range(aligned_dataset, (train_start, train_end))}"
        )
        logger.info(
            f"  VAL:   {format_date_range(aligned_dataset, (train_end, val_end))}"
        )
        logger.info(
            f"  TEST:  {format_date_range(aligned_dataset, (val_end, test_end))}"
        )

        # Create train dataset with time range - preprocessors fitted internally
        train_dataset = cls(
            config=config,
            target_nodes=all_nodes,
            context_nodes=all_nodes,
            biomarker_preprocessor=None,
            mobility_preprocessor=None,
            time_range=(train_start, train_end),
        )

        # Reuse train dataset's fitted preprocessors for val/test
        fitted_bio_preprocessor = train_dataset.biomarker_preprocessor
        fitted_mobility_preprocessor = train_dataset.mobility_preprocessor

        # Create val and test datasets with their time ranges
        val_dataset = cls(
            config=config,
            target_nodes=all_nodes,
            context_nodes=all_nodes,
            biomarker_preprocessor=fitted_bio_preprocessor,
            mobility_preprocessor=fitted_mobility_preprocessor,
            time_range=(train_end, val_end),
        )

        test_dataset = cls(
            config=config,
            target_nodes=all_nodes,
            context_nodes=all_nodes,
            biomarker_preprocessor=fitted_bio_preprocessor,
            mobility_preprocessor=fitted_mobility_preprocessor,
            time_range=(val_end, test_end),
        )

        return train_dataset, val_dataset, test_dataset


def optimized_collate_graphs(batch: list[EpiDatasetItem]) -> Batch:
    """
    Optimized batch construction for dynamic mobility graphs.

    Constructs a PyG Batch object directly from tensor lists, avoiding the overhead
    of Batch.from_data_list().

    Args:
        batch: List of EpiDatasetItem (must contain mob_x, mob_edge_index, mob_edge_weight)

    Returns:
        A single PyG Batch object containing all time-steps for all samples.
    """
    B = len(batch)
    if B == 0:
        return Batch()

    # 1. Flatten Features
    # (B, L, N, F) -> (B*L*N, F)
    # Note: item["mob_x"] is (L, N, F)
    all_x = torch.cat(
        [item["mob_x"].view(-1, item["mob_x"].size(-1)) for item in batch], dim=0
    )

    # 2. Flatten Edge Indices & Weights
    # We assume constant number of nodes per graph in the batch (context nodes)
    # Check first item for dimensions
    L, num_nodes, _ = batch[0]["mob_x"].shape

    # Collect flattened lists
    all_edge_indices = []
    all_edge_weights = []

    # Iterate samples and time steps
    # Offset calculation: graph_idx * num_nodes
    # Total graphs = B * L
    current_graph_idx = 0

    for item in batch:
        # Check consistency of num_nodes
        if item["mob_x"].shape[1] != num_nodes:
            # If variable node counts are needed later, we must track cumulative nodes.
            # For now, EpiDataset guarantees fixed context size.
            pass

        edge_indices = item["mob_edge_index"]  # List of L tensors
        edge_weights = item["mob_edge_weight"]  # List of L tensors

        for t in range(len(edge_indices)):
            offset = current_graph_idx * num_nodes
            # Shift edge indices
            all_edge_indices.append(edge_indices[t] + offset)
            all_edge_weights.append(edge_weights[t])
            current_graph_idx += 1

    # Concatenate all edges
    if all_edge_indices:
        big_edge_index = torch.cat(all_edge_indices, dim=1)
        big_edge_weight = torch.cat(all_edge_weights, dim=0)
    else:
        big_edge_index = torch.empty((2, 0), dtype=torch.long)
        big_edge_weight = torch.empty((0,), dtype=torch.float16)

    # 3. Create Batch Vector
    # Maps each node to its graph index
    total_nodes = current_graph_idx * num_nodes
    # Verify shape matches x
    assert total_nodes == all_x.size(0)

    batch_vec = torch.arange(current_graph_idx, device=all_x.device).repeat_interleave(
        num_nodes
    )

    # 4. Create Batch Object
    mob_batch = Batch(
        x=all_x, edge_index=big_edge_index, edge_weight=big_edge_weight, batch=batch_vec
    )

    # 5. Add custom attributes needed by model
    # Reconstruct target_node tensor: (B*L,)
    # mob_target_node_idx can be int (legacy) or 0-dim tensor (optimized)
    target_nodes_list = []
    for item in batch:
        tgt = item["mob_target_node_idx"]
        # Handle both int (legacy) and tensor (new optimized format)
        if isinstance(tgt, torch.Tensor):
            target_nodes_list.append(tgt)
        else:
            target_nodes_list.append(torch.tensor(tgt, dtype=torch.long))

    # Stack 0-dim tensors and repeat L times for each timestep
    target_nodes_stacked = torch.stack(target_nodes_list).long()
    target_node_tensor = target_nodes_stacked.repeat_interleave(L).to(all_x.device)
    mob_batch.target_node = target_node_tensor

    # Add target_index for model optimization
    # Since we have fixed num_nodes, start index of graph i is i * num_nodes
    # target_index[i] = start[i] + target_node[i]
    num_graphs = target_node_tensor.size(0)  # B * L graphs total
    graph_starts = torch.arange(num_graphs, device=all_x.device) * num_nodes
    mob_batch["target_index"] = graph_starts + target_node_tensor

    # Add ptr for completeness (standard PyG Batch attribute)
    # ptr = [0, N, 2N, ..., num_graphs*N]
    ptr = torch.arange(num_graphs + 1, device=all_x.device) * num_nodes
    mob_batch.ptr = ptr

    return mob_batch


def collate_epiforecaster_batch(
    batch: list[EpiDatasetItem],
    *,
    require_region_index: bool = True,
) -> dict[str, Any]:
    """
    Collate function for EpiForecaster batches.

    This function is shared by curriculum and standard training/evaluation paths.
    It flattens per-time-step graphs into a single PyG Batch for a consistent model
    contract and can enforce region-index availability when region embeddings are used.
    """
    B = len(batch)
    if B == 0:
        return {}

    # 1. Stack standard tensors
    # Clinical series (3-channel: value, mask, age)
    hosp_hist = torch.stack([item["hosp_hist"] for item in batch], dim=0)  # (B, L, 3)
    deaths_hist = torch.stack(
        [item["deaths_hist"] for item in batch], dim=0
    )  # (B, L, 3)
    cases_hist = torch.stack([item["cases_hist"] for item in batch], dim=0)  # (B, L, 3)
    bio_node = torch.stack([item["bio_node"] for item in batch], dim=0)
    target_nodes = torch.tensor(
        [item["target_node"] for item in batch], dtype=torch.long
    )
    window_starts = torch.tensor(
        [item["window_start"] for item in batch], dtype=torch.long
    )
    population = torch.stack([item["population"] for item in batch], dim=0)

    # Stack joint inference targets (log1p per-100k)
    hosp_targets = torch.stack([item["hosp_target"] for item in batch], dim=0)
    ww_targets = torch.stack([item["ww_target"] for item in batch], dim=0)
    cases_targets = torch.stack([item["cases_target"] for item in batch], dim=0)
    deaths_targets = torch.stack([item["deaths_target"] for item in batch], dim=0)
    hosp_target_masks = torch.stack([item["hosp_target_mask"] for item in batch], dim=0)
    ww_target_masks = torch.stack([item["ww_target_mask"] for item in batch], dim=0)
    cases_target_masks = torch.stack(
        [item["cases_target_mask"] for item in batch], dim=0
    )
    deaths_target_masks = torch.stack(
        [item["deaths_target_mask"] for item in batch], dim=0
    )

    # Stack temporal covariates
    temporal_covariates = torch.stack(
        [item["temporal_covariates"] for item in batch], dim=0
    )  # (B, L, cov_dim)

    # 2. Batch Temporal Graphs (Optimized Manual Batching)
    mob_batch = optimized_collate_graphs(batch)
    if hasattr(mob_batch, "x") and mob_batch.x is not None:
        mob_batch.x = _replace_non_finite(mob_batch.x)
    if hasattr(mob_batch, "edge_weight") and mob_batch.edge_weight is not None:
        mob_batch.edge_weight = _replace_non_finite(mob_batch.edge_weight)

    hosp_hist = _replace_non_finite(hosp_hist)
    deaths_hist = _replace_non_finite(deaths_hist)
    cases_hist = _replace_non_finite(cases_hist)
    bio_node = _replace_non_finite(bio_node)
    temporal_covariates = _replace_non_finite(temporal_covariates)

    # Store B and T on the batch for downstream reshaping
    T = batch[0]["mob_x"].shape[0] if B > 0 else 0
    mob_batch.B = torch.tensor([B], dtype=torch.long)  # type: ignore[attr-defined]
    mob_batch.T = torch.tensor([T], dtype=torch.long)  # type: ignore[attr-defined]

    target_region_indices = [item["target_region_index"] for item in batch]
    if require_region_index and any(idx is None for idx in target_region_indices):
        raise ValueError(
            "TargetRegionIndex missing for batch while region matching is required. "
            "Ensure region_id_index is provided when model.type.regions is enabled."
        )

    out = {
        # Clinical series inputs (3-channel: value, mask, age)
        "HospHist": hosp_hist,  # (B, L, 3)
        "DeathsHist": deaths_hist,  # (B, L, 3)
        "CasesHist": cases_hist,  # (B, L, 3)
        "BioNode": bio_node,  # (B, L, B)
        "MobBatch": mob_batch,  # Batched PyG graphs
        "Population": population,  # (B,)
        "B": B,
        "T": T,
        "TargetNode": target_nodes,  # (B,)
        "TargetRegionIndex": torch.tensor(target_region_indices, dtype=torch.long),
        "WindowStart": window_starts,  # (B,)
        "NodeLabels": [item["node_label"] for item in batch],
        # Temporal covariates
        "TemporalCovariates": temporal_covariates,  # (B, L, cov_dim)
        # Joint inference targets (log1p per-100k)
        "HospTarget": hosp_targets,  # (B, H)
        "WWTarget": ww_targets,  # (B, H)
        "CasesTarget": cases_targets,  # (B, H)
        "DeathsTarget": deaths_targets,  # (B, H)
        "HospTargetMask": hosp_target_masks,  # (B, H)
        "WWTargetMask": ww_target_masks,  # (B, H)
        "CasesTargetMask": cases_target_masks,  # (B, H)
        "DeathsTargetMask": deaths_target_masks,  # (B, H)
    }
    if all(idx is not None for idx in target_region_indices):
        out["TargetRegionIndex"] = torch.tensor(target_region_indices, dtype=torch.long)

    return out
