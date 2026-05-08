from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Batch
from utils.precision_policy import MODEL_PARAM_DTYPE

if TYPE_CHECKING:
    from data.epi_dataset import EpiDatasetItem


@dataclass
class EpiBatch:
    """Strongly-typed data structure representing a compiled batch."""

    hosp_hist: torch.Tensor  # (B, L, 3)
    deaths_hist: torch.Tensor  # (B, L, 3)
    cases_hist: torch.Tensor  # (B, L, 3)
    bio_node: torch.Tensor  # (B, L, D)
    mob_batch: Batch
    population: torch.Tensor  # (B,)
    b: int
    t: int
    target_node: torch.Tensor  # (B,)
    target_region_index: torch.Tensor | None  # (B,)
    window_start: torch.Tensor  # (B,)
    node_labels: list[str]
    temporal_covariates: torch.Tensor  # (B, L, cov_dim)
    vaccination_hist: torch.Tensor  # (B, L, 3)

    # Joint inference targets
    ww_hist: torch.Tensor  # (B, L)
    ww_hist_mask: torch.Tensor  # (B, L)
    hosp_target: torch.Tensor  # (B, H)
    ww_target: torch.Tensor  # (B, H)
    cases_target: torch.Tensor  # (B, H)
    deaths_target: torch.Tensor  # (B, H)
    hosp_target_mask: torch.Tensor  # (B, H)
    ww_target_mask: torch.Tensor  # (B, H)
    cases_target_mask: torch.Tensor  # (B, H)
    deaths_target_mask: torch.Tensor  # (B, H)
    S_target: torch.Tensor | None  # (B, H+1)
    I_target: torch.Tensor | None  # (B, H+1)
    R_target: torch.Tensor | None  # (B, H+1)
    D_target: torch.Tensor | None  # (B, H+1)
    S_target_mask: torch.Tensor | None  # (B, H+1)
    I_target_mask: torch.Tensor | None  # (B, H+1)
    R_target_mask: torch.Tensor | None  # (B, H+1)
    D_target_mask: torch.Tensor | None  # (B, H+1)

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = True,
    ) -> "EpiBatch":
        """Move batch to the target device without changing dtypes."""
        if isinstance(device, str):
            device = torch.device(device)
        safe_non_blocking = bool(non_blocking) and device.type == "cuda"

        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                if value.device != device:
                    value = value.to(device=device, non_blocking=safe_non_blocking)
                setattr(self, field_name, value)
            elif isinstance(value, Batch):
                setattr(
                    self,
                    field_name,
                    value.to(device, non_blocking=safe_non_blocking),
                )
        return self

    def pin_memory(self) -> "EpiBatch":
        """Pin CPU tensors recursively so CUDA transfers can be non-blocking."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                if value.device.type == "cpu":
                    setattr(self, field_name, value.pin_memory())
            elif isinstance(value, Batch):
                setattr(self, field_name, value.pin_memory())
        return self

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record the consumer stream on all CUDA tensors in the batch."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                if value.device.type == "cuda":
                    value.record_stream(stream)
            elif isinstance(value, Batch):
                value.record_stream(stream)


def _replace_non_finite(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf values in floating tensors with finite zeros."""
    if not torch.is_floating_point(tensor):
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _promote_batch_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Promote floating tensors to the canonical model input dtype on CPU."""
    if torch.is_floating_point(tensor) and tensor.dtype != MODEL_PARAM_DTYPE:
        return tensor.to(MODEL_PARAM_DTYPE)
    return tensor


def _stack_optional_batch_tensor(
    batch: list[EpiDatasetItem], key: str
) -> torch.Tensor | None:
    first_value = batch[0].get(key)
    if first_value is None:
        if any(item.get(key) is not None for item in batch[1:]):
            raise ValueError(
                f"Inconsistent optional batch key {key!r}: mixed None and tensor"
            )
        return None

    values = [item.get(key) for item in batch]
    if any(value is None for value in values):
        raise ValueError(
            f"Inconsistent optional batch key {key!r}: mixed None and tensor"
        )
    return torch.stack(values, dim=0)  # type: ignore[arg-type]


def optimized_collate_graphs(batch: list[EpiDatasetItem]) -> Batch:
    """
    Optimized dense batch construction for dynamic mobility graphs.

    Args:
        batch: List of EpiDatasetItem (must contain mob_x, mob_t, mob_target_node_idx)

    Returns:
        A PyG Batch-like container with:
        - x_dense: [B*T, N, F]
        - global_t: [B*T]
        - target_node: [B*T]
    """
    B = len(batch)
    if B == 0:
        return Batch()

    # Fixed-size dense contract per item: mob_x is (L, N, F)
    L, num_nodes, _ = batch[0]["mob_x"].shape

    # 1) Dense node features [B*T, N, F]
    x_dense = torch.cat([item["mob_x"] for item in batch], dim=0)

    # 2) Global T indices [B*T]
    global_t_dense = torch.cat([item["mob_t"] for item in batch], dim=0)

    # 3) Target node index per graph [B*T]
    target_nodes_stacked = torch.stack(
        [
            item["mob_target_node_idx"]
            if isinstance(item["mob_target_node_idx"], torch.Tensor)
            else torch.tensor(item["mob_target_node_idx"], dtype=torch.long)
            for item in batch
        ]
    ).long()
    target_node_tensor = target_nodes_stacked.repeat_interleave(L).to(x_dense.device)

    # Extract run_id provenance if the batch is homogeneous (expected with chunked samplers).
    run_ids = [item.get("run_id") for item in batch]
    non_none_run_ids = [run_id for run_id in run_ids if run_id is not None]
    normalized_run_ids = {str(run_id).strip() for run_id in non_none_run_ids}
    batch_run_id = None
    if len(non_none_run_ids) == len(run_ids) and len(normalized_run_ids) == 1:
        batch_run_id = next(iter(normalized_run_ids))

    # Context node mapping is cached once on the dataset and shared by identity
    # across all samples in the batch.
    mob_real_node_idx = None
    if "mob_real_node_idx" in batch[0]:
        mob_real_node_idx = batch[0]["mob_real_node_idx"]
        for item_idx, item in enumerate(batch[1:], start=1):
            if item.get("mob_real_node_idx") is not mob_real_node_idx:
                raise ValueError(
                    "Inconsistent mobility batch: mob_real_node_idx must be the "
                    f"same cached tensor for all samples; sample {item_idx} differs."
                )

    # 5) Batch-like container
    mob_batch = Batch()
    mob_batch.x_dense = x_dense
    mob_batch.global_t = global_t_dense
    mob_batch.target_node = target_node_tensor
    if mob_real_node_idx is not None:
        mob_batch.mob_real_node_idx = mob_real_node_idx
    mob_batch.run_id = batch_run_id

    return mob_batch


def collate_epiforecaster_batch(
    batch: list[EpiDatasetItem],
    *,
    require_region_index: bool = True,
) -> EpiBatch:
    """
    Collate function for EpiForecaster batches.

    This function is shared by curriculum and standard training/evaluation paths.
    It flattens per-time-step graphs into a single PyG Batch for a consistent model
    contract and can enforce region-index availability when region embeddings are used.
    """
    B = len(batch)
    if B == 0:
        return EpiBatch(**{k: None for k in EpiBatch.__dataclass_fields__})  # type: ignore

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
    population = population.to(MODEL_PARAM_DTYPE)

    # Stack joint inference targets (log1p per-100k)
    ww_hist = torch.stack([item["ww_hist"] for item in batch], dim=0)
    ww_hist_mask = torch.stack([item["ww_hist_mask"] for item in batch], dim=0)
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
    S_targets = _stack_optional_batch_tensor(batch, "S_target")
    I_targets = _stack_optional_batch_tensor(batch, "I_target")
    R_targets = _stack_optional_batch_tensor(batch, "R_target")
    D_targets = _stack_optional_batch_tensor(batch, "D_target")
    S_target_masks = _stack_optional_batch_tensor(batch, "S_target_mask")
    I_target_masks = _stack_optional_batch_tensor(batch, "I_target_mask")
    R_target_masks = _stack_optional_batch_tensor(batch, "R_target_mask")
    D_target_masks = _stack_optional_batch_tensor(batch, "D_target_mask")

    # Stack temporal covariates
    temporal_covariates = torch.stack(
        [item["temporal_covariates"] for item in batch], dim=0
    )  # (B, L, cov_dim)

    # Stack vaccination history (3-channel)
    vaccination_hist = torch.stack(
        [item["vaccination_hist"] for item in batch], dim=0
    )  # (B, L, 3)

    # 2. Batch Temporal Graphs (Optimized Manual Batching)
    mob_batch = optimized_collate_graphs(batch)
    if hasattr(mob_batch, "x_dense") and mob_batch.x_dense is not None:
        mob_batch.x_dense = _replace_non_finite(mob_batch.x_dense)
        mob_batch.x_dense = _promote_batch_float_tensor(mob_batch.x_dense)

    hosp_hist = _promote_batch_float_tensor(_replace_non_finite(hosp_hist))
    deaths_hist = _promote_batch_float_tensor(_replace_non_finite(deaths_hist))
    cases_hist = _promote_batch_float_tensor(_replace_non_finite(cases_hist))
    bio_node = _promote_batch_float_tensor(_replace_non_finite(bio_node))
    temporal_covariates = _promote_batch_float_tensor(
        _replace_non_finite(temporal_covariates)
    )
    vaccination_hist = _promote_batch_float_tensor(
        _replace_non_finite(vaccination_hist)
    )
    ww_hist = _promote_batch_float_tensor(_replace_non_finite(ww_hist))
    ww_hist_mask = _promote_batch_float_tensor(_replace_non_finite(ww_hist_mask))
    hosp_targets = _promote_batch_float_tensor(hosp_targets)
    ww_targets = _promote_batch_float_tensor(ww_targets)
    cases_targets = _promote_batch_float_tensor(cases_targets)
    deaths_targets = _promote_batch_float_tensor(deaths_targets)
    hosp_target_masks = _promote_batch_float_tensor(hosp_target_masks)
    ww_target_masks = _promote_batch_float_tensor(ww_target_masks)
    cases_target_masks = _promote_batch_float_tensor(cases_target_masks)
    deaths_target_masks = _promote_batch_float_tensor(deaths_target_masks)
    if S_targets is not None:
        S_targets = _promote_batch_float_tensor(_replace_non_finite(S_targets))
    if I_targets is not None:
        I_targets = _promote_batch_float_tensor(_replace_non_finite(I_targets))
    if R_targets is not None:
        R_targets = _promote_batch_float_tensor(_replace_non_finite(R_targets))
    if D_targets is not None:
        D_targets = _promote_batch_float_tensor(_replace_non_finite(D_targets))
    if S_target_masks is not None:
        S_target_masks = _promote_batch_float_tensor(
            _replace_non_finite(S_target_masks)
        )
    if I_target_masks is not None:
        I_target_masks = _promote_batch_float_tensor(
            _replace_non_finite(I_target_masks)
        )
    if R_target_masks is not None:
        R_target_masks = _promote_batch_float_tensor(
            _replace_non_finite(R_target_masks)
        )
    if D_target_masks is not None:
        D_target_masks = _promote_batch_float_tensor(
            _replace_non_finite(D_target_masks)
        )

    # Store B and T on the batch for downstream reshaping
    T = batch[0]["mob_x"].shape[0] if B > 0 else 0
    mob_batch.B = torch.tensor([B], dtype=torch.long)  # type: ignore[attr-defined]
    mob_batch.T = torch.tensor([T], dtype=torch.long)  # type: ignore[attr-defined]

    target_region_indices = [item["target_region_index"] for item in batch]
    if require_region_index and any(idx is None for idx in target_region_indices):
        raise ValueError(
            "TargetRegionIndex missing for batch while region matching is required. "
            "Ensure region_embedding_store is provided when model.type.regions is enabled."
        )

    return EpiBatch(
        hosp_hist=hosp_hist,
        deaths_hist=deaths_hist,
        cases_hist=cases_hist,
        bio_node=bio_node,
        mob_batch=mob_batch,
        population=population,
        b=B,
        t=T,
        target_node=target_nodes,
        target_region_index=torch.tensor(target_region_indices, dtype=torch.long)
        if all(idx is not None for idx in target_region_indices)
        else None,
        window_start=window_starts,
        node_labels=[item["node_label"] for item in batch],
        temporal_covariates=temporal_covariates,
        vaccination_hist=vaccination_hist,
        ww_hist=ww_hist,
        ww_hist_mask=ww_hist_mask,
        hosp_target=hosp_targets,
        ww_target=ww_targets,
        cases_target=cases_targets,
        deaths_target=deaths_targets,
        hosp_target_mask=hosp_target_masks,
        ww_target_mask=ww_target_masks,
        cases_target_mask=cases_target_masks,
        deaths_target_mask=deaths_target_masks,
        S_target=S_targets,
        I_target=I_targets,
        R_target=R_targets,
        D_target=D_targets,
        S_target_mask=S_target_masks,
        I_target_mask=I_target_masks,
        R_target_mask=R_target_masks,
        D_target_mask=D_target_masks,
    )


def mask_ablated_inputs(
    batch_data: EpiBatch,
    *,
    mask_cases: bool = False,
    mask_ww: bool = False,
    mask_hosp: bool = False,
    mask_deaths: bool = False,
) -> None:
    """
    Apply zero-masking to input data series corresponding to disabled clinical heads.

    This ensures that when an observation head's loss is ablated, its input data
    is also zeroed out in the batch so it does not leak information to other heads
    through the backbone representation.

    Args:
        batch_data: The batch data dictionary. Modified in-place.
        mask_cases: If True, zero out 'CasesHist'.
        mask_ww: If True, zero out 'BioNode'.
        mask_hosp: If True, zero out 'HospHist'.
        mask_deaths: If True, zero out 'DeathsHist'.
    """
    if mask_ww and hasattr(batch_data, "bio_node") and batch_data.bio_node is not None:
        batch_data.bio_node = torch.zeros_like(batch_data.bio_node)
    if (
        mask_hosp
        and hasattr(batch_data, "hosp_hist")
        and batch_data.hosp_hist is not None
    ):
        batch_data.hosp_hist = torch.zeros_like(batch_data.hosp_hist)
    if (
        mask_cases
        and hasattr(batch_data, "cases_hist")
        and batch_data.cases_hist is not None
    ):
        batch_data.cases_hist = torch.zeros_like(batch_data.cases_hist)
    if (
        mask_deaths
        and hasattr(batch_data, "deaths_hist")
        and batch_data.deaths_hist is not None
    ):
        batch_data.deaths_hist = torch.zeros_like(batch_data.deaths_hist)
