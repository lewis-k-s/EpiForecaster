"""Typed overlap and directed-OD context encoder for native support.

The fixed pooled context remains the identity path. Relation-specific
messages are additive behind zero-initialized scalar gates, so the initial
heterogeneous model is exactly equivalent to the pooled baseline. OD graph
states are grouped by forecast origin in :class:`NativeEpiBatch`; one dense
mobility encoding is therefore shared by every query with that origin.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import repeat

from data.native_batch import SOURCE_TYPE_IDS, NativeEpiBatch
from models.mobility_gnn import MobilityDenseEncoder

OVERLAP_RELATIONS = ("muni_abs",)
MOBILITY_TARGET_TYPES = ("municipality", "abs")


class TypedOverlapMessage(nn.Module):
    """A learned relation projection behind an exact zero-init gate."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gate) * self.projection(values)


class HeterogeneousContextEncoder(nn.Module):
    """Add typed overlap and OD messages to the pooled context residual."""

    def __init__(
        self,
        *,
        context_dim: int,
        relation_input_dim: int = 3,
        od_input_dim: int = 3,
        hidden_dim: int = 16,
        mobility_enabled: bool = True,
        wastewater_enabled: bool = False,
        parameter_budget: int = 100_000,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("heterogeneous context_dim must be positive")
        self.context_dim = context_dim
        self.mobility_enabled = mobility_enabled
        overlap_relations = OVERLAP_RELATIONS + (
            ("muni_edar_catchment",) if wastewater_enabled else ()
        )
        self.overlap_messages = nn.ModuleDict(
            {
                relation: TypedOverlapMessage(relation_input_dim, context_dim)
                for relation in overlap_relations
            }
        )
        self.mobility_encoder: MobilityDenseEncoder | None = None
        self.mobility_messages = nn.ModuleDict()
        if mobility_enabled:
            self.mobility_encoder = MobilityDenseEncoder(
                in_dim=od_input_dim,
                hidden_dim=hidden_dim,
                out_dim=context_dim,
                depth=1,
                module_type="gcn",
                dropout=0.0,
            )
            mobility_target_types = MOBILITY_TARGET_TYPES + (
                ("edar",) if wastewater_enabled else ()
            )
            self.mobility_messages = nn.ModuleDict(
                {
                    source_type: TypedOverlapMessage(context_dim, context_dim)
                    for source_type in mobility_target_types
                }
            )
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if self.parameter_count > parameter_budget:
            raise ValueError(
                "heterogeneous encoder parameter count "
                f"{self.parameter_count} exceeds budget {parameter_budget}"
            )

    def forward(self, batch: NativeEpiBatch) -> torch.Tensor:
        """Return ``[B, L, D_ctx]`` with the native-history path untouched."""
        context = batch.ctx_feat
        for relation, message in self.overlap_messages.items():
            values = batch.relation_ctx.get(relation)
            if values is not None:
                context = context + message(values.to(context.dtype))

        if not self.mobility_enabled:
            return context
        if self.mobility_encoder is None or batch.od_node_features.numel() == 0:
            raise ValueError(
                "heterogeneous mobility is enabled but the batch has no OD graph context"
            )
        encoded = self.mobility_encoder(
            batch.od_node_features,
            batch.od_adjacency,
        )
        row_encoded = encoded.index_select(0, batch.od_group_index.long())
        gathered = torch.einsum(
            "bn,bnd->bd", batch.od_target_weights.to(row_encoded.dtype), row_encoded
        )
        sequence = repeat(gathered, "b d -> b l d", l=context.shape[1])
        for source_type, message in self.mobility_messages.items():
            mask = batch.source_type_id.eq(SOURCE_TYPE_IDS[source_type]).to(
                context.dtype
            )
            context = context + message(sequence) * mask[:, None, None]
        return context


__all__ = [
    "HeterogeneousContextEncoder",
    "MOBILITY_TARGET_TYPES",
    "OVERLAP_RELATIONS",
    "TypedOverlapMessage",
]
