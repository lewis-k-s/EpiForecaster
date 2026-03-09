from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from data import dtypes as dtype_utils
from data.epi_batch import _replace_non_finite
from graph.node_encoder import Region2Vec


@dataclass(frozen=True)
class RegionEmbeddingStore:
    """Shared canonical region embedding table and lookup helpers."""

    embeddings: torch.Tensor
    region_id_to_index: dict[str, int]

    @classmethod
    def from_weights(
        cls,
        weights_path: str | Path,
        *,
        expected_dim: int,
    ) -> "RegionEmbeddingStore":
        _, art = Region2Vec.from_weights(weights_path)

        region_ids = [str(region_id) for region_id in art.get("region_ids", [])]
        embeddings = art.get("embeddings")
        if embeddings is None:
            raise ValueError("Region embeddings not found in artifact")
        if len(region_ids) != len(embeddings):
            raise ValueError(
                "Region embedding artifact mismatch: region_ids and embeddings length differ"
            )
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Region embedding dim mismatch: expected {expected_dim}, "
                f"found {embeddings.shape[1]}"
            )

        embedding_tensor = torch.as_tensor(
            embeddings, dtype=dtype_utils.STORAGE_DTYPES["continuous"]
        )
        embedding_tensor = _replace_non_finite(embedding_tensor)

        region_id_to_index = {region_id: idx for idx, region_id in enumerate(region_ids)}
        return cls(
            embeddings=embedding_tensor,
            region_id_to_index=region_id_to_index,
        )

    def build_local_to_global_index(
        self, region_ids: list[str] | tuple[str, ...]
    ) -> torch.Tensor:
        missing = [
            region_id
            for region_id in region_ids
            if region_id not in self.region_id_to_index
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Missing {len(missing)} region IDs in embedding store: {preview}"
            )

        return torch.tensor(
            [self.region_id_to_index[region_id] for region_id in region_ids],
            dtype=torch.long,
        )
