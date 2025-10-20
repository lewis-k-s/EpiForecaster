"""
Attention Mask Processor for EDAR-Municipality Signal Integration

Implements the attention mechanism that uses EDAR-municipality contribution ratios
to mask and aggregate wastewater treatment signals for each municipality.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

logger = logging.getLogger(__name__)


class AttentionMaskProcessor(nn.Module):
    """
    Processes EDAR signals using municipality-specific attention masks.

    This module takes EDAR embeddings and applies the contribution ratio mask
    to generate municipality-specific wastewater signals that can be combined
    with mobility patterns for forecasting.
    """

    def __init__(
        self,
        edar_embedding_dim: int,
        municipality_embedding_dim: int,
        use_learnable_weights: bool = True,
        temperature: float = 1.0,
        dropout: float = 0.1,
        fusion_method: str = "weighted",
    ):
        """
        Initialize attention mask processor.

        Args:
            edar_embedding_dim: Dimension of EDAR embeddings
            municipality_embedding_dim: Dimension of municipality embeddings
            use_learnable_weights: Whether to learn additional attention weights
            temperature: Temperature for attention softmax
            dropout: Dropout probability
            fusion_method: How to combine contributions ('weighted', 'gated', 'mlp')
        """
        super().__init__()

        self.edar_embedding_dim = edar_embedding_dim
        self.municipality_embedding_dim = municipality_embedding_dim
        self.use_learnable_weights = use_learnable_weights
        self.temperature = temperature
        self.fusion_method = fusion_method

        # Learnable attention parameters (optional)
        if use_learnable_weights:
            # Query projection for municipalities
            self.query_proj = nn.Linear(municipality_embedding_dim, edar_embedding_dim)
            # Key projection for EDARs
            self.key_proj = nn.Linear(edar_embedding_dim, edar_embedding_dim)
            # Value projection for EDARs
            self.value_proj = nn.Linear(edar_embedding_dim, edar_embedding_dim)

            # Scale factor for dot product attention
            self.scale = edar_embedding_dim**-0.5

        # Fusion layers based on method
        if fusion_method == "gated":
            # Gating mechanism to combine mask and learned attention
            self.gate = nn.Sequential(
                nn.Linear(edar_embedding_dim * 2, edar_embedding_dim), nn.Sigmoid()
            )
        elif fusion_method == "mlp":
            # MLP to process combined signals
            self.fusion_mlp = nn.Sequential(
                nn.Linear(edar_embedding_dim, edar_embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edar_embedding_dim * 2, edar_embedding_dim),
            )

        # Output projection to match municipality embedding dimension
        if edar_embedding_dim != municipality_embedding_dim:
            self.output_proj = nn.Linear(edar_embedding_dim, municipality_embedding_dim)
        else:
            self.output_proj = nn.Identity()

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(municipality_embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        edar_embeddings: torch.Tensor,
        contribution_mask: torch.Tensor,
        municipality_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Apply attention mask to aggregate EDAR signals for municipalities.

        Args:
            edar_embeddings: EDAR node embeddings [n_edars, edar_embed_dim]
            contribution_mask: Contribution ratios [n_municipalities, n_edars]
            municipality_embeddings: Optional municipality embeddings for attention
                                    [n_municipalities, muni_embed_dim]

        Returns:
            Tuple of:
                - Municipality-specific EDAR signals [n_municipalities, muni_embed_dim]
                - Dictionary with attention weights and statistics
        """
        contribution_mask.shape[0]
        edar_embeddings.shape[0]

        # Initialize attention weights with contribution mask
        attention_weights = contribution_mask.clone()

        # Apply learnable attention if enabled and municipality embeddings provided
        if self.use_learnable_weights and municipality_embeddings is not None:
            # Compute learned attention scores
            learned_attention = self._compute_learned_attention(
                municipality_embeddings, edar_embeddings
            )

            # Combine with contribution mask
            if self.fusion_method == "weighted":
                # Weighted combination
                attention_weights = attention_weights * learned_attention
            elif self.fusion_method == "gated":
                # Gated combination
                gate_input = torch.cat(
                    [
                        attention_weights.unsqueeze(-1).expand(
                            -1, -1, self.edar_embedding_dim
                        ),
                        learned_attention.unsqueeze(-1).expand(
                            -1, -1, self.edar_embedding_dim
                        ),
                    ],
                    dim=-1,
                )
                gate_values = self.gate(gate_input).mean(
                    dim=-1
                )  # Average over embedding dim
                attention_weights = (
                    gate_values * attention_weights
                    + (1 - gate_values) * learned_attention
                )

        # Normalize attention weights
        # Only normalize over non-zero entries to maintain sparsity
        mask_nonzero = attention_weights > 0
        if mask_nonzero.any():
            # Apply temperature scaling
            attention_weights = attention_weights / self.temperature

            # Masked softmax - only over contributing EDARs
            attention_weights_masked = attention_weights.masked_fill(
                ~mask_nonzero, -1e9
            )
            attention_weights = F.softmax(attention_weights_masked, dim=1)

            # Re-apply mask to ensure exact zeros where there's no contribution
            attention_weights = attention_weights * mask_nonzero.float()

        # Apply attention to aggregate EDAR signals using einsum for clarity
        # attention_weights: [n_municipalities, n_edars]
        # edar_embeddings: [n_edars, edar_embed_dim]
        # aggregated_signals: [n_municipalities, edar_embed_dim]
        aggregated_signals = einsum(
            attention_weights,
            edar_embeddings,
            "municipalities edars, edars edar_embed_dim -> municipalities edar_embed_dim",
        )

        # Validate aggregated signals shape
        expected_agg_shape = (attention_weights.shape[0], edar_embeddings.shape[1])
        assert aggregated_signals.shape == expected_agg_shape, (
            f"Aggregated signals shape mismatch: expected {expected_agg_shape}, got {aggregated_signals.shape}"
        )

        # Apply fusion MLP if specified
        if self.fusion_method == "mlp":
            aggregated_signals = self.fusion_mlp(aggregated_signals)

        # Project to municipality embedding dimension
        output = self.output_proj(aggregated_signals)
        # Shape: [n_municipalities, municipality_embedding_dim]
        assert output.shape == (
            attention_weights.shape[0],
            self.municipality_embedding_dim,
        ), (
            f"Output shape mismatch: expected ({attention_weights.shape[0]}, {self.municipality_embedding_dim}), got {output.shape}"
        )

        # Apply layer normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)

        # Prepare attention statistics with shape validation
        attention_stats = {
            "attention_weights": attention_weights,  # [n_municipalities, n_edars]
            "attention_entropy": self._compute_attention_entropy(
                attention_weights
            ),  # scalar
            "effective_edars": (attention_weights > 0.01)
            .sum(dim=1)
            .float()
            .mean(),  # scalar
            "max_attention": attention_weights.max(dim=1)[0].mean(),  # scalar
        }

        return output, attention_stats

    def _compute_learned_attention(
        self, municipality_embeddings: torch.Tensor, edar_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute learned attention scores between municipalities and EDARs.

        Args:
            municipality_embeddings: [n_municipalities, muni_embed_dim]
            edar_embeddings: [n_edars, edar_embed_dim]

        Returns:
            Attention scores [n_municipalities, n_edars]
        """
        # Validate input shapes
        assert municipality_embeddings.ndim == 2, (
            f"Municipality embeddings should be 2D, got {municipality_embeddings.shape}"
        )
        assert edar_embeddings.ndim == 2, (
            f"EDAR embeddings should be 2D, got {edar_embeddings.shape}"
        )

        # Project to common space
        queries = self.query_proj(municipality_embeddings)  # [n_munis, edar_embed_dim]
        keys = self.key_proj(edar_embeddings)  # [n_edars, edar_embed_dim]
        self.value_proj(edar_embeddings)  # [n_edars, edar_embed_dim]

        # Validate projection shapes
        assert queries.shape == (
            municipality_embeddings.shape[0],
            self.edar_embedding_dim,
        ), (
            f"Queries shape mismatch: expected ({municipality_embeddings.shape[0]}, {self.edar_embedding_dim}), got {queries.shape}"
        )
        assert keys.shape == (edar_embeddings.shape[0], self.edar_embedding_dim), (
            f"Keys shape mismatch: expected ({edar_embeddings.shape[0]}, {self.edar_embedding_dim}), got {keys.shape}"
        )

        # Compute attention scores using einsum for clarity
        # queries: [n_munis, edar_embed_dim]
        # keys: [n_edars, edar_embed_dim] -> transpose to [edar_embed_dim, n_edars]
        # attention_scores: [n_munis, n_edars]
        attention_scores = (
            einsum(
                queries,
                keys.transpose(0, 1),
                "munis edar_dim, edar_dim edars -> munis edars",
            )
            * self.scale
        )

        # Validate output shape
        expected_scores_shape = (
            municipality_embeddings.shape[0],
            edar_embeddings.shape[0],
        )
        assert attention_scores.shape == expected_scores_shape, (
            f"Attention scores shape mismatch: expected {expected_scores_shape}, got {attention_scores.shape}"
        )

        return attention_scores

    def _compute_attention_entropy(
        self, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy of attention distribution to measure focus/spread.

        Args:
            attention_weights: Attention weights [n_municipalities, n_edars]

        Returns:
            Mean entropy value (scalar)
        """
        # Validate input shape
        assert attention_weights.ndim == 2, (
            f"Attention weights should be 2D, got {attention_weights.shape}"
        )

        # Add small epsilon to avoid log(0)
        eps = 1e-8

        # Compute entropy using einsum for clarity
        # attention_weights: [n_municipalities, n_edars]
        # entropy: [n_municipalities]
        entropy = -einsum(
            attention_weights,
            torch.log(attention_weights + eps),
            "municipalities edars, municipalities edars -> municipalities",
        )

        # Return mean entropy across all municipalities
        mean_entropy = entropy.mean()
        return mean_entropy


class MultiScaleAttentionMask(nn.Module):
    """
    Multi-scale attention mask that captures different spatial ranges of EDAR influence.
    """

    def __init__(
        self,
        base_processor: AttentionMaskProcessor,
        num_scales: int = 3,
        scale_fusion: str = "concat",
    ):
        """
        Initialize multi-scale attention mask.

        Args:
            base_processor: Base attention mask processor
            num_scales: Number of spatial scales to consider
            scale_fusion: How to combine scales ('concat', 'sum', 'weighted')
        """
        super().__init__()

        self.base_processor = base_processor
        self.num_scales = num_scales
        self.scale_fusion = scale_fusion

        # Scale-specific processors
        self.scale_processors = nn.ModuleList(
            [
                AttentionMaskProcessor(
                    edar_embedding_dim=base_processor.edar_embedding_dim,
                    municipality_embedding_dim=base_processor.municipality_embedding_dim,
                    use_learnable_weights=True,
                    temperature=1.0
                    * (scale + 1),  # Different temperatures for different scales
                    dropout=base_processor.dropout.p,
                )
                for scale in range(num_scales)
            ]
        )

        # Fusion layers
        if scale_fusion == "concat":
            self.fusion_layer = nn.Linear(
                base_processor.municipality_embedding_dim * num_scales,
                base_processor.municipality_embedding_dim,
            )
        elif scale_fusion == "weighted":
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(
        self,
        edar_embeddings: torch.Tensor,
        contribution_mask: torch.Tensor,
        municipality_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Apply multi-scale attention masking.

        Args:
            edar_embeddings: EDAR embeddings
            contribution_mask: Base contribution mask
            municipality_embeddings: Optional municipality embeddings

        Returns:
            Multi-scale aggregated signals and statistics
        """
        scale_outputs = []
        all_stats = {}

        for i, processor in enumerate(self.scale_processors):
            # Apply different spatial scales by modifying the mask
            scale_mask = self._apply_spatial_scale(contribution_mask, scale=i)

            # Process at this scale
            scale_output, scale_stats = processor(
                edar_embeddings, scale_mask, municipality_embeddings
            )

            scale_outputs.append(scale_output)
            all_stats[f"scale_{i}"] = scale_stats

        # Fuse scales
        if self.scale_fusion == "concat":
            fused = torch.cat(scale_outputs, dim=-1)
            output = self.fusion_layer(fused)
        elif self.scale_fusion == "sum":
            output = sum(scale_outputs)
        elif self.scale_fusion == "weighted":
            weights = F.softmax(self.scale_weights, dim=0)
            output = sum(w * out for w, out in zip(weights, scale_outputs))
        else:
            output = scale_outputs[0]  # Default to first scale

        return output, all_stats

    def _apply_spatial_scale(self, mask: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Modify mask to capture different spatial scales.

        Args:
            mask: Original contribution mask
            scale: Scale level (0 = local, higher = broader)

        Returns:
            Modified mask for this scale
        """
        if scale == 0:
            # Local scale - keep original
            return mask

        # Broader scales - expand influence
        # Apply Gaussian-like smoothing to expand spatial influence
        2 * scale + 1

        # Simple averaging kernel for demonstration
        # In practice, could use more sophisticated spatial kernels
        expanded_mask = mask.clone()

        # Increase influence of nearby EDARs
        for _ in range(scale):
            # Simple diffusion-like operation
            expanded_mask = 0.8 * expanded_mask + 0.2 * expanded_mask.mean(
                dim=1, keepdim=True
            )

        # Renormalize
        row_sums = expanded_mask.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        expanded_mask = expanded_mask / row_sums

        return expanded_mask


if __name__ == "__main__":
    # Test the attention mask processor
    logging.basicConfig(level=logging.INFO)

    # Create dummy data
    n_edars = 59
    n_municipalities = 200
    edar_embed_dim = 64
    muni_embed_dim = 128

    # Dummy embeddings
    edar_embeddings = torch.randn(n_edars, edar_embed_dim)
    muni_embeddings = torch.randn(n_municipalities, muni_embed_dim)

    # Dummy contribution mask (sparse)
    contribution_mask = torch.zeros(n_municipalities, n_edars)
    # Each municipality connected to 1-3 EDARs
    for i in range(n_municipalities):
        n_connections = torch.randint(1, 4, (1,)).item()
        connected_edars = torch.randperm(n_edars)[:n_connections]
        weights = torch.rand(n_connections)
        weights = weights / weights.sum()  # Normalize
        contribution_mask[i, connected_edars] = weights

    # Test basic processor
    processor = AttentionMaskProcessor(
        edar_embedding_dim=edar_embed_dim,
        municipality_embedding_dim=muni_embed_dim,
        use_learnable_weights=True,
        fusion_method="weighted",
    )

    output, stats = processor(edar_embeddings, contribution_mask, muni_embeddings)
    print(f"Output shape: {output.shape}")
    print(f"Attention stats: {stats.keys()}")
    print(f"Effective EDARs per municipality: {stats['effective_edars']:.2f}")

    # Test multi-scale processor
    multi_processor = MultiScaleAttentionMask(
        base_processor=processor, num_scales=3, scale_fusion="concat"
    )

    multi_output, multi_stats = multi_processor(
        edar_embeddings, contribution_mask, muni_embeddings
    )
    print(f"\nMulti-scale output shape: {multi_output.shape}")
    print(f"Multi-scale stats: {list(multi_stats.keys())}")

    print("\nAttention mask processor test completed successfully!")
