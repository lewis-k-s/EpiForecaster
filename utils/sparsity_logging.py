"""Sparsity-loss correlation logging utilities for interpretability.

Computes per-sample sparsity metrics from batch masks and per-sample losses
from model outputs, enabling analysis of how input missingness correlates
with prediction quality across observation heads.

Used during training for W&B summary statistics logging at grad_norm_log_frequency intervals.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def compute_batch_sparsity(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Compute per-sample sparsity for all input and target sources.

    Sparsity is defined as 1.0 - (observed_count / total_timesteps).
    Returns values in [0, 1] where 0 = fully observed, 1 = fully missing.

    Args:
        batch: Collated batch from EpiDataset with keys:
            - HospHist, DeathsHist, CasesHist: (B, L, 3) with mask at index 1
            - BioNode: (B, L, bio_dim) with mask channels per variant
            - HospTargetMask, WWTargetMask, etc.: (B, H) target observation masks
            - MobBatch: PyG batch with edge_weight for mobility sparsity

    Returns:
        Dict mapping source names to (B,) tensors of per-sample sparsity:
            - History sparsity: hosp_hist, deaths_hist, cases_hist, bio_hist, mob_hist
            - Target sparsity: hosp_target, ww_target, cases_target, deaths_target
    """
    sparsity: dict[str, torch.Tensor] = {}

    # History sparsity from 3-channel clinical series [value, mask, age]
    # Mask is at index 1, sparsity = 1 - mean(mask)
    for name, key in [
        ("hosp_hist", "HospHist"),
        ("deaths_hist", "DeathsHist"),
        ("cases_hist", "CasesHist"),
    ]:
        if key in batch and batch[key] is not None:
            hist = batch[key]
            if hist.dim() == 3 and hist.shape[-1] >= 2:
                mask = hist[:, :, 1].float()
                sparsity[name] = 1.0 - mask.mean(dim=1)
            else:
                B = batch.get("B", hist.shape[0])
                sparsity[name] = torch.zeros(B, device=hist.device)

    # Biomarker sparsity from BioNode tensor
    # Layout: 4 channels per variant (value, mask, censor, age) + has_data
    # Mask is at index 1 for each variant, plus index 4 for has_data
    if "BioNode" in batch and batch["BioNode"] is not None:
        bio = batch["BioNode"]
        B = batch.get("B", bio.shape[0])
        if bio.dim() == 3 and bio.shape[-1] >= 4:
            # Channels are [value, mask, censor, age] per variant + has_data
            # Use mask channels (indices 1, 5, 9, ...) for sparsity
            num_channels = bio.shape[-1]
            mask_indices = list(range(1, num_channels, 4))
            if mask_indices:
                masks = bio[:, :, mask_indices].float()
                bio_sparsity = 1.0 - masks.mean(dim=[1, 2])
                sparsity["bio_hist"] = bio_sparsity
            else:
                sparsity["bio_hist"] = torch.zeros(B, device=bio.device)
        else:
            sparsity["bio_hist"] = torch.zeros(B, device=bio.device)

    # Mobility sparsity from edge weights
    # Count edges per sample in the batched graph
    if "MobBatch" in batch:
        mob_batch = batch["MobBatch"]
        B = batch.get("B", 1)
        T = batch.get("T", 1)
        device = (
            mob_batch.x.device
            if hasattr(mob_batch, "x") and mob_batch.x is not None
            else "cpu"
        )

        if hasattr(mob_batch, "batch") and mob_batch.batch is not None:
            # PyG Batch: count edges per sample
            edge_batch = mob_batch.batch[mob_batch.edge_index[0]]
            edge_counts = torch.zeros(B * T, device=device)
            ones = torch.ones(mob_batch.edge_index.shape[1], device=device)
            edge_counts.scatter_add_(0, edge_batch, ones)

            # Normalize by expected edges (roughly num_nodes^2 per time step)
            # For now, use relative sparsity: 1 - (actual / max_in_batch)
            max_edges = edge_counts.max().clamp(min=1.0)
            mob_sparsity_per_graph = 1.0 - edge_counts / max_edges

            # Average over time steps per sample
            mob_sparsity = mob_sparsity_per_graph.view(B, T).mean(dim=1)
            sparsity["mob_hist"] = mob_sparsity
        else:
            sparsity["mob_hist"] = torch.zeros(B, device=device)

    # Target sparsity from forecast horizon masks
    for name, key in [
        ("hosp_target", "HospTargetMask"),
        ("ww_target", "WWTargetMask"),
        ("cases_target", "CasesTargetMask"),
        ("deaths_target", "DeathsTargetMask"),
    ]:
        if key in batch and batch[key] is not None:
            mask = batch[key].float()
            sparsity[name] = 1.0 - mask.mean(dim=1)
        else:
            B = batch.get("B", 1)
            device = mask.device if "mask" in dir() else "cpu"
            sparsity[name] = torch.zeros(B, device=device)

    return sparsity


def compute_per_sample_head_losses(
    model_outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    """Compute per-sample MSE loss for each observation head.

    Uses the same masked loss logic as JointInferenceLoss but returns
    per-sample losses instead of a scalar.

    Args:
        model_outputs: Dict from EpiForecaster.forward() with pred_hosp, pred_ww, etc.
        targets: Dict with hosp, hosp_mask, ww, ww_mask, etc.

    Returns:
        Dict mapping head names to (B,) tensors of per-sample losses:
            - loss_hosp, loss_ww, loss_cases, loss_deaths
    """
    losses: dict[str, torch.Tensor] = {}

    head_pairs = [
        ("hosp", "pred_hosp", "hosp", "hosp_mask"),
        ("ww", "pred_ww", "ww", "ww_mask"),
        ("cases", "pred_cases", "cases", "cases_mask"),
        ("deaths", "pred_deaths", "deaths", "deaths_mask"),
    ]

    for head_name, pred_key, target_key, mask_key in head_pairs:
        pred = model_outputs.get(pred_key)
        target = targets.get(target_key)
        mask = targets.get(mask_key)

        if pred is None or target is None:
            continue

        if mask is None:
            mask = torch.ones_like(target)

        pred_f = pred.float()
        target_f = target.float()
        mask_f = mask.float()

        finite_mask = torch.isfinite(target_f).float()
        effective_mask = mask_f * finite_mask

        # Zero out non-finite targets to prevent NaN propagation
        target_clean = torch.where(finite_mask.bool(), target_f, pred_f)
        sq_err = (pred_f - target_clean) ** 2
        masked_sq_err = sq_err * effective_mask

        per_sample_sum = masked_sq_err.sum(dim=1)
        per_sample_count = effective_mask.sum(dim=1).clamp(min=1.0)
        per_sample_loss = per_sample_sum / per_sample_count

        losses[f"loss_{head_name}"] = per_sample_loss

    return losses


def log_sparsity_loss_correlation(
    batch: dict[str, Any],
    model_outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor | None],
    wandb_run,
    step: int,
    epoch: int = 0,
) -> None:
    """Log sparsity vs loss summary statistics for each observation head.

    Computes the product of input sparsity and per-sample loss for each head,
    then logs summary statistics (mean, std, p25, p50, p75, max) as scalar
    metrics. These appear as line charts in W&B UI showing how the sparsity-loss
    relationship evolves over training.

    Args:
        batch: Collated batch from EpiDataset
        model_outputs: Dict from EpiForecaster.forward()
        targets: Dict with observation targets and masks
        wandb_run: Active W&B run (None to skip logging)
        step: Global training step for x-axis
        epoch: Current epoch number (unused, kept for API compatibility)
    """
    if wandb_run is None:
        return

    sparsity = compute_batch_sparsity(batch)
    losses = compute_per_sample_head_losses(model_outputs, targets)

    if not sparsity and not losses:
        return

    import wandb

    # Map observation heads to their input sparsity source
    # hosp head <- hosp_hist sparsity
    # ww head <- bio_hist sparsity (biomarkers are WW input)
    # cases head <- cases_hist sparsity
    # deaths head <- deaths_hist sparsity
    head_to_sparsity = {
        "hosp": "hosp_hist",
        "ww": "bio_hist",
        "cases": "cases_hist",
        "deaths": "deaths_hist",
    }

    stats: dict[str, float] = {}

    for head_name, sparsity_key in head_to_sparsity.items():
        loss_key = f"loss_{head_name}"
        if loss_key not in losses or sparsity_key not in sparsity:
            continue

        loss_vals = losses[loss_key]
        sparsity_vals = sparsity[sparsity_key]

        # Compute product: sparsity * loss
        product = sparsity_vals * loss_vals

        # Compute summary statistics
        stats[f"sparsity_loss_{head_name}_mean"] = product.mean().item()
        stats[f"sparsity_loss_{head_name}_std"] = product.std().item()

        # Percentiles using torch.quantile
        percentiles = torch.tensor([0.25, 0.5, 0.75], device=product.device)
        if product.numel() > 0:
            quantiles = torch.quantile(product, percentiles)
            stats[f"sparsity_loss_{head_name}_p25"] = quantiles[0].item()
            stats[f"sparsity_loss_{head_name}_p50"] = quantiles[1].item()
            stats[f"sparsity_loss_{head_name}_p75"] = quantiles[2].item()
        else:
            stats[f"sparsity_loss_{head_name}_p25"] = 0.0
            stats[f"sparsity_loss_{head_name}_p50"] = 0.0
            stats[f"sparsity_loss_{head_name}_p75"] = 0.0

        stats[f"sparsity_loss_{head_name}_max"] = product.max().item()

    if stats:
        wandb.log(stats, step=step)
