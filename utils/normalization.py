import torch


def unscale_forecasts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unscale predictions and targets using the provided mean and scale.

    Args:
        predictions: Normalized predictions tensor (B, H) or (B, H, 1)
        targets: Normalized targets tensor (B, H) or (B, H, 1)
        target_mean: Mean used for normalization (B, 1)
        target_scale: Scale (std) used for normalization (B, 1)

    Returns:
        tuple[Tensor, Tensor]: Unscaled (predictions, targets)
    """
    # Ensure mean/scale broadcast correctly against (B, H)
    # target_mean/scale are typically (B, 1)
    if target_mean.ndim == 1:
        target_mean = target_mean.unsqueeze(-1)
    if target_scale.ndim == 1:
        target_scale = target_scale.unsqueeze(-1)

    # If predictions/targets are (B, H), we want (B, H) output.
    # If they are (B, H, 1), we want (B, H, 1).
    # scale/mean are (B, 1).
    # If pred is (B, H), broadcasting (B, 1) works for the last dim if H is the last dim?
    # No, (B, H) and (B, 1) broadcast to (B, H).

    # However, safe approach is to align dimensions.
    # We assume standard shape from model is (B, H).
    # But sometimes extended with Feature dim (B, H, F).
    # Let's align on the assumption that scale/mean apply to the sample (B)
    # and broadcast across time (H).

    # Case 1: Inputs are (B, H), stats are (B, 1).
    # Broadcasting: (B, H) op (B, 1) -> (B, H). Correct.

    # Case 2: Inputs are (B, H), stats show up as (B,).
    # We handled the unsqueeze above.

    pred_unscaled = predictions * target_scale + target_mean
    targets_unscaled = targets * target_scale + target_mean

    return pred_unscaled, targets_unscaled
