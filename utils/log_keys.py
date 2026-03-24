"""Shared metric-key builders for logging."""

from __future__ import annotations

OBSERVATION_HEADS: tuple[str, ...] = ("ww", "hosp", "cases", "deaths")
JOINT_LOSS_COMPONENTS: tuple[str, ...] = (*OBSERVATION_HEADS, "sir")
CORE_EVAL_METRICS: tuple[str, ...] = ("mae", "rmse", "r2")

GRAD_SNAPSHOT_HEAD_FIELDS: tuple[str, ...] = (
    "active",
    "n_eff",
    "grad_norm",
    "valid_points",
    "valid_series",
    "expected_zero",
    "unexpected_zero",
    "vanishing_layers",
    "pass_rate",
    "zero_when_active_rate",
    "zero_when_inactive_rate",
)

TENSORBOARD_SCALARS: dict[str, str] = {
    "loss_train": "Loss/Train_step",
    "learning_rate_step": "Learning_Rate/step",
    "gradnorm_total_preclip": "GradNorm/Total_PreClip",
    "gradnorm_mobility_gnn": "GradNorm/MobilityGNN",
    "gradnorm_backbone": "GradNorm/Backbone",
    "gradnorm_other": "GradNorm/Other",
    "gradnorm_clipped_total": "GradNorm/Clipped_Total",
    "time_dataload_s": "Time/DataLoad_s",
    "epoch": "epoch",
    "train_sparsity": "Train/Sparsity",
}


def build_loss_key(
    *,
    split: str | None = None,
    component: str | None = None,
    weighted: bool = False,
) -> str:
    """Build canonical loss metric keys."""
    parts = ["loss"]
    if split is not None:
        parts.append(split)
    if component is not None:
        parts.append(component)
    key = "_".join(parts)
    if weighted:
        key = f"{key}_weighted"
    return key


def build_eval_metric_key(metric_name: str, split: str) -> str:
    """Build keys like ``mae_val`` or ``rmse_test``."""
    return f"{metric_name}_{split}"


def build_horizon_metric_key(metric_name: str, split: str, label: str) -> str:
    """Build keys like ``mae_val_h1`` or ``rmse_test_w2``."""
    return f"{metric_name}_{split}_{label}"


def build_curriculum_metric_key(metric_name: str, scope: str) -> str:
    """Build curriculum keys like ``train_sparsity_epoch``."""
    return f"train_{metric_name}_{scope}"


def build_gradnorm_obs_key(head: str) -> str:
    """Build per-observation-head gradnorm keys."""
    return f"gradnorm_obs_{head}"


def build_grad_snapshot_head_key(head: str, field: str) -> str:
    """Build per-head gradient snapshot keys."""
    return f"grad_snapshot_head_{head}_{field}"


def parse_grad_snapshot_head_key(key: str) -> tuple[str, str] | None:
    """Parse ``grad_snapshot_head_<head>_<field>`` keys into structured parts."""
    prefix = "grad_snapshot_head_"
    if not key.startswith(prefix):
        return None

    remainder = key.removeprefix(prefix)
    for field in sorted(GRAD_SNAPSHOT_HEAD_FIELDS, key=len, reverse=True):
        suffix = f"_{field}"
        if remainder.endswith(suffix):
            head = remainder.removesuffix(suffix)
            if head in OBSERVATION_HEADS:
                return head, field
    return None


def infer_observation_head_from_name(name: str) -> str | None:
    """Infer the observation head alias from a parameter or metric name."""
    for head in OBSERVATION_HEADS:
        if name.startswith(f"{head}_head.") or f".{head}_head." in name:
            return head
    return None
