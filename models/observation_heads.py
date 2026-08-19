"""
Observation Heads for Joint Inference-Observation Framework.

Implements differentiable observation layers that map latent SIR states (I trajectory)
to observable proxy signals via delay kernels and shedding convolution.

These are the emission models (Stage 3) of the Joint Inference architecture.
Learnable kernels and rates are constrained to be positive (softplus/exp) and
kernels are normalized to sum to 1 for probabilistic interpretation.
"""

import logging
import math
from contextlib import nullcontext
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.precision_policy import MODEL_PARAM_DTYPE, model_scalar
from utils.normalization import unscale_forecasts
from utils.device import sync_to_device

logger = logging.getLogger(__name__)


_KERNEL_EPS = 1e-8
_EXP_CLAMP_MIN = -20.0
_EXP_CLAMP_MAX = 20.0
_OBS_CLAMP_MAX = 1.0e12
# Safety clamp for unconstrained kernel logits prior to softplus.
# If we ever see many logits pinned at these bounds, add boundary-hit logging
# to track whether this stabilization starts constraining learned kernels.
_KERNEL_LOGIT_CLAMP_MIN = -30.0
_KERNEL_LOGIT_CLAMP_MAX = 30.0
# Conservative bounds for the two-parameter learnable Gamma delay kernel.
# They keep the reported-delay family epidemiologically plausible
# (mean 3-150 days at these extremes) while preventing runaway shapes.
_GAMMA_SHAPE_MIN = 1.0
_GAMMA_SHAPE_MAX = 30.0
_GAMMA_SCALE_MIN = 0.1
_GAMMA_SCALE_MAX = 5.0
# Wastewater shedding priors need a longer scale box than clinical reporting
# delays: the SARS-CoV-2 shedding literature anchors near a mean of 12 days
# with CV ~0.7 (shape ~2, scale ~6). Shape shares the clinical [1, 30] box so
# the shedding density stays zero at the moment of infection.
_GAMMA_SCALE_WW_MIN = 0.2
_GAMMA_SCALE_WW_MAX = 15.0
_GAMMA_RAW_SANITIZE_LIMIT = 15.0


def _sanitize_free_kernel_weights(weights: torch.Tensor) -> torch.Tensor:
    """Sanitize unconstrained kernel weights before applying them directly."""
    return torch.nan_to_num(
        weights.float(),
        nan=0.0,
        posinf=_KERNEL_LOGIT_CLAMP_MAX,
        neginf=_KERNEL_LOGIT_CLAMP_MIN,
    ).clamp(min=_KERNEL_LOGIT_CLAMP_MIN, max=_KERNEL_LOGIT_CLAMP_MAX)


def _autocast_disabled_for(tensor: torch.Tensor):
    """Disable autocast for numerically sensitive blocks on supported devices."""
    if tensor.device.type in {"cpu", "cuda", "mps"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _safe_exp_from_log(log_param: torch.Tensor) -> torch.Tensor:
    """Exponentiate log-parameter with clamping to avoid overflow."""
    sanitized = torch.nan_to_num(
        log_param.float(),
        nan=0.0,
        posinf=_EXP_CLAMP_MAX,
        neginf=_EXP_CLAMP_MIN,
    )
    clamped = torch.clamp(sanitized, min=_EXP_CLAMP_MIN, max=_EXP_CLAMP_MAX)
    return torch.exp(clamped).clamp_min(_KERNEL_EPS)


def _sanitize_kernel_logits(logits: torch.Tensor) -> torch.Tensor:
    """Sanitize unconstrained kernel logits before softplus."""
    sanitized = torch.nan_to_num(
        logits.float(),
        nan=0.0,
        posinf=_KERNEL_LOGIT_CLAMP_MAX,
        neginf=_KERNEL_LOGIT_CLAMP_MIN,
    )
    return torch.clamp(
        sanitized, min=_KERNEL_LOGIT_CLAMP_MIN, max=_KERNEL_LOGIT_CLAMP_MAX
    )


def _safe_normalize(weights: torch.Tensor) -> torch.Tensor:
    """Normalize positive weights with clamped denominator for stability."""
    weights_f32 = torch.nan_to_num(
        weights.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    denom = weights_f32.sum().clamp_min(_KERNEL_EPS)
    return weights_f32 / denom


def _inverse_softplus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse softplus transform for positive tensors."""
    x = torch.clamp(x, min=eps)
    return torch.log(torch.expm1(x).clamp_min(eps))


def _small_xavier_linear(layer: nn.Linear, gain: float = 0.1) -> None:
    """Initialize a linear layer with small non-zero Xavier weights."""
    nn.init.xavier_normal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _inverse_softplus_scalar(value: float, eps: float = 1e-8) -> float:
    """Scalar inverse softplus used for positive projection biases."""
    return float(_inverse_softplus(model_scalar(value), eps=eps).item())


def _sanitize_gamma_raw(raw: torch.Tensor) -> torch.Tensor:
    """Bound Gamma coordinates while retaining finite-coordinate recovery gradients."""
    finite_raw = torch.nan_to_num(
        raw.float(),
        nan=0.0,
        posinf=_GAMMA_RAW_SANITIZE_LIMIT,
        neginf=-_GAMMA_RAW_SANITIZE_LIMIT,
    )
    bounded_raw = finite_raw.clamp(
        min=-_GAMMA_RAW_SANITIZE_LIMIT,
        max=_GAMMA_RAW_SANITIZE_LIMIT,
    )
    # Keep forward values in the range where sigmoid is representable in
    # float32. The straight-through derivative lets optimization recover a
    # finite coordinate that was loaded beyond this safety limit.
    return finite_raw + (bounded_raw - finite_raw).detach()


def _bounded_sigmoid(raw: torch.Tensor, *, lower: float, upper: float) -> torch.Tensor:
    """Map a finite raw coordinate smoothly into the open interval ``(lower, upper)``."""
    return lower + (upper - lower) * torch.sigmoid(_sanitize_gamma_raw(raw))


def _inverse_bounded_sigmoid(value: float, *, lower: float, upper: float) -> float:
    """Return the raw coordinate whose bounded sigmoid is exactly ``value``."""
    if not math.isfinite(value) or not lower < value < upper:
        raise ValueError(
            f"Gamma prior must be finite and strictly inside ({lower}, {upper}), "
            f"got {value}"
        )
    fraction = (value - lower) / (upper - lower)
    return math.log(fraction) - math.log1p(-fraction)


def _normalized_gamma_pdf_kernel(
    shape: torch.Tensor, scale: torch.Tensor, length: int
) -> torch.Tensor:
    """Compute the normalized Gamma PDF kernel at integer lags ``0..length-1``.

    Differentiable with respect to ``shape`` and ``scale`` so the
    two-parameter Gamma kernel trains end to end.
    """
    x = torch.arange(length, device=shape.device, dtype=torch.float32)

    # Gamma PDF: x^(shape-1) * exp(-x/scale) / (scale^shape * Gamma(shape))
    # Softmax normalizes its log PDF without a hard cutoff, preserving
    # gradients when the finite support catches only a distribution tail.
    log_gamma_pdf = (
        (shape - 1) * torch.log(x + _KERNEL_EPS)  # avoid log(0)
        - x / scale  # exp(-x/scale)
        - shape * torch.log(scale)  # scale^shape
        - torch.lgamma(shape)  # Gamma(shape)
    )
    return torch.softmax(
        torch.nan_to_num(
            log_gamma_pdf,
            nan=0.0,
            posinf=_EXP_CLAMP_MAX,
            neginf=_EXP_CLAMP_MIN,
        ),
        dim=0,
    )


def _gamma_boundary_hit_flags(
    shape: float,
    scale: float,
    *,
    shape_bounds: tuple[float, float],
    scale_bounds: tuple[float, float],
    atol: float,
) -> dict[str, bool]:
    """Report Gamma boundary proximity in normalized open-interval coordinates."""
    shape_coordinate = (shape - shape_bounds[0]) / (shape_bounds[1] - shape_bounds[0])
    scale_coordinate = (scale - scale_bounds[0]) / (scale_bounds[1] - scale_bounds[0])
    return {
        "shape_lower": shape_coordinate <= atol,
        "shape_upper": shape_coordinate >= 1.0 - atol,
        "scale_lower": scale_coordinate <= atol,
        "scale_upper": scale_coordinate >= 1.0 - atol,
    }


def _init_distribution_scale_proj(
    layer: nn.Linear,
    *,
    scale_init: float,
    scale_floor: float,
) -> None:
    """Initialize a positive scale projection to emit ``scale_init`` at startup."""
    margin = max(float(scale_init) - float(scale_floor), _KERNEL_EPS)
    nn.init.zeros_(layer.weight)
    with torch.no_grad():
        layer.bias.fill_(_inverse_softplus_scalar(margin))


# Sentinel stored in the residual-log-location budget buffer: the runtime
# schedule is inactive and the legacy unrestricted residual path applies.
_RESIDUAL_LOG_LOCATION_FRACTION_UNBOUNDED = -1.0


def _apply_residual_log_location_fraction_bound(
    residual_log_location: torch.Tensor,
    mechanistic_log_location: torch.Tensor,
    max_log_location_fraction_buffer: torch.Tensor,
) -> torch.Tensor:
    """Bound residual log location relative to mechanistic log location.

    ``max_log_location_fraction_buffer`` is a scalar runtime buffer: the
    unbounded sentinel restores the unrestricted legacy residual, ``0.0`` is
    an exact residual lockout, and values in ``(0, 1]`` cap the contribution
    at ``max_log_location_fraction * |mechanistic_log_location|`` through a
    tanh squash. The limit is detached, so the bound never trades
    mechanistic-path gradients for residual headroom. A strict relative bound
    means a zero mechanistic log location admits no residual correction.
    Clinical log locations use log1p(per-100k) units.
    """
    fraction = max_log_location_fraction_buffer.clamp_min(0.0)
    limit = fraction * mechanistic_log_location.detach().abs()
    safe_limit = torch.where(limit > 0.0, limit, torch.ones_like(limit))
    bounded = limit * torch.tanh(residual_log_location / safe_limit)
    budgeted = torch.where(
        fraction == 0.0,
        torch.zeros_like(residual_log_location),
        bounded,
    )
    return torch.where(
        max_log_location_fraction_buffer < 0.0,
        residual_log_location,
        budgeted,
    )


def _bound_composed_residual(
    components: dict[str, torch.Tensor],
    max_log_location_fraction_buffer: torch.Tensor,
) -> None:
    """Bound the composed residual contribution to the prediction log location.

    Applies the same detached relative bound to the actual anchored residual
    contribution in ``residual_loc`` and recomposes ``loc`` so the exact
    decomposition ``loc = mechanistic_loc + residual_loc`` is preserved. The
    unbounded sentinel keeps the original composition bit-identical; the
    branch is a tensor select so compiled graphs never specialize on the
    runtime budget value.
    """
    mechanistic_log_location = components["mechanistic_loc"]
    with _autocast_disabled_for(mechanistic_log_location):
        bounded = _apply_residual_log_location_fraction_bound(
            residual_log_location=components["residual_loc"],
            mechanistic_log_location=mechanistic_log_location,
            max_log_location_fraction_buffer=max_log_location_fraction_buffer,
        )
        recomposed_loc = mechanistic_log_location + bounded
    unbounded = max_log_location_fraction_buffer < 0.0
    components["loc"] = torch.where(unbounded, components["loc"], recomposed_loc)
    components["residual_loc"] = torch.where(
        unbounded, components["residual_loc"], bounded
    )


def _register_residual_log_location_fraction_buffer(module: nn.Module) -> None:
    """Register the non-persistent residual log-location budget buffer."""
    module.register_buffer(
        "_residual_max_log_location_fraction",
        model_scalar(_RESIDUAL_LOG_LOCATION_FRACTION_UNBOUNDED),
        persistent=False,
    )


def _set_residual_log_location_fraction_buffer(
    module: nn.Module,
    max_log_location_fraction: float | None,
) -> None:
    """Validate and copy a residual log-location budget into the runtime buffer."""
    if max_log_location_fraction is None:
        value = _RESIDUAL_LOG_LOCATION_FRACTION_UNBOUNDED
    else:
        if (
            not math.isfinite(max_log_location_fraction)
            or not 0.0 <= max_log_location_fraction <= 1.0
        ):
            raise ValueError(
                "residual max_log_location_fraction must be a finite value in "
                f"[0, 1] or None, got {max_log_location_fraction!r}"
            )
        value = float(max_log_location_fraction)
    buffer = module.get_buffer("_residual_max_log_location_fraction")
    with torch.no_grad():
        buffer.fill_(value)


def _compose_observation_prediction(
    *,
    mechanistic_raw: torch.Tensor,
    residual_raw: torch.Tensor,
    delta_forecasting: bool,
    anchor_mode: str,
    last_observed: torch.Tensor | None,
    last_observed_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Compose an exact mechanistic-plus-residual observation decomposition.

    For anchored delta forecasts, the mechanistic output is the prediction obtained
    with the residual path disabled. The residual contribution is defined as the
    exact difference between the full and residual-disabled predictions, including
    interactions with the non-negativity clamp.
    """
    full_raw = mechanistic_raw + residual_raw
    full = full_raw
    mechanistic = mechanistic_raw

    if (
        delta_forecasting
        and anchor_mode != "disabled"
        and last_observed is not None
        and last_observed_mask is not None
    ):
        valid_mask = (last_observed_mask > 0.5).unsqueeze(1)
        last_obs_safe = torch.nan_to_num(
            last_observed,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).unsqueeze(1)
        full_anchored = torch.clamp(
            last_obs_safe + full_raw - full_raw[:, 0:1],
            min=0.0,
        )
        mechanistic_anchored = torch.clamp(
            last_obs_safe + mechanistic_raw - mechanistic_raw[:, 0:1],
            min=0.0,
        )
        full = torch.where(valid_mask, full_anchored, full_raw)
        mechanistic = torch.where(
            valid_mask,
            mechanistic_anchored,
            mechanistic_raw,
        )

    residual = full - mechanistic
    return {
        "loc": full,
        "mechanistic_loc": mechanistic,
        "residual_loc": residual,
    }


class DelayKernel(nn.Module):
    """
    Causal delay kernel for clinical observations (cases, hospitalizations, deaths).

    Models the time delay between infection (I_t) and clinical observation using
    a 1D convolution with a causal, normalized kernel. The kernel represents the
    probability distribution of time from infection to observation.

    Physics:
        - Causal: Output at time t only depends on I_{t-k:t} (no future leakage)
        - Normalized: Kernel sums to 1 (probabilistic interpretation)
        - Gamma-distributed: ``parameterization="gamma"`` uses a bounded,
          two-parameter Gamma(shape, scale) family

    Default initialization (Gamma(5, 2)):
        - Mean delay: shape × scale = 10 days
        - Appropriate for hospitalization delay from infection
        - Supports case reporting delays (~7 days) and death delays (~21 days)

    Args:
        kernel_length: Length of delay kernel (days of history to consider)
        gamma_shape: Shape parameter for Gamma distribution initialization
        gamma_scale: Scale parameter for Gamma distribution initialization
        learnable: If True, kernel weights are trainable; if False, frozen
        parameterization: Kernel parameterization mode. "simplex" enforces
            positivity and sum-to-one normalization; "free" learns unconstrained
            signed weights initialized from the same gamma prior; "gamma" learns
            a two-parameter Gamma(shape, scale) delay distribution whose PDF is
            re-normalized over the kernel support at every forward pass. Its
            shape and scale use a smooth transform into conservative
            epidemiological bounds.
    """

    def __init__(
        self,
        kernel_length: int = 21,
        gamma_shape: float = 5.0,
        gamma_scale: float = 2.0,
        learnable: bool = True,
        parameterization: str = "simplex",
    ):
        super().__init__()
        self.kernel_length = kernel_length
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.learnable = learnable
        self.parameterization = parameterization

        if self.parameterization not in {"simplex", "free", "gamma"}:
            raise ValueError(
                "parameterization must be one of {'simplex', 'free', 'gamma'}, "
                f"got {self.parameterization!r}"
            )

        if learnable and self.parameterization == "gamma":
            # Two raw coordinates transform smoothly into the bounded Gamma
            # family. The logit inverse preserves the configured prior exactly.
            self.gamma_shape_raw = nn.Parameter(
                model_scalar(
                    _inverse_bounded_sigmoid(
                        gamma_shape,
                        lower=_GAMMA_SHAPE_MIN,
                        upper=_GAMMA_SHAPE_MAX,
                    )
                )
            )
            self.gamma_scale_raw = nn.Parameter(
                model_scalar(
                    _inverse_bounded_sigmoid(
                        gamma_scale,
                        lower=_GAMMA_SCALE_MIN,
                        upper=_GAMMA_SCALE_MAX,
                    )
                )
            )
        else:
            # Initialize kernel weights with Gamma distribution
            kernel_weights = self._init_gamma_kernel(
                kernel_length, gamma_shape, gamma_scale
            )

            if learnable:
                if self.parameterization == "simplex":
                    kernel_param = _inverse_softplus(kernel_weights)
                else:
                    kernel_param = kernel_weights
                self.kernel = nn.Parameter(kernel_param)
            elif self.parameterization == "gamma":
                self.register_buffer("gamma_kernel", kernel_weights)
                self.register_buffer("gamma_shape_value", model_scalar(gamma_shape))
                self.register_buffer("gamma_scale_value", model_scalar(gamma_scale))
            else:
                self.register_buffer("kernel", kernel_weights)

        logger.debug(
            f"Initialized DelayKernel: length={kernel_length}, "
            f"gamma({gamma_shape}, {gamma_scale}), learnable={learnable}, "
            f"parameterization={parameterization}"
        )

    def _init_gamma_kernel(
        self, length: int, shape: float, scale: float
    ) -> torch.Tensor:
        """Initialize kernel with Gamma distribution PDF values."""
        return self._compute_gamma_kernel(model_scalar(shape), model_scalar(scale))

    def _compute_gamma_kernel(
        self, shape: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Compute the normalized Gamma PDF kernel at integer lags."""
        return _normalized_gamma_pdf_kernel(shape, scale, self.kernel_length)

    def _current_gamma_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return finite Gamma parameters strictly inside their epidemiological bounds."""
        shape = _bounded_sigmoid(
            self.gamma_shape_raw,
            lower=_GAMMA_SHAPE_MIN,
            upper=_GAMMA_SHAPE_MAX,
        )
        scale = _bounded_sigmoid(
            self.gamma_scale_raw,
            lower=_GAMMA_SCALE_MIN,
            upper=_GAMMA_SCALE_MAX,
        )
        return shape, scale

    def forward(self, I_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Apply causal delay kernel to infection trajectory.

        Args:
            I_trajectory: Infected population trajectory [batch_size, time_steps]
                Should include I_0 at index 0, matching SIR roll-forward output.

        Returns:
            Delayed observations [batch_size, time_steps]
                Output at time t represents expected observations from infections
                at times t-kernel_length+1 through t.

        Note:
            - Uses left-padding to maintain causality at sequence start
            - The first kernel_length-1 outputs will be based on partial history
            - Kernel is re-normalized at each forward pass to ensure it sums to 1
        """
        batch_size, time_steps = I_trajectory.shape

        with _autocast_disabled_for(I_trajectory):
            if self.learnable and self.parameterization == "gamma":
                shape, scale = self._current_gamma_parameters()
                kernel_weights = self._compute_gamma_kernel(shape, scale)
            elif self.learnable and self.parameterization == "simplex":
                kernel_weights = F.softplus(_sanitize_kernel_logits(self.kernel))
                kernel_weights = _safe_normalize(kernel_weights)
            elif self.learnable:
                kernel_weights = _sanitize_free_kernel_weights(self.kernel)
            elif self.parameterization == "gamma":
                kernel_weights = self.gamma_kernel
            else:
                kernel_weights = self.kernel
            flipped_kernel = torch.flip(kernel_weights, dims=[0])

            I_reshaped = torch.nan_to_num(
                I_trajectory.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).unsqueeze(1)
            kernel_reshaped = flipped_kernel.view(1, 1, -1)

            output = F.conv1d(
                I_reshaped,
                kernel_reshaped,
                padding=self.kernel_length - 1,
                groups=1,
            )
            output = output[:, :, :time_steps].squeeze(1)
        return output.to(I_trajectory.dtype)

    def get_kernel_weights(self) -> torch.Tensor:
        """Get current kernel weights in forward-space order."""
        if self.learnable and self.parameterization == "gamma":
            shape, scale = self._current_gamma_parameters()
            kernel_weights = self._compute_gamma_kernel(shape, scale)
            output_dtype = self.gamma_shape_raw.dtype
        elif self.learnable and self.parameterization == "simplex":
            kernel_weights = F.softplus(_sanitize_kernel_logits(self.kernel))
            kernel_weights = _safe_normalize(kernel_weights)
            output_dtype = self.kernel.dtype
        elif self.learnable:
            kernel_weights = _sanitize_free_kernel_weights(self.kernel)
            output_dtype = self.kernel.dtype
        elif self.parameterization == "gamma":
            kernel_weights = self.gamma_kernel
            output_dtype = kernel_weights.dtype
        else:
            kernel_weights = self.kernel
            output_dtype = kernel_weights.dtype
        return kernel_weights.to(output_dtype)

    def get_mean_delay(self) -> float:
        """Compute mean delay in days based on current kernel."""
        if self.parameterization not in {"simplex", "gamma"}:
            raise ValueError("mean delay is only defined for normalized kernels")
        weights = self.get_kernel_weights()
        positions = torch.arange(
            len(weights), device=weights.device, dtype=torch.float32
        )
        return (weights.float() * positions).sum().item()

    def get_delay_std(self) -> float:
        """Compute the delay standard deviation in days of the current kernel."""
        if self.parameterization not in {"simplex", "gamma"}:
            raise ValueError("delay std is only defined for normalized kernels")
        weights = self.get_kernel_weights().float()
        positions = torch.arange(len(weights), device=weights.device)
        mean = (weights * positions).sum()
        return (weights * (positions - mean).square()).sum().sqrt().item()

    def get_gamma_parameters(self) -> tuple[float, float]:
        """Return the current (shape, scale) of a gamma-parameterized kernel."""
        if self.parameterization != "gamma":
            raise ValueError("gamma parameters are only defined for gamma kernels")
        if not self.learnable:
            if hasattr(self, "gamma_shape_value"):
                return (
                    float(self.gamma_shape_value.item()),
                    float(self.gamma_scale_value.item()),
                )
            return float(self.gamma_shape), float(self.gamma_scale)
        shape, scale = self._current_gamma_parameters()
        return float(shape.item()), float(scale.item())

    def get_gamma_boundary_hits(self, *, atol: float = 1.0e-6) -> dict[str, bool]:
        """Report Gamma boundary proximity in normalized open-interval coordinates.

        Learnable Gamma parameters use an open-interior transform, so they never
        exactly equal a closed epidemiological bound. The legacy ``*_bound_hit``
        metric names therefore mean that the normalized coordinate is within
        ``atol`` of zero or one; this gives the same scale-independent definition
        for shape and scale.
        """
        if self.parameterization != "gamma":
            raise ValueError("boundary hits are only defined for gamma kernels")
        if not math.isfinite(atol) or not 0.0 <= atol < 0.5:
            raise ValueError(f"atol must be finite and in [0, 0.5), got {atol}")
        shape, scale = self.get_gamma_parameters()
        return _gamma_boundary_hit_flags(
            shape,
            scale,
            shape_bounds=(_GAMMA_SHAPE_MIN, _GAMMA_SHAPE_MAX),
            scale_bounds=(_GAMMA_SCALE_MIN, _GAMMA_SCALE_MAX),
            atol=atol,
        )

    def __repr__(self) -> str:
        return (
            f"DelayKernel(length={self.kernel_length}, "
            f"gamma({self.gamma_shape}, {self.gamma_scale}), "
            f"learnable={self.learnable}, parameterization={self.parameterization})"
        )


class SheddingConvolution(nn.Module):
    """
    Viral shedding convolution for wastewater observation.

    Models the physical process of viral shedding into wastewater:
        viral_load = conv(I_t, shedding_kernel) × sensitivity_scale / population

    Physics:
        - Shedding: Infected individuals shed virus over time following a profile
        - Dilution: Viral concentration decreases with population (more flow = dilution)
        - Sensitivity: Scale factor absorbs per-capita flow and measurement sensitivity

    Key insight: Population division models DILUTION physics, not per-capita normalization.
        - Village (5k pop): 30 cases = detectable signal
        - Metropolis (500k pop): 30 cases = invisible (100x dilution)

    Shedding kernel (14 days):
        - Day 0-2: Low shedding (incubation)
        - Day 3-7: Peak shedding (acute phase)
        - Day 8-14: Tail shedding (declining)

    Args:
        kernel_length: Length of shedding kernel (days of viral shedding)
        sensitivity_scale: Measurement sensitivity (absorbs FlowPerCapita)
        learnable_kernel: If True, shedding profile is trainable
        learnable_scale: If True, sensitivity scale is trainable
        kernel_parameterization: "simplex" for positive normalized kernels,
            "free" for unconstrained signed kernels with gamma priming,
            "gamma" for a bounded two-parameter Gamma(shape, scale) shedding
            distribution renormalized over the kernel support at every pass
        gamma_shape: Shape prior for the "gamma" shedding family
        gamma_scale: Scale prior (days) for the "gamma" shedding family; the
            wastewater box admits longer scales than the clinical delay box
    """

    def __init__(
        self,
        kernel_length: int = 14,
        sensitivity_scale: float = 1.0,
        learnable_kernel: bool = True,
        learnable_scale: bool = True,
        kernel_parameterization: str = "simplex",
        strict: bool = True,
        gamma_shape: float = 2.0,
        gamma_scale: float = 6.0,
    ):
        super().__init__()
        self.kernel_length = kernel_length
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.learnable_kernel = learnable_kernel
        self.learnable_scale = learnable_scale
        self.parameterization = kernel_parameterization
        self.strict = strict

        if self.parameterization not in {"simplex", "free", "gamma"}:
            raise ValueError(
                "kernel_parameterization must be one of "
                "{'simplex', 'free', 'gamma'}, "
                f"got {self.parameterization!r}"
            )

        if learnable_kernel and self.parameterization == "gamma":
            # Two raw coordinates transform smoothly into the bounded shedding
            # Gamma family. The logit inverse preserves the configured prior.
            self.gamma_shape_raw = nn.Parameter(
                model_scalar(
                    _inverse_bounded_sigmoid(
                        gamma_shape,
                        lower=_GAMMA_SHAPE_MIN,
                        upper=_GAMMA_SHAPE_MAX,
                    )
                )
            )
            self.gamma_scale_raw = nn.Parameter(
                model_scalar(
                    _inverse_bounded_sigmoid(
                        gamma_scale,
                        lower=_GAMMA_SCALE_WW_MIN,
                        upper=_GAMMA_SCALE_WW_MAX,
                    )
                )
            )
        else:
            if self.parameterization == "gamma":
                kernel_weights = _normalized_gamma_pdf_kernel(
                    model_scalar(gamma_shape), model_scalar(gamma_scale), kernel_length
                )
            else:
                # Initialize shedding kernel with biologically-informed profile
                kernel_weights = self._init_shedding_kernel(kernel_length)

            if learnable_kernel:
                if self.parameterization == "simplex":
                    kernel_param = _inverse_softplus(kernel_weights)
                else:
                    kernel_param = kernel_weights
                self.kernel = nn.Parameter(kernel_param)
            elif self.parameterization == "gamma":
                self.register_buffer("gamma_kernel", kernel_weights)
                self.register_buffer("gamma_shape_value", model_scalar(gamma_shape))
                self.register_buffer("gamma_scale_value", model_scalar(gamma_scale))
            else:
                self.register_buffer("kernel", kernel_weights)

        # Sensitivity scale (absorbs FlowPerCapita and measurement characteristics)
        if sensitivity_scale <= 0:
            raise ValueError("sensitivity_scale must be positive")

        if learnable_scale:
            self.sensitivity_scale = nn.Parameter(
                torch.log(model_scalar(sensitivity_scale))
            )
        else:
            self.register_buffer(
                "sensitivity_scale",
                model_scalar(sensitivity_scale),
            )

        logger.debug(
            f"Initialized SheddingConvolution: length={kernel_length}, "
            f"sensitivity_scale={sensitivity_scale:.4f}, "
            f"learnable_kernel={learnable_kernel}, learnable_scale={learnable_scale}, "
            f"kernel_parameterization={kernel_parameterization}"
        )

    def _init_shedding_kernel(self, length: int) -> torch.Tensor:
        """
        Initialize shedding kernel with biologically-informed profile.

        Profile represents viral RNA shedding in feces over days since infection:
        - Days 0-2: Low/ascending (pre-symptomatic, early shedding)
        - Days 3-7: Peak (acute phase, maximum shedding)
        - Days 8-14: Tail (declining, persistent low-level shedding)

        Uses Gamma distribution shifted to create asymmetric profile.
        """
        x = torch.arange(length, dtype=MODEL_PARAM_DTYPE)

        # Use Gamma(3, 2) for ascending then declining profile
        # Shape < scale creates right-skewed distribution (tail)
        shape = 3.0
        scale = 2.0

        # Gamma PDF
        log_gamma_pdf = (
            (shape - 1) * torch.log(x + 1e-8)
            - x / scale
            - shape * math.log(scale)
            - math.lgamma(shape)
        )

        kernel = torch.exp(log_gamma_pdf)
        kernel = _safe_normalize(kernel)

        return kernel

    def forward(
        self, I_trajectory: torch.Tensor, population: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply shedding convolution with dilution physics.

        Args:
            I_trajectory: Infected population trajectory [batch_size, time_steps]
            population: Population time series [batch_size, time_steps]
                Note: Can be constant per batch element or time-varying (mobility-adjusted)

        Returns:
            Viral concentration [batch_size, time_steps]
                Units: arbitrary viral load units (calibrated via sensitivity_scale)

        Formula:
            viral_load_t = sensitivity_scale × conv(I_{t-k:t}, shedding_kernel) / population_t

        Note:
            - Uses causal convolution (no future information)
            - Population in denominator models dilution physics
            - Division by population handles both scalar and time-varying populations
        """
        batch_size, time_steps = I_trajectory.shape

        if self.strict:
            if population.shape == (batch_size,):
                population = population.unsqueeze(1).expand(-1, time_steps)
            elif population.shape != (batch_size, time_steps):
                raise ValueError(
                    "population must be [batch_size, time_steps] or [batch_size]"
                )

            if not torch.all(torch.isfinite(population)):
                raise ValueError("population must be finite and positive")
            if torch.any(population <= 0):
                raise ValueError("population must be positive")
        else:
            if population.shape == (batch_size,):
                population = population.unsqueeze(1).expand(-1, time_steps)

        with _autocast_disabled_for(I_trajectory):
            if self.learnable_kernel and self.parameterization == "gamma":
                shape, scale = self._current_gamma_parameters()
                kernel_weights = self._compute_gamma_kernel(shape, scale)
            elif self.learnable_kernel and self.parameterization == "simplex":
                kernel_weights = F.softplus(_sanitize_kernel_logits(self.kernel))
                kernel_weights = _safe_normalize(kernel_weights)
            elif self.learnable_kernel:
                kernel_weights = _sanitize_free_kernel_weights(self.kernel)
            elif self.parameterization == "gamma":
                kernel_weights = self.gamma_kernel
            else:
                kernel_weights = self.kernel
            flipped_kernel = torch.flip(kernel_weights, dims=[0])

            I_reshaped = torch.nan_to_num(
                I_trajectory.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).unsqueeze(1)
            kernel_reshaped = flipped_kernel.view(1, 1, -1)
            total_shedding = F.conv1d(
                I_reshaped,
                kernel_reshaped,
                padding=self.kernel_length - 1,
                groups=1,
            )
            total_shedding = total_shedding[:, :, :time_steps].squeeze(1)

            population_f32 = population.float().clamp_min(_KERNEL_EPS)
            viral_concentration = total_shedding / population_f32
            viral_concentration = viral_concentration * self.get_sensitivity_scale()
            viral_concentration = torch.nan_to_num(
                viral_concentration,
                nan=0.0,
                posinf=_OBS_CLAMP_MAX,
                neginf=0.0,
            ).clamp(min=0.0, max=_OBS_CLAMP_MAX)

        return viral_concentration.to(I_trajectory.dtype)

    def _compute_gamma_kernel(
        self, shape: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Compute the normalized Gamma PDF kernel at integer lags."""
        return _normalized_gamma_pdf_kernel(shape, scale, self.kernel_length)

    def _current_gamma_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return finite Gamma parameters strictly inside the shedding bounds."""
        shape = _bounded_sigmoid(
            self.gamma_shape_raw,
            lower=_GAMMA_SHAPE_MIN,
            upper=_GAMMA_SHAPE_MAX,
        )
        scale = _bounded_sigmoid(
            self.gamma_scale_raw,
            lower=_GAMMA_SCALE_WW_MIN,
            upper=_GAMMA_SCALE_WW_MAX,
        )
        return shape, scale

    def get_kernel_weights(self) -> torch.Tensor:
        """Get current shedding kernel weights in forward-space order."""
        if self.learnable_kernel and self.parameterization == "gamma":
            shape, scale = self._current_gamma_parameters()
            kernel_weights = self._compute_gamma_kernel(shape, scale)
            output_dtype = self.gamma_shape_raw.dtype
        elif self.learnable_kernel and self.parameterization == "simplex":
            kernel_weights = F.softplus(_sanitize_kernel_logits(self.kernel))
            kernel_weights = _safe_normalize(kernel_weights)
            output_dtype = self.kernel.dtype
        elif self.learnable_kernel:
            kernel_weights = _sanitize_free_kernel_weights(self.kernel)
            output_dtype = self.kernel.dtype
        elif self.parameterization == "gamma":
            kernel_weights = self.gamma_kernel
            output_dtype = kernel_weights.dtype
        else:
            kernel_weights = self.kernel
            output_dtype = kernel_weights.dtype
        return kernel_weights.to(output_dtype)

    def get_mean_delay(self) -> float:
        """Compute mean shedding delay in days."""
        if self.parameterization not in {"simplex", "gamma"}:
            raise ValueError("mean delay is only defined for normalized kernels")
        weights = self.get_kernel_weights()
        positions = torch.arange(
            len(weights), device=weights.device, dtype=torch.float32
        )
        return (weights.float() * positions).sum().item()

    def get_delay_std(self) -> float:
        """Compute the shedding delay standard deviation in days."""
        if self.parameterization not in {"simplex", "gamma"}:
            raise ValueError("delay std is only defined for normalized kernels")
        weights = self.get_kernel_weights().float()
        positions = torch.arange(len(weights), device=weights.device)
        mean = (weights * positions).sum()
        return (weights * (positions - mean).square()).sum().sqrt().item()

    def get_gamma_parameters(self) -> tuple[float, float]:
        """Return the current (shape, scale) of a gamma-parameterized kernel."""
        if self.parameterization != "gamma":
            raise ValueError("gamma parameters are only defined for gamma kernels")
        if not self.learnable_kernel:
            if hasattr(self, "gamma_shape_value"):
                return (
                    float(self.gamma_shape_value.item()),
                    float(self.gamma_scale_value.item()),
                )
            return float(self.gamma_shape), float(self.gamma_scale)
        shape, scale = self._current_gamma_parameters()
        return float(shape.item()), float(scale.item())

    def get_gamma_boundary_hits(self, *, atol: float = 1.0e-6) -> dict[str, bool]:
        """Report Gamma boundary proximity in normalized open-interval coordinates."""
        if self.parameterization != "gamma":
            raise ValueError("boundary hits are only defined for gamma kernels")
        if not math.isfinite(atol) or not 0.0 <= atol < 0.5:
            raise ValueError(f"atol must be finite and in [0, 0.5), got {atol}")
        shape, scale = self.get_gamma_parameters()
        return _gamma_boundary_hit_flags(
            shape,
            scale,
            shape_bounds=(_GAMMA_SHAPE_MIN, _GAMMA_SHAPE_MAX),
            scale_bounds=(_GAMMA_SCALE_WW_MIN, _GAMMA_SCALE_WW_MAX),
            atol=atol,
        )

    def get_sensitivity_scale(self) -> torch.Tensor:
        """Get positive sensitivity scale."""
        if self.learnable_scale:
            return _safe_exp_from_log(self.sensitivity_scale).to(
                self.sensitivity_scale.dtype
            )
        return self.sensitivity_scale.clamp_min(_KERNEL_EPS)

    def get_peak_shedding_day(self) -> int:
        """Get the day of peak shedding based on current kernel."""
        weights = self.get_kernel_weights()
        return int(torch.argmax(weights).item())

    def __repr__(self) -> str:
        return (
            f"SheddingConvolution(length={self.kernel_length}, "
            f"sensitivity_scale={self.get_sensitivity_scale().item():.4f}, "
            f"learnable_kernel={self.learnable_kernel}, learnable_scale={self.learnable_scale}, "
            f"kernel_parameterization={self.parameterization})"
        )


class MLPKernel(nn.Module):
    """Pointwise MLP replacing the causal delay/shedding kernel convolution.

    Instead of a fixed-shape kernel applied via ``F.conv1d``, this module
    extracts a sliding window of the input trajectory and passes each window
    through a small MLP (Linear → GELU → Linear).  The final layer is
    zero-initialised for conservative startup (matching the residual-path
    pattern used elsewhere).
    """

    def __init__(
        self,
        kernel_length: int = 21,
        hidden_dim: int = 32,
        include_population: bool = False,
    ):
        super().__init__()
        self.kernel_length = kernel_length
        self.include_population = include_population
        in_dim = kernel_length + (1 if include_population else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        _small_xavier_linear(self.mlp[0], gain=0.1)  # type: ignore[arg-type]
        # Zero-init final layer for conservative startup
        nn.init.zeros_(self.mlp[-1].weight)  # type: ignore[union-attr]
        nn.init.zeros_(self.mlp[-1].bias)  # type: ignore[union-attr]

    def forward(
        self, I_trajectory: torch.Tensor, population: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T = I_trajectory.shape
        with _autocast_disabled_for(I_trajectory):
            padded = F.pad(I_trajectory.float(), (self.kernel_length - 1, 0))
            windows = padded.unfold(1, self.kernel_length, 1)  # [B, T, K]
            if self.include_population and population is not None:
                if population.shape[0] != B:
                    raise ValueError(
                        "population batch dimension must match I_trajectory: "
                        f"{population.shape[0]} != {B}"
                    )
                pop = population.float().reshape(B, -1)[:, :1]
                pop = pop.unsqueeze(-1).expand(B, T, -1)
                windows = torch.cat([windows, pop], dim=-1)
            output = self.mlp(windows).squeeze(-1)
        return output.to(I_trajectory.dtype)  # [B, T]


class ClinicalObservationHead(nn.Module):
    """
    Observation head wrapper for clinical signals (cases, hospitalizations, deaths).

    Combines DelayKernel with learnable scaling to map infection fractions to
    log1p(per-100k) space for joint inference.

    Unit Contract:
        This head expects I_trajectory as population FRACTIONS (sum of S+I+R = 1.0),
        not absolute counts. The "per-100k" conversion assumes these fractions are
        relative to a normalized population of 100,000. The learnable scale factor
        absorbs the true population-to-fraction relationship during training.

        Input: I_t ∈ [0, 1] (fraction of population infected)
        Output: log1p(cases per 100,000 population)

    Architecture:
        1. DelayKernel: I_trajectory → delayed_infections (fractions)
        2. Convert to per-100k: delayed_infections × 100,000
        3. Apply learnable scale: per_100k × scale
        4. Apply log1p: log1p(scaled)
        5. Add residual: pred_log + alpha × residual(obs_context)

    Args:
        kernel_length: Length of delay kernel
        gamma_shape: Gamma shape parameter for kernel init
        gamma_scale: Gamma scale parameter for kernel init
        learnable_kernel: If True, delay kernel is trainable
        kernel_parameterization: "simplex" for positive normalized kernels,
            "free" for unconstrained signed kernels with gamma priming, "gamma"
            for the two-parameter learnable Gamma delay family
        scale_init: Initial scale for per-100k conversion (default 1.0)
        learnable_scale: If True, scale is trainable
        residual_dim: Dimension of observation context for residual
        alpha_init: Initial weight for residual connection
        residual_enabled: When False, retain the residual projection but freeze its
            contribution at zero.

    """

    def __init__(
        self,
        kernel_length: int = 21,
        gamma_shape: float = 5.0,
        gamma_scale: float = 2.0,
        learnable_kernel: bool = True,
        kernel_parameterization: str = "simplex",
        scale_init: float = 1.0,
        learnable_scale: bool = True,
        residual_dim: int = 0,
        alpha_init: float = 0.1,
        residual_enabled: bool = True,
        delta_forecasting: bool = False,
        anchor_mode: str = "last_valid_step",
        kernel_mlp: bool = False,
        distribution_scale_init: float = 0.3,
        distribution_scale_floor: float = 1.0e-3,
    ):
        super().__init__()

        self.delta_forecasting = delta_forecasting
        self.anchor_mode = anchor_mode
        self.distribution_scale_floor = float(distribution_scale_floor)
        if distribution_scale_init <= 0:
            raise ValueError("distribution_scale_init must be positive")
        if distribution_scale_floor <= 0:
            raise ValueError("distribution_scale_floor must be positive")

        if kernel_mlp:
            self.delay_kernel = MLPKernel(kernel_length=kernel_length, hidden_dim=32)
        else:
            self.delay_kernel = DelayKernel(
                kernel_length=kernel_length,
                gamma_shape=gamma_shape,
                gamma_scale=gamma_scale,
                learnable=learnable_kernel,
                parameterization=kernel_parameterization,
            )

        # Learnable scale for per-100k conversion (constrained positive)
        if learnable_scale:
            self.scale = nn.Parameter(torch.log(model_scalar(scale_init)))
        else:
            self.register_buffer("scale", model_scalar(scale_init))
        self.learnable_scale = learnable_scale

        # Residual connection from observation context
        # Residual projections always exist when context is available. Disabling a
        # head freezes its zero-valued contribution without changing downstream
        # parameter initialization or predictive-scale conditioning.
        self.use_residual = residual_dim > 0
        if self.use_residual:
            self.residual_proj = nn.Linear(residual_dim, 1)
            _small_xavier_linear(self.residual_proj, gain=0.1)
            if learnable_scale and residual_enabled:
                self.alpha = nn.Parameter(model_scalar(alpha_init))
            else:
                self.register_buffer(
                    "alpha", model_scalar(alpha_init if residual_enabled else 0.0)
                )
        else:
            self.residual_proj = None
            self.alpha = None
        # Immutable static disable: survives checkpoint alpha state loading.
        self._residual_location_enabled = residual_enabled
        _register_residual_log_location_fraction_buffer(self)

        self.log_distribution_scale = nn.Parameter(
            torch.log(
                model_scalar(
                    max(distribution_scale_init - distribution_scale_floor, _KERNEL_EPS)
                )
            )
        )
        if residual_dim > 0:
            self.distribution_scale_proj = nn.Linear(residual_dim, 1)
            _init_distribution_scale_proj(
                self.distribution_scale_proj,
                scale_init=distribution_scale_init,
                scale_floor=distribution_scale_floor,
            )
        else:
            self.distribution_scale_proj = None

        logger.info(
            f"Initialized ClinicalObservationHead: scale_init={scale_init}, "
            f"learnable_scale={learnable_scale}, "
            f"residual_location_enabled={self._residual_location_enabled}, "
            f"residual_projection_present={self.residual_proj is not None}, "
            f"delta_forecasting={self.delta_forecasting}, anchor_mode={self.anchor_mode}"
        )

    def forward(
        self,
        I_trajectory: torch.Tensor,
        obs_context: torch.Tensor | None = None,
        last_observed: torch.Tensor | None = None,
        last_observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate clinical observations from infection trajectory in log1p(per-100k) space.

        Args:
            I_trajectory: Infected population fraction trajectory [batch_size, time_steps]
            obs_context: Optional observation context [batch_size, time_steps, residual_dim]
                If provided, adds residual connection.
            last_observed: Optional historically observed value to anchor delta forecasting.
            last_observed_mask: Optional binary mask marking anchor availability.

        Returns:
            Predicted log1p(per-100k) values [batch_size, time_steps]

        Formula:
            pred = log1p(I_delayed × 100,000 × scale) + alpha × residual(obs_context)
        """
        return self.forward_components(
            I_trajectory=I_trajectory,
            obs_context=obs_context,
            last_observed=last_observed,
            last_observed_mask=last_observed_mask,
        )["loc"]

    def forward_components(
        self,
        I_trajectory: torch.Tensor,
        obs_context: torch.Tensor | None = None,
        last_observed: torch.Tensor | None = None,
        last_observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the full, mechanistic-only, and residual prediction components."""
        out_dtype = I_trajectory.dtype
        with _autocast_disabled_for(I_trajectory):
            delayed_infections = self.delay_kernel(I_trajectory.to(torch.float32))
            per_100k = delayed_infections * 100_000.0
            scale = self.get_scale()
            if scale.dtype != per_100k.dtype:
                scale = scale.to(per_100k.dtype)
            scaled = per_100k * scale
            scaled = torch.nan_to_num(
                scaled,
                nan=0.0,
                posinf=_OBS_CLAMP_MAX,
                neginf=0.0,
            ).clamp(min=0.0, max=_OBS_CLAMP_MAX)
            mechanistic_raw = torch.log1p(scaled)

        # Pre-bound residual raw: the parameter-side intermediate probed by
        # pathway-gradient telemetry. A statically disabled head contributes
        # an exact zero regardless of any loaded alpha state.
        residual_raw = torch.zeros_like(mechanistic_raw)
        if (
            self._residual_location_enabled
            and self.use_residual
            and obs_context is not None
            and self.residual_proj is not None
        ):
            residual = self.residual_proj(obs_context).squeeze(-1)
            if residual.dtype != mechanistic_raw.dtype:
                residual = residual.to(mechanistic_raw.dtype)
            alpha = self.get_alpha()
            if alpha is not None:
                if alpha.dtype != mechanistic_raw.dtype:
                    alpha = alpha.to(mechanistic_raw.dtype)
                residual_raw = alpha * residual

        components = _compose_observation_prediction(
            mechanistic_raw=mechanistic_raw,
            residual_raw=residual_raw,
            delta_forecasting=self.delta_forecasting,
            anchor_mode=self.anchor_mode,
            last_observed=last_observed,
            last_observed_mask=last_observed_mask,
        )
        _bound_composed_residual(
            components,
            self.get_buffer("_residual_max_log_location_fraction"),
        )
        output_components = {
            key: value if value.dtype == out_dtype else value.to(out_dtype)
            for key, value in components.items()
        }
        output_components["residual_input"] = residual_raw
        return output_components

    def forward_distribution(
        self,
        I_trajectory: torch.Tensor,
        obs_context: torch.Tensor | None = None,
        last_observed: torch.Tensor | None = None,
        last_observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return predictive distribution parameters in log1p(per-100k) space."""
        components = self.forward_components(
            I_trajectory=I_trajectory,
            obs_context=obs_context,
            last_observed=last_observed,
            last_observed_mask=last_observed_mask,
        )
        loc = components["loc"]
        scale = self.get_distribution_scale(obs_context=obs_context, like=loc)
        return {**components, "scale": scale}

    def get_scale(self) -> torch.Tensor:
        """Get positive scale parameter."""
        if self.learnable_scale:
            return _safe_exp_from_log(self.scale)
        return self.scale.clamp_min(_KERNEL_EPS)

    def get_alpha(self) -> torch.Tensor | None:
        """Get residual weight alpha."""
        if self.alpha is not None and isinstance(self.alpha, nn.Parameter):
            return torch.nan_to_num(
                self.alpha,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
        return self.alpha

    def set_residual_max_log_location_fraction(
        self,
        max_log_location_fraction: float | None,
    ) -> None:
        """Set this head's residual log-location budget.

        ``None`` restores the unrestricted legacy residual, ``0.0`` is an
        exact lockout, and values in ``(0, 1]`` cap the residual contribution
        relative to the mechanistic log-location magnitude. This does not
        change ``residual_disabled_heads`` semantics: a statically disabled
        head keeps its exact zero contribution even though its projection
        module remains present for structural compatibility.
        """
        _set_residual_log_location_fraction_buffer(
            self,
            max_log_location_fraction,
        )

    def get_distribution_scale(
        self,
        *,
        obs_context: torch.Tensor | None,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Get positive predictive scale aligned to ``like``."""
        if self.distribution_scale_proj is not None and obs_context is not None:
            raw_scale = self.distribution_scale_proj(obs_context).squeeze(-1)
            scale = F.softplus(raw_scale.float()) + self.distribution_scale_floor
        else:
            base = _safe_exp_from_log(self.log_distribution_scale).to(like.dtype)
            scale = base + self.distribution_scale_floor
            scale = scale.expand_as(like)
        return scale.to(like.dtype)

    def __repr__(self) -> str:
        return (
            f"ClinicalObservationHead(delay={self.delay_kernel}, "
            f"scale={self.get_scale().item():.4f}, learnable_scale={self.learnable_scale})"
        )


class WastewaterObservationHead(nn.Module):
    """
    Observation head wrapper for wastewater viral concentration.

    Combines SheddingConvolution with learnable scaling to map infection fractions to
    log1p(per-100k) space for joint inference.

    Architecture:
        1. SheddingConvolution: (I_trajectory, population) → diluted shedding proxy
        2. Convert to per-100k scale: proxy × 100,000
        3. Apply learnable positive scale and log1p transform
        4. Optionally add a residual correction from observation context

    Args:
        residual_enabled: When False, retain the residual projection but freeze its
            contribution at zero.

        kernel_length: Length of shedding kernel (default 14 days)
        scale_init: Initial scale for per-100k conversion (default 1.0)
        learnable_kernel: If True, shedding profile is trainable
        learnable_scale: If True, scale is trainable
        residual_dim: Dimension of observation context for residual
        alpha_init: Initial weight for residual connection
    """

    def __init__(
        self,
        kernel_length: int = 14,
        scale_init: float = 1.0,
        learnable_kernel: bool = True,
        kernel_parameterization: str = "simplex",
        gamma_shape: float = 2.0,
        gamma_scale: float = 6.0,
        learnable_scale: bool = True,
        residual_dim: int = 0,
        alpha_init: float = 0.1,
        residual_enabled: bool = True,
        strict: bool = True,
        delta_forecasting: bool = False,
        anchor_mode: str = "last_valid_step",
        kernel_mlp: bool = False,
        distribution_scale_init: float = 0.3,
        distribution_scale_floor: float = 1.0e-3,
    ):
        super().__init__()

        self.delta_forecasting = delta_forecasting
        self.anchor_mode = anchor_mode
        self.distribution_scale_floor = float(distribution_scale_floor)
        if distribution_scale_init <= 0:
            raise ValueError("distribution_scale_init must be positive")
        if distribution_scale_floor <= 0:
            raise ValueError("distribution_scale_floor must be positive")

        if kernel_mlp:
            self.shedding_conv = MLPKernel(
                kernel_length=kernel_length, hidden_dim=32, include_population=True
            )
        else:
            self.shedding_conv = SheddingConvolution(
                kernel_length=kernel_length,
                sensitivity_scale=1.0,  # Fixed, we'll handle scaling separately
                learnable_kernel=learnable_kernel,
                kernel_parameterization=kernel_parameterization,
                gamma_shape=gamma_shape,
                gamma_scale=gamma_scale,
                learnable_scale=False,  # We handle scale separately
                strict=strict,
            )

        # Learnable scale for per-100k conversion (constrained positive)
        if learnable_scale:
            self.scale = nn.Parameter(torch.log(model_scalar(scale_init)))
        else:
            self.register_buffer("scale", model_scalar(scale_init))
        self.learnable_scale = learnable_scale

        # See ClinicalObservationHead: a disabled residual keeps the same
        # projections but has an exact, non-learnable zero contribution.
        self.use_residual = residual_dim > 0
        if self.use_residual:
            self.residual_proj = nn.Linear(residual_dim, 1)
            _small_xavier_linear(self.residual_proj, gain=0.1)
            if learnable_scale and residual_enabled:
                self.alpha = nn.Parameter(model_scalar(alpha_init))
            else:
                self.register_buffer(
                    "alpha", model_scalar(alpha_init if residual_enabled else 0.0)
                )
        else:
            self.residual_proj = None
            self.alpha = None

        # Immutable static disable: survives checkpoint alpha state loading.
        self._residual_location_enabled = residual_enabled
        _register_residual_log_location_fraction_buffer(self)

        self.log_distribution_scale = nn.Parameter(
            torch.log(
                model_scalar(
                    max(distribution_scale_init - distribution_scale_floor, _KERNEL_EPS)
                )
            )
        )
        if residual_dim > 0:
            self.distribution_scale_proj = nn.Linear(residual_dim, 1)
            _init_distribution_scale_proj(
                self.distribution_scale_proj,
                scale_init=distribution_scale_init,
                scale_floor=distribution_scale_floor,
            )
        else:
            self.distribution_scale_proj = None

        logger.info(
            f"Initialized WastewaterObservationHead: scale_init={scale_init}, "
            f"learnable_scale={learnable_scale}, "
            f"residual_location_enabled={self._residual_location_enabled}, "
            f"residual_projection_present={self.residual_proj is not None}, "
            f"delta_forecasting={self.delta_forecasting}, anchor_mode={self.anchor_mode}"
        )

    def forward(
        self,
        I_trajectory: torch.Tensor,
        population: torch.Tensor,
        obs_context: torch.Tensor | None = None,
        last_observed: torch.Tensor | None = None,
        last_observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate wastewater predictions from infection trajectory in log1p(per-100k) space.

        Args:
            I_trajectory: Infected population fraction trajectory [batch_size, time_steps]
            population: Population time series [batch_size, time_steps]
            obs_context: Optional observation context [batch_size, time_steps, residual_dim]
                If provided, adds residual connection.
            last_observed: Optional historically observed value to anchor delta forecasting.
            last_observed_mask: Optional binary mask marking anchor availability.

        Returns:
            Predicted log1p(per-100k) values [batch_size, time_steps]

        Formula:
            viral_load = conv(I, shedding_kernel) / population
            pred = log1p(viral_load × 100,000 × scale) + alpha × residual(obs_context)
        """
        return self.forward_components(
            I_trajectory=I_trajectory,
            population=population,
            obs_context=obs_context,
            last_observed=last_observed,
            last_observed_mask=last_observed_mask,
        )["loc"]

    def forward_components(
        self,
        I_trajectory: torch.Tensor,
        population: torch.Tensor,
        obs_context: torch.Tensor | None = None,
        last_observed: torch.Tensor | None = None,
        last_observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the full, mechanistic-only, and residual prediction components."""
        out_dtype = I_trajectory.dtype
        viral_load = self.shedding_conv(I_trajectory.to(torch.float32), population)

        with _autocast_disabled_for(I_trajectory):
            per_100k = viral_load * 100_000.0
            scale = self.get_scale()
            if scale.dtype != per_100k.dtype:
                scale = scale.to(per_100k.dtype)
            scaled = per_100k * scale
            scaled = torch.nan_to_num(
                scaled,
                nan=0.0,
                posinf=_OBS_CLAMP_MAX,
                neginf=0.0,
            ).clamp(min=0.0, max=_OBS_CLAMP_MAX)
            mechanistic_raw = torch.log1p(scaled)

        # Pre-bound residual raw: the parameter-side intermediate probed by
        # pathway-gradient telemetry. A statically disabled head contributes
        # an exact zero regardless of any loaded alpha state.
        residual_raw = torch.zeros_like(mechanistic_raw)
        if (
            self._residual_location_enabled
            and self.use_residual
            and obs_context is not None
            and self.residual_proj is not None
        ):
            residual = self.residual_proj(obs_context).squeeze(-1)
            if residual.dtype != mechanistic_raw.dtype:
                residual = residual.to(mechanistic_raw.dtype)
            alpha = self.get_alpha()
            if alpha is not None:
                if alpha.dtype != mechanistic_raw.dtype:
                    alpha = alpha.to(mechanistic_raw.dtype)
                residual_raw = alpha * residual

        components = _compose_observation_prediction(
            mechanistic_raw=mechanistic_raw,
            residual_raw=residual_raw,
            delta_forecasting=self.delta_forecasting,
            anchor_mode=self.anchor_mode,
            last_observed=last_observed,
            last_observed_mask=last_observed_mask,
        )
        _bound_composed_residual(
            components,
            self.get_buffer("_residual_max_log_location_fraction"),
        )
        output_components = {
            key: value if value.dtype == out_dtype else value.to(out_dtype)
            for key, value in components.items()
        }
        output_components["residual_input"] = residual_raw
        return output_components

    def forward_distribution(
        self,
        I_trajectory: torch.Tensor,
        population: torch.Tensor,
        obs_context: torch.Tensor | None = None,
        last_observed: torch.Tensor | None = None,
        last_observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return predictive distribution parameters in log1p(per-100k) space."""
        components = self.forward_components(
            I_trajectory=I_trajectory,
            population=population,
            obs_context=obs_context,
            last_observed=last_observed,
            last_observed_mask=last_observed_mask,
        )
        loc = components["loc"]
        scale = self.get_distribution_scale(obs_context=obs_context, like=loc)
        return {**components, "scale": scale}

    def get_scale(self) -> torch.Tensor:
        """Get positive scale parameter."""
        if self.learnable_scale:
            return _safe_exp_from_log(self.scale)
        return self.scale.clamp_min(_KERNEL_EPS)

    def get_alpha(self) -> torch.Tensor | None:
        """Get residual weight alpha."""
        if self.alpha is not None and isinstance(self.alpha, nn.Parameter):
            return torch.nan_to_num(
                self.alpha,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
        return self.alpha

    def set_residual_max_log_location_fraction(
        self,
        max_log_location_fraction: float | None,
    ) -> None:
        """Set this head's residual log-location budget.

        See
        ``ClinicalObservationHead.set_residual_max_log_location_fraction``.
        """
        _set_residual_log_location_fraction_buffer(
            self,
            max_log_location_fraction,
        )

    def get_distribution_scale(
        self,
        *,
        obs_context: torch.Tensor | None,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Get positive predictive scale aligned to ``like``."""
        if self.distribution_scale_proj is not None and obs_context is not None:
            raw_scale = self.distribution_scale_proj(obs_context).squeeze(-1)
            scale = F.softplus(raw_scale.float()) + self.distribution_scale_floor
        else:
            base = _safe_exp_from_log(self.log_distribution_scale).to(like.dtype)
            scale = base + self.distribution_scale_floor
            scale = scale.expand_as(like)
        return scale.to(like.dtype)

    def __repr__(self) -> str:
        return f"WastewaterObservationHead({self.shedding_conv})"


class ObservationLoss(nn.Module):
    """
    Base class for observation head losses.

    Provides a consistent interface for computing losses on observation head outputs,
    with optional support for unscaling normalized predictions/targets.

    Contract:
        - All losses must implement forward(pred, target, *, target_mean=None, target_scale=None, mask=None)
        - Some losses require unscaling; others operate in normalized space
        - Masks are boolean tensors where True = keep, False = ignore
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        target_mean: torch.Tensor | None = None,
        target_scale: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss between prediction and target.

        Args:
            pred: Predicted values [batch_size, time_steps] or [batch_size, time_steps, features]
            target: Target values (same shape as pred)
            target_mean: Mean used for normalization (for unscaling losses)
            target_scale: Scale (std) used for normalization (for unscaling losses)
            mask: Boolean mask where True = include in loss, False = ignore [batch_size, time_steps]

        Returns:
            Scalar loss value (mean over unmasked elements)
        """
        raise NotImplementedError

    def _sync_to_pred_device(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
        target_scale: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        """Sync all tensors to the same device as pred (model output).

        Args:
            pred: Prediction tensor (on accelerator from model forward pass)
            target: Target tensor (may be on CPU from DataLoader)
            mask: Optional mask tensor (may be on CPU)
            target_mean: Optional mean tensor (may be on CPU)
            target_scale: Optional scale tensor (may be on CPU)

        Returns:
            Tuple of (target, mask, target_mean, target_scale) all on pred.device
        """
        return sync_to_device(
            target, mask, target_mean, target_scale, device=pred.device
        )

    def _apply_mask(
        self, loss_values: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply mask to loss values and return mean over unmasked elements.

        Note:
            - Mask shape must broadcast to loss_values shape
            - For multi-feature outputs [B, T, F], provide mask [B, T, 1] or [B, T, F]
            - Division by mask.sum() accounts for all unmasked elements including broadcast dims
        """
        if mask is None:
            return loss_values.mean()
        # Ensure mask broadcasts correctly and compute denominator
        masked_loss = loss_values * mask
        # Count all elements that contribute (accounting for broadcasting)
        denominator = mask.expand_as(loss_values).sum() + 1e-8
        return masked_loss.sum() / denominator


class MSELoss(ObservationLoss):
    """Mean Squared Error in normalized space."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        target_mean: torch.Tensor | None = None,
        target_scale: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE loss operates in normalized space; ignores mean/scale."""
        target, mask, _, _ = self._sync_to_pred_device(pred, target, mask)
        loss_values = (pred - target) ** 2
        return self._apply_mask(loss_values, mask)


class MAELoss(ObservationLoss):
    """Mean Absolute Error in normalized space."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        target_mean: torch.Tensor | None = None,
        target_scale: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MAE loss operates in normalized space; ignores mean/scale."""
        target, mask, _, _ = self._sync_to_pred_device(pred, target, mask)
        loss_values = (pred - target).abs()
        return self._apply_mask(loss_values, mask)


class UnscaledMSELoss(ObservationLoss):
    """
    MSE on unscaled (original scale) values.

    Unscales predictions and targets using target_mean/target_scale before computing MSE.
    Useful when you care about absolute error magnitude in original units.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        target_mean: torch.Tensor | None = None,
        target_scale: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute MSE on unscaled values.

        Requires target_mean and target_scale for unscaling.
        If not provided, computes MSE on normalized values (not recommended).
        """
        target, mask, target_mean, target_scale = self._sync_to_pred_device(
            pred, target, mask, target_mean, target_scale
        )
        if target_mean is not None and target_scale is not None:
            pred_unscaled, target_unscaled = unscale_forecasts(
                pred, target, target_mean, target_scale
            )
        else:
            pred_unscaled, target_unscaled = pred, target

        loss_values = (pred_unscaled - target_unscaled) ** 2
        return self._apply_mask(loss_values, mask)


class CompositeObservationLoss(nn.Module):
    """
    Composite loss across multiple observation heads.

    Supports per-head loss types, weights, and optional unscaling for losses.
    Handles missing targets (None) by skipping heads, and supports per-head masks.

    Example:
        head_specs = {
            "wastewater": (UnscaledMSELoss(), 1.0),
            "hospitalizations": (MSELoss(), 0.5),  # MSE in normalized space
        }
        loss_fn = CompositeObservationLoss(head_specs)

        # During training
        total_loss, per_head = loss_fn(
            preds={"wastewater": ww_pred, "hospitalizations": hosp_pred},
            targets={"wastewater": ww_target, "hospitalizations": hosp_target},
            stats={
                "wastewater": (ww_mean, ww_scale),
                "hospitalizations": (hosp_mean, hosp_scale),
            },
            masks={"wastewater": ww_mask},  # Optional per-head mask
        )

    Args:
        head_specs: Dict mapping head_name -> (loss_fn, weight)
            - loss_fn: ObservationLoss instance
            - weight: Scalar weight for this head's contribution to total loss
    """

    def __init__(self, head_specs: Mapping[str, tuple[ObservationLoss, float]]):
        super().__init__()
        self.head_names = list(head_specs.keys())
        self.head_specs = nn.ModuleDict()
        self.weights: dict[str, float] = {}

        for head_name, (loss_fn, weight) in head_specs.items():
            self.head_specs[head_name] = loss_fn
            self.weights[head_name] = float(weight)

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        stats: dict[str, tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        masks: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute composite loss across all heads.

        Args:
            preds: Dict mapping head_name -> predictions tensor
            targets: Dict mapping head_name -> target tensor OR None (skips head)
            stats: Optional dict mapping head_name -> (mean, scale) tuple for unscaling
            masks: Optional dict mapping head_name -> boolean mask tensor

        Returns:
            Tuple of (total_loss: scalar Tensor, per_head_losses: dict[str, Tensor])
            Heads with None targets are skipped and not included in per_head_losses.

        Note:
            - Total loss includes only heads with non-None targets
            - If all targets are None, returns (0.0, {})
            - Missing keys in stats/masks are treated as None (no unscaling/no masking)
        """
        per_head_losses: dict[str, torch.Tensor] = {}

        stats = stats or {}
        masks = masks or {}

        # Initialize total_loss with correct dtype/device from first prediction
        # This ensures AMP/bfloat16 dtype consistency
        first_pred = next(iter(preds.values())) if preds else None
        total_loss = (
            first_pred.new_zeros(()) if first_pred is not None else torch.tensor(0.0)
        )

        for head_name in self.head_names:
            target = targets.get(head_name)
            if target is None:
                # Skip this head - no target available
                continue

            pred = preds[head_name]
            loss_fn = self.head_specs[head_name]
            weight = self.weights[head_name]

            # Get optional mean/scale for unscaling
            head_stats = stats.get(head_name)
            target_mean, target_scale = (
                head_stats if head_stats is not None else (None, None)
            )

            # Get optional mask
            mask = masks.get(head_name)

            # Compute head loss
            head_loss = loss_fn(
                pred,
                target,
                target_mean=target_mean,
                target_scale=target_scale,
                mask=mask,
            )

            # Store per-head loss for logging (unweighted)
            per_head_losses[head_name] = head_loss.detach()

            # Add weighted contribution to total
            if weight != 0:
                total_loss = total_loss + weight * head_loss

        return total_loss, per_head_losses

    def __repr__(self) -> str:
        specs_str = ", ".join(
            f"{name}={type(self.head_specs[name]).__name__}(w={w})"
            for name, w in self.weights.items()
        )
        return f"CompositeObservationLoss({specs_str})"
