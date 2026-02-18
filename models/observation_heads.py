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

from utils.normalization import unscale_forecasts

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


def _zero_linear(layer: nn.Linear) -> None:
    """Set a linear layer to an exact zero map."""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class DelayKernel(nn.Module):
    """
    Causal delay kernel for clinical observations (cases, hospitalizations, deaths).

    Models the time delay between infection (I_t) and clinical observation using
    a 1D convolution with a causal, normalized kernel. The kernel represents the
    probability distribution of time from infection to observation.

    Physics:
        - Causal: Output at time t only depends on I_{t-k:t} (no future leakage)
        - Normalized: Kernel sums to 1 (probabilistic interpretation)
        - Gamma-distributed: Default kernel follows Gamma(shape, scale) distribution

    Default initialization (Gamma(5, 2)):
        - Mean delay: shape × scale = 10 days
        - Appropriate for hospitalization delay from infection
        - Supports case reporting delays (~7 days) and death delays (~21 days)

    Args:
        kernel_length: Length of delay kernel (days of history to consider)
        gamma_shape: Shape parameter for Gamma distribution initialization
        gamma_scale: Scale parameter for Gamma distribution initialization
        learnable: If True, kernel weights are trainable; if False, frozen
    """

    def __init__(
        self,
        kernel_length: int = 21,
        gamma_shape: float = 5.0,
        gamma_scale: float = 2.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.kernel_length = kernel_length
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.learnable = learnable

        # Initialize kernel weights with Gamma distribution
        kernel_weights = self._init_gamma_kernel(
            kernel_length, gamma_shape, gamma_scale
        )

        if learnable:
            self.kernel = nn.Parameter(_inverse_softplus(kernel_weights))
        else:
            self.register_buffer("kernel", kernel_weights)

        logger.debug(
            f"Initialized DelayKernel: length={kernel_length}, "
            f"gamma({gamma_shape}, {gamma_scale}), learnable={learnable}"
        )

    def _init_gamma_kernel(
        self, length: int, shape: float, scale: float
    ) -> torch.Tensor:
        """Initialize kernel with Gamma distribution PDF values."""
        # Compute Gamma PDF at integer positions
        x = torch.arange(length, dtype=torch.float32)

        # Gamma PDF: x^(shape-1) * exp(-x/scale) / (scale^shape * Gamma(shape))
        # Using log-space for numerical stability
        log_gamma_pdf = (
            (shape - 1) * torch.log(x + 1e-8)  # x^(shape-1), avoid log(0)
            - x / scale  # exp(-x/scale)
            - shape * math.log(scale)  # scale^shape
            - math.lgamma(shape)  # Gamma(shape)
        )

        kernel = torch.exp(log_gamma_pdf)
        kernel = _safe_normalize(kernel)

        return kernel

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
            if self.learnable:
                kernel_weights = F.softplus(_sanitize_kernel_logits(self.kernel))
            else:
                kernel_weights = self.kernel
            normalized_kernel = _safe_normalize(kernel_weights)

            I_reshaped = torch.nan_to_num(
                I_trajectory.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).unsqueeze(1)
            kernel_reshaped = normalized_kernel.view(1, 1, -1)

            output = F.conv1d(
                I_reshaped,
                kernel_reshaped,
                padding=self.kernel_length - 1,
                groups=1,
            )
            output = output[:, :, :time_steps].squeeze(1)
        return output.to(I_trajectory.dtype)

    def get_kernel_weights(self) -> torch.Tensor:
        """Get current kernel weights (normalized)."""
        if self.learnable:
            kernel_weights = F.softplus(_sanitize_kernel_logits(self.kernel))
            output_dtype = self.kernel.dtype
        else:
            kernel_weights = self.kernel
            output_dtype = kernel_weights.dtype
        return _safe_normalize(kernel_weights).to(output_dtype)

    def get_mean_delay(self) -> float:
        """Compute mean delay in days based on current kernel."""
        weights = self.get_kernel_weights()
        positions = torch.arange(
            len(weights), device=weights.device, dtype=torch.float32
        )
        return (weights.float() * positions).sum().item()

    def __repr__(self) -> str:
        return (
            f"DelayKernel(length={self.kernel_length}, "
            f"gamma({self.gamma_shape}, {self.gamma_scale}), "
            f"learnable={self.learnable})"
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
    """

    def __init__(
        self,
        kernel_length: int = 14,
        sensitivity_scale: float = 1.0,
        learnable_kernel: bool = True,
        learnable_scale: bool = True,
    ):
        super().__init__()
        self.kernel_length = kernel_length
        self.learnable_kernel = learnable_kernel
        self.learnable_scale = learnable_scale

        # Initialize shedding kernel with biologically-informed profile
        kernel_weights = self._init_shedding_kernel(kernel_length)

        if learnable_kernel:
            self.kernel = nn.Parameter(_inverse_softplus(kernel_weights))
        else:
            self.register_buffer("kernel", kernel_weights)

        # Sensitivity scale (absorbs FlowPerCapita and measurement characteristics)
        if sensitivity_scale <= 0:
            raise ValueError("sensitivity_scale must be positive")

        if learnable_scale:
            self.sensitivity_scale = nn.Parameter(
                torch.log(torch.tensor(float(sensitivity_scale)))
            )
        else:
            self.register_buffer("sensitivity_scale", torch.tensor(sensitivity_scale))

        logger.debug(
            f"Initialized SheddingConvolution: length={kernel_length}, "
            f"sensitivity_scale={sensitivity_scale:.4f}, "
            f"learnable_kernel={learnable_kernel}, learnable_scale={learnable_scale}"
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
        x = torch.arange(length, dtype=torch.float32)

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

        with _autocast_disabled_for(I_trajectory):
            if self.learnable_kernel:
                kernel_weights = F.softplus(_sanitize_kernel_logits(self.kernel))
            else:
                kernel_weights = self.kernel
            normalized_kernel = _safe_normalize(kernel_weights)

            I_reshaped = torch.nan_to_num(
                I_trajectory.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).unsqueeze(1)
            kernel_reshaped = normalized_kernel.view(1, 1, -1)

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

    def get_kernel_weights(self) -> torch.Tensor:
        """Get current shedding kernel weights (normalized)."""
        if self.learnable_kernel:
            kernel_weights = F.softplus(_sanitize_kernel_logits(self.kernel))
            output_dtype = self.kernel.dtype
        else:
            kernel_weights = self.kernel
            output_dtype = kernel_weights.dtype
        return _safe_normalize(kernel_weights).to(output_dtype)

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
            f"learnable_kernel={self.learnable_kernel}, learnable_scale={self.learnable_scale})"
        )


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
        scale_init: Initial scale for per-100k conversion (default 1.0)
        learnable_scale: If True, scale is trainable
        residual_dim: Dimension of observation context for residual
        alpha_init: Initial weight for residual connection
    """

    def __init__(
        self,
        kernel_length: int = 21,
        gamma_shape: float = 5.0,
        gamma_scale: float = 2.0,
        learnable_kernel: bool = True,
        scale_init: float = 1.0,
        learnable_scale: bool = True,
        residual_dim: int = 0,
        alpha_init: float = 0.1,
    ):
        super().__init__()

        self.delay_kernel = DelayKernel(
            kernel_length=kernel_length,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            learnable=learnable_kernel,
        )

        # Learnable scale for per-100k conversion (constrained positive)
        if learnable_scale:
            self.scale = nn.Parameter(torch.log(torch.tensor(float(scale_init))))
        else:
            self.register_buffer("scale", torch.tensor(scale_init))
        self.learnable_scale = learnable_scale

        # Residual connection from observation context
        self.use_residual = residual_dim > 0
        if self.use_residual:
            self.residual_proj = nn.Linear(residual_dim, 1)
            _zero_linear(self.residual_proj)
            if learnable_scale:
                self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            else:
                self.register_buffer("alpha", torch.tensor(alpha_init))
        else:
            self.residual_proj = None
            self.alpha = None

        logger.info(
            f"Initialized ClinicalObservationHead: scale_init={scale_init}, "
            f"learnable_scale={learnable_scale}, use_residual={self.use_residual}"
        )

    def forward(
        self,
        I_trajectory: torch.Tensor,
        obs_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate clinical observations from infection trajectory in log1p(per-100k) space.

        Args:
            I_trajectory: Infected population fraction trajectory [batch_size, time_steps]
            obs_context: Optional observation context [batch_size, time_steps, residual_dim]
                If provided, adds residual connection.

        Returns:
            Predicted log1p(per-100k) values [batch_size, time_steps]

        Formula:
            pred = log1p(I_delayed × 100,000 × scale) + alpha × residual(obs_context)
        """
        with _autocast_disabled_for(I_trajectory):
            delayed_infections = self.delay_kernel(I_trajectory.float())
            per_100k = delayed_infections * 100_000.0
            scale = self.get_scale().float()
            scaled = per_100k * scale
            scaled = torch.nan_to_num(
                scaled,
                nan=0.0,
                posinf=_OBS_CLAMP_MAX,
                neginf=0.0,
            ).clamp(min=0.0, max=_OBS_CLAMP_MAX)
            pred_log = torch.log1p(scaled)

        # Add residual from observation context if provided
        if (
            self.use_residual
            and obs_context is not None
            and self.residual_proj is not None
        ):
            residual = self.residual_proj(obs_context).squeeze(-1)
            alpha = self.get_alpha()
            if alpha is not None:
                pred_log = pred_log + alpha.float() * residual.float()

        return pred_log.to(I_trajectory.dtype)

    def get_delay_stats(self) -> dict:
        """Get statistics about the delay kernel."""
        return {
            "mean_delay_days": self.delay_kernel.get_mean_delay(),
            "kernel_length": self.delay_kernel.kernel_length,
        }

    def get_scale(self) -> torch.Tensor:
        """Get positive scale parameter."""
        if self.learnable_scale:
            return _safe_exp_from_log(self.scale).to(self.scale.dtype)
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
        1. SheddingConvolution: (I_trajectory, population) → viral_load
        2. Convert to per-100k: viral_load × 100,000 / population
        3. Apply log1p: log1p(per_100k × scale)
        4. Add residual: pred_log + alpha × residual(obs_context)

    Args:
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
        learnable_scale: bool = True,
        residual_dim: int = 0,
        alpha_init: float = 0.1,
    ):
        super().__init__()

        self.shedding_conv = SheddingConvolution(
            kernel_length=kernel_length,
            sensitivity_scale=1.0,  # Fixed, we'll handle scaling separately
            learnable_kernel=learnable_kernel,
            learnable_scale=False,  # We handle scale separately
        )

        # Learnable scale for per-100k conversion (constrained positive)
        if learnable_scale:
            self.scale = nn.Parameter(torch.log(torch.tensor(float(scale_init))))
        else:
            self.register_buffer("scale", torch.tensor(scale_init))
        self.learnable_scale = learnable_scale

        # Residual connection from observation context
        self.use_residual = residual_dim > 0
        if self.use_residual:
            self.residual_proj = nn.Linear(residual_dim, 1)
            _zero_linear(self.residual_proj)
            if learnable_scale:
                self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            else:
                self.register_buffer("alpha", torch.tensor(alpha_init))
        else:
            self.residual_proj = None
            self.alpha = None

        logger.info(
            f"Initialized WastewaterObservationHead: scale_init={scale_init}, "
            f"learnable_scale={learnable_scale}, use_residual={self.use_residual}"
        )

    def forward(
        self,
        I_trajectory: torch.Tensor,
        population: torch.Tensor,
        obs_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate wastewater predictions from infection trajectory in log1p(per-100k) space.

        Args:
            I_trajectory: Infected population fraction trajectory [batch_size, time_steps]
            population: Population time series [batch_size, time_steps]
            obs_context: Optional observation context [batch_size, time_steps, residual_dim]
                If provided, adds residual connection.

        Returns:
            Predicted log1p(per-100k) values [batch_size, time_steps]

        Formula:
            viral_load = conv(I, shedding_kernel) / population
            pred = log1p(viral_load × 100,000 × scale) + alpha × residual(obs_context)
        """
        viral_load = self.shedding_conv(I_trajectory, population)

        with _autocast_disabled_for(I_trajectory):
            per_100k = viral_load.float() * 100_000.0
            scale = self.get_scale().float()
            scaled = per_100k * scale
            scaled = torch.nan_to_num(
                scaled,
                nan=0.0,
                posinf=_OBS_CLAMP_MAX,
                neginf=0.0,
            ).clamp(min=0.0, max=_OBS_CLAMP_MAX)
            pred_log = torch.log1p(scaled)

        # Add residual from observation context if provided
        if (
            self.use_residual
            and obs_context is not None
            and self.residual_proj is not None
        ):
            residual = self.residual_proj(obs_context).squeeze(-1)
            alpha = self.get_alpha()
            if alpha is not None:
                pred_log = pred_log + alpha.float() * residual.float()

        return pred_log.to(I_trajectory.dtype)

    def get_scale(self) -> torch.Tensor:
        """Get positive scale parameter."""
        if self.learnable_scale:
            return _safe_exp_from_log(self.scale).to(self.scale.dtype)
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

    def get_shedding_stats(self) -> dict:
        """Get statistics about the shedding kernel."""
        return {
            "peak_shedding_day": self.shedding_conv.get_peak_shedding_day(),
            "kernel_length": self.shedding_conv.kernel_length,
            "sensitivity_scale": self.shedding_conv.get_sensitivity_scale().item(),
        }

    def calibrate_to_lod(
        self,
        lod_copies_per_liter: float,
        typical_infections: float,
        typical_population: float,
        calibration_window: int = 20,
    ) -> None:
        """
        Calibrate sensitivity scale to match real-world Limit of Detection (LoD).

        Args:
            lod_copies_per_liter: Limit of detection in copies/L
            typical_infections: Reference number of infected individuals
            typical_population: Reference population size
            calibration_window: Number of time steps to use for calibration
                (needs to be >= kernel_length for accurate calibration)

        Note:
            This sets sensitivity_scale such that typical_infections in typical_population
            produces viral concentration equal to lod_copies_per_liter.
        """
        with torch.no_grad():
            if calibration_window < self.shedding_conv.kernel_length:
                raise ValueError(
                    "calibration_window must be >= kernel_length for calibration"
                )

            # Use a trajectory with sufficient history for the convolution
            I_ref = torch.ones(1, calibration_window) * typical_infections
            pop_ref = torch.ones(1, calibration_window) * typical_population

            # Get shedding without sensitivity scale (divide out the scale factor)
            raw_concentration = (
                self.shedding_conv(I_ref, pop_ref)
                / self.shedding_conv.get_sensitivity_scale()
            )

            # Use the middle value where convolution has full history
            mid_point = calibration_window // 2
            raw_value = raw_concentration[0, mid_point].item()
            if raw_value <= 0:
                raise ValueError(
                    "Calibration failed: raw concentration is non-positive"
                )

            # Calibrate: lod = sensitivity_scale × raw_concentration
            # sensitivity_scale = lod / raw_concentration
            new_scale = lod_copies_per_liter / raw_value
            if self.shedding_conv.learnable_scale:
                self.shedding_conv.sensitivity_scale.fill_(math.log(new_scale))
            else:
                self.shedding_conv.sensitivity_scale.fill_(new_scale)

            logger.info(
                f"Calibrated sensitivity_scale to {new_scale:.4f} for LoD={lod_copies_per_liter} copies/L"
            )

    def __repr__(self) -> str:
        return f"WastewaterObservationHead({self.shedding_conv})"


class ObservationLoss(nn.Module):
    """
    Base class for observation head losses.

    Provides a consistent interface for computing losses on observation head outputs,
    with optional support for unscaling normalized predictions/targets.

    Contract:
        - All losses must implement forward(pred, target, *, target_mean=None, target_scale=None, mask=None)
        - Some losses (sMAPE, unscaled MSE) require unscaling; others (MSE, MAE) operate in normalized space
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
        _ = (target_mean, target_scale)
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
        _ = (target_mean, target_scale)
        loss_values = (pred - target).abs()
        return self._apply_mask(loss_values, mask)


class SMAPELoss(ObservationLoss):
    """
    Symmetric Mean Absolute Percentage Error on unscaled values.

    Unscales predictions and targets using target_mean/target_scale before computing sMAPE.
    This matches the evaluation metric used in epiforecaster_eval.py.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

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
        Compute sMAPE on unscaled values.

        Requires target_mean and target_scale for unscaling.
        If not provided, computes sMAPE directly on normalized values.
        """
        if target_mean is not None and target_scale is not None:
            pred_unscaled, target_unscaled = unscale_forecasts(
                pred, target, target_mean, target_scale
            )
        else:
            pred_unscaled, target_unscaled = pred, target

        numerator = 2 * (pred_unscaled - target_unscaled).abs()
        denominator = pred_unscaled.abs() + target_unscaled.abs() + self.epsilon
        loss_values = numerator / denominator
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

    Supports per-head loss types, weights, and optional unscaling for sMAPE/unscaled losses.
    Handles missing targets (None) by skipping heads, and supports per-head masks.

    Example:
        head_specs = {
            "wastewater": (SMAPELoss(), 1.0),  # sMAPE on unscaled values
            "hospitalizations": (MSELoss(), 0.5),  # MSE in normalized space
        }
        loss_fn = CompositeObservationLoss(head_specs)

        # During training
        total_loss, per_head = loss_fn(
            preds={"wastewater": ww_pred, "hospitalizations": hosp_pred},
            targets={"wastewater": ww_target, "hospitalizations": hosp_target},
            stats={
                "wastewater": (ww_mean, ww_scale),  # For unscaling sMAPE
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

    def _get_device(self, preds: dict[str, torch.Tensor]) -> torch.device:
        """Infer device from prediction tensors."""
        if not preds:
            return torch.device("cpu")
        first_pred = next(iter(preds.values()))
        return first_pred.device

    def __repr__(self) -> str:
        specs_str = ", ".join(
            f"{name}={type(self.head_specs[name]).__name__}(w={w})"
            for name, w in self.weights.items()
        )
        return f"CompositeObservationLoss({specs_str})"
