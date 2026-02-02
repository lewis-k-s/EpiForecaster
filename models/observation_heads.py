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
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.normalization import unscale_forecasts

logger = logging.getLogger(__name__)


def _inverse_softplus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse softplus transform for positive tensors."""
    x = torch.clamp(x, min=eps)
    return torch.log(torch.expm1(x).clamp_min(eps))


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

        logger.info(
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

        # Normalize to sum to 1
        kernel = kernel / kernel.sum()

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

        # Normalize kernel (ensure probabilities sum to 1)
        if self.learnable:
            kernel_weights = F.softplus(self.kernel)
        else:
            kernel_weights = self.kernel
        normalized_kernel = kernel_weights / kernel_weights.sum()

        # Reshape for 1D convolution: [batch_size, 1, time_steps]
        I_reshaped = I_trajectory.unsqueeze(1)

        # Prepare kernel for conv1d: [1, 1, kernel_length]
        kernel_reshaped = normalized_kernel.view(1, 1, -1)

        # Apply causal convolution with left padding
        # Padding = kernel_length - 1 ensures causality (output t only sees <= t)
        output = F.conv1d(
            I_reshaped,
            kernel_reshaped,
            padding=self.kernel_length - 1,
            groups=1,
        )

        # Trim to original time length (remove extra padding at end)
        output = output[:, :, :time_steps]

        # Reshape back: [batch_size, time_steps]
        return output.squeeze(1)

    def get_kernel_weights(self) -> torch.Tensor:
        """Get current kernel weights (normalized)."""
        if self.learnable:
            kernel_weights = F.softplus(self.kernel)
        else:
            kernel_weights = self.kernel
        return kernel_weights / kernel_weights.sum()

    def get_mean_delay(self) -> float:
        """Compute mean delay in days based on current kernel."""
        weights = self.get_kernel_weights()
        positions = torch.arange(len(weights), dtype=torch.float32)
        return (weights * positions).sum().item()

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

        logger.info(
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

        # Normalize
        kernel = kernel / kernel.sum()

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

        if torch.any(population <= 0):
            raise ValueError("population must be positive")

        # Normalize kernel
        if self.learnable_kernel:
            kernel_weights = F.softplus(self.kernel)
        else:
            kernel_weights = self.kernel
        normalized_kernel = kernel_weights / kernel_weights.sum()

        # Reshape for 1D convolution: [batch_size, 1, time_steps]
        I_reshaped = I_trajectory.unsqueeze(1)

        # Prepare kernel: [1, 1, kernel_length]
        kernel_reshaped = normalized_kernel.view(1, 1, -1)

        # Causal convolution with left padding
        total_shedding = F.conv1d(
            I_reshaped,
            kernel_reshaped,
            padding=self.kernel_length - 1,
            groups=1,
        )

        # Trim to original length
        total_shedding = total_shedding[:, :, :time_steps]
        total_shedding = total_shedding.squeeze(1)  # [batch_size, time_steps]

        # Apply dilution physics: divide by population
        # This is the key physical insight: more people = more wastewater flow = dilution
        viral_concentration = total_shedding / (population + 1e-8)  # Avoid div by zero

        # Apply sensitivity scale (measurement calibration)
        viral_concentration = viral_concentration * self.get_sensitivity_scale()

        return viral_concentration

    def get_kernel_weights(self) -> torch.Tensor:
        """Get current shedding kernel weights (normalized)."""
        if self.learnable_kernel:
            kernel_weights = F.softplus(self.kernel)
        else:
            kernel_weights = self.kernel
        return kernel_weights / kernel_weights.sum()

    def get_sensitivity_scale(self) -> torch.Tensor:
        """Get positive sensitivity scale."""
        if self.learnable_scale:
            return torch.exp(self.sensitivity_scale)
        return self.sensitivity_scale

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

    Combines DelayKernel with optional learnable scaling factors for observation rates.
    Supports variant-specific adjustments if variant proportions are provided.

    Architecture:
        1. DelayKernel: I_trajectory → delayed_infections
        2. Observation scaling: delayed_infections × observation_rate
        3. (Optional) Variant adjustment: Σ_v (I × Prop_v × Rate_v)

    Args:
        kernel_length: Length of delay kernel
        gamma_shape: Gamma shape parameter for kernel init
        gamma_scale: Gamma scale parameter for kernel init
        learnable_kernel: If True, delay kernel is trainable
        observation_rate_init: Initial observation rate (fraction of infections observed)
        learnable_rate: If True, observation rate is trainable
    """

    def __init__(
        self,
        kernel_length: int = 21,
        gamma_shape: float = 5.0,
        gamma_scale: float = 2.0,
        learnable_kernel: bool = True,
        observation_rate_init: float = 0.1,
        learnable_rate: bool = True,
    ):
        super().__init__()

        self.delay_kernel = DelayKernel(
            kernel_length=kernel_length,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            learnable=learnable_kernel,
        )

        # Observation rate: fraction of infections that become observed cases/hosp/deaths
        if learnable_rate:
            rate_tensor = torch.tensor(float(observation_rate_init))
            self.observation_rate = nn.Parameter(_inverse_softplus(rate_tensor))
        else:
            self.register_buffer(
                "observation_rate", torch.tensor(observation_rate_init)
            )

        self.learnable_rate = learnable_rate

        logger.info(
            f"Initialized ClinicalObservationHead: rate_init={observation_rate_init}, "
            f"learnable_rate={learnable_rate}"
        )

    def forward(
        self,
        I_trajectory: torch.Tensor,
        variant_proportions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate clinical observations from infection trajectory.

        Args:
            I_trajectory: Infected population trajectory [batch_size, time_steps]
            variant_proportions: Optional variant proportions [batch_size, time_steps, n_variants]
                If provided, applies variant-specific observation rates.

        Returns:
            Observed counts [batch_size, time_steps]

        Note:
            - For variant-aware prediction, use variant-specific observation rates
            - Without variant info, uses single observation_rate for all infections
        """
        # Apply delay kernel
        delayed_infections = self.delay_kernel(I_trajectory)
        observation_rate = self.get_observation_rate()

        if variant_proportions is not None:
            # Variant-aware: weighted by variant proportions (placeholder for future)
            # For now, use same observation rate (extensible structure)
            observed = delayed_infections * observation_rate
        else:
            # Single observation rate
            observed = delayed_infections * observation_rate

        return observed

    def get_delay_stats(self) -> dict:
        """Get statistics about the delay kernel."""
        return {
            "mean_delay_days": self.delay_kernel.get_mean_delay(),
            "kernel_length": self.delay_kernel.kernel_length,
        }

    def get_observation_rate(self) -> torch.Tensor:
        """Get positive observation rate."""
        if self.learnable_rate:
            return F.softplus(self.observation_rate)
        return self.observation_rate

    def __repr__(self) -> str:
        return (
            f"ClinicalObservationHead(delay={self.delay_kernel}, "
            f"rate={self.get_observation_rate().item():.4f}, learnable_rate={self.learnable_rate})"
        )


class WastewaterObservationHead(nn.Module):
    """
    Observation head wrapper for wastewater viral concentration.

    Combines SheddingConvolution with sensitivity scaling for wastewater surveillance.
    Models the physical dilution process and measurement sensitivity.

    Architecture:
        1. SheddingConvolution: (I_trajectory, population) → viral_concentration
        2. Observation noise model: (Optional) Add measurement noise calibration

    Args:
        kernel_length: Length of shedding kernel (default 14 days)
        sensitivity_scale: Measurement sensitivity (default 1.0)
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

        self.shedding_conv = SheddingConvolution(
            kernel_length=kernel_length,
            sensitivity_scale=sensitivity_scale,
            learnable_kernel=learnable_kernel,
            learnable_scale=learnable_scale,
        )

        logger.info(f"Initialized WastewaterObservationHead: {self.shedding_conv}")

    def forward(
        self, I_trajectory: torch.Tensor, population: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate wastewater viral concentration from infection trajectory.

        Args:
            I_trajectory: Infected population trajectory [batch_size, time_steps]
            population: Population time series [batch_size, time_steps]
                Time-varying population accounts for mobility-adjusted flow

        Returns:
            Viral concentration [batch_size, time_steps]
                Calibrated units via sensitivity_scale (e.g., copies/L)

        Formula:
            viral_concentration_t = sensitivity_scale × Σ_k(I_{t-k} × shedding_k) / population_t

        Key Physical Properties:
            - Same infections in small population → high concentration (detectable)
            - Same infections in large population → low concentration (diluted)
            - Calibratable to match real-world LoD (e.g., 375 copies/L)
        """
        return self.shedding_conv(I_trajectory, population)

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
