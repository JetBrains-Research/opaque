"""Bootstrap resampling for confidence intervals in privacy auditing.

Bootstrap resampling is used to estimate confidence intervals for privacy metrics
(epsilon bounds, AUROC, etc.) by repeatedly resampling the canary scores with
replacement and computing the metric on each resample.

This module implements the bias-corrected and accelerated (BCa) bootstrap method
from Efron (1987), which provides more accurate confidence intervals than the
standard percentile method.

Reference:
    B. Efron, "Bootstrap Confidence Intervals", Statist. Sci. 2(3), 189-228 (1987)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapParams:
    """Parameters for bootstrap confidence interval estimation.

    Bootstrap resampling estimates confidence intervals by:
    1. Repeatedly resampling the held-in and held-out canary scores with replacement
    2. Computing the metric (epsilon, AUROC, etc.) on each resample
    3. Estimating quantiles from the empirical distribution

    Optionally applies bias correction and acceleration (BCa) from Efron (1987)
    for more accurate intervals. Note: acceleration requires jackknife resampling,
    which computes the metric n times (once per canary), increasing computation.

    Attributes:
        num_samples: Number of bootstrap resamples. More samples give more accurate
            quantile estimates. Default: 1000.
        quantiles: Array-like of quantiles to report. For example, (0.025, 0.975)
            gives a 95% confidence interval. Values must be in (0, 1).
        bias_correction: If True, apply bias correction to adjust for systematic
            bias in the bootstrap distribution. Default: True.
        acceleration: If True, apply acceleration to adjust for non-constant variance.
            Requires bias_correction=True. Increases computation significantly
            (requires jackknife resampling). Default: False.
        seed: Random seed for reproducibility. If None, a non-deterministic seed
            is chosen.

    Example:
        >>> # 95% confidence interval with bias correction (default)
        >>> params = BootstrapParams.confidence_interval(confidence=0.95, seed=42)
        >>> auditor = CanaryScoreAuditor(in_scores, out_scores)
        >>> eps, (eps_lo, eps_hi) = auditor.epsilon_clopper_pearson(
        ...     threshold_strategy=Split(),
        ...     bootstrap_params=params,
        ... )

        >>> # Custom quantiles (e.g., 90% CI)
        >>> params = BootstrapParams(
        ...     num_samples=2000,
        ...     quantiles=(0.05, 0.95),
        ...     seed=42,
        ... )

        >>> # No bias correction (faster but less accurate)
        >>> params = BootstrapParams(bias_correction=False, seed=42)
    """

    num_samples: int = 1000
    quantiles: tuple[float, ...] = (0.025, 0.975)
    bias_correction: bool = True
    acceleration: bool = False
    seed: int | None = None

    def __post_init__(self):
        """Validate parameters."""
        # Convert quantiles to array for validation
        quantile_arr = np.asarray(self.quantiles)

        if quantile_arr.size == 0:
            raise ValueError("quantiles cannot be empty")

        if not np.all((0 < quantile_arr) & (quantile_arr < 1)):
            raise ValueError(
                f"quantiles must be in (0, 1), got {self.quantiles}"
            )

        if self.num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, got {self.num_samples}"
            )

        if self.acceleration and not self.bias_correction:
            raise ValueError("Cannot use acceleration without bias correction")

    @classmethod
    def confidence_interval(
        cls,
        num_samples: int = 1000,
        confidence: float = 0.95,
        seed: int | None = None,
    ) -> "BootstrapParams":
        """Create BootstrapParams for a symmetric confidence interval.

        This is a convenience method for creating common confidence intervals
        (e.g., 95%, 99%) without manually computing the quantiles.

        Args:
            num_samples: Number of bootstrap resamples. Default: 1000.
            confidence: Desired confidence level in (0, 1). For example, 0.95
                for a 95% confidence interval. Default: 0.95.
            seed: Random seed for reproducibility. If None, a non-deterministic
                seed is chosen.

        Returns:
            BootstrapParams configured for the specified confidence interval.

        Example:
            >>> # 95% confidence interval
            >>> params = BootstrapParams.confidence_interval(confidence=0.95)
            >>> # Equivalent to:
            >>> # params = BootstrapParams(quantiles=(0.025, 0.975))

            >>> # 99% confidence interval
            >>> params = BootstrapParams.confidence_interval(confidence=0.99)
            >>> # Equivalent to:
            >>> # params = BootstrapParams(quantiles=(0.005, 0.995))
        """
        if not 0 < confidence < 1:
            raise ValueError(
                f"confidence must be in (0, 1), got {confidence}"
            )

        significance = 1 - confidence
        quantiles = (significance / 2, 1 - significance / 2)

        return cls(num_samples=num_samples, quantiles=quantiles, seed=seed)


__all__ = ["BootstrapParams"]
