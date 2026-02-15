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
        >>> from opaque.auditing import bootstrap, attack_auroc
        >>> auroc_ci = bootstrap(attack_auroc, in_scores, out_scores, params)

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
            raise ValueError(f"quantiles must be in (0, 1), got {self.quantiles}")

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        if self.acceleration and not self.bias_correction:
            raise ValueError("Cannot use acceleration without bias correction")

    @classmethod
    def confidence_interval(
        cls,
        confidence: float = 0.95,
        num_samples: int = 1000,
        bias_correction: bool = True,
        acceleration: bool = False,
        seed: int | None = None,
    ) -> "BootstrapParams":
        """Create BootstrapParams for a symmetric confidence interval.

        This is a convenience method for creating common confidence intervals
        (e.g., 95%, 99%) without manually computing the quantiles.

        Args:
            confidence: Desired confidence level in (0, 1). For example, 0.95
                for a 95% confidence interval. Default: 0.95.
            num_samples: Number of bootstrap resamples. Default: 1000.
            bias_correction: If True, apply bias correction. Default: True.
            acceleration: If True, apply acceleration (BCa). Default: False.
            seed: Random seed for reproducibility. If None, a non-deterministic
                seed is chosen.

        Returns:
            BootstrapParams configured for the specified confidence interval.

        Example:
            >>> # 95% confidence interval
            >>> params = BootstrapParams.confidence_interval(confidence=0.95)
            >>> # Equivalent to:
            >>> # params = BootstrapParams(quantiles=(0.025, 0.975))

            >>> # 99% CI with BCa acceleration
            >>> params = BootstrapParams.confidence_interval(
            ...     confidence=0.99,
            ...     bias_correction=True,
            ...     acceleration=True,
            ... )
        """
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        significance = 1 - confidence
        quantiles = (significance / 2, 1 - significance / 2)

        return cls(
            num_samples=num_samples,
            quantiles=quantiles,
            bias_correction=bias_correction,
            acceleration=acceleration,
            seed=seed,
        )


def bootstrap(
    fn,
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    params: BootstrapParams,
) -> np.ndarray:
    """Compute bootstrapped quantiles for any auditing function.

    Args:
        fn: Function with signature fn(in_scores, out_scores, ...) -> float
        in_scores: Attack scores for held-in canaries.
        out_scores: Attack scores for held-out canaries.
        params: Bootstrap parameters.

    Returns:
        Array of quantiles specified in params.quantiles.

    Example:
        >>> from opaque.auditing import bootstrap, attack_auroc, BootstrapParams
        >>> params = BootstrapParams(num_samples=1000, seed=42)
        >>> auroc_ci = bootstrap(attack_auroc, in_scores, out_scores, params)
        >>> print(f"AUROC 95% CI: [{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]")
    """
    import scipy.stats

    in_arr = np.asarray(in_scores)
    out_arr = np.asarray(out_scores)
    n_in, n_out = len(in_arr), len(out_arr)

    rng = np.random.default_rng(seed=params.seed)

    values = np.empty(params.num_samples)
    for i in range(params.num_samples):
        in_sample = rng.choice(in_arr, size=n_in)
        out_sample = rng.choice(out_arr, size=n_out)
        values[i] = fn(in_sample, out_sample)

    if not params.bias_correction:
        return np.quantile(values, params.quantiles, method="linear")

    # Bias-corrected bootstrap (BCa)
    full_estimate = fn(in_arr, out_arr)
    prop_less = (np.sum(values < full_estimate) + 1) / (params.num_samples + 2)
    z0 = scipy.stats.norm.ppf(prop_less)

    if params.acceleration:
        # Jackknife for acceleration
        jk = np.empty(n_in + n_out)
        for i in range(n_in):
            jk[i] = fn(np.delete(in_arr, i), out_arr)
        for i in range(n_out):
            jk[n_in + i] = fn(in_arr, np.delete(out_arr, i))

        jk_mean = np.mean(jk)
        num = np.sum((jk_mean - jk) ** 3)
        denom = 6 * np.sum((jk_mean - jk) ** 2) ** 1.5
        accel = 0.0 if denom == 0 else num / denom
    else:
        accel = 0.0

    z_q = scipy.stats.norm.ppf(params.quantiles)
    corrected = scipy.stats.norm.cdf(z0 + (z0 + z_q) / (1 - accel * (z0 + z_q)))

    return np.quantile(values, corrected, method="linear")


__all__ = ["BootstrapParams", "bootstrap"]
