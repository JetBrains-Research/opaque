"""Bounded Gaussian noise mechanism for differential privacy.

This module implements the Bounded Gaussian Mechanism from:

    Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for Differential
    Privacy," Journal of Privacy and Confidentiality, 14(1), 2024.
    https://arxiv.org/abs/2211.17230

The bounded Gaussian mechanism uses a truncated normal distribution restricted
to a given domain [lower, upper], ensuring all privatized outputs are valid.
Unlike the standard Gaussian mechanism which has unbounded support (and may
produce invalid values that require post-hoc projection), this mechanism
confines noise to a bounded region from the start.

The functional API provides:
1. ``bounded_gaussian(stddev, bounds)`` -- stateless noise function (recommended)
2. ``bounded_gaussian_stateful(stddev, bounds, seed)`` -- (fn, state) for reproducibility
"""

import math
from collections.abc import Callable

import torch

from opaque.utils.pytree import tree_map

_SQRT2 = math.sqrt(2.0)


def _truncated_normal_around(
    center: torch.Tensor,
    stddev: float,
    lower: float,
    upper: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from N^T(center, stddev^2, [lower, upper]) element-wise.

    For each element *c* in ``center``, draws a sample from the truncated
    Gaussian centred at *c* with support [lower, upper] using the inverse-CDF
    method:

        1. alpha_i = Phi((lower - c_i) / sigma)
        2. beta_i  = Phi((upper - c_i) / sigma)
        3. u_i     ~ Uniform(alpha_i, beta_i)
        4. sample_i = c_i + sigma * sqrt(2) * erfinv(2 * u_i - 1)

    Args:
        center: Centre (mean) of the truncated Gaussian for each element.
        stddev: Standard deviation of the underlying Gaussian.
        lower: Lower bound of the output domain.
        upper: Upper bound of the output domain.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        Tensor of same shape as *center* with values in [lower, upper].
    """
    dtype = center.dtype
    device = center.device

    # Per-element CDF bounds:
    #   alpha_i = Phi((lower - c_i) / sigma)
    #   beta_i  = Phi((upper - c_i) / sigma)
    z_lower = (lower - center) / stddev
    z_upper = (upper - center) / stddev

    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    alpha = 0.5 * (1.0 + torch.erf(z_lower / _SQRT2))
    beta = 0.5 * (1.0 + torch.erf(z_upper / _SQRT2))

    # u ~ Uniform(alpha, beta) element-wise
    u = torch.rand(center.shape, dtype=dtype, device=device, generator=generator)
    u = alpha + u * (beta - alpha)

    # Clamp for numerical safety (avoid +/-inf from erfinv at 0 or 1)
    eps = torch.finfo(dtype).tiny
    u = torch.clamp(u, min=eps, max=1.0 - eps)

    # Inverse CDF: sample = c + sigma * sqrt(2) * erfinv(2u - 1)
    samples = center + stddev * _SQRT2 * torch.erfinv(2.0 * u - 1.0)

    # Hard clamp as a floating-point safety net
    return torch.clamp(samples, min=lower, max=upper)


def bounded_gaussian(
    stddev: float,
    bounds: tuple[float, float],
) -> Callable:
    """Create a stateless bounded Gaussian noise function.

    Returns a function that adds noise from a truncated normal distribution
    centred at each input value, with support restricted to [lower, upper].
    This implements the Bounded Gaussian Mechanism (Chen & Hale, 2024).

    For a gradient element *g*, the noisy output is sampled from
    ``N^T(g, sigma^2, [lower, upper])`` -- a Gaussian centred at *g*,
    truncated to [lower, upper].  This guarantees all outputs lie within the
    valid domain.

    For reproducible noise (e.g., testing, debugging), use
    ``bounded_gaussian_stateful()``.

    Args:
        stddev: Standard deviation of the underlying Gaussian noise
            (usually ``noise_multiplier * clip_norm``).
        bounds: ``(lower, upper)`` bounds for the noisy output domain.
            Must satisfy ``lower < upper``.

    Returns:
        A function ``noise_fn(grads) -> noisy_grads`` where every element of
        the output is guaranteed to lie in [lower, upper].

    Raises:
        ValueError: If ``stddev`` is negative, or bounds are invalid.

    Example:
        >>> import torch
        >>> from opaque.noise import bounded_gaussian
        >>>
        >>> noise_fn = bounded_gaussian(stddev=1.0, bounds=(-3.0, 3.0))
        >>> grads = torch.zeros(1000)
        >>> noisy = noise_fn(grads)
        >>> assert noisy.min() >= -3.0
        >>> assert noisy.max() <= 3.0

    References:
        Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
        Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
        https://arxiv.org/abs/2211.17230
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    lower, upper = bounds
    if lower >= upper:
        raise ValueError(f"bounds must satisfy lower < upper, got ({lower}, {upper})")

    if stddev == 0:
        return lambda grads: tree_map(
            lambda t: torch.clamp(t, min=lower, max=upper), grads
        )

    def noise_fn(grads):
        """Add bounded Gaussian noise to gradients."""

        def add_bounded_noise(tensor: torch.Tensor) -> torch.Tensor:
            return _truncated_normal_around(
                tensor,
                stddev=stddev,
                lower=lower,
                upper=upper,
            )

        return tree_map(add_bounded_noise, grads)

    return noise_fn


def bounded_gaussian_stateful(
    stddev: float,
    bounds: tuple[float, float],
    seed: int,
) -> tuple[Callable, torch.Generator]:
    """Create a bounded Gaussian noise function with explicit state management.

    Returns a tuple ``(noise_fn, state)`` where *state* is a
    ``torch.Generator`` for reproducible noise.  This follows the functional
    pattern of explicit state passing used throughout Opaque.

    Use this when you need reproducible noise (e.g., for testing, debugging, or
    deterministic training).  For typical use cases, use ``bounded_gaussian()``.

    Args:
        stddev: Standard deviation of the underlying Gaussian noise.
        bounds: ``(lower, upper)`` bounds for the noisy output domain.
            Must satisfy ``lower < upper``.
        seed: Random seed for the PRNG state.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> noisy_grads``
        - ``state`` is a ``torch.Generator`` initialised with *seed*

    Raises:
        ValueError: If ``stddev`` is negative, or bounds are invalid.

    Example:
        >>> import torch
        >>> from opaque.noise import bounded_gaussian_stateful
        >>>
        >>> noise_fn, state = bounded_gaussian_stateful(
        ...     stddev=1.0, bounds=(-3.0, 3.0), seed=42
        ... )
        >>> grads = torch.zeros(100)
        >>> noisy = noise_fn(grads, state)
        >>> assert noisy.min() >= -3.0 and noisy.max() <= 3.0

    References:
        Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
        Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
        https://arxiv.org/abs/2211.17230
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    lower, upper = bounds
    if lower >= upper:
        raise ValueError(f"bounds must satisfy lower < upper, got ({lower}, {upper})")

    state = torch.Generator().manual_seed(seed)

    if stddev == 0:
        return (
            lambda grads, gen: tree_map(
                lambda t: torch.clamp(t, min=lower, max=upper), grads
            ),
            state,
        )

    def noise_fn(grads, generator: torch.Generator):
        """Add bounded Gaussian noise using the provided generator."""

        def add_bounded_noise(tensor: torch.Tensor) -> torch.Tensor:
            return _truncated_normal_around(
                tensor,
                stddev=stddev,
                lower=lower,
                upper=upper,
                generator=generator,
            )

        return tree_map(add_bounded_noise, grads)

    return noise_fn, state


__all__ = ["bounded_gaussian", "bounded_gaussian_stateful"]
