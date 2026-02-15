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

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> noise_fn, state = bounded_gaussian_noise(stddev=1.0, bounds=(-3.0, 3.0), generator=42)
    >>> noisy_grads, state = noise_fn(grads, state)

References:
    Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
    Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
    https://arxiv.org/abs/2211.17230
"""

import math
from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.gaussian_noise import GaussianNoiseState, _resolve_generator
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

    z_lower = (lower - center) / stddev
    z_upper = (upper - center) / stddev

    alpha = 0.5 * (1.0 + torch.erf(z_lower / _SQRT2))
    beta = 0.5 * (1.0 + torch.erf(z_upper / _SQRT2))

    u = torch.rand(center.shape, dtype=dtype, device=device, generator=generator)
    u = alpha + u * (beta - alpha)

    eps = torch.finfo(dtype).tiny
    u = torch.clamp(u, min=eps, max=1.0 - eps)

    samples = center + stddev * _SQRT2 * torch.erfinv(2.0 * u - 1.0)

    return torch.clamp(samples, min=lower, max=upper)


def bounded_gaussian_noise(
    stddev: float,
    bounds: tuple[float, float],
    *,
    generator: None | int | torch.Generator = None,
) -> tuple[
    Callable[[Any, GaussianNoiseState], tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a bounded Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds noise from a
    truncated normal distribution centred at each input value, with support
    restricted to [lower, upper]. This implements the Bounded Gaussian
    Mechanism (Chen & Hale, 2024).

    Args:
        stddev: Standard deviation of the underlying Gaussian noise
            (usually ``noise_multiplier * clip_norm``).
        bounds: ``(lower, upper)`` bounds for the noisy output domain.
            Must satisfy ``lower < upper``.
        generator: RNG configuration:
            - ``None``: new unseeded generator (non-reproducible)
            - ``int``: seeded generator (reproducible)
            - ``torch.Generator``: use directly

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.gaussian_noise.GaussianNoiseState`

    Raises:
        ValueError: If ``stddev`` is negative, or bounds are invalid.

    Example:
        >>> import torch
        >>> from opaque.noise import bounded_gaussian_noise
        >>>
        >>> noise_fn, state = bounded_gaussian_noise(
        ...     stddev=1.0, bounds=(-3.0, 3.0), generator=42,
        ... )
        >>> grads = torch.zeros(100)
        >>> noisy, state = noise_fn(grads, state)
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

    gen = _resolve_generator(generator)
    state = GaussianNoiseState(rng_state=gen)

    if stddev == 0:

        def zero_noise_fn(grads, st):
            return tree_map(lambda t: torch.clamp(t, min=lower, max=upper), grads), st

        return zero_noise_fn, state

    def noise_fn(grads, st):
        """Add bounded Gaussian noise to gradients."""
        g = st.rng_state

        def add_bounded_noise(tensor: torch.Tensor) -> torch.Tensor:
            return _truncated_normal_around(
                tensor,
                stddev=stddev,
                lower=lower,
                upper=upper,
                generator=g,
            )

        noisy = tree_map(add_bounded_noise, grads)
        return noisy, GaussianNoiseState(rng_state=g)

    return noise_fn, state


__all__ = ["bounded_gaussian_noise"]
