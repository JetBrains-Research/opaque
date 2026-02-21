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

    >>> from opaque.random import key
    >>> noise_fn, state = bounded_gaussian_noise(stddev=1.0, bounds=(-3.0, 3.0), key=key(42))
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

from opaque.noise.gaussian_noise import GaussianNoiseState, _create_rng_state
from opaque.random import RngKey, generator_from_key
from opaque.random import fold_in as rng_fold_in
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

    # Move center to CPU for computation with generator (CPU-only)
    center_cpu = center.cpu()

    z_lower = (lower - center_cpu) / stddev
    z_upper = (upper - center_cpu) / stddev

    alpha = 0.5 * (1.0 + torch.erf(z_lower / _SQRT2))
    beta = 0.5 * (1.0 + torch.erf(z_upper / _SQRT2))

    # Generate random values on CPU with generator, then move to device
    u = torch.rand(center_cpu.shape, dtype=dtype, generator=generator)
    u = alpha + u * (beta - alpha)

    eps = torch.finfo(dtype).tiny
    u = torch.clamp(u, min=eps, max=1.0 - eps)

    samples = center_cpu + stddev * _SQRT2 * torch.erfinv(2.0 * u - 1.0)
    samples = torch.clamp(samples, min=lower, max=upper)

    # Move result back to original device
    return samples.to(device=device)


def bounded_gaussian_noise(
    stddev: float,
    bounds: tuple[float, float],
    *,
    key: RngKey,
    synchronized: str | bool = "auto",
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
        key: Explicit RNG key for deterministic, functional randomness.
        synchronized: Synchronization mode for distributed training:
            - ``"auto"`` (default): Auto-detect and sync if distributed
            - ``True``: Force synchronized noise (same seed across devices)
            - ``False``: Independent noise per device (seed + rank offset)

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.gaussian_noise.GaussianNoiseState`

    Raises:
        ValueError: If ``stddev`` is negative, or bounds are invalid.

    Example:
        >>> import torch
        >>> from opaque.noise import bounded_gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = bounded_gaussian_noise(
        ...     stddev=1.0, bounds=(-3.0, 3.0), key=key(42),
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

    base_key, resolved_seed, is_sync = _create_rng_state(key, synchronized)
    state = GaussianNoiseState(
        seed=resolved_seed,
        synchronized=is_sync,
        step_counter=0,
        rng_key=base_key,
    )

    if stddev == 0:

        def zero_noise_fn(grads, st):
            return tree_map(lambda t: torch.clamp(t, min=lower, max=upper), grads), st

        return zero_noise_fn, state

    def noise_fn(grads, st):
        """Add bounded Gaussian noise to gradients."""
        step_key = rng_fold_in(st.rng_key, st.step_counter)
        g = generator_from_key(step_key)

        def add_bounded_noise(tensor: torch.Tensor) -> torch.Tensor:
            return _truncated_normal_around(
                tensor,
                stddev=stddev,
                lower=lower,
                upper=upper,
                generator=g,
            )

        noisy = tree_map(add_bounded_noise, grads)

        # Return updated state with incremented step counter
        return noisy, GaussianNoiseState(
            seed=st.seed,
            synchronized=st.synchronized,
            step_counter=st.step_counter + 1,
            rng_key=st.rng_key,
        )

    return noise_fn, state


__all__ = ["bounded_gaussian_noise"]
