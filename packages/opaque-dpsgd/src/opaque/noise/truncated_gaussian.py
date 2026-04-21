"""Truncated Gaussian noise mechanism for differential privacy.

This module implements the truncated Gaussian noise mechanism, based on the
Bounded Gaussian Mechanism from:

    Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for Differential
    Privacy," Journal of Privacy and Confidentiality, 14(1), 2024.
    https://arxiv.org/abs/2211.17230

The mechanism uses a truncated normal distribution restricted to the symmetric
domain [−radius·stddev, radius·stddev], ensuring all privatized outputs stay
bounded.  Unlike the standard Gaussian mechanism which has unbounded support
(and may produce invalid values that require post-hoc projection), this
mechanism confines noise to a bounded region from the start.

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.core.random import key
    >>> noise_fn, state = truncated_gaussian_noise(stddev=1.0, radius=5.0, key=key(42))
    >>> noisy_grads, state = noise_fn(grads, state)

The ``stddev`` can be overridden per call for adaptive clipping:

    >>> noisy_grads, state = noise_fn(grads, state, stddev=new_stddev)

References:
    Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
    Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
    https://arxiv.org/abs/2211.17230
"""

import math
from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.gaussian import GaussianNoiseState
from opaque.core.random import RngKey, generator_from_key
from opaque.core.random import fold_in as rng_fold_in
from opaque.core.utils.per_group import PerGroup
from opaque.core.utils.pytree import tree_map

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


def _validate_truncated_stddev(stddev: float | PerGroup) -> None:
    """Validate that stddev is non-negative (scalar or per-group)."""
    if isinstance(stddev, PerGroup):
        for gname, val in stddev.values.items():
            if val < 0:
                raise ValueError(
                    f"stddev must be non-negative for all groups, "
                    f"got {val} for group '{gname}'"
                )
    else:
        if stddev < 0:
            raise ValueError(f"stddev must be non-negative, got {stddev}")


def truncated_gaussian_noise(
    stddev: float | PerGroup,
    radius: float = 3.0,
    *,
    key: RngKey,
) -> tuple[
    Callable[..., tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a truncated Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds noise from a
    truncated normal distribution centred at each input value, with support
    restricted to [−radius·stddev, radius·stddev]. This implements the
    Bounded Gaussian Mechanism (Chen & Hale, 2024).

    The ``stddev`` provided here is the default. It can be overridden on each
    call via ``noise_fn(grads, state, stddev=new_stddev)`` — useful when the
    noise scale changes between steps (e.g., with adaptive clipping). The
    ``radius`` (in σ-units) is fixed at creation and the truncation bounds
    adjust automatically: [−radius·stddev, radius·stddev].

    When ``stddev`` is a :class:`~opaque.utils.per_group.PerGroup`, each
    parameter receives noise scaled by its group's stddev value, with
    per-group truncation bounds [−radius·σ_g, radius·σ_g].

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank.

    Args:
        stddev: Standard deviation of the underlying Gaussian noise
            (usually ``noise_multiplier * clip_state.sensitivity``).
            When ``PerGroup``, each parameter group gets its own noise scale
            and truncation bounds.
        radius: Truncation radius in units of standard deviations.
            Noise is truncated to [−radius·stddev, radius·stddev].
            Must be positive. Typical values: 3–10.
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state, *, stddev=None) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.gaussian.GaussianNoiseState`

    Raises:
        ValueError: If ``stddev`` is negative, or ``radius`` is not positive.

    Example:
        >>> import torch
        >>> from opaque.noise.truncated_gaussian import truncated_gaussian_noise
        >>> from opaque.core.random import key
        >>>
        >>> noise_fn, state = truncated_gaussian_noise(
        ...     stddev=1.0, radius=3.0, key=key(42),
        ... )
        >>> grads = torch.zeros(100)
        >>> noisy, state = noise_fn(grads, state)
        >>> assert noisy.min() >= -3.0 and noisy.max() <= 3.0

    Example (per-call override for adaptive clipping):
        >>> noise_fn, state = truncated_gaussian_noise(stddev=1.0, radius=5.0, key=key(42))
        >>> noisy, state = noise_fn(grads, state, stddev=0.8)  # bounds become ±4.0

    References:
        Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
        Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
        https://arxiv.org/abs/2211.17230
    """
    _validate_truncated_stddev(stddev)

    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    state = GaussianNoiseState(
        _step_counter=0,
        _rng_key=key,
    )

    default_stddev = stddev

    def noise_fn(grads, st, *, stddev=None):
        """Add truncated Gaussian noise to gradients."""
        effective_stddev = stddev if stddev is not None else default_stddev
        _validate_truncated_stddev(effective_stddev)

        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )

        # Per-group noise path
        if isinstance(effective_stddev, PerGroup):
            if all(v == 0 for v in effective_stddev.values.values()):
                return grads, next_state

            step_key = rng_fold_in(st._rng_key, st._step_counter)
            g = generator_from_key(step_key)

            noisy = {}
            for param_key, tensor in grads.items():
                group_std = effective_stddev.for_key(param_key)
                bound = group_std * radius
                noisy[param_key] = _truncated_normal_around(
                    tensor,
                    stddev=group_std,
                    lower=-bound,
                    upper=bound,
                    generator=g,
                )

            return noisy, next_state

        # Global (scalar) noise path
        bound = effective_stddev * radius

        if effective_stddev == 0:
            return tree_map(
                lambda t: torch.clamp(t, min=-bound, max=bound), grads
            ), next_state

        step_key = rng_fold_in(st._rng_key, st._step_counter)
        g = generator_from_key(step_key)

        def add_bounded_noise(tensor: torch.Tensor) -> torch.Tensor:
            return _truncated_normal_around(
                tensor,
                stddev=effective_stddev,
                lower=-bound,
                upper=bound,
                generator=g,
            )

        noisy = tree_map(add_bounded_noise, grads)

        return noisy, next_state

    return noise_fn, state


__all__ = ["truncated_gaussian_noise"]
