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

    >>> from opaque.random import key
    >>> from opaque.bounded import bounded
    >>> noise_fn, state = truncated_gaussian_noise(
    ...     noise_multiplier=1.0, radius=5.0, key=key(42)
    ... )
    >>> noisy_grads, state = noise_fn(bounded(grads, bound=1.0), state)

References:
    Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
    Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
    https://arxiv.org/abs/2211.17230
"""

import math
from collections.abc import Callable
from typing import Any

import torch

from opaque.bounded import BoundedPytree, NoisyPytree
from opaque.dpsgd.noise.gaussian import GaussianNoiseState
from opaque.dpsgd.noise.per_group_noise import per_group_noise_stddev
from opaque.random import RngKey, generator_from_key
from opaque.random import fold_in as rng_fold_in
from opaque.clipping.per_group import PerGroup
from opaque.core.pytree import tree_map

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


def _validate_truncated_stddev(noise_stddev: float | PerGroup) -> None:
    """Validate that realized noise stddev is non-negative."""
    if isinstance(noise_stddev, PerGroup):
        for gname, val in noise_stddev.values.items():
            if val < 0:
                raise ValueError(
                    f"noise standard deviation must be non-negative for all groups, "
                    f"got {val} for group '{gname}'"
                )
    else:
        if noise_stddev < 0:
            raise ValueError(
                f"noise standard deviation must be non-negative, got {noise_stddev}"
            )


def _resolve_noise_multiplier(noise_multiplier: float | None) -> float:
    if noise_multiplier is None:
        raise ValueError("truncated_gaussian_noise() requires noise_multiplier.")
    multiplier = float(noise_multiplier)
    if multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    return multiplier


def truncated_gaussian_noise(
    *,
    noise_multiplier: float,
    key: RngKey,
    radius: float = 3.0,
) -> tuple[
    Callable[..., tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a truncated Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds noise from a
    truncated normal distribution centred at each bounded input value. The
    realized standard deviation is derived from the input
    :class:`opaque.bounded.BoundedPytree` metadata and carried by the returned
    :class:`opaque.bounded.NoisyPytree`.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank.

    Args:
        noise_multiplier: Gaussian noise multiplier. The realized standard
            deviation is derived from ``noise_multiplier`` and the input bound.
            For ``PerGroup`` bounds, the same MSE-optimal allocation used by
            :func:`gaussian_noise` is applied.
        radius: Truncation radius in units of standard deviations.
            Noise is truncated to [−radius·stddev, radius·stddev].
            Must be positive. Typical values: 3–10.
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(bounded_grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.dpsgd.noise.gaussian.GaussianNoiseState`

    Raises:
        ValueError: If ``noise_multiplier`` or the realized bound-derived
            standard deviation is negative, or ``radius`` is not positive.

    Example:
        >>> import torch
        >>> from opaque.bounded import bounded
        >>> from opaque.dpsgd.noise.truncated_gaussian import truncated_gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = truncated_gaussian_noise(
        ...     noise_multiplier=1.0, radius=3.0, key=key(42),
        ... )
        >>> grads = torch.zeros(100)
        >>> noisy, state = noise_fn(bounded(grads, bound=1.0), state)
        >>> assert noisy.pytree.min() >= -3.0 and noisy.pytree.max() <= 3.0

    References:
        Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
        Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
        https://arxiv.org/abs/2211.17230
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)

    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    state = GaussianNoiseState(
        _step_counter=0,
        _rng_key=key,
    )

    def _bounded_stddev(bounded: BoundedPytree) -> float | PerGroup:
        if isinstance(bounded.bound, PerGroup):
            return per_group_noise_stddev(bounded.bound, resolved_noise_multiplier)
        effective = resolved_noise_multiplier * bounded.bound
        _validate_truncated_stddev(effective)
        return effective

    def noise_fn(grads, st):
        """Add truncated Gaussian noise to a bounded pytree."""
        if isinstance(grads, NoisyPytree):
            raise TypeError(
                "truncated_gaussian_noise expects BoundedPytree inputs, not "
                "NoisyPytree values that have already passed through a noise "
                "mechanism."
            )
        if not isinstance(grads, BoundedPytree):
            raise TypeError(
                "truncated_gaussian_noise expects BoundedPytree inputs. Wrap "
                "manual values with opaque.bounded.bounded(...)."
            )

        effective_stddev = _bounded_stddev(grads)
        _validate_truncated_stddev(effective_stddev)

        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )

        # Per-group noise path
        if isinstance(effective_stddev, PerGroup):
            if all(v == 0 for v in effective_stddev.values.values()):
                return NoisyPytree(
                    pytree=grads.pytree,
                    bound=grads.bound,
                    noise_stddev=effective_stddev,
                ), next_state

            step_key = rng_fold_in(st._rng_key, st._step_counter)
            g = generator_from_key(step_key)

            noisy = {}
            for param_key, tensor in grads.pytree.items():
                group_std = effective_stddev.for_key(param_key)
                bound = group_std * radius
                noisy[param_key] = _truncated_normal_around(
                    tensor,
                    stddev=group_std,
                    lower=-bound,
                    upper=bound,
                    generator=g,
                )

            return NoisyPytree(
                pytree=noisy,
                bound=grads.bound,
                noise_stddev=effective_stddev,
            ), next_state

        # Global (scalar) noise path
        bound = effective_stddev * radius

        if effective_stddev == 0:
            noisy = tree_map(
                lambda t: torch.clamp(t, min=-bound, max=bound), grads.pytree
            )
            return NoisyPytree(
                pytree=noisy,
                bound=grads.bound,
                noise_stddev=effective_stddev,
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

        noisy = tree_map(add_bounded_noise, grads.pytree)

        return NoisyPytree(
            pytree=noisy,
            bound=grads.bound,
            noise_stddev=effective_stddev,
        ), next_state

    return noise_fn, state


__all__ = ["truncated_gaussian_noise"]
