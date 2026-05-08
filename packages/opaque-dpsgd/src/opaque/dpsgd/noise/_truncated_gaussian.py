"""Truncated Gaussian noise mechanism for differential privacy.

This module implements the truncated Gaussian noise mechanism, based on the
Bounded Gaussian Mechanism from:

    Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for Differential
    Privacy," Journal of Privacy and Confidentiality, 14(1), 2024.
    https://arxiv.org/abs/2211.17230

The mechanism uses a truncated normal distribution restricted to the symmetric
domain [−radius·stddev, radius·stddev], ensuring all privatized outputs stay
clipped.  Unlike the standard Gaussian mechanism which has unclipped support
(and may produce invalid values that require post-hoc projection), this
mechanism confines noise to a clipped region from the start.

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.random import key
    >>> from opaque.clipping.types import clipped
    >>> noise_fn, state = truncated_gaussian_noise(
    ...     noise_multiplier=1.0, radius=5.0, key=key(42)
    ... )
    >>> noisy_grads, state = noise_fn(clipped(grads, max_norm=1.0), state)

References:
    Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
    Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
    https://arxiv.org/abs/2211.17230
"""

import math
from collections.abc import Callable
from typing import Any

import torch

from opaque.types import (
    ClippedPytree,
    NoisedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

from opaque.dpsgd.noise._gaussian import GaussianNoiseState
from opaque.noise_allocation import paired_noise_stddevs, per_group_noise_stddev
from opaque.random import generator_from_key
from opaque.random.types import RngKey
from opaque.random import fold_in as rng_fold_in
from opaque.pytree import tree_map

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
    truncated normal distribution centred at each clipped input value. The
    realized standard deviation is derived from the input
    :class:`opaque.types.ClippedPytree` metadata and carried by the returned
    :class:`opaque.types.NoisedPytree`.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank.

    Args:
        noise_multiplier: Gaussian noise multiplier. The realized standard
            deviation is derived from ``noise_multiplier`` and the input max_norm.
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

        - ``noise_fn(clipped_grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.dpsgd.noise._gaussian.GaussianNoiseState`

    Raises:
        ValueError: If ``noise_multiplier`` or the realized max_norm-derived
            standard deviation is negative, or ``radius`` is not positive.

    Example:
        >>> import torch
        >>> from opaque.clipping.types import clipped
        >>> from opaque.dpsgd.noise import truncated_gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = truncated_gaussian_noise(
        ...     noise_multiplier=1.0, radius=3.0, key=key(42),
        ... )
        >>> grads = torch.zeros(100)
        >>> noised, state = noise_fn(clipped(grads, max_norm=1.0), state)
        >>> assert noised.pytree.min() >= -3.0 and noised.pytree.max() <= 3.0

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

    def _clipped_stddev(clipped: ClippedPytree) -> float | PerGroup:
        if isinstance(clipped.max_norm, PerGroup):
            return per_group_noise_stddev(clipped.max_norm, resolved_noise_multiplier)
        effective = resolved_noise_multiplier * clipped.max_norm
        _validate_truncated_stddev(effective)
        return effective

    def _add_truncated(
        tensor: torch.Tensor, std: float, generator: torch.Generator
    ) -> torch.Tensor:
        if std == 0:
            bound = 0.0
            return torch.clamp(tensor, min=-bound, max=bound)
        bound = std * radius
        return _truncated_normal_around(
            tensor,
            stddev=std,
            lower=-bound,
            upper=bound,
            generator=generator,
        )

    def _add_truncated_tree(
        grads: Any,
        stddev: float | PerGroup,
        generator: torch.Generator,
    ) -> Any:
        """Apply truncated Gaussian noise; dispatch on scalar vs PerGroup.

        For ``σ_g = 0`` the truncated support collapses to ``{0}``; we
        match the scalar zero-σ path and clamp the per-group tensor to 0
        rather than returning it unchanged (which would violate the
        bounded-support guarantee).
        """
        if isinstance(stddev, PerGroup):
            noised: dict[str, torch.Tensor] = {}
            for param_key, tensor in grads.items():
                group_std = stddev.for_key(param_key)
                if group_std == 0:
                    noised[param_key] = torch.clamp(tensor, min=-0.0, max=0.0)
                    continue
                bound = group_std * radius
                noised[param_key] = _truncated_normal_around(
                    tensor,
                    stddev=group_std,
                    lower=-bound,
                    upper=bound,
                    generator=generator,
                )
            return noised
        return tree_map(lambda t: _add_truncated(t, stddev, generator), grads)

    def _paired_stddevs(
        first_clipped: ClippedPytree,
        second_clipped: ClippedPytree,
    ) -> tuple[float | PerGroup, float | PerGroup]:
        """Resolve (σ_first, σ_second) for the paired truncated-Gaussian release.

        Routes through :func:`paired_noise_stddevs`, which implements the
        sensitivity-proportional Mahalanobis allocation that satisfies the
        joint privacy budget with equality (joint PLD = single Gaussian
        release at ``noise_multiplier``).  Both streams must carry the
        same kind of ``max_norm`` (both scalar or both PerGroup); mixed
        kinds are a configuration error.
        """
        return paired_noise_stddevs(
            resolved_noise_multiplier,
            first=first_clipped.max_norm,
            second=second_clipped.max_norm,
        )

    def _add_paired(
        clipped_input: SecondMomentClippingOutput, st: GaussianNoiseState
    ) -> tuple[SecondMomentNoiseOutput, GaussianNoiseState]:
        first_clipped = clipped_input.grads
        second_clipped = clipped_input.squared_grads
        if not isinstance(first_clipped, ClippedPytree):
            raise TypeError("SecondMomentClippingOutput.grads must be a ClippedPytree.")
        if not isinstance(second_clipped, ClippedPytree):
            raise TypeError(
                "SecondMomentClippingOutput.squared_grads must be a ClippedPytree."
            )
        first_stddev, second_stddev = _paired_stddevs(first_clipped, second_clipped)
        # Two independent noise streams; fold-in 1 / 2 namespaces them so
        # they don't collide with the single-stream key derivation.
        first_step_key = rng_fold_in(rng_fold_in(st._rng_key, 1), st._step_counter)
        second_step_key = rng_fold_in(rng_fold_in(st._rng_key, 2), st._step_counter)
        first_gen = generator_from_key(first_step_key)
        second_gen = generator_from_key(second_step_key)
        noisy_grads = _add_truncated_tree(first_clipped.pytree, first_stddev, first_gen)
        noisy_squared = _add_truncated_tree(
            second_clipped.pytree, second_stddev, second_gen
        )
        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )
        return (
            SecondMomentNoiseOutput(
                NoisedPytree(
                    pytree=noisy_grads,
                    max_norm=first_clipped.max_norm,
                    noise_stddev=first_stddev,
                ),
                NoisedPytree(
                    pytree=noisy_squared,
                    max_norm=second_clipped.max_norm,
                    noise_stddev=second_stddev,
                ),
            ),
            next_state,
        )

    def noise_fn(grads, st):
        """Add truncated Gaussian noise to a clipped pytree (or paired stream)."""
        if isinstance(grads, SecondMomentClippingOutput):
            return _add_paired(grads, st)

        if isinstance(grads, NoisedPytree):
            raise TypeError(
                "truncated_gaussian_noise expects ClippedPytree inputs, not "
                "NoisedPytree values that have already passed through a noise "
                "mechanism."
            )
        if not isinstance(grads, ClippedPytree):
            raise TypeError(
                "truncated_gaussian_noise expects ClippedPytree inputs. Wrap "
                "manual values with opaque.types.clipped(...)."
            )

        effective_stddev = _clipped_stddev(grads)
        _validate_truncated_stddev(effective_stddev)

        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )

        # Per-group noise path: per-key zero-σ short-circuits to ±0 to
        # match the scalar zero-σ branch (truncated support collapses to
        # ``{0}`` when σ=0).
        if isinstance(effective_stddev, PerGroup):
            step_key = rng_fold_in(st._rng_key, st._step_counter)
            g = generator_from_key(step_key)

            noised = {}
            for param_key, tensor in grads.pytree.items():
                group_std = effective_stddev.for_key(param_key)
                if group_std == 0:
                    noised[param_key] = torch.clamp(tensor, min=-0.0, max=0.0)
                    continue
                max_norm = group_std * radius
                noised[param_key] = _truncated_normal_around(
                    tensor,
                    stddev=group_std,
                    lower=-max_norm,
                    upper=max_norm,
                    generator=g,
                )

            return NoisedPytree(
                pytree=noised,
                max_norm=grads.max_norm,
                noise_stddev=effective_stddev,
            ), next_state

        # Global (scalar) noise path
        max_norm = effective_stddev * radius

        if effective_stddev == 0:
            noised = tree_map(
                lambda t: torch.clamp(t, min=-max_norm, max=max_norm), grads.pytree
            )
            return NoisedPytree(
                pytree=noised,
                max_norm=grads.max_norm,
                noise_stddev=effective_stddev,
            ), next_state

        step_key = rng_fold_in(st._rng_key, st._step_counter)
        g = generator_from_key(step_key)

        def add_clipped_noise(tensor: torch.Tensor) -> torch.Tensor:
            return _truncated_normal_around(
                tensor,
                stddev=effective_stddev,
                lower=-max_norm,
                upper=max_norm,
                generator=g,
            )

        noised = tree_map(add_clipped_noise, grads.pytree)

        return NoisedPytree(
            pytree=noised,
            max_norm=grads.max_norm,
            noise_stddev=effective_stddev,
        ), next_state

    return noise_fn, state


__all__ = ["truncated_gaussian_noise"]
